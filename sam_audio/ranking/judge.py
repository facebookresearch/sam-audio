import torch

from sam_audio.model.config import JudgeRankerConfig
from sam_audio.ranking.ranker import Ranker

try:
    from transformers import SamAudioJudgeModel, SamAudioJudgeProcessor

    __sam_audio_judge_exists__ = True
except ImportError:
    get_audio_features = int16_to_float32_torch = float32_to_int16_torch = lambda x: x
    __sam_audio_judge_exists__ = False


class JudgeRanker(Ranker):
    def __init__(self, config: JudgeRankerConfig):
        if not __sam_audio_judge_exists__:
            raise ImportError(
                'Install reranking dependencies: `pip install "sam-audio[reranking]"`'
            )
        super().__init__()
        self.config = config
        self.model = SamAudioJudgeModel.from_pretrained(config.checkpoint_or_model_id)
        self.processor = SamAudioJudgeProcessor.from_pretrained(
            config.checkpoint_or_model_id
        )

    @torch.inference_mode()
    def forward(
        self,
        input_audio: list[torch.Tensor],
        extracted_audio: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        bsz, ncandidates = len(input_audio), len(input_audio[0])
        input_seqs = [x.cpu().numpy() for candidates in input_audio for x in candidates]
        extracted_seqs = [
            x.cpu().numpy() for candidates in extracted_audio for x in candidates
        ]
        repeated_descriptions = [x for x in descriptions for _ in range(ncandidates)]
        processed = self.processor(
            text=repeated_descriptions,
            input_audio=input_seqs,
            separated_audio=extracted_seqs,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate,
        )
        res = self.model(**processed.to(input_audio[0].device))
        return res.overall.view(bsz, ncandidates)
