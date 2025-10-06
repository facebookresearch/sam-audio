import torch
import torchaudio
from audiobox.models.clap_models.laion_clap_model import laion_clap

from sam_audio.model.config import ClapRankerConfig
from sam_audio.ranking.ranker import Ranker

try:
    from laion_clap.training.data import (
        float32_to_int16_torch,
        get_audio_features,
        int16_to_float32_torch,
    )

    __laion_clap_exists__ = True
except ImportError:
    get_audio_features = int16_to_float32_torch = float32_to_int16_torch = lambda x: x
    __laion_clap_exists__ = False


class ClapRanker(Ranker):
    def __init__(self, config: ClapRankerConfig):
        if not __laion_clap_exists__:
            raise ImportError(
                'Install reranking dependencies: `pip install "sam-audio[reranking]"`'
            )
        super().__init__()
        self.config = config
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        self.model.load_ckpt(ckpt=config.checkpoint, model_id=0)
        self.model = self.model.eval()

    def _prepare_audio(self, audio, sample_rate):
        audio_features = []
        for candidates in audio:
            if sample_rate != 48_000:
                candidates = torchaudio.functional.resample(
                    candidates, sample_rate, 48000
                )

            quantized = int16_to_float32_torch(
                float32_to_int16_torch(candidates)
            ).float()
            for sample in quantized:
                temp_dict = {}
                temp_dict = get_audio_features(
                    temp_dict,
                    sample,
                    480000,
                    data_truncating=(
                        "fusion" if self.model.enable_fusion else "rand_trunc"
                    ),
                    data_filling="repeatpad",
                    audio_cfg=self.model.model_cfg["audio_cfg"],
                    require_grad=False,
                )
                audio_features.append(temp_dict)
        return audio_features

    @torch.inference_mode()
    def forward(
        self,
        extracted_audio: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        audio_embed = self.model.model.get_audio_embedding(
            self._prepare_audio(extracted_audio, sample_rate)
        )
        text_embed = self.model.get_text_embedding(descriptions, use_tensor=True)
        bsz = len(extracted_audio)
        candidates = len(audio_embed) // bsz
        audio_embed = audio_embed.reshape(bsz, candidates, -1)
        text_embed = text_embed.reshape(bsz, -1, 1)
        scores = audio_embed @ text_embed
        return scores.squeeze(-1)
