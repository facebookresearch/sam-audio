from typing import Optional

import torch
from transformers import SamAudioJudgeModel, SamAudioJudgeProcessor


class Judge(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = "facebook/sam-audio-judge",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = SamAudioJudgeModel.from_pretrained(checkpoint).to(device)
        self.processor = SamAudioJudgeProcessor.from_pretrained(checkpoint)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(
        self,
        input_wavs: list[torch.Tensor],
        target_wavs: list[torch.Tensor],
        descriptions: list[str],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            processed = self.processor(
                text=descriptions,
                input_audio=[x.cpu().numpy() for x in input_wavs],
                separated_audio=[x.cpu().numpy() for x in target_wavs],
                padding=True,
                return_tensors="pt",
                sampling_rate=target_wavs_sample_rate,
            ).to(self.device)
            result = self.model(**processed)
            return {
                "JudgeOverall": result.overall.squeeze(-1).cpu().tolist(),
                "JudgeFaithfulness": result.faithfulness.squeeze(-1).cpu().tolist(),
                "JudgeRecall": result.recall.squeeze(-1).cpu().tolist(),
                "JudgePrecision": result.precision.squeeze(-1).cpu().tolist(),
            }
