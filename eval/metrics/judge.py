from typing import Optional

import torch
from transformers import AutoTokenizer, SamAudioJudgeModel


class Judge(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = "facebook/sam-audio-judge",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = SamAudioJudgeModel.from_pretrained(checkpoint).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(
        self,
        input_wavs: list[torch.Tensor],
        target_wavs: list[torch.Tensor],
        descriptions: list[str],
        **kwargs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            processed = self.tokenizer(
                descriptions, padding=True, return_tensors="pt"
            ).to(self.device)
            result = self.model(
                [x.to(self.device) for x in input_wavs],
                [x.to(self.device) for x in target_wavs],
                **processed,
            )
            return {
                "JudgeOverall": result.overall.squeeze(-1).cpu().tolist(),
                "JudgeFaithfulness": result.faithfulness.squeeze(-1).cpu().tolist(),
                "JudgeRecall": result.recall.squeeze(-1).cpu().tolist(),
                "JudgePrecision": result.precision.squeeze(-1).cpu().tolist(),
            }
