from typing import Optional

import torch
from audiobox_aesthetics.infer import AesPredictor

COLUMN_MAP = {
    "CE": "ContentEnjoyment",
    "CU": "ContentUsefulness",
    "PC": "ProductionComplexity",
    "PQ": "ProductionQuality",
}


class Aesthetic(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = AesPredictor(
            checkpoint_pth=checkpoint,
            data_col="wav",
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __call__(
        self,
        target_wavs: list[torch.Tensor],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> dict[str, list[float]]:
        result = self.model.forward(
            [
                {"wav": wav[None], "sample_rate": target_wavs_sample_rate}
                for wav in target_wavs
            ]
        )
        return {
            long_name: [x[shortname] for x in result]
            for shortname, long_name in COLUMN_MAP.items()
        }
