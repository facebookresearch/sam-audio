from tempfile import TemporaryDirectory
from typing import Optional

import laion_clap
import torch
import torchaudio


class CLAP(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False).to(device)
        self.model.load_ckpt(ckpt=checkpoint)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def __call__(
        self,
        target_wavs: list[torch.Tensor],
        descriptions: list[str],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> list[dict[str, float]]:
        with TemporaryDirectory() as tdir, torch.inference_mode():
            file_list = []
            for i, wav in enumerate(target_wavs):
                file_list.append(f"{tdir}/hyp_{i}.wav")
                torchaudio.save(file_list[-1], wav.cpu()[None], target_wavs_sample_rate)
            audio_embs = self.model.get_audio_embedding_from_filelist(
                file_list, use_tensor=True
            )

            text_embs = self.model.get_text_embedding(descriptions, use_tensor=True)
            sims = audio_embs.unsqueeze(1) @ text_embs.unsqueeze(2)
            return {"CLAPSimilarity": sims.cpu()[:, 0, 0].tolist()}
