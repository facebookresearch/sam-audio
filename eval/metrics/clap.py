# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from tempfile import TemporaryDirectory
from typing import Optional

import torch
import torchaudio

from sam_audio.ranking.clap import get_model


class CLAP(torch.nn.Module):
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = get_model(device)
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
                wav_to_save = wav.cpu()[None] if wav.ndim == 1 else wav.cpu()
                torchaudio.save(file_list[-1], wav_to_save, target_wavs_sample_rate)
            audio_embs = self.model.get_audio_embedding_from_filelist(
                file_list, use_tensor=True
            )

            text_embs = self.model.get_text_embedding(descriptions, use_tensor=True)
            sims = audio_embs.unsqueeze(1) @ text_embs.unsqueeze(2)
            return {"CLAPSimilarity": sims.cpu()[:, 0, 0].tolist()}
