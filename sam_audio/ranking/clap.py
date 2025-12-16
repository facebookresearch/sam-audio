# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from sam_audio.model.config import ClapRankerConfig
from sam_audio.ranking.ranker import Ranker


def get_model(device="cpu"):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny").to(device)
    checkpoint_file = hf_hub_download(
        repo_id="lukewys/laion_clap", filename="630k-best.pt"
    )
    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)[
        "state_dict"
    ]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if "text_branch.embeddings.position_ids" in state_dict:
        del state_dict["text_branch.embeddings.position_ids"]

    model.model.load_state_dict(state_dict)
    return model.eval()


class ClapRanker(Ranker):
    def __init__(self, config: ClapRankerConfig):
        from laion_clap.training import data

        self.laion_data_module = data
        super().__init__()
        self.config = config
        self.model = get_model()

    def _prepare_audio(self, audio, sample_rate):
        audio_features = []
        for candidates in audio:
            if sample_rate != 48_000:
                candidates = torchaudio.functional.resample(
                    candidates, sample_rate, 48000
                )

            quantized = self.laion_data_module.int16_to_float32_torch(
                self.laion_data_module.float32_to_int16_torch(candidates)
            ).float()
            for sample in quantized:
                temp_dict = {}
                temp_dict = self.laion_data_module.get_audio_features(
                    temp_dict,
                    sample,
                    480000,
                    data_truncating=(
                        "fusion" if self.model.model.enable_fusion else "rand_trunc"
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
