from dataclasses import asdict, dataclass
from typing import Optional

import torch

from sam_audio.inputs import batch_audio, mask_from_sizes
from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import JudgeConfig
from sam_audio.model.text_encoder import ModernBERTEncoder
from sam_audio.model.transformer import Transformer


class MeanPool(torch.nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x has shape B x T x C
        if mask is None:
            return x.mean(dim=1)
        else:
            sizes = mask.sum(-1)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / sizes.unsqueeze(-1)


class Head(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        self.proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.cls_emb = MeanPool()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.cls_emb(self.proj(x), mask)


@dataclass
class JudgeOutput:
    overall: torch.Tensor
    recall: torch.Tensor
    precision: torch.Tensor
    faithfulness: torch.Tensor


class Judge(BaseModel):
    def __init__(self, cfg: JudgeConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = ModernBERTEncoder(cfg.text_encoder)
        self.audio_encoder = Transformer(**asdict(cfg.audio_encoder))
        self.finetune_encoder = Transformer(**asdict(cfg.finetune_encoder))
        self.text_proj = torch.nn.Linear(
            cfg.text_encoder.dim, cfg.audio_encoder.dim, bias=False
        )
        self.cat_audio_proj = torch.nn.Linear(
            2 * cfg.audio_encoder.dim, cfg.finetune_encoder.in_channels
        )
        self.aligner = AlignModalities(
            cfg.audio_encoder.dim, cfg.finetune_encoder.in_channels, with_gate=False
        )
        self.cat_aligned = torch.nn.Linear(
            2 * cfg.finetune_encoder.in_channels, cfg.finetune_encoder.in_channels
        )
        self.head = torch.nn.Linear(cfg.finetune_encoder.out_channels, 5, bias=False)
        self.pool = MeanPool()
        self.mean = torch.nn.Parameter(torch.zeros(4, requires_grad=False))
        self.std = torch.nn.Parameter(torch.ones(4, requires_grad=False))

    def predict(self, input_wavs, hyp_wavs, descriptions):
        text_features = self.text_proj(
            self.text_encoder([x.lower().strip() for x in descriptions])
        )
        device = text_features.device
        hyp_wavs, hyp_sizes = batch_audio(hyp_wavs)
        input_wavs, input_sizes = batch_audio(input_wavs)

        assert (hyp_sizes == input_sizes).all().item(), (
            "Input and separated audio must be the same size"
        )

        mask = mask_from_sizes(self.audio_codec.wav_idx_to_feature_idx(hyp_sizes))
        if mask is not None:
            mask = mask.to(device)

        input_codec_feats, hyp_codec_feats = (
            self.audio_codec(torch.cat([input_wavs, hyp_wavs], dim=0).to(device))
            .transpose(1, 2)
            .chunk(2, 0)
        )
        input_features, _ = self.audio_encoder(input_codec_feats, padding_mask=mask)
        hyp_features, _ = self.audio_encoder(hyp_codec_feats, padding_mask=mask)
        audio_features = self.cat_audio_proj(
            torch.cat([input_features, hyp_features], dim=2)
        )
        aligned = self.aligner(audio_features, text_features.unsqueeze(-1))
        finetune_inp = self.cat_aligned(
            torch.cat([audio_features, aligned.expand_as(audio_features)], dim=2)
        )
        final_features, _ = self.finetune_encoder(finetune_inp, padding_mask=mask)
        result = self.pool(self.head(final_features), mask)[:, :4]
        de_normalized = result * self.std + self.mean
        return JudgeOutput(*de_normalized.chunk(4, dim=1))

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=strict
            )
            # We load this directly from HF, not in checkpoint
            missing_keys = [x for x in missing_keys if not x.startswith("text_encoder")]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )
