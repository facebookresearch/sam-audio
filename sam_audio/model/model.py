import math
import re
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, Optional

import torch
from torchdiffeq import odeint

from sam_audio.inputs import Batch, prepare_inputs
from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import (
    MetaCLIPConfig,
    PerceptionEncoderConfig,
    SAMAudioConfig,
)
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import MetaCLIPEncoder, PerceptionEncoder
from sam_audio.ranking import create_ranker

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 2 / 32}}


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


class SAMAudio(BaseModel):
    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        if isinstance(cfg.vision_encoder, MetaCLIPConfig):
            self.vision_encoder = MetaCLIPEncoder(cfg.vision_encoder)
        elif isinstance(cfg.vision_encoder, PerceptionEncoderConfig):
            self.vision_encoder = PerceptionEncoder(cfg.vision_encoder)
        else:
            raise ValueError(
                f"Unrecognized vision_encoder config: {type(cfg.vision_encoder)}"
            )
        self.transformer = DiT(**asdict(cfg.transformer))

        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)

        self.align_video = AlignModalities(cfg.vision_encoder.dim, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self.visual_ranker = create_ranker(cfg.visual_ranker)
        self.text_ranker = create_ranker(cfg.text_ranker)

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        video_features: Optional[torch.Tensor] = None,
        video_mask_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_video(projected, video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        aligned = self.align_masked_video(aligned, video_mask_features)
        return aligned

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        video_features: Optional[torch.Tensor] = None,
        video_mask_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            video_features (Optional[torch.Tensor], optional): Video features tensor [B x C x T]
            video_mask_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            video_features=video_features,
            video_mask_features=video_mask_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def get_transform(self):
        return partial(
            prepare_inputs,
            wav_to_feature_idx=self.audio_codec.wav_idx_to_feature_idx,
            feature_to_wav_idx=self.audio_codec.feature_idx_to_wav_idx,
        )

    def _get_audio_features(self, audios: torch.Tensor):
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_video_features(self, video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, self.vision_encoder.dim, T)
        else:
            return self.vision_encoder(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        audio_features = self._get_audio_features(batch.audios)
        text_features, text_mask = self.text_encoder(batch.descriptions)
        video_features = self._get_video_features(batch.video, audio_features)
        video_mask_features = self._get_video_features(batch.video_mask, audio_features)

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "video_features": self._repeat_for_reranking(video_features, candidates),
            "video_mask_features": self._repeat_for_reranking(
                video_mask_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
    ) -> SeparationResult:
        # Encode audio
        forward_args = self._get_forward_args(batch, candidates=reranking_candidates)
        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if noise is None:
            noise = torch.randn_like(audio_features)

        noise = audio_features + noise

        def vector_field(t, noisy_audio):
            res = self.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )
            return res

        states = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=noise.device),
            **ode_opt,
        )
        generated_features = states[-1].transpose(1, 2)
        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
            B, 2, -1
        )

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = self.unbatch_wavs(
            wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
        )
        residual_wavs = self.unbatch_wavs(
            wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
        )

        if (
            reranking_candidates > 1
            and batch.video_paths is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                video_paths=batch.video_paths,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs)],
            residual=[wavs[idx] for wavs, idx in zip(residual_wavs, idxs)],
            noise=noise,
        )

    def unbatch_wavs(self, wavs: torch.Tensor, sizes: torch.Tensor):
        result = []
        for row, size in zip(wavs, sizes):
            result.append(row.narrow(dim=-1, start=0, length=size))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile("(^text_encoder|^visual_ranker|^text_ranker)")
            missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


__all__ = ["SAMAudio"]
