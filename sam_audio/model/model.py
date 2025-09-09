import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torchdiffeq import odeint

from sam_audio.inputs import Anchor, prepare_inputs
from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAM_AUDIO_CONFIGS, SAMAudioConfig
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import MetaCLIPEncoder

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 1 / 32}}


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


class SAMAudio(BaseModel):
    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        self.vision_encoder = MetaCLIPEncoder(cfg.vision_encoder)
        self.transformer = DiT(**asdict(cfg.transformer))

        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)

        self.align_video = AlignModalities(cfg.video_feature_dim, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.video_feature_dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)

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
        aligned = self.align_masked_video(aligned, video_mask_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
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

    @torch.inference_mode()
    def separate(
        self,
        audio_paths: list[str],
        descriptions: list[str],
        video_paths: Optional[list[str]] = None,
        video_mask_paths: Optional[list[str]] = None,
        anchors: Optional[list[list[Anchor]]] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
    ) -> SeparationResult:
        assert len(audio_paths) == len(descriptions)
        batch = prepare_inputs(
            descriptions=descriptions,
            audio_paths=audio_paths,
            wav_to_feature_idx=self.audio_codec.wav_idx_to_feature_idx,
            anchors=anchors,
            video_paths=video_paths,
            video_mask_paths=video_mask_paths,
        )
        batch = batch.to(self.device())

        # Encode audio
        audio_features = self.audio_codec(batch.audios).transpose(1, 2)
        B, T, C = audio_features.shape

        audio_features = torch.cat([audio_features, audio_features], dim=2)
        text_features, text_mask = self.text_encoder(batch.descriptions)

        if batch.video is None:
            video_features = audio_features.new_zeros(B, self.vision_encoder.dim, T)
        else:
            video_features = self.vision_encoder(batch.video)

        if batch.video_mask is None:
            video_mask_features = audio_features.new_zeros(
                B, self.vision_encoder.dim, T
            )
        else:
            video_mask_features = self.vision_encoder(batch.video_mask)

        noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            return self.forward(
                noisy_audio=noisy_audio,
                audio_features=audio_features,
                text_features=text_features,
                text_mask=text_mask,
                video_features=video_features,
                video_mask_features=video_mask_features,
                anchor_ids=batch.anchor_ids,
                anchor_alignment=batch.anchor_alignment,
                audio_pad_mask=batch.audio_pad_mask,
                time=t.expand(noisy_audio.size(0)),
            )

        generated_features = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=audio_features.device),
            **ode_opt,
        )[-1].transpose(1, 2)

        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        wavs = self.audio_codec.decode(generated_features.view(2 * B, C, T)).view(
            B, 2, -1
        )
        return SeparationResult(
            target=wavs[:, [0]],
            residual=wavs[:, [1]],
        )

    @classmethod
    def get_configs(cls):
        return SAM_AUDIO_CONFIGS
