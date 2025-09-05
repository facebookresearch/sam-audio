from dataclasses import dataclass, field
from typing import Optional

SAM_AUDIO_CONFIGS = {}
JUDGE_CONFIGS = {}


@dataclass
class DACVAEConfig:
    encoder_dim: int = 64
    encoder_rates: list[int] = field(default_factory=lambda: [2, 8, 10, 12])
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [12, 10, 8, 2])
    n_codebooks: int = 16
    codebook_size: int = 1024
    codebook_dim: int = 128
    quantizer_dropout: bool = False
    sample_rate: int = 48_000
    mean: float = 0.0
    std: float = 1.0


@dataclass
class T5EncoderConfig:
    name: str = "t5-base"
    max_length: Optional[int] = 512
    pad_mode: str = "longest"
    dim: int = 768


@dataclass
class VisionEncoderConfig:
    name: str = "ViT-H-14"
    dim: int = 1024
    resize_type = "aspect_variant"
    batch_size: int = 300
    normalize_features: bool = True


@dataclass
class TransformerConfig:
    dim: int = 2048
    n_heads: int = 16
    n_layers: int = 16
    dropout: float = 0.1
    norm_eps: float = 1.0e-05
    qk_norm: bool = True
    fc_bias: bool = False
    ffn_exp: int = 4
    ffn_dim_multiplier: int = 1
    multiple_of: int = 64
    non_linearity: str = "swiglu"
    use_rope: bool = True
    max_positions: int = 10000
    frequency_embedding_dim: int = 256
    timestep_non_linearity: str = "swiglu"
    t_block_non_linearity: str = "silu"
    t_block_bias: bool = True
    context_dim: int = 2048
    context_non_linearity: str = "swiglu"
    context_embedder_dropout: float = 0.0
    context_norm: bool = False
    out_channels: int = 256
    in_channels: Optional[int] = None
    no_cross_attention: bool = False


@dataclass(kw_only=True)
class SAMAudioConfig:
    in_channels: int = 768
    audio_codec: DACVAEConfig = field(default_factory=DACVAEConfig)
    text_encoder: T5EncoderConfig = field(default_factory=T5EncoderConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    video_feature_dim: int = 1024  # metaclip dim
    num_anchors: int = 3
    anchor_embedding_dim: int = 128


@dataclass
class ModernBERTConfig:
    model_id: str = "answerdotai/ModernBERT-large"
    pad_mode: str = "longest"
    max_length: int = 512
    dim: int = 1024
    nth_layer: int = 22


@dataclass(kw_only=True)
class JudgeConfig:
    audio_codec: DACVAEConfig = field(default_factory=DACVAEConfig)
    text_encoder: ModernBERTConfig = field(default_factory=ModernBERTConfig)
    audio_encoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            dim=1792,
            out_channels=1792,
            in_channels=128,
            n_heads=14,
            n_layers=28,
            no_cross_attention=True,
        )
    )
    finetune_encoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            dim=192,
            out_channels=192,
            in_channels=256,
            n_heads=3,
            n_layers=6,
            no_cross_attention=True,
        )
    )


SAM_AUDIO_CONFIGS["base"] = SAMAudioConfig()
JUDGE_CONFIGS["base"] = JudgeConfig()
