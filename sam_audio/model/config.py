from dataclasses import dataclass, field, is_dataclass
from typing import Optional

_CONFIG_CLASSES = {}


def config(cls=None, **kwargs):
    def wrap(cls):
        _CONFIG_CLASSES[cls.__name__] = cls
        return dataclass(**kwargs)(cls)

    if cls is None:
        return wrap
    return wrap(cls)


def serialize_config(cfg):
    if is_dataclass(cfg):
        cls_name = cfg.__class__.__name__
        return {
            "_target_": cls_name,
            **{k: serialize_config(v) for k, v in cfg.__dict__.items()},
        }
    elif isinstance(cfg, list):
        return [serialize_config(x) for x in cfg]
    return cfg


def deserialize_config(config):
    if isinstance(config, dict):
        target = config.pop("_target_", None)
        res = {k: deserialize_config(v) for k, v in config.items()}
        if target is not None:
            cls = _CONFIG_CLASSES[target]
            res = cls(**res)
        return res
    elif isinstance(config, list):
        return [deserialize_config(x) for x in config]
    else:
        return config


@config(kw_only=True)
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


@config(kw_only=True)
class TextEncoderConfig:
    dim: int = 768


@config(kw_only=True)
class T5EncoderConfig(TextEncoderConfig):
    name: str = "t5-base"
    max_length: Optional[int] = 512
    pad_mode: str = "longest"


@config(kw_only=True)
class ModernBERTConfig(TextEncoderConfig):
    model_id: str = "answerdotai/ModernBERT-large"
    pad_mode: str = "longest"
    max_length: int = 512
    nth_layer: Optional[int] = 22
    dim: int = 1024


@config(kw_only=True)
class VisionEncoderConfig:
    dim: int = 1024
    batch_size: int = 300


@config(kw_only=True)
class MetaCLIPConfig(VisionEncoderConfig):
    dim: int = 1024
    name: str = "ViT-H-14"
    resize_type = "aspect_variant"
    normalize_features: bool = True


@config(kw_only=True)
class PerceptionEncoderConfig(VisionEncoderConfig):
    dim: int = 1024
    name: str = "PE-Core-L14-336"
    normalize_feature: bool = True
    interpolation_mode: str = "BICUBIC"
    image_size: int = 336


@config(kw_only=True)
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
class RankerConfig: ...


@config(kw_only=True)
class ImageBindRankerConfig(RankerConfig):
    checkpoint: Optional[str] = (
        None  # Optional local checkpoint, otherwise download from internet
    )


@config(kw_only=True)
class ClapRankerConfig(RankerConfig):
    checkpoint: Optional[str] = (
        None  # Optional local checkpoint, otherwise download from internet
    )


@config(kw_only=True)
class JudgeRankerConfig(RankerConfig):
    checkpoint_or_model_id: str = "facebook/sam-audio-judge"


@config(kw_only=True)
class EnsembleRankerConfig(RankerConfig):
    rankers: list[RankerConfig] = field(
        default_factory=lambda: [
            ClapRankerConfig(),
            JudgeRankerConfig(),
        ]
    )
    weights: list[float] = field(default_factory=lambda: [5.0, 1.0])


@config(kw_only=True)
class SAMAudioConfig:
    in_channels: int = 768
    audio_codec: DACVAEConfig = field(default_factory=DACVAEConfig)
    text_encoder: T5EncoderConfig = field(default_factory=T5EncoderConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=PerceptionEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    num_anchors: int = 3
    anchor_embedding_dim: int = 128
    visual_ranker: ImageBindRankerConfig = field(default_factory=ImageBindRankerConfig)
    text_ranker: EnsembleRankerConfig = field(default_factory=EnsembleRankerConfig)
