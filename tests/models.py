import os
from contextlib import contextmanager
from unittest import mock

import torch
from audiobox.models.transformer_layers.modules.inner_attention.flash_attention2 import (
    FlashAttentionSDPA,
    FlashCrossAttentionSDPA,
)
from audiobox.models.transformer_layers.modules.inner_attention.scaled_dot_product import (
    PTScaledDotProductAttention,
)

MODELS = {}


@contextmanager
def deterministic_randn_liked(mock_path, original_method):
    def side_effect(self, *args, **kwargs):
        torch.manual_seed(0)
        return original_method(self, *args, **kwargs)

    patcher = mock.patch(mock_path, autospec=True, side_effect=side_effect)
    yield patcher.start()
    patcher.stop()


def replace_flash_attn(module):
    # This allows us to run inference in full precision
    for name, child in module.named_children():
        if isinstance(child, (FlashAttentionSDPA, FlashCrossAttentionSDPA)):
            attn = PTScaledDotProductAttention(qk_scale=child.scale)
            setattr(module, name, attn)
        else:
            replace_flash_attn(child)


def get_model(name, additional_overrides=None):
    JUDGE_CHECKPOINT = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_large_300hrs_base_regression_joint_posttrain/runs/2025-09-18-20-41-51/lightning_logs/version_0/checkpoints/epoch-000_step-100000.ckpt"

    additional_overrides = additional_overrides or []
    if name in MODELS:
        return MODELS[name]
    if name == "audiobox":
        import configs.resolvers  # noqa F401
        from audiobox.e2e.e2e import SeparationE2EModel

        checkpoint_path = (
            "/home/mattle/checkpoints/separation/demo/models/vts_20250826/model.ckpt"
        )

        overrides = [
            "data.feature_extractor.repository=/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
            "data.dataset.conditioning.video.height=336",
            "data.dataset.conditioning.video.width=336",
            "data.dataset.conditioning.video_mask.height=336",
            "data.dataset.conditioning.video_mask.width=336",
        ] + additional_overrides
        model = SeparationE2EModel(
            device=torch.device("cuda"),
            audio_checkpoint=checkpoint_path,
            overrides=overrides,
            precision="bf16",
            vocoder_checkpoint="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth",
            max_positions=10_000,
            guidance_weight=None,
        )
        replace_flash_attn(model.method)
        MODELS[name] = model
        return model
    elif name == "sam":
        from sam_audio.model.model import SAMAudio

        checkpoint_path = (
            "/home/mattle/checkpoints/separation/demo/models/vts_mitigated_v1/oss.ckpt"
        )
        config = "base"
        checkpoint_path = (
            "/home/mattle/checkpoints/separation/demo/models/vts_20250826/oss.pth"
        )
        config = "base-pe"

        model = SAMAudio.from_config(
            config, pretrained=True, checkpoint_path=checkpoint_path
        )
        model = model.to("cuda").eval()
        MODELS[name] = model
        return model
    elif name == "audiobox-judge":
        from audiobox.models.audio_understanding.sam_audio_judge import E2ESAMAudioJudge

        model = E2ESAMAudioJudge(
            batch_size=4,
            checkpoint_path=JUDGE_CHECKPOINT,
            dacvae_repository="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
            num_workers=0,
            precision="fp32",
        )
        replace_flash_attn(model.model)
        return model
    elif name == "sam-judge":
        from transformers import SamAudioJudgeModel, SamAudioJudgeProcessor

        checkpoint_path = os.path.join(os.path.dirname(JUDGE_CHECKPOINT), "hf")
        model = SamAudioJudgeModel.from_pretrained(checkpoint_path)
        processor = SamAudioJudgeProcessor.from_pretrained(checkpoint_path)
        MODELS[name] = model.eval().cuda(), processor
        return MODELS[name]
    else:
        raise RuntimeError(f"Unknown model {name}")
