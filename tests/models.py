import torch
from audiobox.models.transformer_layers.modules.inner_attention.flash_attention2 import (
    FlashAttentionSDPA,
    FlashCrossAttentionSDPA,
)
from audiobox.models.transformer_layers.modules.inner_attention.scaled_dot_product import (
    PTScaledDotProductAttention,
)

MODELS = {}


def replace_flash_attn(module):
    # This allows us to run inference in full precision
    for name, child in module.named_children():
        if isinstance(child, (FlashAttentionSDPA, FlashCrossAttentionSDPA)):
            attn = PTScaledDotProductAttention(qk_scale=child.scale)
            setattr(module, name, attn)
        else:
            replace_flash_attn(child)


def get_model(name):
    if name in MODELS:
        return MODELS[name]
    if name == "audiobox":
        import configs.resolvers  # noqa F401
        from audiobox.e2e.e2e import SeparationE2EModel

        checkpoint_pth = "/home/mattle/checkpoints/separation/demo/models/vts_mitigated_v1/passrm_video_two_stream_mitigated_scratch_higher_span_ratio_300k_r1.ckpt"
        overrides = [
            "data.batch_feature_extractors.0.pretrained=/home/mattle/checkpoints/metaclip/v2/metaclipv2_h14_genai.pt",
            "data.batch_feature_extractors.0.cache_dir=/home/mattle/.cache/openclip",
            "data.feature_extractor.repository=/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
            "data.dataset.conditioning.video.height=224",
            "data.dataset.conditioning.video.width=224",
            "data.dataset.conditioning.video.height=224",
            "data.dataset.conditioning.video.width=224",
        ]
        model = SeparationE2EModel(
            device=torch.device("cuda"),
            audio_checkpoint=checkpoint_pth,
            overrides=overrides,
            precision="bf16",
            vocoder_checkpoint="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth",
            max_positions=10_000,
        )
        MODELS[name] = model
        return model
    elif name == "sam":
        from sam_audio.model.model import SAMAudio

        model = SAMAudio.from_config(
            "base",
            pretrained=True,
            checkpoint_path="/home/mattle/checkpoints/separation/demo/models/vts_mitigated_v1/oss.ckpt",
        )
        model = model.to("cuda").eval()
        MODELS[name] = model
        return model
    elif name == "audiobox-judge":
        from audiobox.models.audio_understanding.sam_audio_judge import E2ESAMAudioJudge

        checkpoint_path = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_300hrs_base_regression_joint_posttrain/runs/2025-08-14-13-48-43/epoch-000_step-100000.ckpt"
        checkpoint_path = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_300hrs_base_regression_joint_posttrain_v6/runs/2025-08-29-02-11-21/epoch-000_step-100000.ckpt"
        model = E2ESAMAudioJudge(
            batch_size=4,
            checkpoint_path=checkpoint_path,
            dacvae_repository="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
            num_workers=0,
            precision="fp32",
        )
        replace_flash_attn(model.model)
        return model
    elif name == "sam-judge":
        from sam_audio.model.judge import Judge

        checkpoint_path = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_300hrs_base_regression_joint_posttrain/runs/2025-08-14-13-48-43/oss.pth"
        checkpoint_path = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_300hrs_base_regression_joint_posttrain_v6/runs/2025-08-29-02-11-21/oss.pth"
        model = Judge.from_config(
            "base", pretrained=True, checkpoint_path=checkpoint_path
        )
        MODELS[name] = model
        return MODELS[name].eval().cuda()
    else:
        raise RuntimeError(f"Unknown model {name}")
