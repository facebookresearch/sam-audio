# from tests.utils import load_audiobox_model
import re

import torch
from audiobox.e2e.e2e import SeparationE2EModel
from dac.model import dac
from open_clip import create_model_and_transforms

model_or_checkpoint = "/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth"
dacvae = dac.DACVAE.load(model_or_checkpoint).eval()

checkpoint = {"audio_codec.model." + k: v for k, v in dacvae.state_dict().items()}

pth = "/home/mattle/checkpoints/metaclip/v2/metaclipv2_h14_genai.pt"
model, _, _ = create_model_and_transforms("ViT-H-14", pretrained=pth)

checkpoint.update(
    {"vision_encoder.model." + k: v for k, v in model.state_dict().items()}
)

checkpoint_pth = "/home/mattle/checkpoints/separation/demo/models/vts_mitigated_v1/passrm_video_two_stream_mitigated_scratch_higher_span_ratio_300k_r1.ckpt"
overrides = [
    "data.batch_feature_extractors.0.pretrained=/home/mattle/checkpoints/metaclip/v2/metaclipv2_h14_genai.pt",
    "data.batch_feature_extractors.0.cache_dir=/home/mattle/.cache/openclip",
    "data.feature_extractor.repository=/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
]
ab_model = SeparationE2EModel(
    device=torch.device("cuda"),
    audio_checkpoint=checkpoint_pth,
    overrides=overrides,
    precision="bf16",
    vocoder_checkpoint="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth",
)


def remap_audio_encoder(key, prefix="transformer."):
    res = prefix + re.sub(r"^model\.", "", key)  # rename top level module

    # We remove GroupedLayers
    match = re.search(r"layers\.(\d+)\.layers\.(\d+)\.", res)
    if match:
        layers_per_group = ab_model.method.model.layer_grouping
        level1 = int(match.group(1))
        level2 = int(match.group(2))
        new_level = layers_per_group * level1 + level2
        res = re.sub(
            r"layers\.(\d+)\.layers\.(\d+)\.", f"layers.{new_level}.", res
        )  # remove `GroupedLayers`
    return res


def remap_vid_aligner(key):
    if key == "value":
        return "gate"
    else:
        return re.sub(r"^encoder\.", "", key)


checkpoint.update(
    {
        # Note, we filter out the _ema parameters, since `SeparationE2EModel` calls .eval(), copying the ema params to the main params
        **{
            remap_audio_encoder(k): v
            for k, v in ab_model.method.model.model.model.state_dict().items()
            if not k.endswith("_ema")
        },
        # Alignment layers
        **{
            "align_video." + remap_vid_aligner(k): v
            for k, v in ab_model.method.model.model.time_aligned_encoders[2]
            .state_dict()
            .items()
            if not k.endswith("_ema")
        },
        **{
            "align_masked_video." + remap_vid_aligner(k): v
            for k, v in ab_model.method.model.model.time_aligned_encoders[3]
            .state_dict()
            .items()
            if not k.endswith("_ema")
        },
        **{
            "embed_anchors." + remap_vid_aligner(k): v
            for k, v in ab_model.method.model.model.time_aligned_encoders[4]
            .state_dict()
            .items()
            if not k.endswith("_ema")
        },
        # aligned input projection layer
        **{
            "proj." + k: v
            for k, v in ab_model.method.model.model.data_proj.state_dict().items()
            if not k.endswith("_ema")
        },
        # memory encoder
        **{
            k.replace("proj", "memory_proj"): v
            for k, v in ab_model.method.model.model.memory_encoders[0]
            .state_dict()
            .items()
            if not k.endswith("_ema")
        },
    }
)

torch.save(
    checkpoint,
    "/home/mattle/checkpoints/separation/demo/models/vts_mitigated_v1/oss.ckpt",
)

breakpoint()
