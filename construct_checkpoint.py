# from tests.utils import load_audiobox_model
import argparse
import json
import os
import re

import configs.resolvers  # noqa
import torch
from audiobox.e2e.e2e import SeparationE2EModel
from audiobox.models.ema import EMA
from dac.model import dac
from omegaconf import OmegaConf

from sam_audio.model.config import (
    SAMAudioConfig,
    TransformerConfig,
    serialize_config,
)
from sam_audio.model.model import SAMAudio


def main(audiobox_path: str):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(audiobox_path), "config.yaml"))
    video_encoder = cfg.data.batch_feature_extractors[0]._target_.split(".")[-1]
    extra_overrides = []
    if video_encoder == "MetaCLIPVideoExtractor":
        extra_overrides = [
            "data.batch_feature_extractors.0.pretrained=/home/mattle/checkpoints/metaclip/v2/metaclipv2_h14_genai.pt",
            "data.batch_feature_extractors.0.cache_dir=/home/mattle/.cache/openclip",
        ]

    overrides = [
        "data.feature_extractor.repository=/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
    ] + extra_overrides

    ab_model = SeparationE2EModel(
        device=torch.device("cuda"),
        audio_checkpoint=audiobox_path,
        overrides=overrides,
        vocoder_checkpoint="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth",
    )

    model_or_checkpoint = "/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth"
    dacvae = dac.DACVAE.load(model_or_checkpoint).eval()
    checkpoint = {"audio_codec.model." + k: v for k, v in dacvae.state_dict().items()}

    checkpoint.update(
        {
            "vision_encoder.model." + k: v
            for k, v in ab_model.method.data_module.batch_feature_extractors[0]
            .model.state_dict()
            .items()
        }
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

    if isinstance(ab_model.method.model, EMA):
        model = ab_model.method.model.model
    else:
        model = ab_model.method.model

    checkpoint.update(
        {
            # Note, we filter out the _ema parameters, since `SeparationE2EModel` calls .eval(), copying the ema params to the main params
            **{
                remap_audio_encoder(k): v
                for k, v in model.model.state_dict().items()
                if not k.endswith("_ema")
            },
            # aligned input projection layer
            **{
                "proj." + k: v
                for k, v in model.data_proj.state_dict().items()
                if not k.endswith("_ema")
            },
            # memory encoder
            **{
                k.replace("proj", "memory_proj"): v
                for k, v in model.memory_encoders[0].state_dict().items()
                if not k.endswith("_ema")
            },
        }
    )

    for module in model.time_aligned_encoders:
        sd = module.state_dict()
        if len(sd) == 0:
            continue
        if module._target_.endswith("Gate") and module.encoder._target_.endswith(
            "AlignVideo"
        ):
            prefix = (
                "align_video."
                if module.encoder.key == "video_features"
                else "align_masked_video."
            )
            checkpoint.update(
                {
                    prefix + remap_vid_aligner(k): v
                    for k, v in sd.items()
                    if not k.endswith("_ema")
                }
            )
        if module._target_.endswith("Gate") and module.encoder._target_.endswith(
            "EmbedAndAlignPhonemes"
        ):
            checkpoint.update(
                {
                    "embed_anchors." + remap_vid_aligner(k): v
                    for k, v in sd.items()
                    if not k.endswith("_ema")
                }
            )

    config = SAMAudioConfig(
        in_channels=model.in_channels,
        transformer=TransformerConfig(
            dim=model.d_model,
            n_heads=model.nhead,
            n_layers=model.num_layers,
        ),
    )

    output_path = os.path.join(os.path.dirname(audiobox_path), "hf/checkpoint.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)

    config_path = os.path.join(os.path.dirname(audiobox_path), "hf/config.json")
    with open(config_path, "w") as fout:
        print(
            json.dumps(serialize_config(config), indent=4),
            file=fout,
        )

    model = SAMAudio.from_pretrained(os.path.join(os.path.dirname(audiobox_path), "hf"))
    breakpoint()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiobox_path", type=str)
    args = parser.parse_args()
    main(args.audiobox_path)
