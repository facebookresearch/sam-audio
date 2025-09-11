import argparse
import json
import os
import re

import torch
from dac.model import dac

from sam_audio.model.config import JudgeConfig, TransformerConfig, serialize_config
from sam_audio.model.judge import Judge
from tests.models import get_model


def main(audiobox_path: str):
    model_or_checkpoint = "/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth"
    dacvae = dac.DACVAE.load(model_or_checkpoint).eval()
    checkpoint = {
        **{"audio_codec.model." + k: v for k, v in dacvae.state_dict().items()},
    }

    ab_model = get_model("audiobox-judge")

    def remap_audio_encoder(key, prefix="audio_encoder."):
        res = prefix + re.sub(r"^model\.", "", key)  # rename top level module

        # We remove GroupedLayers
        match = re.search(r"layers\.(\d+)\.layers\.(\d+)\.", res)
        if match:
            layers_per_group = (
                ab_model.model.modality_encoders.audio_understanding.layer_grouping
            )
            level1 = int(match.group(1))
            level2 = int(match.group(2))
            new_level = layers_per_group * level1 + level2
            res = re.sub(
                r"layers\.(\d+)\.layers\.(\d+)\.", f"layers.{new_level}.", res
            )  # remove `GroupedLayers`
        return res

    checkpoint = {
        **checkpoint,
        # audio understanding modality encoder
        **{
            remap_audio_encoder(k): v
            for k, v in ab_model.model.modality_encoders.audio_understanding.state_dict().items()
        },
        # fientune audio understanding modality encoder
        **{
            remap_audio_encoder(k, "finetune_encoder."): v
            for k, v in ab_model.model.finetune_audio_understanding.state_dict().items()
        },
        # Concat input/hyp audio features
        **{
            "cat_audio_proj." + k: v
            for k, v in ab_model.model.finetune_pre_alignment_proj.proj.state_dict().items()
        },
        # text_projector (modality postprocessors)
        **{
            "text_proj." + k: v
            for k, v in ab_model.model.modality_postprocessors.text_projector.proj.state_dict().items()
        },
        # modality aligner
        **{
            "aligner." + k: v
            for k, v in ab_model.model.finetune_modality_aligners[0]
            .state_dict()
            .items()
        },
        **{
            "cat_aligned." + k: v
            for k, v in ab_model.model.finetune_post_alignment_proj.proj.state_dict().items()
        },
        "mean": torch.tensor(ab_model.model.mean),
        "std": torch.tensor(ab_model.model.std),
    }

    # Create a merged fientune head
    heads = [x.proj.weight for x in ab_model.model.finetune_heads]
    checkpoint["head.weight"] = torch.cat(heads, dim=0)

    if "ModernBertEmbeddingModel" in str(
        type(ab_model.model.modality_encoders.text_encoder)
    ):
        # Add state dict for modern BERT
        checkpoint.update(
            {
                f"text_encoder.model.{k}": v
                for k, v in ab_model.model.modality_encoders.text_encoder.transformer.state_dict().items()
            }
        )

    config = JudgeConfig(
        audio_encoder=TransformerConfig(
            dim=1792,
            out_channels=1792,
            in_channels=128,
            n_heads=14,
            n_layers=28,
            no_cross_attention=True,
        ),
        finetune_encoder=TransformerConfig(
            dim=192,
            out_channels=192,
            in_channels=256,
            n_heads=3,
            n_layers=6,
            no_cross_attention=True,
        ),
    )

    output_path = os.path.join(os.path.dirname(audiobox_path), "hf/checkpoint.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)

    config_path = os.path.join(os.path.dirname(audiobox_path), "hf/config.json")
    with open(config_path, "w") as fout:
        print(json.dumps(serialize_config(config), indent=4), file=fout)

    model = Judge.from_pretrained(os.path.join(os.path.dirname(audiobox_path), "hf"))

    breakpoint()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiobox_path", type=str)
    args = parser.parse_args()
    main(args.audiobox_path)
