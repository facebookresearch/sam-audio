import argparse
import json
import os
import re
import shutil

import torch
from dac.model import dac
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from sam_audio.model.config import TransformerConfig, serialize_config
from tests.models import get_model


def get_dacvae_encoder_state_dict(dacvae):
    for module in dacvae.modules():
        if "weight_g" in module.state_dict():
            torch.nn.utils.remove_weight_norm(module)

    names = ["res_unit1", "res_unit2", "res_unit3", "snake1", "conv1"]

    state_dict = {}

    for block_idx, block in enumerate(dacvae.encoder.block[1:-2]):
        for name, sub_block in zip(names, block.block):
            if hasattr(sub_block, "block"):
                for sub_name, sub_sub_block in zip(
                    ["snake1", "conv1", "snake2", "conv2"], sub_block.block
                ):
                    state_dict.update(
                        {
                            f"block.{block_idx}.{name}.{sub_name}." + k: v
                            for k, v in sub_sub_block.state_dict().items()
                        }
                    )
            else:
                state_dict.update(
                    {
                        f"block.{block_idx}.{name}." + k: v
                        for k, v in sub_block.state_dict().items()
                    }
                )

    state_dict.update(
        {"conv1." + k: v for k, v in dacvae.encoder.block[0].state_dict().items()}
    )
    state_dict.update(
        {"snake1." + k: v for k, v in dacvae.encoder.block[-2].state_dict().items()}
    )
    state_dict.update(
        {"conv2." + k: v for k, v in dacvae.encoder.block[-1].state_dict().items()}
    )
    return state_dict


def convert_decoder_layer_sd(state_dict, nheads):
    def rewrite(k):
        k = re.sub(r"(^|\.)feed_forward.w1", r"\1mlp.gate_proj", k)
        k = re.sub(r"(^|\.)feed_forward.w3", r"\1mlp.up_proj", k)
        k = re.sub(r"(^|\.)feed_forward.w2", r"\1mlp.down_proj", k)
        k = re.sub(r"(^|\.)attention.w([a-z])", r"\1self_attn.\2_proj", k)
        k = re.sub(r"(^|\.)attention_norm", r"\1input_layernorm", k)
        k = re.sub(r"(^|\.)ffn_norm", r"\1post_attention_layernorm", k)
        k = re.sub(r"(^|\.)attention.q_norm", r"\1self_attn.q_norm", k)
        k = re.sub(r"(^|\.)attention.k_norm", r"\1self_attn.k_norm", k)
        return k

    rewritten = {rewrite(k): v for k, v in state_dict.items()}

    weight = [v for k, v in rewritten.items() if "k_proj.weight" in k][0]
    head_dim = weight.shape[0] // nheads
    idxs = torch.arange(weight.shape[0])
    idxs = idxs.reshape(head_dim, nheads)
    idxs = idxs.transpose(0, 1)
    idxs = idxs.flatten()
    for k in rewritten.keys():
        if "k_proj.weight" in k or "q_proj.weight" in k or "v_proj.weight" in k:
            rewritten[k] = rewritten[k][idxs]

    return rewritten


def convert_transformer(module, num_heads, prefix):
    sd = module.state_dict()

    def rename_key(key):
        res = prefix + re.sub(r"^model\.", "", key)  # rename top level module

        # We remove GroupedLayers
        match = re.search(r"layers\.(\d+)\.layers\.(\d+)\.", res)
        if match:
            layers_per_group = module.layer_grouping
            level1 = int(match.group(1))
            level2 = int(match.group(2))
            new_level = layers_per_group * level1 + level2
            res = re.sub(
                r"layers\.(\d+)\.layers\.(\d+)\.", f"layers.{new_level}.", res
            )  # remove `GroupedLayers`
        return res

    sd = {rename_key(k): v for k, v in sd.items()}
    return convert_decoder_layer_sd(sd, num_heads)


def transformers_rewrites(state_dict):
    # Additional rewrites for `transformers` refactoring
    def rewrite(k):
        k = re.sub(r"(^|\.)cls_token.weight", r"\1embeddings.cls_token", k)
        k = re.sub(r"(^|\.)x_embedder.block", r"\1embeddings.resnet_block", k)
        k = re.sub(r"(^|\.)transformer.data_proj", r"\1data_proj", k)
        k = re.sub(r"(^|\.)finetune_transformer.data_proj", r"\1finetune_data_proj", k)
        return k

    return {rewrite(k): v for k, v in state_dict.items()}


def convert_config(cfg):
    intermediate_size = int(2 * (cfg.dim * cfg.ffn_exp) / 3)
    # custom dim factor multiplier
    intermediate_size = int(cfg.ffn_dim_multiplier * intermediate_size)
    # round hidden dimension to `multiple_of`
    intermediate_size = cfg.multiple_of * (
        (intermediate_size + cfg.multiple_of - 1) // cfg.multiple_of
    )

    return Qwen3Config(
        hidden_size=cfg.dim,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_heads,
        head_dim=cfg.dim // cfg.n_heads,
        max_position_embeddings=cfg.max_positions,
        rope_theta=max(10_000, 2 * cfg.max_positions),
        rms_norm_eps=cfg.norm_eps,
        num_hidden_layers=cfg.n_layers,
    ).to_dict()


def main(audiobox_path: str):
    model_or_checkpoint = "/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae/weights.pth"
    dacvae = dac.DACVAE.load(model_or_checkpoint).eval()

    checkpoint = {
        **{
            "dac_vae_encoder.encoder." + k: v
            for k, v in get_dacvae_encoder_state_dict(dacvae).items()
        },
        **{
            "dac_vae_encoder.bottleneck." + k: v
            for k, v in dacvae.quantizer.state_dict().items()
        },
    }

    ab_model = get_model("audiobox-judge")

    checkpoint = {
        **checkpoint,
        # audio understanding modality encoder
        **convert_transformer(
            module=ab_model.model.modality_encoders.audio_understanding,
            num_heads=ab_model.model.modality_encoders.audio_understanding.nhead,
            prefix="transformer.",
        ),
        # fientune audio understanding modality encoder
        **convert_transformer(
            module=ab_model.model.finetune_audio_understanding,
            num_heads=ab_model.model.finetune_audio_understanding.nhead,
            prefix="finetune_transformer.",
        ),
        # Concat input/hyp audio features
        **{
            "cat_audio_proj." + k: v
            for k, v in ab_model.model.finetune_pre_alignment_proj.proj.state_dict().items()
        },
        # text_projector (modality postprocessors)
        **{
            "text_proj1." + k: v
            for k, v in ab_model.model.modality_postprocessors.text_projector.proj.state_dict().items()
        },
        # modality aligner
        "text_proj2.bias": ab_model.model.finetune_modality_aligners[0].conv.bias,
        "text_proj2.weight": ab_model.model.finetune_modality_aligners[
            0
        ].conv.weight.squeeze(-1),
        "layer_norm.weight": ab_model.model.finetune_modality_aligners[
            0
        ].layer_norm.weight,
        "layer_norm.bias": ab_model.model.finetune_modality_aligners[0].layer_norm.bias,
        **{
            "proj_audio_and_text." + k: v
            for k, v in ab_model.model.finetune_post_alignment_proj.proj.state_dict().items()
        },
        "mean": torch.tensor(ab_model.model.mean),
        "std": torch.tensor(ab_model.model.std),
    }

    # Create a merged fientune head
    heads = [x.proj.weight for x in ab_model.model.finetune_heads[:4]]
    checkpoint["head.weight"] = torch.cat(heads, dim=0)

    if "ModernBertEmbeddingModel" in str(
        type(ab_model.model.modality_encoders.text_encoder)
    ):
        # Add state dict for modern BERT
        checkpoint.update(
            {
                f"text_model.{k}": v
                for k, v in ab_model.model.modality_encoders.text_encoder.transformer.state_dict().items()
            }
        )

    checkpoint = transformers_rewrites(checkpoint)

    model_size = "base"  # or "light"

    config = {
        "transformer": convert_config(
            TransformerConfig(
                dim=1792 if model_size == "base" else 1024,
                out_channels=1792 if model_size == "base" else 1024,
                in_channels=128,
                n_heads=14 if model_size == "base" else 8,
                n_layers=28 if model_size == "base" else 16,
            ),
        ),
        "finetune_transformer": convert_config(
            TransformerConfig(
                dim=192,
                out_channels=192,
                in_channels=256,
                n_heads=3,
                n_layers=6,
            ),
        ),
        "dac_vae_encoder": {
            "encoder_hidden_size": 64,
            "downsampling_ratios": [2, 8, 10, 12],
            "decoder_hidden_size": 1536,
            "n_codebooks": 16,
            "codebook_size": 1024,
            "codebook_dim": 128,
            "quantizer_dropout": 0,
            "sampling_rate": 48_000,
        },
        "text_model": ab_model.model.modality_encoders.text_encoder.transformer.config.to_dict(),
        "bottleneck_dim": 256,
    }

    hf_dir = os.path.join(os.path.dirname(audiobox_path), "hf")
    os.makedirs(hf_dir, exist_ok=True)
    save_file(checkpoint, os.path.join(hf_dir, "model.safetensors"))

    with open(os.path.join(hf_dir, "config.json"), "w") as fout:
        print(json.dumps(serialize_config(config), indent=4), file=fout)

    preproc_config = {
        "feature_extractor_type": "PerceptionEncoderAudioVideoFeatureExtractor",
        "feature_size": 1,
        "hop_length": 1920,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
        "sampling_rate": 48000,
    }
    with open(os.path.join(hf_dir, "preprocessor_config.json"), "w") as fout:
        print(
            json.dumps(preproc_config, indent=4),
            file=fout,
        )

    snapshot_dir = snapshot_download(
        repo_id=ab_model.model.modality_encoders.text_encoder.model,
        allow_patterns=["tokenizer*.json", "special_tokens_map.json"],
    )
    for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copyfile(os.path.join(snapshot_dir, file), os.path.join(hf_dir, file))

    # model = Judge.from_pretrained(os.path.join(os.path.dirname(audiobox_path), "hf"))
    from transformers import SamAudioJudgeModel

    model = SamAudioJudgeModel.from_pretrained(hf_dir)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    breakpoint()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiobox_path", type=str)
    args = parser.parse_args()
    main(args.audiobox_path)
