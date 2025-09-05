import unittest

import torch
import torch.nn.functional as F
from audiobox.models.transformer_layers.modules.inner_attention.flash_attention2 import (
    FlashAttentionSDPA,
    FlashCrossAttentionSDPA,
)
from audiobox.models.xpos_relative_position import CachedXposWithCustomKernel

from sam_audio.model.rope import RotaryEmbedding


class TestAttention(unittest.TestCase):
    def test_attention(self):
        rope = RotaryEmbedding(
            theta=20_000,
            head_dim=128,
            max_seqlen=10_000,
        )
        xpos_cached = CachedXposWithCustomKernel(
            128,
            freq_base=20_000,
            max_len=10_000,
            scale_base=None,
            inplace=False,
            interleaved=True,
        )
        xpos_cached = xpos_cached.cuda()
        rope.reset_parameters()
        rope = rope.cuda()
        xq = torch.rand(5, 8, 32, 128).cuda().bfloat16()  # B x H x T x C/H
        xk = torch.rand(5, 8, 32, 128).cuda().bfloat16()
        xv = torch.rand(5, 8, 32, 128).cuda().bfloat16()
        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.bfloat16):
            inner_attention = FlashAttentionSDPA(attn_drop=0.0, head_dim=128)
            fa_res = inner_attention(xq, xk, xv, xpos=xpos_cached)

            xq = rope(xq, bhle=True)
            xk = rope(xk, bhle=True)

            sdpa_res = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=None,
            )
            diff = (fa_res - sdpa_res).abs().max().item()
            print(f"Difference: {diff}")

            self.assertLess(diff, 0.05)

    def test_attention_batched(self):
        sizes = torch.tensor([8, 16, 24, 30, 32], device="cuda")
        mask = torch.arange(32, device="cuda")[None, :] >= sizes[:, None]

        xq = torch.rand(5, 8, 32, 128).cuda().bfloat16()  # B x H x T x C/H
        xk = torch.rand(5, 8, 32, 128).cuda().bfloat16()
        xv = torch.rand(5, 8, 32, 128).cuda().bfloat16()
        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.bfloat16):
            inner_attention = FlashAttentionSDPA(attn_drop=0.0, head_dim=128)
            fa_res = inner_attention(xq, xk, xv, padding_mask=mask)

            sdpa_res = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=~mask[:, None, None],
            )
            diff = (fa_res - sdpa_res).abs()
            max_diff = diff[~mask[:, None, :, None].expand_as(diff)].max()
            print(f"Difference: {max_diff}")
            self.assertLess(max_diff, 0.05)

    def test_cross_attention(self):
        xq = torch.rand(5, 8, 32, 128).cuda().bfloat16()  # B x H x T x C/H
        xk = torch.rand(5, 8, 30, 128).cuda().bfloat16()
        xv = torch.rand(5, 8, 30, 128).cuda().bfloat16()

        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.bfloat16):
            inner_attention = FlashCrossAttentionSDPA(attn_drop=0.0, head_dim=128)
            fa_res = inner_attention(xq, xk, xv)

            sdpa_res = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
            )
            diff = (fa_res - sdpa_res).abs().max().item()
            print(f"Difference: {diff}")

            self.assertLess(diff, 0.05)

    def test_cross_attention_batched(self):
        sizes = torch.tensor([8, 16, 24, 28, 30], device="cuda")
        key_padding_mask = torch.arange(30, device="cuda")[None, :] >= sizes[:, None]

        sizes = torch.tensor([8, 16, 24, 28, 32], device="cuda")
        mask = torch.arange(32, device="cuda")[None, :] >= sizes[:, None]

        xq = torch.rand(5, 8, 32, 128).cuda().bfloat16()  # B x H x T x C/H
        xk = torch.rand(5, 8, 30, 128).cuda().bfloat16()
        xv = torch.rand(5, 8, 30, 128).cuda().bfloat16()

        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.bfloat16):
            inner_attention = FlashCrossAttentionSDPA(attn_drop=0.0, head_dim=128)
            fa_res = inner_attention(
                xq,
                xk,
                xv,
                padding_mask=mask,
                key_padding_mask=key_padding_mask,
            )

            bias = ~mask[:, None, :, None] & ~key_padding_mask[:, None, None, :]

            sdpa_res = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=bias)
            diff = (fa_res - sdpa_res).abs()
            max_diff = diff[~mask[:, None, :, None].expand_as(diff)].max()
            print(f"Difference: {max_diff}")
            self.assertLess(max_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
