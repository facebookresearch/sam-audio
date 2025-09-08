import unittest

import torch
import torch.nn.functional as F
from audiobox.models.transformer_layers.modules.inner_attention.flash_attention2 import (
    FlashAttentionSDPA,
    FlashCrossAttentionSDPA,
)
from audiobox.models.xpos_relative_position import CachedXposWithCustomKernel

from sam_audio.model.rope import RotaryEmbedding
from sam_audio.model.transformer import Attention


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
        torch.manual_seed(0)
        head_dim = 128
        n_heads = 8
        sizes = torch.tensor([8, 16, 24, 30, 32])
        mask = torch.arange(32)[None, :] < sizes[:, None]
        x = torch.rand(5, 32, head_dim * n_heads)
        for use_qk_norm in [False, True]:
            attn = Attention(
                head_dim * n_heads, head_dim, n_heads, n_heads, use_qk_norm=use_qk_norm
            )
            batched = attn(x, key_padding_mask=mask)
            single = attn(x[[0], : sizes[0]], key_padding_mask=None)
            self.assertTrue(
                torch.allclose(batched[[0], : sizes[0]], single, atol=1e-6, rtol=1e-6)
            )

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
        torch.manual_seed(0)
        head_dim = 128
        n_heads = 8
        sizes = torch.tensor([8, 16, 24, 30, 32])
        mem_sizes = sizes - 2
        key_pad_mask = torch.arange(30)[None, :] < mem_sizes[:, None]

        x = torch.rand(5, 32, head_dim * n_heads)
        y = torch.rand(5, 30, head_dim * n_heads)
        for use_qk_norm in [False, True]:
            attn = Attention(
                head_dim * n_heads, head_dim, n_heads, n_heads, use_qk_norm=use_qk_norm
            )
            batched = attn(x, cross_x=y, key_padding_mask=key_pad_mask)
            single = attn(
                x[[0], : sizes[0]],
                cross_x=y[[0], : mem_sizes[0]],
                key_padding_mask=None,
            )
            self.assertTrue(
                torch.allclose(batched[[0], : sizes[0]], single, atol=1e-6, rtol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
