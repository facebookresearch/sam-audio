import unittest

import torch

from sam_audio.model.rope import RotaryEmbedding


class TestRope(unittest.TestCase):
    def test_rope(self):
        from audiobox.models.xpos_relative_position import CachedXposWithCustomKernel

        xpos_cached = CachedXposWithCustomKernel(
            128,
            freq_base=20_000,
            max_len=10_000,
            scale_base=None,
            inplace=False,
            interleaved=True,
        )
        rope = RotaryEmbedding(
            theta=20_000,
            head_dim=128,
            max_seqlen=10_000,
        )
        xpos_cached = xpos_cached.cuda()
        rope.reset_parameters()
        rope = rope.cuda()
        input = torch.rand(5, 8, 128, 128).cuda()
        xpos_xq = xpos_cached(input, bhle=True, downscale=False)
        xpos_xk = xpos_cached(input, bhle=True, downscale=True)
        rope_xq = rope(input, bhle=True)
        rope_xk = rope(input, bhle=True)
        diff_xq = (xpos_xq - rope_xq).abs().max().item()
        diff_xk = (xpos_xk - rope_xk).abs().max().item()
        self.assertAlmostEqual(diff_xk, 0.0, places=5)
        self.assertAlmostEqual(diff_xq, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
