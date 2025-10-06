import os
import unittest
from unittest.mock import patch

import torch
from audiobox.models.audio_understanding.sam_audio_judge import SAMAudioJudgeRanker
from audiobox.models.audio_visual_models.imagebind import ImageBindScorer
from audiobox.rankers.clap import ClapRanker as ABClapRanker

from sam_audio.model.config import (
    ClapRankerConfig,
    ImageBindRankerConfig,
    JudgeRankerConfig,
)
from sam_audio.ranking.clap import ClapRanker
from sam_audio.ranking.imagebind import ImageBindRanker
from sam_audio.ranking.judge import JudgeRanker


class TestReRanking(unittest.TestCase):
    def get_file(self, basename):
        dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir, "data", basename)

    def test_imagebind(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranker = ImageBindRanker(ImageBindRankerConfig()).to(device)
        ab_ranker = ImageBindScorer().to(device)
        video_file = self.get_file("15pi8h_bHQE_173000_183000.mp4")
        sr = 48_000
        with torch.no_grad():
            wavs = torch.rand(8, 1, sr * 10, device=device)
            scores = ranker(
                [wavs.squeeze(1), wavs.squeeze(1)], [video_file, video_file]
            )
            ab_scores = ab_ranker.score_audio_video(
                torch.cat([wavs, wavs]), [video_file] * len(wavs) * 2, sample_rate=sr
            )
            self.assertTrue(torch.allclose(scores.view(-1), ab_scores, atol=1e-6))

    def test_clap(self):
        import laion_clap

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranker = ClapRanker(ClapRankerConfig()).to(device)
        checkpoint = os.path.join(os.path.dirname(laion_clap.__file__), "630k-best.pt")
        ab_ranker = ABClapRanker(
            clap_path=checkpoint, enable_fusion=False, device=device
        )
        description = "a cat is sitting on a chair"
        sr = 48_000
        with torch.no_grad():
            wavs = torch.rand(8, 1, sr * 10, device=device)
            scores = ranker(
                [wavs.squeeze(1), wavs.squeeze(1)], [description, description]
            )
            ab_scores = ab_ranker.score(
                torch.cat([wavs, wavs]).cpu(),
                [description] * len(wavs) * 2,
                sample_rate=sr,
            )
            self.assertTrue(torch.allclose(scores.view(-1), ab_scores, atol=1e-6))

    @patch("torch.randn_like", orig_randn_like=torch.randn_like)
    def test_judge(self, randn_like_mock):
        def f(like, **kwargs):
            # The only place that we use this is in `DACVAE` latent sampling
            return torch.zeros_like(like)

        randn_like_mock.side_effect = f
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranker = JudgeRanker(JudgeRankerConfig()).to(device)

        checkpoint = "/home/mattle/checkpoints/separation/samjudge/v2/runs/peaudio_large_300hrs_base_regression_joint_posttrain/runs/2025-09-18-20-41-51/lightning_logs/version_0/checkpoints/epoch-000_step-100000.ckpt"
        ab_ranker = SAMAudioJudgeRanker(
            checkpoint=checkpoint,
            dacvae_repository="/home/mattle/checkpoints/dacvae/vae_large_scale_pretrain_v2_48000_hop1920_ld128/100k/dacvae",
            batch_size=8,
            metrics=["overall"],
        )

        description = "a cat is sitting on a chair"
        sr = 48_000
        with torch.no_grad():
            wavs = torch.rand(8, 1, sr * 10, device=device)
            ex_wavs = torch.rand(8, 1, sr * 10, device=device)
            scores = ranker(
                [wavs.squeeze(1), wavs.squeeze(1)],
                [ex_wavs.squeeze(1), ex_wavs.squeeze(1)],
                [description, description],
            )
            hyp_wavs = torch.cat([ex_wavs, ex_wavs]).cpu()
            texts = [(w.cpu(), description) for w in torch.cat([wavs, wavs])]
            ab_scores = ab_ranker.score(hyp_wavs, texts, sample_rate=sr)
            ab_idxs = ab_scores.view(2, -1).argmax(1)
            idxs = scores.argmax(1)
            self.assertTrue((idxs.cpu() == ab_idxs.cpu()).all())


if __name__ == "__main__":
    unittest.main()
