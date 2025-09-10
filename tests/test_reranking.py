import os
import unittest

import torch
from audiobox.models.audio_visual_models.imagebind import ImageBindScorer

from sam_audio.model.config import ImageBindRankerConfig
from sam_audio.ranking.imagebind import ImageBindRanker


class TestReRanking(unittest.TestCase):
    def get_file(self, basename):
        dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir, "data", basename)

    def test_imagebind(self):
        ranker = ImageBindRanker(ImageBindRankerConfig())
        ab_ranker = ImageBindScorer()
        video_file = self.get_file("15pi8h_bHQE_173000_183000.mp4")
        sr = 48_000
        wavs = torch.rand(8, 1, sr * 10)
        scores = ranker(wavs, video_file)
        ab_scores = ab_ranker.score_audio_video(
            wavs, [video_file] * len(wavs), sample_rate=sr
        )
        self.assertTrue(torch.allclose(scores, ab_scores))

        breakpoint()


if __name__ == "__main__":
    unittest.main()
