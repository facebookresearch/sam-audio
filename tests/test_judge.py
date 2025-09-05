import os
import unittest
from unittest.mock import patch

import torch
import torchaudio
from models import get_model


class TestJudge(unittest.TestCase):
    def get_audio(self, file):
        dir = os.path.dirname(os.path.realpath(__file__))
        return torchaudio.load(os.path.join(dir, file))

    @patch("torch.randn_like", orig_randn_like=torch.randn_like)
    @patch("torch.compile")
    @torch.no_grad()
    def test_judge(self, compile_mock, randn_like_mock):
        def f(like, **kwargs):
            # Mock this so that we get deterministic output from `randn` given the same input
            nonlocal randn_like_mock
            torch.manual_seed(0)
            return randn_like_mock.orig_randn_like(like, **kwargs)

        randn_like_mock.side_effect = f
        compile_mock.side_effect = lambda *args, **kwargs: lambda x, *args, **kwargs: x
        description = "Raindrops are falling heavily, splashing on the ground."
        target_sr = 48_000
        wav, sr = self.get_audio("data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav")
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        hyp_wav, sr2 = self.get_audio(
            "data/702459_6464538-hq_690345_1453392-hq_snr-3.0_separated.wav"
        )
        ab_model = get_model("audiobox-judge")
        sam_model = get_model("sam-judge")

        with torch.inference_mode():
            ab_res = ab_model.predict([wav], [hyp_wav], [description])
            sam_result = sam_model.predict([wav], [hyp_wav], [description])

        self.assertAlmostEqual(
            ab_res[0]["overall"], sam_result.overall.item(), places=2
        )
        self.assertAlmostEqual(ab_res[0]["recall"], sam_result.recall.item(), places=2)
        self.assertAlmostEqual(
            ab_res[0]["precision"], sam_result.precision.item(), places=2
        )
        self.assertAlmostEqual(
            ab_res[0]["faithfulness"], sam_result.faithfulness.item(), places=2
        )


if __name__ == "__main__":
    unittest.main()
