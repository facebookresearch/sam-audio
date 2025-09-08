import os
import unittest
from unittest.mock import patch

import torch
import torchaudio
from models import get_model


class TestJudge(unittest.TestCase):
    def get_audio(self, file, sample_rate: int = 48_000):
        dir = os.path.dirname(os.path.realpath(__file__))
        wav, sr = torchaudio.load(os.path.join(dir, file))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate).mean(
                dim=0, keepdim=True
            )
        return wav

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
        wav = self.get_audio("data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav")
        hyp_wav = self.get_audio(
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

    @patch("torch.randn_like", orig_randn_like=torch.randn_like)
    @patch("torch.compile")
    @torch.no_grad()
    def test_batched(self, compile_mock, randn_like_mock):
        def f(like, **kwargs):
            # Mock this so that we get deterministic output from `randn` given the same input
            nonlocal randn_like_mock
            torch.manual_seed(0)
            return randn_like_mock.orig_randn_like(like, **kwargs)

        randn_like_mock.side_effect = f
        compile_mock.side_effect = lambda *args, **kwargs: lambda x, *args, **kwargs: x
        description = "Raindrops are falling heavily, splashing on the ground."
        wav = self.get_audio("data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav")
        wav2 = self.get_audio(
            "data/702459_6464538-hq_690345_1453392-hq_snr-3.0_shortened.wav"
        )
        hyp_wav = self.get_audio(
            "data/702459_6464538-hq_690345_1453392-hq_snr-3.0_separated.wav"
        )
        hyp_wav2 = self.get_audio(
            "data/702459_6464538-hq_690345_1453392-hq_snr-3.0_separated_shortened.wav"
        )
        sam_model = get_model("sam-judge")

        with torch.inference_mode():
            batched = sam_model.predict(
                [wav, wav2], [hyp_wav, hyp_wav2], [description, description]
            )
            single1 = sam_model.predict([wav], [hyp_wav], [description])
            single2 = sam_model.predict([wav2], [hyp_wav2], [description])

        self.assertAlmostEqual(
            batched.overall[0].item(), single1.overall.item(), delta=0.1
        )
        self.assertAlmostEqual(
            batched.overall[1].item(), single2.overall.item(), delta=0.1
        )

        self.assertAlmostEqual(
            batched.recall[0].item(), single1.recall.item(), delta=0.1
        )
        self.assertAlmostEqual(
            batched.recall[1].item(), single2.recall.item(), delta=0.1
        )

        self.assertAlmostEqual(
            batched.precision[0].item(), single1.precision.item(), delta=0.1
        )
        self.assertAlmostEqual(
            batched.precision[1].item(), single2.precision.item(), delta=0.1
        )

        self.assertAlmostEqual(
            batched.faithfulness[0].item(), single1.faithfulness.item(), delta=0.1
        )
        self.assertAlmostEqual(
            batched.faithfulness[1].item(), single2.faithfulness.item(), delta=0.1
        )


if __name__ == "__main__":
    unittest.main()
