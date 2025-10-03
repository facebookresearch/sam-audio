import os
import unittest
from unittest.mock import patch

import torch
import torchaudio
from models import get_model


class TestJudge(unittest.TestCase):
    def get_file(self, basename):
        dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir, "data", basename)

    def get_audio(self, file, sample_rate: int = 48_000):
        wav, sr = torchaudio.load(file)
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
            # The only place that we use this is in `DACVAE` latent sampling
            return torch.zeros_like(like)

        randn_like_mock.side_effect = f
        compile_mock.side_effect = lambda *args, **kwargs: lambda x, *args, **kwargs: x
        description = (
            "Raindrops are falling heavily, splashing on the ground.".lower().strip()
        )
        wav_file = self.get_file(
            "702459_6464538-hq_690345_1453392-hq_snr-3.0_resampled.wav"
        )
        hyp_wav_file = self.get_file(
            "702459_6464538-hq_690345_1453392-hq_snr-3.0_separated.wav"
        )
        wav = self.get_audio(wav_file)
        hyp_wav = self.get_audio(hyp_wav_file)
        ab_model = get_model("audiobox-judge")
        sam_model, processor = get_model("sam-judge")

        with torch.inference_mode():
            ab_res = ab_model.predict([wav], [hyp_wav], [description])
            processed = processor(
                input_audio=[wav_file],
                separated_audio=[hyp_wav_file],
                text=[description],
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            sam_result = sam_model(**processed)

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
            # The only place that we use this is in `DACVAE` latent sampling
            return torch.zeros_like(like)

        randn_like_mock.side_effect = f
        compile_mock.side_effect = lambda *args, **kwargs: lambda x, *args, **kwargs: x
        description = (
            "Raindrops are falling heavily, splashing on the ground.".lower().strip()
        )
        description2 = "Raindrops are falling heavily".lower().strip()

        wav_files = [
            self.get_file("702459_6464538-hq_690345_1453392-hq_snr-3.0_resampled.wav"),
            self.get_file("702459_6464538-hq_690345_1453392-hq_snr-3.0_shortened.wav"),
        ]
        hyp_files = [
            self.get_file("702459_6464538-hq_690345_1453392-hq_snr-3.0_separated.wav"),
            self.get_file(
                "702459_6464538-hq_690345_1453392-hq_snr-3.0_separated_shortened.wav"
            ),
        ]

        sam_model, processor = get_model("sam-judge")

        with torch.inference_mode():
            processed = processor(
                input_audio=wav_files,
                separated_audio=hyp_files,
                text=[description, description2],
                return_tensors="pt",
                padding=True,
            )
            batched = sam_model(**processed.to("cuda"))

            processed1 = processor(
                input_audio=[wav_files[0]],
                separated_audio=[hyp_files[0]],
                text=[description],
                return_tensors="pt",
            )
            processed2 = processor(
                input_audio=[wav_files[1]],
                separated_audio=[hyp_files[1]],
                text=[description2],
                return_tensors="pt",
            )
            single1 = sam_model(**processed1.to("cuda"))
            single2 = sam_model(**processed2.to("cuda"))

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
