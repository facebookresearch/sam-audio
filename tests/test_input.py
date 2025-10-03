import os
import unittest

import torchaudio
from audiobox.e2e.use_case.audio_editing import Separation

from tests.models import get_model


class TestInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = os.path.dirname(os.path.realpath(__file__))
        cls.model = get_model("audiobox")
        cls.sam = get_model("sam")

    def test_text_based_separation(self):
        file = os.path.join(
            self.dir, "data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"
        )
        description = "Raindrops are falling heavily, splashing on the ground."
        info = torchaudio.info(file)
        mask_times = [0, info.num_frames / info.sample_rate]
        use_case = Separation(
            input_paths=[file], descriptions=[description], mask_times=[mask_times]
        )
        batch = next(use_case.prepare_batch(self.model.method, None, self.model.dset))
        res = self.sam.get_transform()(
            descriptions=[description],
            audio_paths=[file],
        )
        diff = (
            res.audios
            - batch["edit_audio"]["spec_or_wav"].squeeze().mean(0, keepdim=True).cpu()
        )
        self.assertAlmostEqual(diff.abs().max().item(), 0, places=5)
        self.assertEqual(batch["description"], res.descriptions)

    def test_video_based_separation(self):
        file = os.path.join(
            self.dir, "data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"
        )
        video_file = os.path.join(self.dir, "data/15pi8h_bHQE_173000_183000.mp4")
        description = "Raindrops are falling heavily, splashing on the ground."
        info = torchaudio.info(file)
        mask_times = [0, info.num_frames / info.sample_rate]
        use_case = Separation(
            input_paths=[file],
            descriptions=[description],
            mask_times=[mask_times],
            video_paths=[video_file],
        )
        batch = next(use_case.prepare_batch(self.model.method, None, self.model.dset))

        res = self.sam.get_transform()(
            descriptions=[description],
            audio_paths=[file],
            video_paths=[video_file],
        )
        diff = (
            res.audios
            - batch["edit_audio"]["spec_or_wav"].squeeze().mean(0, keepdim=True).cpu()
        )
        self.assertAlmostEqual(diff.abs().max().item(), 0, places=5)
        self.assertEqual(batch["description"], res.descriptions)

        res = res.to("cuda")
        encoded_video = self.sam.vision_encoder(res.video)

        diff = (encoded_video - batch["video_features"]["data"].transpose(1, 2)).abs()

        self.assertLess(diff[0, 0].max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
