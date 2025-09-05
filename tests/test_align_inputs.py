import os
import unittest

import torch
import torchaudio
from audiobox.e2e.use_case.audio_editing import Separation

from segment_anything_audio.inputs import prepare_inputs
from tests.models import get_model


class TestAlignInputs(unittest.TestCase):
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

        noise = torch.randn_like(batch["x"])

        with torch.no_grad():
            aligned_input = self.model.method.model.model.prepare_aligned_input(
                {**batch, "noisy_x": noise}
            )

        sam_batch = prepare_inputs(
            [description], [file], self.sam.audio_codec.wav_idx_to_feature_idx
        )
        sam_batch = sam_batch.to("cuda")
        with torch.no_grad():
            video_features = batch["video_features"]["data"]
            video_mask_features = batch["video_mask_features"]["data"]
            sam_aligned_input = self.sam.align_inputs(
                noisy_audio=noise.transpose(1, 2),
                audio_features=batch["edit_audio_embedding"]["seq"],
                video_features=video_features,
                video_mask_features=video_mask_features,
                anchor_ids=sam_batch.anchor_ids,
                anchor_alignment=sam_batch.anchor_alignment,
            )
        self.assertTrue(torch.allclose(aligned_input, sam_aligned_input, atol=1e-5))

    def test_video_separation(self):
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

        noise = torch.randn_like(batch["x"])

        with torch.no_grad():
            aligned_input = self.model.method.model.model.prepare_aligned_input(
                {**batch, "noisy_x": noise}
            )

        sam_batch = prepare_inputs(
            [description],
            [file],
            self.sam.audio_codec.wav_idx_to_feature_idx,
            video_paths=[video_file],
        )
        sam_batch = sam_batch.to("cuda")
        with torch.no_grad():
            video_features = batch["video_features"]["data"]
            video_mask_features = batch["video_mask_features"]["data"]
            sam_aligned_input = self.sam.align_inputs(
                noisy_audio=noise.transpose(1, 2),
                audio_features=batch["edit_audio_embedding"]["seq"],
                video_features=video_features,
                video_mask_features=video_mask_features,
                anchor_ids=sam_batch.anchor_ids,
                anchor_alignment=sam_batch.anchor_alignment,
            )
        self.assertTrue(torch.allclose(aligned_input, sam_aligned_input, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
