import os
import unittest
from unittest.mock import patch

import torch
import torchaudio
from audiobox.e2e.use_case.audio_editing import Separation

from sam_audio.inputs import batch_audio
from sam_audio.model.model import DFLT_ODE_OPT
from tests.models import get_model, get_transformer


class TestAlignInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = os.path.dirname(os.path.realpath(__file__))
        overrides = [
            "data.dataset.conditioning.video.av_alignment=false",
            "data.dataset.conditioning.video_mask.av_alignment=false",
        ]
        cls.model = get_model("audiobox", additional_overrides=overrides)
        cls.sam = get_model("sam")

    def test_transformer(self):
        time = torch.tensor([0.5], device="cuda")
        ali_inp = torch.rand(1, 250, 2048, device="cuda")
        cross_x = torch.rand(1, 125, 2048, device="cuda")
        model = get_transformer(self.model)

        with torch.no_grad():
            ab_output = model.model_forward(
                ali_inp=ali_inp, batch={}, memory=cross_x, time=time
            )

        with torch.no_grad():
            sam_output = self.sam.transformer(ali_inp, time, memory=cross_x)
        self.assertLess((sam_output.transpose(1, 2) - ab_output).abs().max(), 1e-4)

    def test_text_based_separation_forward(self):
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

        transform = self.sam.get_transform()
        sam_batch = transform(
            descriptions=use_case.descriptions, audios=use_case.input_paths
        ).to("cuda")

        with torch.no_grad():
            sam_kwargs = self.sam._get_forward_args(sam_batch)

        for t in [0.0, 0.5, 1.0]:
            time = torch.tensor([t], device=batch["x"].device)

            with torch.no_grad():
                ab_output = self.model.method.model(
                    {**batch, "noisy_x": noise}, timesteps=time
                )

            with torch.no_grad():
                sam_kwargs["audio_features"] = batch["edit_audio_embedding"]["seq"]
                sam_output = self.sam.forward(
                    noisy_audio=noise.transpose(1, 2), time=time, **sam_kwargs
                )
            self.assertLess((sam_output.transpose(1, 2) - ab_output).abs().max(), 1e-4)

    def get_file(self, basename):
        return os.path.join(self.dir, "data", basename)

    def test_batched(self):
        files = [
            self.get_file("702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"),
            self.get_file("702459_6464538-hq_690345_1453392-hq_snr-3.0_shortened.wav"),
        ]
        video_files = [
            self.get_file("15pi8h_bHQE_173000_183000.mp4"),
            self.get_file("15pi8h_bHQE_173000_183000_shortened.mp4"),
        ]
        video_mask_files = [
            self.get_file("15pi8h_bHQE_173000_183000_mask.mp4"),
            self.get_file("15pi8h_bHQE_173000_183000_shortened_mask.mp4"),
        ]
        descriptions = [
            "Raindrops are falling heavily, splashing on the ground.",
            "Raindrops are falling heavily",
        ]
        anchors = [
            [["+", 0.567, 0.795], ["+", 3.173, 3.591]],
            [["+", 0.567, 0.795]],
        ]

        transform = self.sam.get_transform()

        def forward(descriptions, paths, videos, video_masks, anchors, features, noise):
            batch = transform(
                descriptions=descriptions,
                audios=paths,
                video_paths=videos,
                video_mask_paths=video_masks,
                anchors=anchors,
            ).to("cuda")
            time = torch.tensor([0.5] * len(descriptions), device="cuda")
            with torch.no_grad():
                sam_kwargs = self.sam._get_forward_args(batch)
                sam_kwargs["audio_features"] = features
                sam_output = self.sam.forward(
                    noisy_audio=noise, time=time, **sam_kwargs
                )
            return sam_output

        wavs, sizes = batch_audio(files)
        sizes = self.sam.audio_codec.wav_idx_to_feature_idx(sizes)
        features = self.sam.audio_codec(wavs.cuda())
        features = torch.cat([features, features], dim=1).transpose(1, 2)
        noise = torch.randn_like(features)

        # x_embedder does not respect masking
        old_x_embedder = self.sam.transformer.x_embedder
        self.sam.transformer.x_embedder = torch.nn.Identity()
        batched = forward(
            descriptions, files, video_files, video_mask_files, anchors, features, noise
        )
        single = forward(
            descriptions[1:],
            files[1:],
            video_files[1:],
            video_mask_files[1:],
            anchors[1:],
            features[[1], : sizes[1]],
            noise[[1], : sizes[1]],
        )
        self.sam.transformer.x_embedder = old_x_embedder

        diff = (batched[[1], : sizes[1]] - single).abs()
        self.assertLess(diff.max(), 1e-4)

    def check_wav(self, generated, hyp, places=2):
        generated = generated[None]
        diff = (generated - hyp).abs()
        corr = torch.corrcoef(torch.cat([generated, hyp]))
        self.assertAlmostEqual(diff.max().item(), 0, places=places)
        self.assertGreater(corr.min().item(), 0.99)

    def _test_e2e(self, use_case, batch, places=2):
        with torch.no_grad():
            # Use the same noise
            ab_batch = next(
                use_case.prepare_batch(self.model.method, None, self.model.dset)
            )
            noise = torch.randn_like(ab_batch["x"])
            ab_res = self.model.samplers["audio_sampler"].sample(
                self.model.method, noise, extra=ab_batch, ode_opts=DFLT_ODE_OPT
            )

        # Note that we patch _get_audio_features to cope with randomness from dacvae
        with torch.no_grad(), patch.object(
            self.sam,
            "_get_audio_features",
            return_value=ab_batch["edit_audio_embedding"]["seq"],
        ):
            sam_res = self.sam.separate(batch, noise=noise.transpose(1, 2))

        self.check_wav(sam_res.target[0], ab_res["wav"][0], places=places)
        self.check_wav(sam_res.residual[0], ab_res["rest_wav"][0], places=places)

    def test_text_based_separation_e2e(self):
        torch.manual_seed(0)
        file = os.path.join(
            self.dir, "data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"
        )
        description = "Raindrops are falling heavily, splashing on the ground."
        info = torchaudio.info(file)
        mask_times = [0, info.num_frames / info.sample_rate]

        transform = self.sam.get_transform()

        batch = transform(
            descriptions=[description],
            audios=[file],
        )
        batch = batch.to("cuda")
        use_case = Separation(
            input_paths=[file], descriptions=[description], mask_times=[mask_times]
        )
        self._test_e2e(use_case, batch)

    def test_text_based_separation_w_anchors_e2e(self):
        torch.manual_seed(0)
        file = os.path.join(
            self.dir, "data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"
        )
        description = "Raindrops are falling heavily, splashing on the ground."
        info = torchaudio.info(file)
        mask_times = [0, info.num_frames / info.sample_rate]
        transform = self.sam.get_transform()
        anchors = [[["+", 0.567, 0.795], ["+", 3.173, 3.591]]]
        batch = transform(descriptions=[description], audios=[file], anchors=anchors)
        batch = batch.to("cuda")
        use_case = Separation(
            input_paths=[file],
            descriptions=[description],
            mask_times=[mask_times],
            anchors=anchors,
        )
        self._test_e2e(use_case, batch)

    def test_video_based_separation_e2e(self):
        torch.manual_seed(0)
        file = os.path.join(
            self.dir, "data/702459_6464538-hq_690345_1453392-hq_snr-3.0.wav"
        )
        description = "Raindrops are falling heavily, splashing on the ground."
        video_file = os.path.join(self.dir, "data/15pi8h_bHQE_173000_183000.mp4")
        mask_file = os.path.join(self.dir, "data/15pi8h_bHQE_173000_183000_mask.mp4")
        info = torchaudio.info(file)
        mask_times = [0, info.num_frames / info.sample_rate]
        transform = self.sam.get_transform()
        anchors = [[["+", 0.567, 0.795], ["+", 3.173, 3.591]]]
        batch = transform(
            descriptions=[description],
            audios=[file],
            anchors=anchors,
            video_paths=[video_file],
            video_mask_paths=[mask_file],
        )
        batch = batch.to("cuda")
        use_case = Separation(
            input_paths=[file],
            descriptions=[description],
            mask_times=[mask_times],
            anchors=anchors,
            video_paths=[video_file],
            extra_items=[{"video_mask_path": mask_file}],
        )
        self._test_e2e(use_case, batch, places=2)


if __name__ == "__main__":
    unittest.main()
