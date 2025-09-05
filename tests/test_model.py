import os
import unittest

import torch
import torchaudio
from audiobox.e2e.use_case.audio_editing import Separation

from tests.models import get_model


class TestAlignInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = os.path.dirname(os.path.realpath(__file__))
        cls.model = get_model("audiobox")
        cls.sam = get_model("sam")

    def test_transformer(self):
        time = torch.tensor([0.5], device="cuda")
        ali_inp = torch.rand(1, 250, 2048, device="cuda")
        cross_x = torch.rand(1, 125, 2048, device="cuda")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ab_output = self.model.method.model.model.model_forward(
                ali_inp=ali_inp, batch={}, memory=cross_x, time=time
            )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            sam_output = self.sam.transformer(ali_inp, time, memory=cross_x)

        self.assertLess((sam_output.transpose(1, 2) - ab_output).abs().max(), 0.05)

    def test_text_based_separation(self):
        ab_acts = {}
        sam_acts = {}

        from functools import partial

        def hook(acts, name, module, args, output):
            acts[name] = output

        for name, module in self.model.method.named_modules():
            module.register_forward_hook(partial(hook, ab_acts, name))

        for name, module in self.sam.named_modules():
            module.register_forward_hook(partial(hook, sam_acts, name))

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

        time = torch.tensor([0.5], device=batch["x"].device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ab_output = self.model.method.model(
                {**batch, "noisy_x": noise}, timesteps=time
            )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            sam_output = self.sam.forward(
                noisy_audio=noise.transpose(1, 2),
                audio_features=batch["edit_audio_embedding"]["seq"],
                text_features=batch["description_embedding"]["seq"],
                time=time,
                video_features=batch["video_features"]["data"],
                video_mask_features=batch["video_mask_features"]["data"],
                text_mask=batch["description_embedding"]["mask"],
                anchor_ids=batch["phonemes"],
                anchor_alignment=batch["alignment"],
                audio_pad_mask=~batch["edit_audio_embedding"]["mask"],
            )

        self.assertLess((sam_output.transpose(1, 2) - ab_output).abs().max(), 0.05)


if __name__ == "__main__":
    unittest.main()
