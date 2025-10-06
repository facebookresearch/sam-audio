from typing import List, Union

import torch
import torchaudio

from sam_audio.model.config import ImageBindRankerConfig
from sam_audio.ranking.ranker import Ranker

try:
    from imagebind.data import (
        ConstantClipsPerVideoSampler,
        get_clip_timepoints,
        load_and_transform_video_data,
        transforms,
        waveform2melspec,
    )
    from imagebind.models.imagebind_model import ModalityType, imagebind_huge

    __imagebind_exists__ = True
except ImportError:
    __imagebind_exists__ = False


def load_and_transform_audio_data(
    audios: List[Union[str, torch.Tensor]],
    input_sample_rate=None,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
    device=None,
):
    if audios is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio in audios:
        if isinstance(audio, str):
            waveform, input_sample_rate = torchaudio.load(audio)
        else:
            assert torch.is_tensor(audio)
            assert sample_rate is not None
            # Preprocessing needs to be done in full precision
            waveform = audio.float()
            if waveform.ndim == 1:
                waveform = waveform[None]
        if sample_rate != input_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=input_sample_rate, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


class ImageBindRanker(Ranker):
    def __init__(self, cfg: ImageBindRankerConfig):
        super().__init__()
        assert __imagebind_exists__, (
            "Install ImageBind in order to use this ranker: https://github.com/facebookresearch/ImageBind/tree/main"
        )

        self.model = imagebind_huge(pretrained=cfg.checkpoint is None)
        if cfg.checkpoint is not None:
            self.model.load_state_dict(torch.load(cfg.checkpoint, map_location="cpu"))
        self.model = self.model.eval()

    @torch.inference_mode()
    def forward(
        self,
        extracted_audio: list[torch.Tensor],
        video_paths: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        audio_data = torch.cat(
            [
                load_and_transform_audio_data(x, input_sample_rate=sample_rate)
                for x in extracted_audio
            ],
            dim=0,
        )
        video_data = load_and_transform_video_data(video_paths, audio_data.device)

        inputs = {ModalityType.AUDIO: audio_data, ModalityType.VISION: video_data}
        embs = self.model(inputs)
        audio_embs, video_embs = embs[ModalityType.AUDIO], embs[ModalityType.VISION]
        audio_embs, video_embs = (
            audio_embs / ((audio_embs**2).sum(dim=-1, keepdims=True) ** 0.5),
            video_embs / ((video_embs**2).sum(dim=-1, keepdims=True) ** 0.5),
        )
        bsz = len(extracted_audio)
        candidates = len(audio_embs) // bsz
        scores = audio_embs.view(bsz, candidates, -1) @ video_embs.view(bsz, -1, 1)
        return scores
