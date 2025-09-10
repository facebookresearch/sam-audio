from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import decord
import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def batch_audio(
    audios: list[str | torch.Tensor], audio_sampling_rate: int = 48_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs = []
    for audio in audios:
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
            if sr != audio_sampling_rate:
                wav = torchaudio.functional.resample(wav, sr, audio_sampling_rate)
        else:
            wav = audio
        wavs.append(wav.mean(0))
    sizes = torch.tensor([wav.size(-1) for wav in wavs])
    return pad_sequence(wavs, batch_first=True).unsqueeze(1), sizes


@dataclass(kw_only=True)
class Batch:
    audios: torch.Tensor
    audio_feature_mask: torch.Tensor
    descriptions: list[str]
    anchor_ids: torch.Tensor
    anchor_alignment: torch.Tensor
    audio_pad_mask: Optional[torch.Tensor] = None
    video: Optional[list[torch.Tensor]] = None
    video_mask: Optional[list[torch.Tensor]] = None

    def __post_init__(self):
        assert self.audios.size(0) == len(self.descriptions)

    def to(self, device: torch.device):
        self.audios = self.audios.to(device)
        self.audio_feature_mask = self.audio_feature_mask.to(device)
        self.anchor_ids = self.anchor_ids.to(device)
        self.anchor_alignment = self.anchor_alignment.to(device)
        if self.audio_pad_mask is not None:
            self.audio_pad_mask = self.audio_pad_mask.to(device)
        if self.video is not None:
            self.video = [v.to(device) for v in self.video]
        if self.video_mask is not None:
            self.video_mask = [v.to(device) for v in self.video_mask]

        return self


def mask_from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    return torch.arange(sizes.max()).expand(len(sizes), -1) < sizes.unsqueeze(1)


Anchor = Tuple[str, float, float]


def load_video(sizes: torch.Tensor, video_paths: List[str], fps: Optional[int] = None):
    video = []
    for size, video_path in zip(sizes, video_paths):
        vr = decord.VideoReader(video_path, height=224, width=224)
        frames = vr[:]
        T, H, W, C = frames.shape
        interpolated = F.interpolate(
            rearrange(frames, "t h w c -> c (h w) t"), size=size, mode="nearest"
        )
        video.append(rearrange(interpolated, "c (h w) t -> t c h w", h=H, w=W))
    return video


def prepare_inputs(
    descriptions: list[str],
    audio_paths: list[str],
    wav_to_feature_idx: Callable[[torch.Tensor], torch.Tensor],
    audio_sampling_rate: int = 48_000,
    anchors: Optional[list[list[Anchor]]] = None,
    video_paths: Optional[list[str]] = None,
    video_mask_paths: Optional[list[str]] = None,
):
    decord.bridge.set_bridge("torch")
    assert len(descriptions) == len(audio_paths)
    assert anchors is None or len(descriptions) == len(anchors)
    assert video_paths is None or len(descriptions) == len(video_paths)

    anchor_dict = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}

    audios, wav_sizes = batch_audio(audio_paths, audio_sampling_rate)

    sizes = wav_to_feature_idx(wav_sizes)
    audio_pad_mask = mask_from_sizes(sizes)
    audio_feature_mask = torch.zeros_like(audio_pad_mask)
    if anchors is None:
        anchor_ids = torch.full(
            (len(descriptions), 2), anchor_dict["<null>"], dtype=torch.long
        )
        anchor_ids[:, 1] = anchor_dict["<pad>"]
        anchor_alignment = torch.full(
            (
                len(descriptions),
                audio_pad_mask.size(-1),
            ),
            0,
            dtype=torch.long,
        )
        anchor_alignment[~audio_pad_mask] = 1  # point to pad token
    else:
        anchor_alignment = torch.full(
            (
                len(descriptions),
                audio_pad_mask.size(-1),
            ),
            0,
            dtype=torch.long,
        )
        anchor_alignment[~audio_pad_mask] = 1  # point to pad token
        ids = []

        for i, anchor_list in enumerate(anchors):
            current = [anchor_dict["<null>"], anchor_dict["<pad>"]]
            for token, start_time, end_time in anchor_list:
                start_idx = wav_to_feature_idx(start_time * audio_sampling_rate)
                end_idx = wav_to_feature_idx(end_time * audio_sampling_rate)
                anchor_alignment[i, start_idx:end_idx] = len(current)
                current.append(anchor_dict[token])
            ids.append(torch.tensor(current))
        anchor_ids = pad_sequence(
            ids, batch_first=True, padding_value=anchor_dict["<pad>"]
        )

    video = video_mask = None
    if video_paths is not None:
        video = load_video(sizes, video_paths)
    if video_mask_paths is not None:
        video = load_video(sizes, video_mask_paths)

    return Batch(
        audios=audios,
        audio_feature_mask=audio_feature_mask,
        descriptions=descriptions,
        audio_pad_mask=audio_pad_mask,
        anchor_ids=anchor_ids,
        anchor_alignment=anchor_alignment,
        video=video,
        video_mask=video_mask,
    )
