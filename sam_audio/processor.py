# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import logging
import math
import os
from typing import Callable, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from torchcodec.decoders import AudioDecoder, VideoDecoder
from transformers import AutoTokenizer, BatchFeature

from sam_audio.model.config import SAMAudioConfig, SAMAudioJudgeConfig

logger = logging.getLogger(__name__)

Anchor = Tuple[str, float, float]


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


class Batch:
    def __init__(
        self,
        audios: torch.Tensor,
        sizes: torch.Tensor,
        wav_sizes: torch.Tensor,
        descriptions: list[str],
        hop_length: int,
        audio_sampling_rate: int,
        anchors: Optional[list[list[Anchor]]] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
        masked_video: Optional[torch.Tensor] = None,
    ):
        self.audios = audios
        self.sizes = sizes
        self.wav_sizes = wav_sizes
        self.descriptions = descriptions
        self.audio_pad_mask = audio_pad_mask
        self.masked_video = masked_video
        self.hop_length = hop_length
        self.audio_sampling_rate = audio_sampling_rate
        self.process_anchors(anchors)
        assert self.audios.size(0) == len(self.descriptions)

    def _wav_to_feature_idx(self, wav_idx: int):
        return math.ceil(wav_idx / self.hop_length)

    def to(self, device: torch.device):
        self.audios = self.audios.to(device)
        self.anchor_ids = self.anchor_ids.to(device)
        self.anchor_alignment = self.anchor_alignment.to(device)
        self.sizes = self.sizes.to(device)
        self.wav_sizes = self.wav_sizes.to(device)
        if self.audio_pad_mask is not None:
            self.audio_pad_mask = self.audio_pad_mask.to(device)
        if self.masked_video is not None:
            self.masked_video = [v.to(device) for v in self.masked_video]
        return self

    def process_anchors(self, anchors: Optional[list[list[Anchor]]]):
        batch_size = len(self.audios)
        anchor_dict = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}
        if anchors is None:
            anchor_ids = torch.full(
                (batch_size, 2), anchor_dict["<null>"], dtype=torch.long
            )
            anchor_ids[:, 1] = anchor_dict["<pad>"]
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
        else:
            anchor_alignment = torch.full(
                (
                    batch_size,
                    self.audio_pad_mask.size(-1),
                ),
                0,
                dtype=torch.long,
            )
            anchor_alignment[~self.audio_pad_mask] = 1  # point to pad token
            ids = []

            for i, anchor_list in enumerate(anchors):
                current = [anchor_dict["<null>"], anchor_dict["<pad>"]]
                for token, start_time, end_time in anchor_list:
                    start_idx = self._wav_to_feature_idx(
                        start_time * self.audio_sampling_rate
                    )
                    end_idx = self._wav_to_feature_idx(
                        end_time * self.audio_sampling_rate
                    )
                    anchor_alignment[i, start_idx:end_idx] = len(current)
                    current.append(anchor_dict[token])
                ids.append(torch.tensor(current))
            anchor_ids = pad_sequence(
                ids, batch_first=True, padding_value=anchor_dict["<pad>"]
            )
        self.anchor_ids = anchor_ids.to(self.audios.device)
        self.anchor_alignment = anchor_alignment.to(self.audios.device)
        self.anchors = anchors


def mask_from_sizes(sizes: torch.Tensor) -> torch.Tensor:
    return torch.arange(sizes.max()).expand(len(sizes), -1) < sizes.unsqueeze(1)


def load_video(
    sizes: torch.Tensor,
    videos: List[str],
    feature_idx_to_wav_idx: Callable[[torch.Tensor], torch.Tensor],
    audio_sampling_rate: int,
) -> list[torch.Tensor]:
    all_frames = []
    for size, video in zip(sizes, videos, strict=False):
        audio_timestamps = (
            feature_idx_to_wav_idx(torch.arange(size)) / audio_sampling_rate
        )
        if isinstance(video, str):
            decoder = VideoDecoder(video, dimension_order="NCHW")
            data = decoder.get_frames_in_range(0, len(decoder))
            diffs = (audio_timestamps[None] - data.pts_seconds[:, None]).abs()
            frame_idxs = diffs.argmin(dim=0)
            frames = data.data[frame_idxs]
        else:
            assert video.size(1) == 3, (
                f"Expected video tensor to be in NCHW format, but found {video.size(1)} channels"
            )
            idx = torch.linspace(0, video.size(0) - 1, int(size)).round().long()
            frames = video[idx]
        all_frames.append(frames)
    return all_frames


class Processor:
    config_cls: Callable

    def __init__(self, audio_hop_length: int, audio_sampling_rate: int):
        self.audio_hop_length = audio_hop_length
        self.audio_sampling_rate = audio_sampling_rate

    @classmethod
    def _get_config(cls, model_name_or_path: str):
        if os.path.exists(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json",
                revision=cls.revision,
            )
        with open(config_path) as fin:
            config = cls.config_cls(**json.load(fin))
        return config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Processor":
        config = cls._get_config(model_name_or_path)
        return cls(
            audio_hop_length=config.audio_codec.hop_length,
            audio_sampling_rate=config.audio_codec.sample_rate,
        )

    def feature_to_wav_idx(self, feature_idx):
        return feature_idx * self.audio_hop_length

    def wav_to_feature_idx(self, wav_idx):
        if torch.is_tensor(wav_idx):
            ceil = torch.ceil
        else:
            ceil = math.ceil
        return ceil(wav_idx / self.audio_hop_length)

    def mask_videos(
        self,
        videos: List[str | torch.Tensor],
        masks: List[str | torch.Tensor],
    ) -> list[torch.Tensor]:
        video = [VideoDecoder(v)[:] if isinstance(v, str) else v for v in videos]
        video_mask = [VideoDecoder(v)[:] if isinstance(v, str) else v for v in masks]
        return [v * m.eq(0) for v, m in zip(video, video_mask, strict=False)]


class SAMAudioProcessor(Processor):
    config_cls = SAMAudioConfig
    revision = None

    def __call__(
        self,
        descriptions: list[str],
        audios: list[str | torch.Tensor],
        anchors: Optional[list[list[Anchor]]] = None,
        masked_videos: Optional[list[str | torch.Tensor]] = None,
    ):
        """
        Processes input data for the model.

        Args:
            descriptions (list[str]): List of text descriptions corresponding to each audio sample.
            audios (list[str]): List of audio file paths or tensors.
                If a tensor:
                    - should have shape (channels, time) where channels=1 for mono and 2 for stereo.
                    - should be resampled to 48_000 hz
            anchors (Optional[list[list[Anchor]]], optional): List of anchors for each sample,
                where each anchor is a tuple (token, start_time, end_time).
            masked_videos (Optional[list[str | torch.Tensor]], optional): List of masked video file paths or tensors.
                If a tensor, should have shape (N, C, H, W)

        Returns:
            Batch: A Batch object containing processed audio, sizes, descriptions, anchor ids, anchor alignment, audio pad mask, and optionally masked video.
        """

        assert len(descriptions) == len(audios)
        assert anchors is None or len(descriptions) == len(anchors)
        assert masked_videos is None or len(descriptions) == len(masked_videos)

        audios, wav_sizes = batch_audio(audios, self.audio_sampling_rate)

        sizes = self.wav_to_feature_idx(wav_sizes)
        audio_pad_mask = mask_from_sizes(sizes)
        masked_video = None
        if masked_videos is not None:
            masked_video = load_video(
                sizes, masked_videos, self.feature_to_wav_idx, self.audio_sampling_rate
            )

        return Batch(
            audios=audios,
            sizes=sizes,
            descriptions=descriptions,
            audio_pad_mask=audio_pad_mask,
            anchors=anchors,
            masked_video=masked_video,
            hop_length=self.audio_hop_length,
            audio_sampling_rate=self.audio_sampling_rate,
            wav_sizes=wav_sizes,
        )


class SAMAudioJudgeProcessor(Processor):
    config_cls = SAMAudioJudgeConfig
    revision = "sam_audio"

    def __init__(
        self,
        audio_hop_length: int,
        audio_sampling_rate: int,
        tokenizer: AutoTokenizer,
    ):
        super().__init__(audio_hop_length, audio_sampling_rate)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "SAMAudioJudgeProcessor":
        config = cls._get_config(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(
            audio_hop_length=config.audio_codec.hop_length,
            audio_sampling_rate=config.audio_codec.sample_rate,
            tokenizer=tokenizer,
        )

    def _reflect_pad(self, wav):
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.size(-1) % self.audio_hop_length == 0:
            return wav
        p1d = (0, self.audio_hop_length - (wav.size(-1) % self.audio_hop_length))
        return torch.nn.functional.pad(wav, p1d, mode="reflect")

    def _load_audio(self, path: str):
        ad = AudioDecoder(path, sample_rate=self.audio_sampling_rate, num_channels=1)
        return ad.get_all_samples().data

    def _process_audio(
        self,
        raw_audio,
        sampling_rate: Optional[int] = None,
    ):
        from_file = False
        if isinstance(raw_audio, str):
            raw_audio = [raw_audio]

        if isinstance(raw_audio, (list, tuple)) and isinstance(raw_audio[0], str):
            loaded = []
            for audio_file in raw_audio:
                loaded.append(self._load_audio(audio_file))
            raw_audio = loaded
            from_file = True

        if sampling_rate is not None:
            if sampling_rate != self.audio_sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.audio_sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.audio_sampling_rate} and not {sampling_rate}."
                )
        elif not from_file:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if isinstance(raw_audio, list):
            raw_audio = [self._reflect_pad(x).T for x in raw_audio]
        else:
            raw_audio = self._reflect_pad(raw_audio).T

        # verify inputs are valid
        for example in raw_audio:
            if example.ndim > 2:
                raise ValueError(
                    f"Expected input shape (channels, num_samples), but got shape ({example.shape})"
                )

        lengths = torch.tensor([x.size(0) for x in raw_audio])
        input_values = pad_sequence(raw_audio, batch_first=True).transpose(1, 2)
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]

        return BatchFeature(
            {"input_values": input_values, "padding_mask": padding_mask}
        )

    def __call__(
        self,
        text: Optional[str] = None,
        input_audio: Optional[
            str | list[str] | torch.Tensor | list[torch.Tensor]
        ] = None,
        separated_audio: Optional[
            str | list[str] | torch.Tensor | list[torch.Tensor]
        ] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
        batch = BatchFeature()
        if text is not None:
            batch.update(
                self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    max_length=512,
                    truncation=True,
                )
            )

        if input_audio is not None:
            batch.update(self._process_audio(input_audio, sampling_rate))

        if separated_audio is not None:
            batch["separated_values"] = self._process_audio(
                separated_audio, sampling_rate
            )["input_values"]

        return batch


__all__ = ["SAMAudioProcessor", "SAMAudioJudgeProcessor", "Batch"]
