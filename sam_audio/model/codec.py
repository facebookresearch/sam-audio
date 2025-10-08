from abc import ABCMeta, abstractmethod
from dataclasses import asdict
from typing import Union

import torch
from dac.model import dac

from sam_audio.model.config import DACVAEConfig


class Codec(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def wav_idx_to_feature_idx(
        self, wav_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @abstractmethod
    def feature_idx_to_wav_idx(
        self, feature_idx: Union[torch.Tensor, int], sample_rate=None
    ) -> Union[torch.Tensor, int]: ...

    @staticmethod
    def cast_to_int(
        x: Union[int, torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return x.int()
        else:
            return int(x)


class DACVAE(Codec):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        kwargs = asdict(config)
        self.mean = kwargs.pop("mean")
        self.std = kwargs.pop("std")
        self.model = dac.DACVAE(**kwargs).eval()
        self.hop_length = self.model.hop_length
        self.sample_rate = config.sample_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.model.encoder(self._pad(waveform))
            mean, scale = self.model.quantizer.in_proj(z).chunk(2, dim=1)
            encoded_frames, _ = self.model.quantizer._vae_sample(mean, scale)
            encoded_frames = (encoded_frames - self.mean) / self.std
        return encoded_frames

    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        emb = self.model.quantizer.out_proj(encoded_frames)
        return self.model.decoder(emb)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs

    def feature_idx_to_wav_idx(self, feature_idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        wav_chunklen = feature_idx * self.hop_length * (orig_freq / new_freq)
        return self.cast_to_int(wav_chunklen)

    def wav_idx_to_feature_idx(self, wav_idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        target_length = torch.ceil(new_freq * wav_idx / orig_freq)
        res = torch.ceil(target_length / self.hop_length)
        return self.cast_to_int(res)
