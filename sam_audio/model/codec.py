# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import dacvae
import torch

from sam_audio.model.config import DACVAEConfig


def _run_with_cudnn(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor):
    """
    Run a codec pass with cuDNN, which is several times faster than PyTorch's
    fallback kernels for the codec's convolutions and the watermarker's LSTMs.
    TF32 stays off to keep full FP32 precision. cuDNN's LSTM rejects sequences
    of 65,536+ steps (inputs longer than ~7 minutes), so on a cuDNN error the
    pass is re-run with cuDNN disabled, as before.
    """
    if x.is_cuda and torch.version.hip is None:
        try:
            with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
                return fn(x)
        except RuntimeError as e:
            if "cudnn" not in str(e).lower():
                raise
    with torch.backends.cudnn.flags(enabled=False):
        return fn(x)


class Encoder(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...


class Codec(Encoder):
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


class DACVAEEncoder(Encoder):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        model = dacvae.DACVAE(
            encoder_dim=config.encoder_dim,
            encoder_rates=config.encoder_rates,
            latent_dim=config.latent_dim,
            decoder_dim=config.decoder_dim,
            decoder_rates=config.decoder_rates,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
        ).eval()
        self._setup_model(model)
        self.hop_length = config.hop_length
        self.sample_rate = config.sample_rate

    def _setup_model(self, model):
        self.encoder = model.encoder
        self.quantizer = model.quantizer

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self._pad(waveform))
        mean, _ = self.quantizer.in_proj(z).chunk(2, dim=1)
        return mean

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _run_with_cudnn(self._encode, waveform)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs


class DACVAE(DACVAEEncoder, Codec):
    def _setup_model(self, model):
        super()._setup_model(model)
        self.decoder = model.decoder

    def _decode(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        emb = self.quantizer.out_proj(encoded_frames)
        return self.decoder(emb)

    def decode(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        return _run_with_cudnn(self._decode, encoded_frames)

    def feature_idx_to_wav_idx(self, feature_idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        wav_chunklen = feature_idx * self.hop_length * (orig_freq / new_freq)
        return self.cast_to_int(wav_chunklen)

    def wav_idx_to_feature_idx(self, wav_idx, sample_rate=None):
        ceil = math.ceil
        if torch.is_tensor(wav_idx):
            ceil = torch.ceil
        if sample_rate is None:
            sample_rate = self.sample_rate
        orig_freq = sample_rate
        new_freq = self.sample_rate
        target_length = ceil(new_freq * wav_idx / orig_freq)
        res = ceil(target_length / self.hop_length)
        return self.cast_to_int(res)
