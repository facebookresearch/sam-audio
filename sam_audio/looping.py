"""Utilities to loop audio segments with crossfades for long runs.

Functions:
 - loop_segment_with_crossfade: repeat a time range until a target duration, applying crossfades
 - crossfade_concat: concatenate segments with a fixed crossfade length

Accepts mono or multi-channel tensors (shape (T,) or (C, T)).
"""
from typing import Optional

import math
import torch
import torchaudio


def _ensure_chan_time(wav: torch.Tensor) -> torch.Tensor:
    """Return tensor with shape (C, T)."""
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    return wav


def crossfade_concat(segments: list[torch.Tensor], crossfade_samples: int) -> torch.Tensor:
    """Concatenate a list of segments with crossfade overlap.

    Args:
        segments: list of tensors shape (C, T) or (T,)
        crossfade_samples: number of samples to overlap between adjacent segments

    Returns:
        tensor shape (C, T_out)
    """
    if not segments:
        raise ValueError("No segments provided")

    segs = [_ensure_chan_time(s) for s in segments]
    channels = segs[0].size(0)
    for s in segs:
        if s.size(0) != channels:
            raise ValueError("All segments must have same number of channels")

    if crossfade_samples <= 0:
        return torch.cat(segs, dim=-1)

    fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=segs[0].device)
    fade_out = 1.0 - fade_in

    out = segs[0].clone()
    for seg in segs[1:]:
        if seg.size(-1) < crossfade_samples:
            # pad the segment to be at least crossfade length
            pad = crossfade_samples - seg.size(-1)
            seg = torch.nn.functional.pad(seg, (0, pad))

        # apply crossfade on overlap region
        out_end = out[..., -crossfade_samples :]
        seg_start = seg[..., :crossfade_samples]
        mixed = out_end * fade_out + seg_start * fade_in
        out = torch.cat([out[..., :-crossfade_samples], mixed, seg[..., crossfade_samples :]], dim=-1)

    return out


def loop_segment_with_crossfade(
    wav: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    target_duration_sec: float,
    crossfade_sec: Optional[float] = 0.05,
) -> torch.Tensor:
    """Loop the time range [start_sec, end_sec) from `wav` until reaching `target_duration_sec`.

    The function overlaps adjacent repeats by `crossfade_sec` seconds and applies smooth fades
    to avoid clicks.

    Args:
        wav: waveform tensor (T,) or (C, T) sampled at `sample_rate`.
        sample_rate: sampling rate in Hz.
        start_sec: segment start time in seconds (inclusive).
        end_sec: segment end time in seconds (exclusive).
        target_duration_sec: desired output duration in seconds.
        crossfade_sec: crossfade overlap in seconds (default 50 ms).

    Returns:
        waveform tensor shape (C, T_out) where T_out ~= target_duration_sec * sample_rate.
    """
    if end_sec <= start_sec:
        raise ValueError("end_sec must be > start_sec")
    if target_duration_sec <= 0:
        raise ValueError("target_duration_sec must be > 0")

    wav = _ensure_chan_time(wav)
    start = int(round(start_sec * sample_rate))
    end = int(round(end_sec * sample_rate))
    seg = wav[..., start:end]
    seg_len = seg.size(-1)
    if seg_len == 0:
        raise ValueError("Selected segment is empty")

    target_samples = int(round(target_duration_sec * sample_rate))
    crossfade_samples = int(round((crossfade_sec or 0.0) * sample_rate))

    # If target shorter or equal to segment, just trim and return
    if target_samples <= seg_len:
        return seg[..., :target_samples]

    # Effective advance per repeat when overlapping = seg_len - crossfade_samples
    advance = seg_len - crossfade_samples
    if advance <= 0:
        raise ValueError("crossfade_sec is too long relative to segment length")

    # Compute needed repeats N: total = seg_len + (N-1)*advance >= target_samples
    needed = math.ceil((target_samples - seg_len) / advance) + 1

    segments = [seg.clone() for _ in range(needed)]
    out = crossfade_concat(segments, crossfade_samples)
    # Trim to exact target_samples
    return out[..., :target_samples]


def save_tensor_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Save waveform tensor using torchaudio.save. Accepts (C,T) or (T,)"""
    if wav.ndim == 2 and wav.size(0) <= 4 and wav.size(1) > wav.size(0):
        # assume (C, T)
        data = wav
    elif wav.ndim == 1:
        data = wav.unsqueeze(0)
    else:
        # conservative reshape: ensure (C, T)
        data = wav
    torchaudio.save(path, data, sample_rate)
