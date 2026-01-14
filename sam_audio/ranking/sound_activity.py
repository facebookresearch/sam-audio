# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from io import BytesIO
from typing import Tuple, Union

import torch
import torchaudio

from ..model.config import SoundActivityRankerConfig
from .ranker import Ranker

try:
    import pydub
except ImportError:
    pydub = None


def get_peak_rms(audio, win_ms=250, hop_ms=100):
    """
    win_length and hop_length are in ms
    """
    last_slice_start = len(audio) - win_ms
    slice_starts = range(0, last_slice_start + 1, hop_ms)
    peak_rms = -1
    for i in slice_starts:
        audio_slice = audio[i : i + win_ms]
        peak_rms = max(peak_rms, audio_slice.rms / audio.max_possible_amplitude)
    # Ensure peak_rms is positive
    peak_rms = max(peak_rms, 0)
    return peak_rms


def torch_tensor_to_pydub(wav: torch.Tensor, sample_rate: int):
    bytesio = BytesIO()
    # Ensure wav has correct shape for torchaudio.save (channels, samples)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(bytesio, wav, sample_rate, format="wav")
    bytesio.seek(0)
    audio = pydub.AudioSegment.from_file(bytesio, format="wav")
    return audio


def detect_nonsilent(
    path: Union[str, Tuple[torch.Tensor, int]],  # either a file path or pair wav & sr
    min_sil_ms=250,
    sil_threshold=-40,
    threshold_mode="rel_to_max",
):
    TH_MODES = {"abs", "rel_to_max"}
    SAMPLE_RATE = 24_000
    assert threshold_mode in TH_MODES, f"{threshold_mode=} not in {TH_MODES}"
    if isinstance(path, str):
        audio = pydub.AudioSegment.from_file(path)
    else:  # tuple of (tensor, sr)
        audio = torch_tensor_to_pydub(path[0], path[1])
    audio = audio.set_frame_rate(SAMPLE_RATE)
    if threshold_mode == "rel_to_max":
        peak_rms = get_peak_rms(audio)
        sil_threshold = sil_threshold + pydub.utils.ratio_to_db(
            peak_rms
        )  # convert to absolute db threshold
    elif threshold_mode == "abs":
        pass
    else:
        raise NotImplementedError(f"Unknown threshold_mode '{threshold_mode}'")
    spans = pydub.silence.detect_nonsilent(
        audio, min_silence_len=min_sil_ms, silence_thresh=sil_threshold, seek_step=10
    )
    spans = [(round(start / 1000, 3), round(end / 1000, 3)) for start, end in spans]
    return spans


def compute_iou_recall_precision(hyp_spans, ref_spans):
    def span_length(span):
        return span[1] - span[0]

    def intersection_length(span1, span2):
        return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0]))

    total_hyp_length = sum(span_length(span) for span in hyp_spans)
    total_ref_length = sum(span_length(span) for span in ref_spans)
    total_intersection = 0
    for hyp_span in hyp_spans:
        for ref_span in ref_spans:
            total_intersection += intersection_length(hyp_span, ref_span)

    union_spans = hyp_spans + ref_spans  # Combine both lists to compute union
    union_length = sum(span_length(span) for span in union_spans) - total_intersection

    iou = total_intersection / union_length if union_length > 0 else 0
    recall = total_intersection / total_ref_length if total_ref_length > 0 else 0
    precision = total_intersection / total_hyp_length if total_hyp_length > 0 else 0

    return {"iou": iou, "recall": recall, "precision": precision}


class SoundActivityRanker(Ranker):
    def __init__(self, config: SoundActivityRankerConfig):
        if pydub is None:
            raise ImportError(
                'Install reranking dependencies: `pip install "sam-audio[reranking]"`'
            )
        super().__init__()
        self.config = config

    @torch.inference_mode()
    def forward(
        self,
        extracted_audio: list[torch.Tensor],
        spans: list[list[list[float]]],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        device = extracted_audio[0].device
        scores = []
        for wav, current_spans in zip(extracted_audio, spans, strict=True):
            wav = wav.to(torch.float32).cpu()
            # get non-silent spans
            hyp_spans = detect_nonsilent(
                (wav, sample_rate),
                sil_threshold=self.config.sil_threshold,
                threshold_mode=self.config.threshold_mode,
            )
            timestamps = [[span[1], span[2]] for span in current_spans]
            result = compute_iou_recall_precision(hyp_spans, timestamps)
            scores.append(result[self.config.metric])

        # convert to tensor
        scores = torch.tensor(scores, device=device)
        return scores
