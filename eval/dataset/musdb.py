# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import os
from subprocess import check_call
from typing import Optional, Tuple

import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset


def _load_audio_with_range(
    path: str,
    start_seconds: Optional[float] = None,
    stop_seconds: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio from file and extract a time range.

    Args:
        path: Path to audio file.
        start_seconds: Start time in seconds (None for beginning).
        stop_seconds: End time in seconds (None for end).

    Returns:
        Tuple of (audio tensor, sample rate).
    """
    wav, sr = torchaudio.load(path)
    if start_seconds is not None or stop_seconds is not None:
        start_sample = int(start_seconds * sr) if start_seconds else 0
        end_sample = int(stop_seconds * sr) if stop_seconds else wav.size(-1)
        wav = wav[:, start_sample:end_sample]
    return wav, sr


def cache_file(url, outfile):
    if not os.path.exists(outfile):
        print("Downloading musdb18hq dataset...")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        check_call(["curl", "--url", url, "--output", outfile + ".tmp"])
        os.rename(outfile + ".tmp", outfile)


class MUSDB(Dataset):
    def __init__(
        self,
        collate_fn,
        sample_rate: int = 48_000,
        cache_path: str = os.path.expanduser("~/.cache/sam_audio"),
    ):
        self.cache_path = os.path.join(cache_path, "musdb18hq")
        self.ds = self.get_dataset(cache_path)
        self.captions = ["bass", "drums", "vocals"]
        self.collate_fn = collate_fn
        self.sample_rate = sample_rate

    @property
    def visual(self):
        return False

    def get_dataset(self, cache_path):
        zip_file = os.path.join(cache_path, "musdb18hq.zip")
        url = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
        cache_file(url, zip_file)
        extracted_dir = os.path.join(cache_path, "musdb18hq")
        if not os.path.exists(extracted_dir):
            check_call(["unzip", zip_file, "-d", extracted_dir + ".tmp"])
            os.rename(extracted_dir + ".tmp", extracted_dir)
        return load_dataset("facebook/sam-audio-musdb18hq-test")["test"]

    def __len__(self):
        return len(self.ds)

    def collate(self, items):
        audios, descriptions = zip(*items, strict=False)
        return self.collate_fn(
            audios=audios,
            descriptions=descriptions,
        )

    def __getitem__(self, idx):
        item = self.ds[idx]
        path = os.path.join(self.cache_path, "test", item["id"], "mixture.wav")
        assert os.path.exists(path), f"{path} does not exist!"
        wav, sample_rate = _load_audio_with_range(
            path, item["start_time"], item["end_time"]
        )
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sample_rate, self.sample_rate
            )
        wav = wav.mean(0, keepdim=True)
        return wav, item["description"]


if __name__ == "__main__":
    dataset = MUSDB(lambda **kwargs: None)
    print(len(dataset))
    print(dataset[0])
