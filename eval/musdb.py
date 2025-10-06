import os
from io import BytesIO

import torchaudio
from datasets import Audio, load_dataset, load_from_disk
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from tqdm import tqdm


class MUSDB(Dataset):
    def __init__(
        self,
        collate_fn,
        sample_rate: int = 48_000,
        chunk_size_secs: float = 30.0,
        cache_path: str = os.path.expanduser("~/.cache/sam_audio"),
    ):
        self.ds = self.get_dataset(cache_path, chunk_size_secs, sample_rate)
        self.captions = ["bass", "drums", "vocals"]
        self.collate_fn = collate_fn
        self.sample_rate = sample_rate

    def get_dataset(self, cache_path, chunk_size_secs, sample_rate):
        cached_filter_ds = os.path.join(
            cache_path,
            f"filtered_chunk_size_{chunk_size_secs}_sample_rate_{sample_rate}.hf",
        )
        if os.path.exists(cached_filter_ds):
            return load_from_disk(cached_filter_ds)
        else:
            ds = load_dataset(
                "danjacobellis/musdb18HQ", split="validation", cache_dir=cache_path
            )
            ds = ds.filter(lambda x: x["instrument"] == "mixture")
            ds = ds.cast_column("audio", Audio())
            data = {"audio": [], "path": [], "instrument": []}
            for row in tqdm(ds):
                audio = row["audio"]
                samples = audio.get_all_samples()
                stride = int(chunk_size_secs * samples.sample_rate)
                for i in range(0, samples.data.size(-1), stride):
                    bio = BytesIO()
                    torchaudio.save(
                        bio,
                        samples.data[:, i : i + stride],
                        samples.sample_rate,
                        format="wav",
                    )
                    data["audio"].append(bio.getvalue())
                    data["path"].append(row["path"])
                    data["instrument"].append(row["instrument"])

            chunked_ds = HFDataset.from_dict(data).cast_column("audio", Audio())
            chunked_ds.save_to_disk(cached_filter_ds)
            return chunked_ds

    def __len__(self):
        return len(self.ds) * len(self.captions)

    def collate(self, items):
        audios, descriptions = zip(*items)
        return self.collate_fn(
            audios=audios,
            descriptions=descriptions,
        )

    def __getitem__(self, idx):
        file_idx = idx // len(self.captions)
        caption_idx = idx % len(self.captions)
        wav = self.ds[file_idx]["audio"].get_all_samples()
        wav = torchaudio.functional.resample(
            wav.data, wav.sample_rate, self.sample_rate
        )
        return wav.mean(0, keepdim=True), self.captions[caption_idx]


if __name__ == "__main__":
    dataset = MUSDB(lambda **kwargs: None)
    print(len(dataset))
    print(dataset[0])
