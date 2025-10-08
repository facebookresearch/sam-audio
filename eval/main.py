import argparse
import json
import os

import pandas as pd
import torch
import torch.distributed as dist
from metrics import CLAP, Aesthetic, Judge
from musdb import MUSDB
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sam_audio.model.model import SAMAudio

DSETS = {"musdb": MUSDB}


def gather_and_average_results(results, world_size):
    if world_size == 1:
        return json.loads(results.mean().to_json())

    # 1. Gather all dictionaries to all ranks
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(
        all_results, {"sum": results.sum().to_json(), "count": len(results)}
    )

    summed = {}
    counts = 0

    for res in all_results:
        for k, v in json.loads(res["sum"]).items():
            if k not in summed:
                summed[k] = 0.0
            summed[k] += v
        counts += res["count"]

    # 3. Compute average for keys that appeared at least once
    averaged = {k: summed[k] / counts for k in summed}

    return averaged


def main(
    dataset_name: str,
    cache_path: str,
    batch_size: int,
    checkpoint_path: str,
    num_workers: int = 4,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    model = SAMAudio.from_pretrained(checkpoint_path)
    model = model.eval().to(device)
    transform = model.get_transform()

    dset = DSETS[dataset_name](cache_path=cache_path, collate_fn=transform)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dset)

    dl = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dset.collate,
        num_workers=num_workers,
        sampler=sampler,
    )

    all_metrics = [
        Judge(device=device),
        Aesthetic(device=device),
        CLAP(device=device),
    ]

    dfs = []
    for batch in tqdm(dl, disable=rank > 1):
        batch = batch.to(device)
        result = model.separate(batch)
        mets = {}
        sizes = model.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        for metric in all_metrics:
            input_wavs = model.unbatch_wavs(batch.audios.squeeze(1), sizes)
            mets.update(
                metric(
                    target_wavs=result.target,
                    target_wavs_sample_rate=model.sample_rate,
                    descriptions=batch.descriptions,
                    input_wavs=input_wavs,
                )
            )

        dfs.append(pd.DataFrame.from_dict(mets))

    df = pd.concat(dfs)
    averaged_results = gather_and_average_results(df, world_size)
    if rank == 0:
        print(
            json.dumps({k: f"{v:.2f}" for k, v in averaged_results.items()}, indent=4)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset",
        choices=DSETS.keys(),
        help=f"Which dataset to evaluate.  Choices: {DSETS.keys()}",
        default="musdb",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.expanduser("~/.cache/sam_audio"),
        help="Where to cache downloaded datasets",
    )
    parser.add_argument("--checkpoint-path", "-p", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    opt = parser.parse_args()
    main(
        dataset_name=opt.dset,
        cache_path=opt.cache_path,
        batch_size=opt.batch_size,
        checkpoint_path=opt.checkpoint_path,
    )
