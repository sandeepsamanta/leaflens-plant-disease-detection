"""Run the frozen stem once over images and cache its output as float16.

    python -m ml.cache_features --view A --splits train val test      # original photos
    python -m ml.cache_features --view B --splits train               # leaf pasted onto field backgrounds
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.backbone import FEATURE_SHAPE, Stem, load_pretrained
from ml.preprocess import to_tensor_array

ROOT = Path(__file__).resolve().parents[1]
COLOR = ROOT / "data" / "raw" / "raw" / "color"
CACHE = ROOT / "data" / "cache"


class ViewA(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.from_numpy(to_tensor_array(Image.open(COLOR / self.items[i][0])))


def run(view: str, split: str, items: list, batch: int, workers: int) -> None:
    out = CACHE / f"{view}_{split}.npy"
    if out.exists() and (CACHE / f"{view}_{split}.done").exists():
        print(f"{out.name}: already done")
        return
    if view == "A":
        ds = ViewA(items)
    else:
        from ml.composite import ViewB
        ds = ViewB(items)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(items), *FEATURE_SHAPE))
    np.save(CACHE / f"labels_{split}.npy", np.array([l for _, l in items], dtype=np.int64))
    stem = Stem(load_pretrained()).eval().to(memory_format=torch.channels_last)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=False)
    i, t0 = 0, time.time()
    with torch.inference_mode():
        for x in dl:
            f = stem(x.contiguous(memory_format=torch.channels_last))
            mm[i : i + len(f)] = f.numpy().astype(np.float16)
            i += len(f)
            if (i // batch) % 20 == 0:
                print(f"{view}/{split}: {i}/{len(items)}  {i / (time.time() - t0):.0f} img/s", flush=True)
    mm.flush()
    (CACHE / f"{view}_{split}.done").write_text("ok")
    print(f"{view}/{split}: finished {len(items)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["A", "B"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=13)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(ROOT / "artifacts" / "splits.json"))
    for s in a.splits:
        run(a.view, s, splits[s], a.batch, a.workers)
