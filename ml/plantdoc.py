"""PlantDoc test set (real field photos): crop every labelled leaf and cache crops + stem features.

PlantDoc is a detection dataset (COCO boxes). We crop each box to a square (box centre, side = 1.1 x the
longer box side) so the leaf is not cut off, then run it through the same preprocessing as everything else.
Only the PlantDoc *test* split is used for evaluation; backgrounds for view B come from its *train* split.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ml.backbone import Stem, load_pretrained
from ml.classes import CLASS_INDEX, PLANTDOC_TO_PV
from ml.preprocess import SIZE, to_tensor_array

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def square_crop(img: Image.Image, bbox) -> Image.Image:
    x, y, w, h = [float(v) for v in bbox]
    side = min(max(w, h) * 1.1, img.width, img.height)
    cx, cy = x + w / 2, y + h / 2
    left = min(max(cx - side / 2, 0), img.width - side)
    top = min(max(cy - side / 2, 0), img.height - side)
    return img.crop((int(left), int(top), int(left + side), int(top + side))).resize((SIZE, SIZE), Image.BILINEAR)


def build(split: str = "test") -> None:
    parts = ["test"] if split == "test" else ["train0", "train1"]
    df = pd.concat([pd.read_parquet(ROOT / "data" / "plantdoc" / f"{p}.parquet") for p in parts], ignore_index=True)
    crops, labels, names, image_ids = [], [], [], []
    for img_no, row in df.iterrows():
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        for bbox, cat in zip(row["objects"]["bbox"], row["objects"]["category"]):
            if cat not in PLANTDOC_TO_PV or min(bbox[2], bbox[3]) < 40:  # skip unmapped and tiny boxes
                continue
            crops.append(np.asarray(square_crop(img, bbox)))
            labels.append(CLASS_INDEX[PLANTDOC_TO_PV[cat]])
            names.append(cat)
            image_ids.append(img_no)
    crops = np.stack(crops)
    stem = Stem(load_pretrained()).eval().to(memory_format=torch.channels_last)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(crops), 64):
            x = np.stack([to_tensor_array(Image.fromarray(c)) for c in crops[i : i + 64]])
            feats.append(stem(torch.from_numpy(x).contiguous(memory_format=torch.channels_last)).numpy().astype(np.float16))
    prefix = "plantdoc" if split == "test" else "plantdoc_train"
    np.save(CACHE / f"{prefix}_crops.npy", crops)
    np.save(CACHE / f"{prefix}_feats.npy", np.concatenate(feats))
    np.save(CACHE / f"{prefix}_labels.npy", np.array(labels))
    np.save(CACHE / f"{prefix}_imgid.npy", np.array(image_ids))
    (CACHE / f"{prefix}_names.txt").write_text("\n".join(names))
    print("plantdoc crops", crops.shape, "classes", len(set(labels)))


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "test")
