"""Collect natural background crops from PlantDoc *train* photos (never from the test split we evaluate on)."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main(n: int = 900, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    images = []
    for part in ("train0", "train1"):
        df = pd.read_parquet(ROOT / "data" / "plantdoc" / f"{part}.parquet", columns=["image"])
        images += [d["bytes"] for d in df["image"]]
    out = np.zeros((n, 256, 256, 3), dtype=np.uint8)
    for k in range(n):
        img = Image.open(io.BytesIO(images[rng.integers(len(images))])).convert("RGB")
        w, h = img.size
        side = int(min(w, h) * rng.uniform(0.3, 0.8))
        x, y = rng.integers(0, w - side + 1), rng.integers(0, h - side + 1)
        out[k] = np.asarray(img.crop((x, y, x + side, y + side)).resize((256, 256), Image.BILINEAR))
    np.save(ROOT / "data" / "cache" / "backgrounds.npy", out)
    print("saved", out.shape)


if __name__ == "__main__":
    main()
