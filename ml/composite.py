"""View B: paste the leaf onto a real field background so the model learns that the background is irrelevant.

PlantVillage leaves were photographed on a plain grey/black surface. A classifier can partly "cheat" on
that background, which is one reason PlantVillage models drop sharply on real field photos.

The dataset's own segmentation masks cut away the diseased (brown, necrotic) tissue as "background", so we
cannot paste them directly: the lesions would disappear. We close and fill the mask first so lesions stay.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from ml.preprocess import to_tensor_array

ROOT = Path(__file__).resolve().parents[1]
COLOR = ROOT / "data" / "raw" / "raw" / "color"
SEG = ROOT / "data" / "raw" / "raw" / "segmented"


def leaf_mask(seg_path: Path) -> np.ndarray:
    m = np.asarray(Image.open(seg_path).convert("RGB")).max(2) > 25
    m = ndi.binary_opening(m, iterations=1)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=7)  # bridge the cut-out lesions
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:  # drop specks, keep components that are at least 5% of the biggest one
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.05 * sizes.max()]
        m = np.isin(lab, keep)
    return m


def composite(color: Image.Image, mask: np.ndarray, bg: np.ndarray, rng: np.random.Generator) -> Image.Image:
    color = color.convert("RGB")
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    # random flip / rot90 (leaf orientation is arbitrary), random scale and position
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    scale = rng.uniform(0.65, 1.0)
    size = int(256 * scale)
    leaf, a = color, alpha
    for _ in range(k):
        leaf, a = leaf.transpose(Image.ROTATE_90), a.transpose(Image.ROTATE_90)
    if flip:
        leaf, a = leaf.transpose(Image.FLIP_LEFT_RIGHT), a.transpose(Image.FLIP_LEFT_RIGHT)
    leaf, a = leaf.resize((size, size), Image.BILINEAR), a.resize((size, size), Image.BILINEAR)
    canvas = Image.fromarray(bg)
    if rng.random() < 0.5:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    slack = 256 - size
    canvas.paste(leaf, (int(rng.integers(0, slack + 1)), int(rng.integers(0, slack + 1))), a)
    # photometric jitter on the whole picture, like a different phone / light
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.7, 1.3))
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.75, 1.25))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return canvas


def _suffix(name: str) -> str:
    """'<uuid>___RS_Rust 1563_final_masked.jpg' and 'RS_Rust 1563.JPG' -> 'rs_rust 1563'."""
    return name.split("___", 1)[-1].replace("_final_masked", "").rsplit(".", 1)[0].lower()


class ViewB(Dataset):
    def __init__(self, items, seed: int = 123):
        self.items, self.seed = items, seed
        # some colour files lack the uuid prefix that their segmented twin has, so match on the photo name
        self.seg_index = {}
        for cls in {rel.split("/", 1)[0] for rel, _ in items}:
            self.seg_index[cls] = {_suffix(p.name): p.name for p in (SEG / cls).iterdir()}
        self.bgs = np.load(ROOT / "data" / "cache" / "backgrounds.npy", mmap_mode="r")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel = self.items[i][0]
        cls, fn = rel.split("/", 1)
        rng = np.random.default_rng([self.seed, i])
        seg = self.seg_index[cls].get(_suffix(fn))
        if seg is None:  # no segmentation available: fall back to the plain photo
            return torch.from_numpy(to_tensor_array(Image.open(COLOR / rel)))
        mask = leaf_mask(SEG / cls / seg)
        img = composite(Image.open(COLOR / rel), mask, np.array(self.bgs[int(rng.integers(len(self.bgs)))]), rng)
        return torch.from_numpy(to_tensor_array(img))
