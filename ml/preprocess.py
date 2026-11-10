"""Image preprocessing shared by training, evaluation and (mirrored in) the API.

Every image is centre-cropped to a square on its shorter side, resized to 224x224 and normalised with
ImageNet statistics. The API has its own copy of this function (so it can deploy without torch);
a test asserts the two stay identical.
"""
import numpy as np
from PIL import Image

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_square(img: Image.Image, size: int = SIZE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)


def to_tensor_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 CHW array, normalised."""
    a = np.asarray(to_square(img), dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)
