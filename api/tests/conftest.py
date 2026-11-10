import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main  # noqa: E402


@pytest.fixture()
def client():
    main._hits.clear()
    return TestClient(main.app)


def image_bytes(size=(300, 220), fmt="JPEG", color=None, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) if color is None else np.full((size[1], size[0], 3), color, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def jpg():
    return image_bytes()
