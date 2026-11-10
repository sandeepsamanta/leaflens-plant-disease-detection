import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import inference
from app.inference import Classifier, softmax, to_input, to_square

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "web" / "public" / "samples"


def test_square_crop_keeps_the_centre_and_resizes():
    img = Image.new("RGB", (400, 200))
    img.paste((255, 0, 0), (100, 0, 300, 200))  # red centre square, blue-ish edges stay black
    out = to_square(img)
    assert out.size == (224, 224)
    assert np.asarray(out)[112, 112].tolist() == [255, 0, 0]
    assert np.asarray(out)[5, 5].tolist() == [255, 0, 0]  # the crop covers only the red square


def test_softmax_temperature_flattens():
    z = np.array([4.0, 1.0, 0.0])
    sharp, flat = softmax(z, 1.0), softmax(z, 3.0)
    assert sharp.sum() == pytest.approx(1) and flat.sum() == pytest.approx(1)
    assert flat.max() < sharp.max()


def test_preprocessing_matches_the_training_code():
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT))
    from ml.preprocess import to_tensor_array

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (180, 260, 3), dtype=np.uint8))
    np.testing.assert_allclose(to_input(to_square(img))[0], to_tensor_array(img), atol=1e-6)


def test_model_card_is_consistent_with_the_weights():
    c = Classifier()
    assert len(c.labels) == 38 and c.fc.shape == (38, 1280)
    assert set(c.card["class_info"]) == set(c.labels) == set(c.advice)
    assert 0 < c.threshold < 1 and c.temperature > 0


def test_heatmap_highlights_something():
    c = Classifier()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    out = c.predict(img, explain=True)
    assert out["heatmap"].startswith("data:image/png;base64,")


samples = json.loads((SAMPLES / "manifest.json").read_text()) if (SAMPLES / "manifest.json").exists() else []


@pytest.mark.parametrize("s", [s for s in samples if s["kind"] == "lab"], ids=lambda s: s["file"])
def test_known_lab_samples_are_classified_correctly(s):
    """Regression guard: the bundled lab sample photos must keep their labels."""
    out = Classifier().predict(Image.open(SAMPLES / s["file"]))
    assert out["prediction"]["label"] == s["label"], out["top"]
