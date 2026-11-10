"""Unit tests for the training-side code. They need no dataset and no trained model."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml import evaluate
from ml.classes import CLASSES, CLASS_INDEX, PLANTDOC_TO_PV, display_name, is_healthy, split_label
from ml.plantdoc import square_crop
from ml.prepare_splits import CAP, GAP, assign_runs, parse_key, photo_key
from ml.preprocess import SIZE, to_square, to_tensor_array
from ml.train_head import symmetries

ROOT = Path(__file__).resolve().parents[1]


def test_class_list_and_names():
    assert len(CLASSES) == 38 == len(set(CLASSES))
    assert sum(is_healthy(c) for c in CLASSES) == 12
    assert split_label("Tomato___Late_blight") == ("Tomato", "Late blight")
    assert display_name("Pepper,_bell___Bacterial_spot") == "Bell pepper - Bacterial spot"
    assert display_name("Corn_(maize)___Common_rust_") == "Corn (maize) - Common rust"


def test_plantdoc_mapping_targets_exist_and_cover_all_27_categories():
    assert len(PLANTDOC_TO_PV) == 27
    assert all(v in CLASS_INDEX for v in PLANTDOC_TO_PV.values())
    assert len(set(PLANTDOC_TO_PV.values())) == 27  # one PlantDoc category per PlantVillage class


def test_every_class_has_advice():
    advice = json.loads((ROOT / "api" / "app" / "advice.json").read_text())
    assert set(advice) == set(CLASSES)
    assert all(a["actions"] for a in advice.values())


def test_preprocess_shapes_and_range():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (123, 301, 3), dtype=np.uint8))
    assert to_square(img).size == (SIZE, SIZE)
    a = to_tensor_array(img)
    assert a.shape == (3, SIZE, SIZE) and a.dtype == np.float32 and -3 < a.min() and a.max() < 3


def test_photo_key_and_number_parsing():
    assert photo_key("0003faa8-4b27___RS_Late.B 4946.JPG") == "rs_late.b 4946"
    assert photo_key("RS_Rust 1563.JPG") == "rs_rust 1563"  # some files have no uuid prefix
    assert parse_key("rs_late.b 4946") == ("rs_late.b", 4946)


def _recs(numbers):
    return [{"key": f"rs_x {n}"} for n in numbers]


def test_runs_split_on_gaps():
    recs = _recs([1, 2, 3, 3 + GAP + 1, 3 + GAP + 2])
    assign_runs(recs, "g")
    assert [r["group"] for r in recs] == ["g0", "g0", "g0", "g1", "g1"]


def test_runs_are_capped_so_dense_series_do_not_become_one_giant_group():
    recs = _recs(range(1, 3 * CAP + 1))
    assign_runs(recs, "g")
    sizes = np.bincount([int(r["group"][1:]) for r in recs])
    assert sizes.max() == CAP and len(sizes) == 3


def test_wilson_interval_brackets_the_estimate():
    lo, hi = evaluate.wilson(65, 100)
    assert lo < 0.65 < hi and 0 < lo and hi < 1
    assert evaluate.wilson(0, 50)[0] == 0.0


def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence():
    y = np.array([0] * 8 + [1] * 2)
    perfect = np.tile([0.8, 0.2], (10, 1))
    assert evaluate.ece(perfect, perfect.argmax(1), y) == pytest.approx(0.0, abs=1e-9)
    overconfident = np.tile([0.99, 0.01], (10, 1))
    wrong = np.array([1] * 10)
    assert evaluate.ece(overconfident, overconfident.argmax(1), wrong) > 0.9


def test_temperature_fit_softens_an_overconfident_model():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 5, (2000,), generator=g)
    logits = torch.randn(2000, 5, generator=g) * 3
    logits[torch.arange(2000), y] += 4 * (torch.rand(2000, generator=g) < 0.5)  # right only about half the time
    assert evaluate.fit_temperature(logits, y) > 1.0


def test_threshold_suggestion_hits_the_target_accuracy():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 2000)
    y = rng.integers(0, 3, 2000)
    prob = np.zeros((2000, 3))
    right = rng.random(2000) < conf  # higher confidence -> more often right
    pred = np.where(right, y, (y + 1) % 3)
    prob[np.arange(2000), pred] = conf
    t = evaluate.suggest_threshold(prob, y, target=0.85)
    keep = conf >= t
    assert 0.5 <= t < 0.99 and (pred[keep] == y[keep]).mean() >= 0.84


def test_symmetries_preserve_shape_and_content():
    x = torch.arange(2 * 4 * 6 * 6, dtype=torch.float32).reshape(2, 4, 6, 6)
    out = symmetries(x)
    assert out.shape == x.shape
    for i in range(2):  # every output is one of the 8 symmetries of the square
        cands = [torch.flip(x[i], list(d)).transpose(1, 2) if t else torch.flip(x[i], list(d)) for d in [(), (1,), (2,), (1, 2)] for t in (0, 1)]
        assert any(torch.equal(out[i], c) for c in cands)


def test_square_crop_stays_inside_the_image():
    img = Image.new("RGB", (800, 600), (10, 200, 10))
    out = square_crop(img, (700, 500, 90, 80))  # box in the corner
    assert out.size == (SIZE, SIZE)
