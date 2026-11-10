import base64
import io

import pytest
from PIL import Image

from app import main
from tests.conftest import image_bytes


def post(client, data, name="leaf.jpg", ctype="image/jpeg", **params):
    return client.post("/api/predict", files={"file": (name, data, ctype)}, params=params)


def test_health_and_model_card(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["classes"] == 38
    card = client.get("/api/model").json()
    assert card["metrics"]["plantdoc_field"]["accuracy"] < card["metrics"]["plantvillage_test"]["accuracy"]
    assert len(card["limitations"]) >= 3


def test_classes_endpoint(client):
    rows = client.get("/api/classes").json()
    assert len(rows) == 38
    assert {"label", "plant", "disease", "healthy"} <= rows[0].keys()
    assert sum(r["healthy"] for r in rows) == 12


def test_predict_returns_a_complete_result(client, jpg):
    r = post(client, jpg)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"confident", "uncertain"}
    top = body["top"]
    assert len(top) == 3
    confs = [t["confidence"] for t in top]
    assert confs == sorted(confs, reverse=True) and 0 < sum(confs) <= 1.0001
    assert body["prediction"]["label"] == top[0]["label"]
    assert {"about", "symptoms", "actions"} <= body["advice"].keys()
    png = base64.b64decode(body["heatmap"].split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (224, 224)


def test_explain_false_skips_the_heatmap_and_top_k_is_honoured(client, jpg):
    body = post(client, jpg, explain="false", top_k=5).json()
    assert "heatmap" not in body and len(body["top"]) == 5


@pytest.mark.parametrize("k", [0, 6])
def test_top_k_is_bounded(client, jpg, k):
    assert post(client, jpg, top_k=k).status_code == 422


def test_png_and_webp_are_accepted(client):
    assert post(client, image_bytes(fmt="PNG"), "a.png", "image/png").status_code == 200
    assert post(client, image_bytes(fmt="WEBP"), "a.webp", "image/webp").status_code == 200


def test_rejects_wrong_content_type(client):
    r = post(client, b"%PDF-1.4", "a.pdf", "application/pdf")
    assert r.status_code == 415 and "JPEG" in r.json()["detail"]


def test_rejects_bytes_that_are_not_an_image(client):
    r = post(client, b"definitely not an image", "a.jpg", "image/jpeg")
    assert r.status_code == 400 and "could not be read" in r.json()["detail"]


def test_rejects_tiny_images(client):
    assert post(client, image_bytes(size=(32, 32))).status_code == 400


def test_rejects_oversize_uploads(client, jpg, monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 1000)
    r = post(client, jpg)
    assert r.status_code == 413


def test_rate_limit(client, jpg, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 3)
    codes = [post(client, jpg, explain="false").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert client.get("/api/health").status_code == 200  # only /predict is limited


def test_abstains_below_the_threshold(client, jpg, monkeypatch):
    clf = main.classifier()
    monkeypatch.setattr(clf, "threshold", 1.01)
    body = post(client, jpg).json()
    assert body["decision"] == "uncertain" and "not sure" in body["message"]
    monkeypatch.setattr(clf, "threshold", 0.0)
    body = post(client, jpg).json()
    assert body["decision"] == "confident" and "message" not in body


def test_prediction_is_deterministic(client, jpg):
    a, b = post(client, jpg).json(), post(client, jpg).json()
    assert a["top"] == b["top"] and a["heatmap"] == b["heatmap"]
