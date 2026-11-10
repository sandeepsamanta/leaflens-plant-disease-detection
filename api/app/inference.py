"""ONNX inference: preprocessing, top-k, calibrated confidence, abstention and a class-activation heatmap.

This module deliberately does not import torch. Its preprocessing mirrors ml/preprocess.py (a test checks
the two give identical tensors).
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def to_square(img: Image.Image, size: int = SIZE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)


def to_input(square: Image.Image) -> np.ndarray:
    a = np.asarray(square, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None]


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _jet(x: np.ndarray) -> np.ndarray:
    """Tiny jet-like colormap, x in [0,1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class Classifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.card = json.loads((model_dir / "model_card.json").read_text())
        self.advice = json.loads((Path(__file__).parent / "advice.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_dir / "leaflens.onnx"), opts, providers=["CPUExecutionProvider"])
        self.fc = np.load(model_dir / "fc_weight.npy")  # (classes, 1280)
        self.labels: list[str] = self.card["classes"]
        self.temperature: float = self.card["temperature"]
        self.threshold: float = self.card["confidence_threshold"]

    def _meta(self, i: int) -> dict:
        c = self.card["class_info"][self.labels[i]]
        return {"label": self.labels[i], "plant": c["plant"], "disease": c["disease"], "healthy": c["healthy"]}

    def heatmap(self, square: Image.Image, fmap: np.ndarray, cls: int) -> str:
        """Class activation map: sum_k w[cls,k] * fmap[k], relu, normalise, overlay on the photo (PNG data URL)."""
        cam = np.maximum(np.tensordot(self.fc[cls], fmap, axes=(0, 0)), 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
        heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255.0)
        alpha = (np.asarray(cam_img, dtype=np.float32) / 255.0)[..., None] * 0.55
        over = (np.asarray(square, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(over).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def predict(self, img: Image.Image, top_k: int = 3, explain: bool = False) -> dict:
        square = to_square(img)
        logits, fmap = self.session.run(None, {"input": to_input(square)})
        logits, fmap = logits[0], fmap[0]
        prob = softmax(logits, self.temperature)
        order = np.argsort(-prob)[:top_k]
        top = [{**self._meta(int(i)), "confidence": round(float(prob[i]), 4)} for i in order]
        best = top[0]
        confident = best["confidence"] >= self.threshold
        out = {
            "decision": "confident" if confident else "uncertain",
            "prediction": best,
            "top": top,
            "advice": self.advice[best["label"]],
            "confidence_threshold": self.threshold,
        }
        if not confident:
            out["message"] = ("The model is not sure about this photo. Retake it in daylight with a single leaf filling most of the frame, "
                              "or treat the suggestions below as hints only.")
        if explain:
            out["heatmap"] = self.heatmap(square, fmap, int(order[0]))
        return out
