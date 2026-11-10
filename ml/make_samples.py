"""Pick sample photos for the web demo from the held-out test sets, using the exported ONNX model.

Lab samples are test-split PlantVillage photos the model gets right (they double as regression tests).
Field samples are PlantDoc test crops: two it gets right, one it is unsure about, one it gets wrong,
so visitors see honest behaviour on real photos, not only the flattering cases.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from app.inference import Classifier  # noqa: E402

from ml.classes import CLASSES, display_name  # noqa: E402

OUT = ROOT / "web" / "public" / "samples"
COLOR = ROOT / "data" / "raw" / "raw" / "color"
LAB_CLASSES = ["Tomato___Late_blight", "Apple___Apple_scab", "Potato___Early_blight", "Corn_(maize)___Common_rust_", "Grape___Black_rot", "Tomato___healthy"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()
    clf = Classifier()
    manifest = []
    test = json.load(open(ROOT / "artifacts" / "splits.json"))["test"]
    for cls in LAB_CLASSES:
        idx = CLASSES.index(cls)
        for rel, lab in [t for t in test if t[1] == idx][::7]:
            img = Image.open(COLOR / rel)
            out = clf.predict(img, explain=False)
            if out["prediction"]["label"] == cls and out["decision"] == "confident":
                name = f"lab-{len(manifest):02d}.jpg"
                img.convert("RGB").save(OUT / name, quality=88)
                manifest.append({"file": name, "kind": "lab", "label": cls, "title": display_name(cls), "source": "PlantVillage", "license": "CC BY-SA 3.0"})
                break
        else:
            raise SystemExit(f"no correctly classified lab sample for {cls}")

    crops = np.load(ROOT / "data" / "cache" / "plantdoc_crops.npy")
    labels = np.load(ROOT / "data" / "cache" / "plantdoc_labels.npy")
    preds = [clf.predict(Image.fromarray(c), explain=False) for c in crops]
    want = {"right": [], "unsure": [], "wrong": []}
    for i, (p, y) in enumerate(zip(preds, labels)):
        ok = p["prediction"]["label"] == CLASSES[y]
        if ok and p["decision"] == "confident" and p["prediction"]["confidence"] > 0.85:
            want["right"].append(i)
        elif p["decision"] == "uncertain":
            want["unsure"].append(i)
        elif not ok:
            want["wrong"].append(i)
    rng = np.random.default_rng(7)
    picks = [("right", 2), ("unsure", 1), ("wrong", 1)]
    for kind, n in picks:
        pool = [i for i in want[kind] if CLASSES[labels[i]] not in {m["label"] for m in manifest if m["kind"] == "field"}]
        for i in rng.choice(pool, size=min(n, len(pool)), replace=False):
            name = f"field-{len(manifest):02d}.jpg"
            Image.fromarray(crops[i]).save(OUT / name, quality=90)
            manifest.append({"file": name, "kind": "field", "label": CLASSES[labels[i]], "title": display_name(CLASSES[labels[i]]), "source": "PlantDoc", "license": "CC BY 4.0", "model_outcome": kind})
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(len(manifest), "samples;", {k: len(v) for k, v in want.items()}, "field candidates")


if __name__ == "__main__":
    main()
