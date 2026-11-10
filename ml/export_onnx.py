"""Export stem + trained tail to ONNX (logits and the 1280x7x7 feature map for the heatmap) and write the model card.

    python -m ml.export_onnx --tag field --threshold 0.7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from ml.backbone import FullModel, Stem, Tail, load_pretrained
from ml.classes import CLASSES, is_healthy, split_label
from ml.preprocess import to_tensor_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "api" / "model"
COLOR = ROOT / "data" / "raw" / "raw" / "color"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="default: the threshold suggested by ml.evaluate from the calibration crops")
    a = ap.parse_args()
    if a.threshold is None:
        a.threshold = json.load(open(ROOT / "artifacts" / "metrics.json"))[a.tag]["suggested_threshold"]
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_pretrained()
    tail = Tail(load_pretrained(), 38)
    tail.load_state_dict(torch.load(ROOT / "artifacts" / f"tail_{a.tag}.pt"))
    model = FullModel(Stem(base), tail).eval()
    path = OUT / "leaflens.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(path), input_names=["input"], output_names=["logits", "fmap"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "fmap": {0: "batch"}}, opset_version=17, dynamo=False)
    np.save(OUT / "fc_weight.npy", tail.fc.weight.detach().numpy().astype(np.float32))

    # parity check on real test images: ONNX vs the PyTorch model
    items = json.load(open(ROOT / "artifacts" / "splits.json"))["test"][::400][:20]
    x = np.stack([to_tensor_array(Image.open(COLOR / p)) for p, _ in items])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ol, _ = sess.run(None, {"input": x})
    with torch.no_grad():
        tl, _ = model(torch.from_numpy(x))
    diff = float(np.abs(ol - tl.numpy()).max())
    assert diff < 1e-3 and (ol.argmax(1) == tl.numpy().argmax(1)).all(), f"ONNX mismatch: {diff}"
    print(f"ONNX matches PyTorch (max logit diff {diff:.2e}); size {path.stat().st_size / 1e6:.1f} MB")

    all_metrics = json.load(open(ROOT / "artifacts" / "metrics.json"))
    metrics = all_metrics[a.tag]
    variants = {
        "lab": "PlantVillage lab photos only",
        "field": "+ leaves pasted onto real field backgrounds",
        "plus": "+ real labelled field crops (PlantDoc train)",
    }
    comparison = [
        {"variant": t, "description": variants.get(t, t), "plantvillage_test_accuracy": m["plantvillage_test"]["accuracy"],
         "plantdoc_field_accuracy": m["plantdoc_field"]["accuracy"], "plantdoc_field_accuracy_ci95": m["plantdoc_field"]["accuracy_ci95"]}
        for t, m in all_metrics.items()
    ]
    pv, pd_ = metrics["plantvillage_test"], metrics["plantdoc_field"]
    card = {
        "name": "LeafLens MobileNetV2",
        "version": "1.0.0",
        "architecture": "MobileNetV2 (ImageNet pretrained); blocks 14-18 and the classifier fine-tuned on PlantVillage",
        "input": "224 x 224 RGB; centre-crop to square, resize, ImageNet normalisation",
        "trained_variant": a.tag,
        "classes": CLASSES,
        "class_info": {c: {"plant": split_label(c)[0], "disease": split_label(c)[1], "healthy": is_healthy(c)} for c in CLASSES},
        "temperature": metrics["temperature"],
        "confidence_threshold": a.threshold,
        "metrics": {
            "plantvillage_test": {k: pv[k] for k in ("n", "accuracy", "accuracy_ci95", "macro_f1", "top3_accuracy")},
            "plantdoc_field": {k: pd_[k] for k in ("n", "accuracy", "accuracy_ci95", "top3_accuracy", "plant_species_accuracy", "healthy_vs_diseased_accuracy")},
            "selective_prediction_plantdoc": pd_["selective_prediction"],
            "training_variants": comparison,
        },
        "limitations": [
            "Mostly trained on lab photos (plain background); field-photo accuracy is much lower than lab accuracy, see metrics.plantdoc_field.",
            "Only 38 crop-disease classes (14 crops). Anything else is forced into one of them; low confidence is the main warning sign.",
            "One leaf per photo works best. Multiple diseases on one leaf are not modelled.",
            "A decision aid, not a diagnosis: confirm with a local agricultural extension service before treating.",
        ],
        "data": "PlantVillage (CC BY-SA 3.0) and the PlantDoc train split (CC BY 4.0) for training; the PlantDoc test split for field evaluation.",
    }
    json.dump(card, open(OUT / "model_card.json", "w"), indent=2)
    print("wrote model card; threshold", a.threshold)


if __name__ == "__main__":
    main()
