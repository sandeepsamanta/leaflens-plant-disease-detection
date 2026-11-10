"""Evaluate trained tails on (a) the leak-free PlantVillage test split and (b) PlantDoc field crops.

    python -m ml.evaluate --tags lab field

Writes artifacts/metrics.json and figures/*.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ml.backbone import Tail, load_pretrained
from ml.classes import CLASSES, display_name, is_healthy, split_label
from ml.train_head import CACHE, predict

ROOT = Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
NICE = {"lab": "Lab photos only", "field": "+ field backgrounds", "plus": "+ real field crops"}
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    logits = logits.clone()  # leave inference mode so autograd can run
    t = torch.zeros(1, requires_grad=True)  # log T
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp().detach())


def ece(prob: np.ndarray, pred: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = prob.max(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def selective(prob: np.ndarray, y: np.ndarray) -> list[dict]:
    conf, pred = prob.max(1), prob.argmax(1)
    rows = []
    for t in THRESHOLDS:
        keep = conf >= t
        rows.append({"threshold": t, "coverage": round(float(keep.mean()), 4), "accuracy": round(float((pred[keep] == y[keep]).mean()), 4) if keep.any() else None})
    return rows


def suggest_threshold(prob: np.ndarray, y: np.ndarray, target: float = 0.85) -> float:
    """Smallest confidence threshold at which the held-out field calibration crops are answered with >= target accuracy."""
    conf, pred = prob.max(1), prob.argmax(1)
    for t in np.arange(0.5, 0.99, 0.025):
        keep = conf >= t
        if keep.sum() >= 50 and (pred[keep] == y[keep]).mean() >= target:
            return round(float(t), 3)
    return 0.9


def plant_of(i: int) -> str:
    return split_label(CLASSES[i])[0]


def summarise(logits: torch.Tensor, y: np.ndarray, T: float, field: bool) -> dict:
    pred = logits.argmax(1).numpy()
    prob = F.softmax(logits / T, 1).numpy()
    prob_raw = F.softmax(logits, 1).numpy()
    n = len(y)
    top3 = float(np.mean([y[i] in np.argsort(-logits[i].numpy())[:3] for i in range(n)]))
    present = sorted(set(y.tolist()))
    out = {
        "n": n,
        "accuracy": round(accuracy_score(y, pred), 4),
        "accuracy_ci95": wilson(int((pred == y).sum()), n),
        "top3_accuracy": round(top3, 4),
        "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
        "plant_species_accuracy": round(float(np.mean([plant_of(a) == plant_of(b) for a, b in zip(pred, y)])), 4),
        "healthy_vs_diseased_accuracy": round(float(np.mean([is_healthy(CLASSES[a]) == is_healthy(CLASSES[b]) for a, b in zip(pred, y)])), 4),
        "ece_raw": round(ece(prob_raw, pred, y), 4),
        "ece_temperature_scaled": round(ece(prob, pred, y), 4),
        "selective_prediction": selective(prob, y),
    }
    if not field:
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(38), zero_division=0)
        out["per_class"] = {CLASSES[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(s[i])} for i in range(38)}
    else:
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
        out["per_class"] = {CLASSES[c]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(s[i])} for i, c in enumerate(present)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--temperature", type=float, default=1.0, help="confidence temperature used for everything below (1.0 = raw softmax)")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    torch.set_num_threads(16)
    xt, yt = np.load(CACHE / "A_test.npy"), np.load(CACHE / "labels_test.npy")
    xv, yv = np.load(CACHE / "A_val.npy"), np.load(CACHE / "labels_val.npy")
    xf, yf = np.load(CACHE / "plantdoc_feats.npy"), np.load(CACHE / "plantdoc_labels.npy")
    # Calibration set: 20% of PlantDoc *train* crops (split by source image), never used for fitting weights or for the test.
    feats, labels, ids = (np.load(CACHE / f"plantdoc_train_{k}.npy") for k in ("feats", "labels", "imgid"))
    xc, yc = feats[ids % 5 == 0], labels[ids % 5 == 0]
    metrics, store = {}, {}
    for tag in a.tags:
        tail = Tail(load_pretrained(), 38).to(memory_format=torch.channels_last)
        tail.load_state_dict(torch.load(ART / f"tail_{tag}.pt"))
        lv, lt, lf, lc = predict(tail, xv), predict(tail, xt), predict(tail, xf), predict(tail, xc)
        T_lab = fit_temperature(lv, torch.from_numpy(yv))  # fitted on lab photos: shown only to demonstrate why it is wrong for the field
        T_fit = fit_temperature(lc, torch.from_numpy(yc))  # fitted on held-out field crops, reported for reference
        T = a.temperature  # what is actually used (see README: the fitted value did not improve calibration on the test set)
        pf = F.softmax(lf / T_lab, 1).numpy()
        metrics[tag] = {
            "temperature": round(T, 3),
            "temperature_field_fitted": round(T_fit, 3),
            "temperature_lab_fitted": round(T_lab, 3),
            "field_ece_with_lab_fitted_temperature": round(ece(pf, pf.argmax(1), yf), 4),
            "calibration_set": {"n": int(len(yc)), "accuracy": round(float((lc.argmax(1).numpy() == yc).mean()), 4), "selective_prediction": selective(F.softmax(lc / T, 1).numpy(), yc)},
            "suggested_threshold": suggest_threshold(F.softmax(lc / T, 1).numpy(), yc),
            "plantvillage_val_accuracy": round(accuracy_score(yv, lv.argmax(1).numpy()), 4),
            "plantvillage_test": summarise(lt, yt, T, field=False),
            "plantdoc_field": summarise(lf, yf, T, field=True),
        }
        store[tag] = {"lt": lt, "lf": lf, "T": T}
        m, f = metrics[tag]["plantvillage_test"], metrics[tag]["plantdoc_field"]
        print(f"[{tag}] suggested threshold {metrics[tag]['suggested_threshold']} | T_field_fitted={T_fit:.2f} (lab-fitted {T_lab:.2f}, field ECE with it {metrics[tag]['field_ece_with_lab_fitted_temperature']:.3f})   PlantVillage test acc {m['accuracy']:.4f} (F1 {m['macro_f1']:.4f}) | "
              f"PlantDoc field acc {f['accuracy']:.4f} CI {f['accuracy_ci95']} top3 {f['top3_accuracy']:.3f} species {f['plant_species_accuracy']:.3f} healthy/diseased {f['healthy_vs_diseased_accuracy']:.3f}")
    json.dump(metrics, open(ART / "metrics.json", "w"), indent=2)

    # ---- figures for the last tag (the deployed model is evaluated last) ----
    tag = a.tags[-1]
    pred = store[tag]["lt"].argmax(1).numpy()
    cm = confusion_matrix(yt, pred, labels=range(38))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(38)); ax.set_yticks(range(38))
    ax.set_xticklabels([display_name(c) for c in CLASSES], rotation=90, fontsize=6)
    ax.set_yticklabels([display_name(c) for c in CLASSES], fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"PlantVillage test (leaf-grouped), row-normalised - model '{tag}'")
    fig.colorbar(im, fraction=0.04); fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in zip(axes, ["plantvillage_test", "plantdoc_field"], ["PlantVillage test (lab photos)", "PlantDoc (real field photos)"]):
        for t in a.tags:
            rows = [r for r in metrics[t][key]["selective_prediction"] if r["accuracy"] is not None]
            ax.plot([r["coverage"] for r in rows], [r["accuracy"] for r in rows], marker="o", label=NICE.get(t, t))
        ax.set_xlabel("Coverage (share of photos the model answers)"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy on answered photos"); axes[1].legend(title="model")
    fig.tight_layout(); fig.savefig(FIG / "selective_prediction.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.8 / len(a.tags)
    for k, t in enumerate(a.tags):
        vals = [metrics[t]["plantvillage_test"]["accuracy"], metrics[t]["plantdoc_field"]["accuracy"]]
        ci = metrics[t]["plantdoc_field"]["accuracy_ci95"]
        errs = [[0, vals[1] - ci[0]], [0, ci[1] - vals[1]]]
        bars = ax.bar(np.arange(2) + k * w, vals, w, label=NICE.get(t, t), yerr=errs, capsize=3)
        tops = [vals[0], ci[1]]
        for b, v, top in zip(bars, vals, tops):
            ax.text(b.get_x() + b.get_width() / 2, top + 0.015, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(2) + w * (len(a.tags) - 1) / 2); ax.set_xticklabels(["PlantVillage test\n(lab photos)", "PlantDoc\n(field photos)"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy"); ax.legend(title="Training data", loc="upper right", fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "lab_vs_field.png", dpi=140); plt.close(fig)
    print("wrote artifacts/metrics.json and figures/")


if __name__ == "__main__":
    main()
