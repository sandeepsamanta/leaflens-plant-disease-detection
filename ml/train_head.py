"""Stage 2: fine-tune the tail (MobileNetV2 blocks 14-18 + classifier) on cached stem features.

    python -m ml.train_head --tag lab   --pb 0.0     # lab photos only
    python -m ml.train_head --tag field --pb 0.5     # half of each epoch uses the field-background view

Augmentation happens on the cached 14x14 feature maps (random flips / transpose = the 8 symmetries of the
square), plus feature dropout. Pixel-space augmentation is baked into view B instead.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.backbone import Tail, load_pretrained

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
ART = ROOT / "artifacts"


def load(view: str, split: str):
    return np.load(CACHE / f"{view}_{split}.npy"), np.load(CACHE / f"labels_{split}.npy")


def symmetries(x: torch.Tensor) -> torch.Tensor:
    """Random element of the dihedral group D4 applied per sample to (B, C, H, W) feature maps."""
    b = x.shape[0]
    pick = lambda: (torch.rand(b) < 0.5).view(b, 1, 1, 1)
    x = torch.where(pick(), x.flip(3), x)
    x = torch.where(pick(), x.flip(2), x)
    return torch.where(pick(), x.transpose(2, 3), x)


@torch.inference_mode()
def predict(tail: Tail, feats: np.ndarray, batch: int = 512) -> torch.Tensor:
    tail.eval()
    out = []
    for i in range(0, len(feats), batch):
        out.append(tail(torch.from_numpy(feats[i : i + batch]).float().contiguous(memory_format=torch.channels_last)))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pb", type=float, default=0.0, help="probability that a training image uses view B")
    ap.add_argument("--field-frac", type=float, default=0.0, help="share of each batch drawn from real PlantDoc-train crops")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "16")))

    xa, y = load("A", "train")
    xb = np.load(CACHE / "B_train.npy") if a.pb > 0 else None
    xv, yv = load("A", "val")
    yv_t = torch.from_numpy(yv)
    n = len(y)
    xf = yf = xc = yc = None
    if a.field_frac > 0:  # real field crops, split by source image: 80% train, 20% held out for calibration
        feats, labels, ids = (np.load(CACHE / f"plantdoc_train_{k}.npy") for k in ("feats", "labels", "imgid"))
        calib = ids % 5 == 0
        xf, yf, xc, yc = feats[~calib], labels[~calib], feats[calib], labels[calib]
        print(f"field crops: {len(yf)} train, {len(yc)} calibration", flush=True)
    tail = Tail(load_pretrained(), 38).to(memory_format=torch.channels_last)
    body = [p for n_, p in tail.named_parameters() if n_.startswith("body")]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": a.lr * 0.3}, {"params": tail.fc.parameters(), "lr": a.lr}], weight_decay=1e-4
    )
    steps = a.epochs * (n // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr * 0.3, a.lr], total_steps=steps, pct_start=0.15, anneal_strategy="cos"
    )
    history, best = [], 0.0
    for ep in range(a.epochs):
        tail.train()
        perm = np.random.permutation(n)
        t0, run_loss, run_acc, seen = time.time(), 0.0, 0.0, 0
        for k in range(n // a.batch):
            idx = np.sort(perm[k * a.batch : (k + 1) * a.batch])
            xs = xa[idx]
            if xb is not None:
                use_b = np.random.rand(len(idx)) < a.pb
                if use_b.any():
                    xs = np.where(use_b[:, None, None, None], xb[idx], xs)
            ys = y[idx]
            if xf is not None:
                k_f = int(round(len(idx) * a.field_frac))
                pick = np.random.randint(0, len(yf), k_f)
                xs = np.concatenate([xs[: len(idx) - k_f], xf[pick]])
                ys = np.concatenate([ys[: len(idx) - k_f], yf[pick]])
            x = F.dropout2d(symmetries(torch.from_numpy(xs).float()), 0.1).contiguous(memory_format=torch.channels_last)
            t = torch.from_numpy(ys)
            logits = tail(x)
            loss = F.cross_entropy(logits, t, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(t)
            run_acc += (logits.argmax(1) == t).sum().item()
            seen += len(t)
        vl = predict(tail, xv)
        vacc = (vl.argmax(1) == yv_t).float().mean().item()
        vloss = F.cross_entropy(vl, yv_t).item()
        rec = {"epoch": ep + 1, "train_loss": run_loss / seen, "train_acc": run_acc / seen, "val_loss": vloss, "val_acc": vacc, "secs": time.time() - t0}
        score = vacc
        if xc is not None:
            rec["field_calib_acc"] = (predict(tail, xc).argmax(1).numpy() == yc).mean().item()
            score = 0.5 * vacc + 0.5 * rec["field_calib_acc"]
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if score >= best:
            best = score
            torch.save(tail.state_dict(), ART / f"tail_{a.tag}.pt")
    json.dump({"args": vars(a), "best_val_acc": best, "history": history}, open(ART / f"train_{a.tag}.json", "w"), indent=2)
    print("best val acc", best)


if __name__ == "__main__":
    main()
