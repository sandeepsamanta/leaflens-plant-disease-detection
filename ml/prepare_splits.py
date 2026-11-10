"""Build leak-free train/val/test splits for PlantVillage.

PlantVillage contains several near-duplicate photos (rotations, flips, crops) of every physical leaf.
A random image-level split therefore puts siblings of the same leaf in both train and test and inflates
accuracy. We split by *leaf group* instead:

  1. use the official leaf id from `leaf_grouping/leaf-map.json` when the image is in the map (76% of images);
  2. otherwise group by runs of consecutive photo numbers within a capture series: a run ends at a gap
     above 12 or after 60 photos (the cap stops dense series from chaining into one giant group). The rule
     is validated on the mapped images, where the true leaf is known; see `fallback_check` in the stats.

Splits are stratified by class with StratifiedGroupKFold (7 folds: fold 0 = test, fold 1 = val, rest = train).

    python -m ml.prepare_splits
"""
import collections
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ml.classes import CLASSES, CLASS_INDEX

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COLOR = RAW / "raw" / "color"
GAP = 12
CAP = 60


def photo_key(filename: str) -> str:
    return filename.split("___", 1)[-1].rsplit(".", 1)[0].lower()


def parse_key(key: str) -> tuple[str, int]:
    m = re.match(r"^(.*?)\s*(\d+)$", key)
    return (m.group(1), int(m.group(2))) if m else (key, 0)


def assign_runs(recs: list[dict], tag: str) -> None:
    """Group photos of one capture series into runs of consecutive numbers (see module docstring)."""
    recs.sort(key=lambda r: parse_key(r["key"])[1])
    run, size, prev = 0, 0, None
    for r in recs:
        n = parse_key(r["key"])[1]
        if prev is not None and (n - prev > GAP or size >= CAP):
            run, size = run + 1, 0
        prev, size = n, size + 1
        r["group"] = f"{tag}{run}"


def fallback_check() -> dict:
    """Apply the fallback rule to images whose true leaf IS known, and count how many leaves it splits."""
    leaf_map = json.load(open(RAW / "leaf_grouping" / "leaf-map.json"))
    series = collections.defaultdict(list)
    for key, cand in leaf_map.items():
        if len(cand) == 1:
            series[(cand[0].split(":::")[0], parse_key(key)[0])].append({"key": key, "truth": cand[0]})
    total = split_imgs = 0
    for (cls, prefix), recs in series.items():
        assign_runs(recs, "x")
        runs_of_leaf = collections.defaultdict(set)
        for r in recs:
            runs_of_leaf[r["truth"]].add(r["group"])
        for r in recs:
            total += 1
            split_imgs += len(runs_of_leaf[r["truth"]]) > 1
    return {"images_checked": total, "share_whose_leaf_is_split_across_groups": round(split_imgs / total, 4)}


def build_groups() -> list[dict]:
    leaf_map = json.load(open(RAW / "leaf_grouping" / "leaf-map.json"))
    records, unmapped = [], collections.defaultdict(list)
    folders = sorted(os.listdir(COLOR))
    assert folders == sorted(CLASSES), "class folders do not match ml/classes.py"
    for cls in folders:
        for fn in sorted(os.listdir(COLOR / cls)):
            rec = {"path": f"{cls}/{fn}", "label": CLASS_INDEX[cls], "cls": cls, "key": photo_key(fn)}
            cand = leaf_map.get(rec["key"], [])
            exact = [c for c in cand if c.split(":::")[0] == cls]
            pick = exact[0] if exact else (cand[0] if len(cand) == 1 else None)
            if pick:
                rec["group"] = "leaf:" + pick
            else:
                unmapped[(cls, parse_key(rec["key"])[0])].append(rec)
            records.append(rec)
    # fallback: consecutive-number runs inside one capture series (same class, same filename prefix)
    for (cls, prefix), recs in unmapped.items():
        assign_runs(recs, f"run:{cls}|{prefix}|")
    return records


def main() -> None:
    records = build_groups()
    y = np.array([r["label"] for r in records])
    group_ids = {g: i for i, g in enumerate(sorted({r["group"] for r in records}))}
    g = np.array([group_ids[r["group"]] for r in records])
    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g))
    idx = {"test": folds[0][1], "val": folds[1][1], "train": np.concatenate([f[1] for f in folds[2:]])}
    splits = {k: [[records[i]["path"], int(records[i]["label"])] for i in v] for k, v in idx.items()}

    gsets = {k: {records[i]["group"] for i in v} for k, v in idx.items()}
    assert not (gsets["train"] & gsets["test"]) and not (gsets["train"] & gsets["val"]) and not (gsets["val"] & gsets["test"])

    # How badly would a plain random image split have leaked? (same test size, random images)
    rng = random.Random(0)
    n_test = len(idx["test"])
    order = list(range(len(records)))
    rng.shuffle(order)
    rand_test, rand_train = order[:n_test], order[n_test:]
    train_groups = {records[i]["group"] for i in rand_train}
    leaked = sum(records[i]["group"] in train_groups for i in rand_test) / n_test

    stats = {
        "images": len(records),
        "groups": len(group_ids),
        "groups_from_official_leaf_map": sum(r["group"].startswith("leaf:") for r in records),
        "sizes": {k: len(v) for k, v in splits.items()},
        "fallback_check": fallback_check(),
        "classes_missing_from_split": {k: sorted({CLASSES[i] for i in range(38)} - {CLASSES[l] for _, l in v}) for k, v in splits.items()},
        "random_split_test_images_with_a_sibling_in_train": round(leaked, 4),
    }
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    json.dump({"classes": CLASSES, **splits, "stats": stats}, open(out / "splits.json", "w"))
    json.dump(stats, open(out / "split_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
