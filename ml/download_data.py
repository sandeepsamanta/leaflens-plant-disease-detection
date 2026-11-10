"""Download and unpack the datasets (about 3.2 GB). Safe to re-run: existing files are skipped.

  PlantVillage (CC BY-SA 3.0)  https://huggingface.co/datasets/mohanty/PlantVillage
  PlantDoc     (CC BY 4.0)     https://huggingface.co/datasets/agyaatcoder/PlantDoc
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, DOC = ROOT / "data" / "raw", ROOT / "data" / "plantdoc"
PV = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/"
PD = "https://huggingface.co/datasets/agyaatcoder/PlantDoc/resolve/main/data/"

FILES = [
    (PV + "leaf_grouping/leaf-map.json", RAW / "leaf_grouping" / "leaf-map.json"),
    (PV + "data.zip", RAW / "data.zip"),
    (PD + "test-00000-of-00001.parquet", DOC / "test.parquet"),
    (PD + "train-00000-of-00002.parquet", DOC / "train0.parquet"),
    (PD + "train-00001-of-00002.parquet", DOC / "train1.parquet"),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print("have", dest.relative_to(ROOT))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)


def main() -> None:
    for url, dest in FILES:
        fetch(url, dest)
    if not (RAW / "raw" / "color").exists() or not (RAW / "raw" / "segmented").exists():
        print("unpacking colour and segmented images")
        with zipfile.ZipFile(RAW / "data.zip") as z:
            z.extractall(RAW, [n for n in z.namelist() if n.startswith(("raw/color/", "raw/segmented/"))])
    print("done")


if __name__ == "__main__":
    main()
