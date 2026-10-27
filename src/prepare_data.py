"""
Download CamVid (from the SegNet tutorial repository on GitHub) and convert it to the
folder layout Ultralytics expects:

    datasets/camvid/images/{train,val,test}/*.png
    datasets/camvid/masks/{train,val,test}/*.png    (pixel value = class id, 11 = unlabelled)

    python src/prepare_data.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from ultralytics.utils import SETTINGS

REPO = "https://github.com/alexgkendall/SegNet-Tutorial.git"


def dataset_root() -> Path:
    return Path(SETTINGS["datasets_dir"]) / "camvid"


def prepare(force: bool = False) -> Path:
    root = dataset_root()
    if (root / "images" / "test").is_dir() and not force:
        print(f"CamVid already prepared in {root}")
        return root

    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading CamVid (~170 MB)...")
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO, tmp], check=True)
        src = Path(tmp) / "CamVid"
        for split in ("train", "val", "test"):
            img_dst, mask_dst = root / "images" / split, root / "masks" / split
            img_dst.mkdir(parents=True, exist_ok=True)
            mask_dst.mkdir(parents=True, exist_ok=True)
            images = sorted((src / split).glob("*.png"))
            for img in images:
                shutil.copy(img, img_dst / img.name)
                shutil.copy(src / f"{split}annot" / img.name, mask_dst / img.name)
            print(f"  {split}: {len(images)} images")

    print(f"Done. Dataset ready in {root}")
    return root


if __name__ == "__main__":
    prepare()
