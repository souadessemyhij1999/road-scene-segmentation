"""
Fine-tune YOLO26-sem on CamVid.

The starting model was pretrained by Ultralytics on Cityscapes (street scenes from 50 German
cities). Fine-tuning adapts it to CamVid's camera, roads and 11 classes.

Examples:
    python src/train.py                               # full run (GPU recommended)
    python src/train.py --epochs 1 --fraction 0.1     # quick test that everything works
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from prepare_data import prepare


def parse_args():
    p = argparse.ArgumentParser(description="Train the road-scene segmentation model.")
    p.add_argument("--data", default="configs/camvid.yaml")
    p.add_argument("--model", default="yolo26n-sem.pt", help="starting weights (n=nano, s=small, ...)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=480, help="CamVid frames are 480x360 pixels")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fraction", type=float, default=1.0, help="share of training images to use (0-1)")
    p.add_argument("--device", default=None, help="0 for first GPU, 'cpu' for CPU (default: auto)")
    p.add_argument("--name", default="camvid_yolo26n_sem")
    return p.parse_args()


def main():
    args = parse_args()
    prepare()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        project=str(Path("runs/semantic").resolve()),
        name=args.name,
        exist_ok=True,
        patience=20,
        seed=42,
        plots=True,
    )

    best = Path(model.trainer.best)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/best.pt")
    print("\nTraining finished. Best weights copied to weights/best.pt")


if __name__ == "__main__":
    main()
