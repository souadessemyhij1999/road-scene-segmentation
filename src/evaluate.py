"""
Score segmentation quality on the CamVid test set (233 images never used in training)
and compare two models:

  * baseline   - YOLO26-sem pretrained on Cityscapes (German cities), used as-is.
                 Its 19 classes are mapped onto CamVid's 11 classes.
  * fine-tuned - the same model after training on CamVid (weights/best.pt)

Metric: IoU (intersection over union) per class = overlap between predicted and true pixels.
mIoU is the average over classes. 1.0 = perfect.

Outputs:
  results/metrics_per_class.csv   per-class IoU for both models (open in Excel / Power BI)
  results/iou_comparison.png      bar chart for the README
  results/summary.md              markdown tables to paste into the README

    python src/evaluate.py
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

IGNORE = 11  # CamVid "unlabelled"

# Cityscapes class name -> CamVid class name (classes CamVid merges together)
CITYSCAPES_TO_CAMVID = {
    "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "building",
    "fence": "fence", "pole": "pole", "traffic light": "sign", "traffic sign": "sign",
    "vegetation": "vegetation", "terrain": "vegetation", "sky": "sky",
    "person": "pedestrian", "rider": "cyclist", "bicycle": "cyclist", "motorcycle": "cyclist",
    "car": "car", "truck": "car", "bus": "car", "train": "car",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation models on CamVid.")
    p.add_argument("--model", default="weights/best.pt")
    p.add_argument("--baseline", default="yolo26n-sem.pt", help="pretrained model to compare against ('' to skip)")
    p.add_argument("--data", default="configs/camvid.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640, help="inference size; the model was trained at 480 but scores best at 640")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results")
    return p.parse_args()


def class_lut(model, camvid_names):
    """Lookup table: model class id -> CamVid class id (identity for CamVid-trained models)."""
    camvid_ids = {n: i for i, n in camvid_names.items()}
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for i, name in model.names.items():
        target = name if name in camvid_ids else CITYSCAPES_TO_CAMVID.get(name)
        if target in camvid_ids:
            lut[i] = camvid_ids[target]
    return lut


def score(model, images, masks_dir, names, imgsz, device):
    nc = len(names)
    lut = class_lut(model, names)
    cm = np.zeros((nc, nc), dtype=np.int64)
    times = []
    for img_path in images:
        truth = cv2.imread(str(masks_dir / img_path.name), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        result = model.predict(str(img_path), imgsz=imgsz, device=device, verbose=False)[0]
        times.append(result.speed["inference"])
        pred = lut[result.semantic_mask.data.cpu().numpy().astype(np.uint8)]
        valid = (truth != IGNORE) & (pred != IGNORE)
        cm += np.bincount(nc * truth[valid].astype(int) + pred[valid], minlength=nc * nc).reshape(nc, nc)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return {
        "iou": iou,
        "miou": float(iou.mean()),
        "pixel_acc": float(inter.sum() / cm.sum()),
        "ms": float(np.median(times)),
        "pixels": cm.sum(1),
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.data))
    names = {int(k): v for k, v in cfg["names"].items()}
    root = Path(SETTINGS["datasets_dir"]) / cfg["path"]
    images = sorted((root / "images" / args.split).glob("*.png"))
    masks_dir = root / cfg.get("masks_dir", "masks") / args.split
    print(f"Evaluating on {len(images)} {args.split} images")

    runs = {}
    if args.baseline:
        print(f"Baseline: {args.baseline}")
        runs["baseline"] = score(YOLO(args.baseline), images, masks_dir, names, args.imgsz, args.device)
    print(f"Fine-tuned: {args.model}")
    runs["finetuned"] = score(YOLO(args.model), images, masks_dir, names, args.imgsz, args.device)

    ft = runs["finetuned"]
    df = pd.DataFrame({"class": list(names.values()), "pixel_share": ft["pixels"] / ft["pixels"].sum()})
    if "baseline" in runs:
        df["iou_baseline"] = runs["baseline"]["iou"]
    df["iou_finetuned"] = ft["iou"]
    df = df.round(3).sort_values("iou_finetuned", ascending=False)
    df.to_csv(out / "metrics_per_class.csv", index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(df))
    if "baseline" in runs:
        ax.barh(y + 0.2, df["iou_baseline"], 0.4, color="#b8c4d6", label="Pretrained on Cityscapes (as-is)")
    ax.barh(y - 0.2, df["iou_finetuned"], 0.4, color="#1c69d4", label="Fine-tuned on CamVid")
    for yi, v in zip(y, df["iou_finetuned"]):
        ax.text(v + 0.01, yi - 0.2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, df["class"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("IoU (higher is better)")
    ax.set_title(f"Segmentation quality per class (CamVid {args.split} set, {len(images)} images)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "iou_comparison.png", dpi=150)

    lines = [f"**Model:** `{args.model}` | **Data:** CamVid {args.split} ({len(images)} images) | **Image size:** {args.imgsz}", ""]
    if "baseline" in runs:
        b = runs["baseline"]
        lines += ["| Metric | Pretrained (Cityscapes, as-is) | Fine-tuned on CamVid |", "|---|---|---|",
                  f"| mIoU | {b['miou']:.3f} | **{ft['miou']:.3f}** |",
                  f"| Pixel accuracy | {b['pixel_acc']:.3f} | **{ft['pixel_acc']:.3f}** |",
                  f"| Inference time | {b['ms']:.1f} ms | {ft['ms']:.1f} ms |", ""]
    else:
        lines += ["| Metric | Value |", "|---|---|", f"| mIoU | {ft['miou']:.3f} |",
                  f"| Pixel accuracy | {ft['pixel_acc']:.3f} |", f"| Inference time | {ft['ms']:.1f} ms |", ""]
    cols = ["class"] + [c for c in ("iou_baseline", "iou_finetuned") if c in df]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in df.to_dict("records")]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {out}/metrics_per_class.csv, iou_comparison.png, summary.md")


if __name__ == "__main__":
    main()
