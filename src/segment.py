"""
Road, lane and obstacle segmentation for images, folders, videos or a webcam.

For every frame:
  1. Deep learning (YOLO26-sem) labels every pixel: road, sidewalk, car, pedestrian, ...
  2. OpenCV finds lane markings inside the predicted road area (brightness threshold +
     Canny edges + Hough line transform) and fits a left and right lane line.
  3. The driving corridor (ego lane, or a default zone ahead if no lanes are visible) is
     checked for obstacles: cars, pedestrians and cyclists inside it.

Outputs an annotated image/video plus frame_metrics.csv for analysis in Excel / Power BI.

Examples:
    python src/segment.py --source samples/street.png
    python src/segment.py --source samples/drive.mp4
    python src/segment.py --source 0 --show          # webcam
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OBSTACLES = {"car", "pedestrian", "cyclist", "person", "rider", "truck", "bus", "train", "motorcycle", "bicycle"}

# Overlay colours (BGR), close to the usual Cityscapes palette
COLOURS = {
    "road": (128, 64, 128), "sidewalk": (232, 35, 244), "building": (70, 70, 70), "wall": (156, 102, 102),
    "fence": (153, 153, 190), "pole": (153, 153, 153), "sign": (0, 220, 220), "traffic light": (30, 170, 250),
    "traffic sign": (0, 220, 220), "vegetation": (35, 142, 107), "terrain": (152, 251, 152), "sky": (180, 130, 70),
    "pedestrian": (60, 20, 220), "person": (60, 20, 220), "cyclist": (0, 0, 255), "rider": (0, 0, 255),
    "car": (142, 0, 0), "truck": (70, 0, 0), "bus": (100, 60, 0), "train": (100, 80, 0),
    "motorcycle": (230, 0, 0), "bicycle": (32, 11, 119),
}

HORIZON = 0.60  # lane lines and corridor are drawn from the bottom up to 60% of image height


def default_model() -> str:
    trained = Path("weights/best.pt")
    return str(trained) if trained.exists() else "yolo26n-sem.pt"


class RoadSceneAnalyser:
    def __init__(self, model_path: str, imgsz: int = 640, alpha: float = 0.45):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.alpha = alpha
        names = self.model.names
        self.road_id = next(i for i, n in names.items() if n == "road")
        self.obstacle_ids = {i: n for i, n in names.items() if n in OBSTACLES}
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        for i, n in names.items():
            self.palette[i] = COLOURS.get(n, (0, 0, 0))

    # ---------- step 1: deep-learning segmentation ----------
    def segment(self, frame):
        result = self.model.predict(frame, imgsz=self.imgsz, verbose=False)[0]
        return result.semantic_mask.data.cpu().numpy().astype(np.uint8)

    # ---------- step 2: classic OpenCV lane detection ----------
    def find_lanes(self, frame, road):
        h, w = road.shape
        top = int(h * HORIZON)
        # Lane markings are usually labelled "road", so search inside a slightly grown road mask
        region = cv2.dilate(road.astype(np.uint8), np.ones((15, 15), np.uint8))
        region[:top] = 0
        if region.sum() < 0.02 * h * w:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_brightness = np.median(gray[region > 0])
        white = gray > min(road_brightness + 45, 235)
        yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255)) > 0
        marks = ((white | yellow) & (region > 0)).astype(np.uint8) * 255
        marks = cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        edges = cv2.Canny(marks, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=18, maxLineGap=30)
        if lines is None:
            return None, None

        left, right = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not 0.35 < abs(slope) < 5:  # ignore near-horizontal lines (stop lines, kerbs far away)
                continue
            length = np.hypot(x2 - x1, y2 - y1)
            intercept = y1 - slope * x1
            x_bottom = (h - intercept) / slope
            if slope < 0 and x_bottom < w * 0.6:
                left.append((slope, intercept, length))
            elif slope > 0 and x_bottom > w * 0.4:
                right.append((slope, intercept, length))

        def fit(group):
            if not group:
                return None
            g = np.array(group)
            slope, intercept = np.average(g[:, :2], axis=0, weights=g[:, 2])
            return (int((h - intercept) / slope), h), (int((top - intercept) / slope), top)

        left_line, right_line = fit(left), fit(right)
        if left_line and right_line:
            (lb, _), (lt, _) = left_line
            (rb, _), (rt, _) = right_line
            # Sanity check: real lane borders never cross and are at least ~20% of the image apart at the
            # bottom. Crossing lines happen at junctions and on painted symbols, so reject both.
            if lt >= rt or rb - lb < 0.2 * w:
                return None, None
        return left_line, right_line

    # ---------- step 3: corridor + obstacle check ----------
    @staticmethod
    def corridor(shape, left, right):
        h, w = shape
        top = int(h * HORIZON)
        if left and right and left[0][0] < right[0][0]:
            pts = [left[0], left[1], right[1], right[0]]
        else:  # default zone straight ahead of the car
            pts = [(int(w * 0.25), h), (int(w * 0.46), top), (int(w * 0.54), top), (int(w * 0.75), h)]
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
        return mask, pts

    def check_path(self, classes, corridor_mask):
        h = classes.shape[0]
        top = int(h * HORIZON)
        area = corridor_mask.sum()
        drivable = float((classes[corridor_mask > 0] == self.road_id).mean()) if area else 0.0
        found = {}
        for i, name in self.obstacle_ids.items():
            px = (classes == i) & (corridor_mask > 0)
            n = int(px.sum())
            if n > max(0.004 * area, 60):
                found[name] = (n, int(np.where(px)[0].max()))  # pixel count, lowest (nearest) row
        if not found:
            return {"status": "clear", "obstacle": "", "obstacle_pct": 0.0, "free_distance_pct": 100.0,
                    "drivable_pct": round(100 * drivable, 1)}
        name, (n, row) = max(found.items(), key=lambda kv: kv[1][1])  # nearest obstacle
        free = 100 * (h - row) / max(h - top, 1)
        return {
            "status": "obstacle_near" if free < 35 else "obstacle_ahead",
            "obstacle": name,
            "obstacle_pct": round(100 * sum(v[0] for v in found.values()) / area, 1),
            "free_distance_pct": round(free, 1),
            "drivable_pct": round(100 * drivable, 1),
        }

    # ---------- drawing ----------
    def draw(self, frame, classes, left, right, pts, info):
        out = cv2.addWeighted(frame, 1 - self.alpha, self.palette[classes], self.alpha, 0)
        lane_layer = out.copy()
        cv2.fillPoly(lane_layer, [np.array(pts, dtype=np.int32)], (0, 200, 0) if info["status"] == "clear" else (0, 140, 255))
        out = cv2.addWeighted(out, 0.75, lane_layer, 0.25, 0)
        for line in (left, right):
            if line:
                cv2.line(out, line[0], line[1], (0, 255, 255), 4, cv2.LINE_AA)

        status_text = {"clear": "PATH CLEAR", "obstacle_ahead": f"{info['obstacle'].upper()} AHEAD",
                       "obstacle_near": f"CAUTION: {info['obstacle'].upper()} CLOSE"}[info["status"]]
        status_colour = {"clear": (0, 170, 0), "obstacle_ahead": (0, 140, 255), "obstacle_near": (0, 0, 220)}[info["status"]]
        lanes = (left is not None) + (right is not None)
        rows = [(status_text, status_colour), (f"Drivable area in path: {info['drivable_pct']:.0f}%", (40, 40, 40)),
                (f"Lane lines found: {lanes}", (40, 40, 40))]
        cv2.rectangle(out, (8, 8), (300, 18 + 24 * len(rows)), (255, 255, 255), -1)
        for k, (text, colour) in enumerate(rows):
            cv2.putText(out, text, (16, 30 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2 if k == 0 else 1, cv2.LINE_AA)
        return out

    def process(self, frame):
        classes = self.segment(frame)
        left, right = self.find_lanes(frame, classes == self.road_id)
        corridor_mask, pts = self.corridor(classes.shape, left, right)
        info = self.check_path(classes, corridor_mask)
        info["lane_lines"] = (left is not None) + (right is not None)
        info["road_pct_frame"] = round(100 * float((classes == self.road_id).mean()), 1)
        return self.draw(frame, classes, left, right, pts, info), info


FIELDS = ["source", "frame", "status", "obstacle", "obstacle_pct", "free_distance_pct", "drivable_pct", "lane_lines", "road_pct_frame"]


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyser = RoadSceneAnalyser(args.model, args.imgsz, args.alpha)
    print(f"Model: {args.model} ({len(analyser.model.names)} classes)")

    with open(out_dir / "frame_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        src = Path(args.source)

        if src.is_dir() or src.suffix.lower() in IMAGE_EXTS:
            images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
            for i, path in enumerate(images):
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                annotated, info = analyser.process(frame)
                cv2.imwrite(str(out_dir / path.name), annotated)
                writer.writerow({"source": path.name, "frame": i, **info})
                print(f"{path.name}: {info['status']} {info['obstacle']} | lanes {info['lane_lines']} | drivable {info['drivable_pct']}%")
        else:
            cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            if not cap.isOpened():
                raise SystemExit(f"Could not open video source: {args.source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            name = "webcam.mp4" if args.source.isdigit() else f"{src.stem}_segmented.mp4"
            video = cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, info = analyser.process(frame)
                video.write(annotated)
                writer.writerow({"source": name, "frame": i, **info})
                if args.show:
                    cv2.imshow("Road scene segmentation (q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                i += 1
                if i % 50 == 0:
                    print(f"processed {i} frames")
            cap.release()
            video.release()
            cv2.destroyAllWindows()
    print(f"\nDone. Output and frame_metrics.csv saved in {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Road, lane and obstacle segmentation.")
    p.add_argument("--source", required=True, help="image, folder, video file, or webcam index (0)")
    p.add_argument("--model", default=default_model(), help="default: weights/best.pt if present")
    p.add_argument("--imgsz", type=int, default=640, help="inference size; the model was trained at 480 but scores best at 640")
    p.add_argument("--alpha", type=float, default=0.45, help="overlay strength (0-1)")
    p.add_argument("--out", default="results/segmented")
    p.add_argument("--show", action="store_true", help="show a live window (local machine only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
