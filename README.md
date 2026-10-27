# 🛣️ Road, Lane & Obstacle Segmentation

Pixel-level understanding of driving scenes: for every camera frame the system labels **each pixel** (road, sidewalk, car, pedestrian, cyclist, …), finds the **lane lines** on the road, and checks whether the **path ahead is clear**: the perception building blocks behind lane keeping, free-space detection and collision warning.

It combines **deep learning** (a YOLO26 semantic segmentation model, pretrained on German city streets and fine-tuned on CamVid) with **classic OpenCV** (edge detection and Hough lines for lane markings), and logs every frame to CSV for analysis in Excel or Power BI.

![Demo](assets/demo.gif)
<sub>Unseen CamVid test drives. Green corridor = path clear, orange = obstacle in path. Yellow = detected lane lines.</sub>

---

## Results

Scored on the **233 CamVid test images**, which were never used in training. Metric: **IoU**, the overlap between predicted and true pixels (1.0 = perfect). **mIoU** averages it over all 11 classes.

| Metric (640 px) | Pretrained on Cityscapes, used as-is | **Fine-tuned on CamVid** |
|---|---|---|
| **mIoU** | 0.682 | **0.732** |
| Pixel accuracy | 0.905 | **0.929** |
| Inference time | 47 ms / image | **47 ms / image** on a 2-core CPU, no GPU |

![IoU per class](assets/iou_comparison.png)

| Class | Pretrained | Fine-tuned | | Class | Pretrained | Fine-tuned |
|---|---|---|---|---|---|---|
| road | 0.941 | **0.960** | | vegetation | 0.720 | **0.780** |
| sky | 0.831 | **0.915** | | cyclist | 0.536 | **0.699** |
| car | 0.857 | **0.886** | | pedestrian | 0.550 | **0.641** |
| sidewalk | 0.816 | **0.865** | | fence | 0.568 | **0.570** |
| building | 0.842 | **0.858** | | sign | 0.519 | **0.539** |
| | | | | pole | 0.327 | **0.343** |

### Speed vs. accuracy

| Model | 480 px | 640 px | 1024 px |
|---|---|---|---|
| Pretrained (Cityscapes) | 0.627 · 30 ms | 0.682 · 47 ms | 0.705 · 120 ms |
| **Fine-tuned (CamVid)** | 0.697 · 31 ms | **0.732 · 47 ms** | 0.748 · 121 ms |

![Speed vs accuracy](assets/speed_vs_accuracy.png)

<sub>Training: YOLO26n-sem (1.6 M parameters, 3.4 MB), 30 epochs on 367 images at 480 px, CPU only, ~1.5 hours. Full numbers in [`results/`](results/).</sub>

### Key observations

- **Knowledge transfers between countries.** The model trained only on German cities already reached 0.68 mIoU on English streets it had never seen. Fine-tuning on 367 local images added 5 points, and **the biggest gains were for the classes that matter most for safety: cyclists +16 points (0.54 → 0.70) and pedestrians +9 points (0.55 → 0.64)**. CamVid's camera, viewpoint and class definitions differ from Cityscapes (for example, a cyclist is one class including the bike), and fine-tuning adapts to those differences.
- **640 px is the sweet spot.** The fine-tuned model at 640 px beats the pretrained model at its full 1024 px resolution (0.732 vs 0.705) while being **2.5× faster** (47 vs 120 ms). Going to 1024 px adds only 0.016 mIoU for 2.6× the computing time, a poor trade for an in-vehicle system with a fixed compute budget.
- **Thin objects are the hardest.** Poles (0.34) and signs (0.54) cover about 1% of all pixels and are only a few pixels wide, so small boundary errors cost a large share of their IoU. Higher resolution helps them most (pole 0.34 → 0.39 at 1024 px).
- **Pretraining does most of the work.** Validation mIoU jumped from 0.51 to 0.73 in the first 5 epochs, then rose slowly to 0.77 by epoch 30.

![Training curves](assets/training_curves.png)

### Lane and obstacle analysis

![Examples](assets/demo_grid.jpg)
<sub>Left to right, top to bottom: both lane lines found · pedestrian crossing in path · car ahead · car close (caution).</sub>

- **Lane lines** were found in **80% of frames** of the 171-frame Seq05VD test drive.
- **Bug found and fixed:** at junctions, painted symbols (such as a cycle-lane bicycle) produced left and right lines that *crossed*, which turned the driving corridor into an X shape and flagged pedestrians on the pavement as "in path". A sanity check now rejects crossing or implausibly narrow lane pairs and falls back to a fixed zone ahead. That cut false obstacle warnings on this drive from 7 frames to 2, both of them genuine (a pedestrian crossing, a car turning in).
- **Following a car** (0001TP drive, 62 frames): as the car ahead got closer, the free distance in the corridor fell from 65% to under 1% and the status escalated from *car ahead* to *caution*, then dropped back to *car ahead* or *clear* as the gap opened again.

![Drive report](assets/drive_report.png)

---

## How it works

```
 camera frame
     │
     ▼
 YOLO26-sem (pretrained on Cityscapes → fine-tuned on CamVid)
     │  class for every pixel: road, sidewalk, car, pedestrian, cyclist, ...
     ├──────────────► coloured overlay
     ▼
 OpenCV lane detection inside the predicted road
     │  bright / yellow pixels → Canny edges → Hough lines → left + right lane fit → sanity check
     ▼
 Driving corridor (ego lane, or fixed zone ahead if no lanes)
     │  cars / pedestrians / cyclists inside it? how close?
     ▼
 PATH CLEAR · OBSTACLE AHEAD · CAUTION      +  frame_metrics.csv
```

| Decision | Why |
|---|---|
| **Semantic segmentation** instead of boxes | Labels every pixel, so the system knows exactly where the road ends; boxes can't describe road shape or free space. |
| **YOLO26-sem nano** | Small (3.4 MB) and fast on CPU: the kind of efficiency in-vehicle hardware needs. |
| **Start from Cityscapes** | Cityscapes is recorded in 50 German cities with a car-mounted camera, the closest public match to the target domain, so fine-tuning needs little data. |
| **Classic OpenCV for lanes** | CamVid has no separate lane-marking labels. Simple image processing inside the predicted road is cheap, explainable and easy to debug. |
| **CSV per frame** | Turns video into data: share of time with a clear path, obstacle types, lane visibility, drivable area over time. |

---

## Project structure

```
road-scene-segmentation/
├── configs/camvid.yaml           # dataset definition (11 classes)
├── src/
│   ├── prepare_data.py           # download CamVid from GitHub and convert it
│   ├── train.py                  # fine-tune YOLO26-sem on CamVid
│   ├── evaluate.py               # per-class IoU vs. pretrained baseline → CSV, chart, markdown
│   ├── segment.py                # image / folder / video / webcam → overlay + lanes + obstacle check + CSV
│   └── drive_report.py           # drive summary chart from frame_metrics.csv
├── notebooks/train_on_colab.ipynb  # retrain on a free Colab GPU
├── weights/best.pt               # trained model (3.4 MB)
├── results/                      # metrics, training log, speed/accuracy table, sample drive
└── assets/                       # images and demo animation for this README
```

---

## Quick start

Download or clone this repository, then:

```bash
pip install -r requirements.txt

python src/segment.py --source path/to/dashcam.mp4       # road, lanes and obstacles with the trained model
python src/drive_report.py                               # drive summary from that run

python src/prepare_data.py                               # download CamVid (~170 MB)
python src/evaluate.py                                   # reproduce the results above
python src/train.py --epochs 60 --device 0               # retrain (GPU recommended)
```

Or open `notebooks/train_on_colab.ipynb` in Google Colab to retrain on a free GPU.

---

## Limitations & next steps

- **Distance is an image-space estimate.** "Free distance" is the share of the corridor below the nearest obstacle, not metres. Converting it to real distance needs camera calibration.
- **Lane detection is rule-based** and was checked visually, not scored against lane ground truth. A learned lane model (for example trained on CULane or TuSimple) would handle worn markings, curves and night driving better.
- **Frames are analysed independently.** Smoothing lanes and obstacle status over consecutive frames would remove flicker.
- **Domain:** CamVid is daytime driving in Cambridge, UK. The next step is testing on German roads, at night and in rain.

---

## Tech stack

Python · Ultralytics YOLO26 · PyTorch · OpenCV · NumPy · pandas · matplotlib · Google Colab

## Credits & licences

- Model and training framework: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). The starting weights were pretrained by Ultralytics on [Cityscapes](https://www.cityscapes-dataset.com/) (non-commercial licence)
- Dataset and demo images: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), University of Cambridge (Brostow, Fauqueur & Cipolla, 2009), via the [SegNet tutorial](https://github.com/alexgkendall/SegNet-Tutorial) repository
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning)

This project is released under the AGPL-3.0 licence to match Ultralytics.

**Author:** Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck · [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/) · [GitHub](https://github.com/souadessemyhij1999)
