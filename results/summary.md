**Model:** `weights/best.pt` | **Data:** CamVid test (233 images) | **Image size:** 640

| Metric | Pretrained (Cityscapes, as-is) | Fine-tuned on CamVid |
|---|---|---|
| mIoU | 0.682 | **0.732** |
| Pixel accuracy | 0.905 | **0.929** |
| Inference time | 47.2 ms | 47.1 ms |

| class | iou_baseline | iou_finetuned |
|---|---|---|
| road | 0.941 | 0.96 |
| sky | 0.831 | 0.915 |
| car | 0.857 | 0.886 |
| sidewalk | 0.816 | 0.865 |
| building | 0.842 | 0.858 |
| vegetation | 0.72 | 0.78 |
| cyclist | 0.536 | 0.699 |
| pedestrian | 0.55 | 0.641 |
| fence | 0.568 | 0.57 |
| sign | 0.519 | 0.539 |
| pole | 0.327 | 0.343 |
