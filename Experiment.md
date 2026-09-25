## v0 — Zero-shot baseline  (2026-09-25)
- Hypothesis: an off-the-shelf open-vocabulary detector gives a usable baseline without any training.
- Change: YOLO-World v2 (yolov8s-worldv2), text prompts = our 6 class names, no fine-tuning.
- Eval set: data-v1 val — 349 images, 501 objects.
- Result: mAP50 = 0.386 | mAP50-95 = 0.278 | Precision = 0.321 | Recall = 0.569
- Speed: 11.6 ms/image (~87 FPS) on Tesla T4
- Per-class AP50: Glasses __, Sunglasses __, Headphones __, Watch __, Pen __, Hat __
- Weakest classes: __
- Regression gate: BASELINE (first version)
- Observations:
  - Low precision (0.32) means many false positives; recall is moderate (0.57).
  - Data issue: 63 val images had duplicate boxes (auto-removed by Ultralytics).
    Likely cause: merging the rare-class top-up into the original pull duplicated labels
    for overlapping images. No impact on metrics, but fix in data-v2.
  - Infra issue: reading images from Google Drive is slow (val scan took ~4 min).
    Mitigation: copy data to local Colab disk before training.
- Decision: fine-tune a YOLOv8n detector on our data (v1).
