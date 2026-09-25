## v0 - Zero-shot baseline 
- Hypothesis: an off-the-shelf open-vocabulary detector gives a usable baseline without any training.
- Change: YOLO-World v2 (yolov8s-worldv2), text prompts = our 6 class names, no fine-tuning.
- Eval set: data-v1 val - 349 images, 501 objects.
- Result: mAP50 = 0.386 | mAP50-95 = 0.278 | Precision = 0.321 | Recall = 0.569
- Speed: 11.6 ms/image (Approximately 87 FPS) on Tesla T4
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


## v1 - First fine-tuned model  
- Hypothesis: fine-tuning a small detector on our data will beat the zero-shot baseline by a wide margin.
- Change: YOLOv8n (COCO-pretrained), 30 epochs, imgsz 640, batch 16, seed 42.
- Infra: data copied to local Colab disk before training (fixes slow Drive I/O).
- Result: mAP50 = 0.676 | mAP50-95 = 0.437 (v0: 0.278, +57%) | Precision = 0.746 | Recall = 0.611
- Speed: Approximately 118 FPS on T4 (v0: 87)
- Per-class AP50 (v0 -> v1): Glasses 0.75-> 0.81, Sunglasses 0.45 -> 0.59, Headphones 0.23 -> 0.72,
  Watch 0.18→0.73, Pen 0.13→0.64, Hat 0.57→0.55
- Regression gate: PASS vs v0 (Hat −0.02, within 0.03 tolerance - watch item)
- Training curves: train AND val losses still falling at epoch 30; mAP still rising → under-trained, no overfitting.
- Decision: v2 = same setup with 60 epochs.
