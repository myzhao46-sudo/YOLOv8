# EWC Incremental Training (YOLOv8 8.4.32)

This folder contains a minimal EWC integration on top of Ultralytics `DetectionTrainer`.

## Files

- `train_ewc.py`: entry script (Step1 Fisher, Step2 EWC train)
- `custom_trainer.py`: `EWC_DetectionTrainer` that wraps native detection loss with EWC penalty
- `ewc_utils.py`: Fisher/old-params save-load utilities

## Quick start

Run from repository root:

```bash
python scripts/train_ewc.py
```

Example with explicit arguments:

```bash
python scripts/train_ewc.py \
  --old-model-path ultralytics/runs/detect/ship_teacher/weights/best.pt \
  --old-data-yaml ultralytics/tmp_data/ship_teacher_abs.yaml \
  --new-data-yaml ultralytics/tmp_data/oiltank_new_eval_abs.yaml \
  --joint-val-yaml ultralytics/tmp_data/ship_oiltank_joint_abs.yaml \
  --epochs 100 --batch 16 --imgsz 640 --device 0 \
  --lambda-ewc 500 --topk-ratio 0.2 --extra-eval-interval 10
```

Smoke test:

```bash
python scripts/train_ewc.py --device cpu --smoke
```

## Notes

- This integration keeps the native YOLO detection training loop and only injects EWC into `model.loss`.
- Fisher and old params are saved under `--fisher-cache-dir` as `old_params.pth` and `fisher.pth`.
- If class count changes between old and new tasks, detection head parameters may be shape-mismatched and skipped by EWC (reported in logs).
- By default, script reports extra old/new/joint validation metrics every 10 epochs (and final epoch).
