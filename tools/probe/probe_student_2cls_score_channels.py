# tools/probe/probe_student_2cls_score_channels.py
# -*- coding: utf-8 -*-

"""
Probe whether a candidate YOLOE student truly emits 2 class score channels after
set_classes(["ship", "bridge"]).

This script does not train, evaluate, modify best.pt, save checkpoints, convert the
model to detect, or replace the YOLOESegment head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
DEFAULT_WEIGHTS = REPO_ROOT / "ultralytics" / "best.pt"
STUDENT_NAMES = ["ship", "bridge"]


def setup_paths() -> None:
    for p in [str(PACKAGE_ROOT.resolve()), str(REPO_ROOT.resolve())]:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(REPO_ROOT.resolve()))


def parse_args():
    parser = argparse.ArgumentParser(description="Probe YOLOE student 2-class score channels.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Candidate student .pt weights.")
    parser.add_argument("--model-yaml", default="", help="Optional YOLOE model yaml to instantiate instead of weights.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_paths()

    from ultralytics import YOLO
    from tools.eval.eval_teacher_ship_external import (
        get_names_state,
        get_task_state,
        head_state,
        obj_type,
        raw_yoloeseg_fused_forward,
        safe_getattr,
        safe_signature,
        shape_of,
    )

    source = Path(args.model_yaml).resolve() if args.model_yaml else Path(args.weights).resolve()

    print("[IMPORT CHECK]", flush=True)
    import ultralytics as ultralytics_pkg

    print(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}", flush=True)
    print(f"first sys.path entries: {sys.path[:3]}", flush=True)

    print("", flush=True)
    print("[LOAD CANDIDATE STUDENT]", flush=True)
    print(f"student weights / model yaml: {source}", flush=True)
    yolo = YOLO(str(source))
    inner = safe_getattr(yolo, "model", None)
    before_head = head_state(inner)
    print(f"YOLO wrapper type: {obj_type(yolo)}", flush=True)
    print(f"model class: {obj_type(inner)}", flush=True)
    print(f"task before: {json.dumps(get_task_state(yolo, inner), ensure_ascii=False)}", flush=True)
    print(f"names before: {json.dumps(get_names_state(yolo, inner), ensure_ascii=False)}", flush=True)
    print(f"last layer type: {before_head['last_layer_type']}", flush=True)
    print(f"last_layer.nc before: {before_head['last_layer.nc']}", flush=True)
    print(f"last_layer.is_fused before: {before_head.get('last_layer.is_fused')}", flush=True)

    print("", flush=True)
    print("[SET_CLASSES 2CLS]", flush=True)
    get_text_pe = safe_getattr(yolo, "get_text_pe", None) or safe_getattr(inner, "get_text_pe", None)
    set_classes = safe_getattr(yolo, "set_classes", None) or safe_getattr(inner, "set_classes", None)
    if not callable(get_text_pe) or not callable(set_classes):
        raise AttributeError("Candidate does not expose callable get_text_pe and set_classes.")

    print(f"student_names = {STUDENT_NAMES}", flush=True)
    print(f"get_text_pe signature: {safe_signature(get_text_pe)}", flush=True)
    embeddings = get_text_pe(STUDENT_NAMES)
    print(f"pe shape: {shape_of(embeddings)}", flush=True)
    print(f"set_classes signature: {safe_signature(set_classes)}", flush=True)
    set_classes(STUDENT_NAMES, embeddings)

    after_head = head_state(inner)
    print(f"names after: {json.dumps(get_names_state(yolo, inner), ensure_ascii=False)}", flush=True)
    print(f"task after: {json.dumps(get_task_state(yolo, inner), ensure_ascii=False)}", flush=True)
    print(f"last layer type after: {after_head['last_layer_type']}", flush=True)
    print(f"last_layer.nc after: {after_head['last_layer.nc']}", flush=True)
    print(f"last_layer.is_fused after: {after_head.get('last_layer.is_fused')}", flush=True)

    print("", flush=True)
    print("[SCORE CHANNEL PROBE]", flush=True)
    inner.eval().to(args.device)
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        _, score_nc = raw_yoloeseg_fused_forward(inner, x)
    score_channels_seen = [int(score_nc)]
    print(f"score channels seen: {score_channels_seen}", flush=True)
    print(f"whether score channels == 2: {score_channels_seen == [2]}", flush=True)

    print("", flush=True)
    print("[CONCLUSION]", flush=True)
    if score_channels_seen == [2] and after_head["last_layer.nc"] == 2:
        print("OK: candidate student appears to emit 2 class score channels.", flush=True)
        return 0

    print("NOT OK: candidate student does not emit 2 class score channels.", flush=True)
    print("If this is best.pt, it is still the original fused 3-class head and is not a 2-class student.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
