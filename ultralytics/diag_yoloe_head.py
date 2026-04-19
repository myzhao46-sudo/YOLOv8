#!/usr/bin/env python
"""Dump set_classes source + cv4 structure so we know how to inject embeddings."""
import sys, os, inspect, torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics")]:
    if os.path.isdir(os.path.join(_c, "ultralytics")):
        sys.path.insert(0, _c)
        break

MODEL = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\ultralytics\best.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
inner = model.model
head = list(inner.model)[-1]

print("=" * 80)
print("1) set_classes source code")
print("=" * 80)
if hasattr(inner, "set_classes"):
    try:
        print(inspect.getsource(inner.set_classes))
    except Exception as e:
        print(f"Cannot get source: {e}")
else:
    print("No set_classes method found")

print("\n" + "=" * 80)
print("2) head.cv4 structure")
print("=" * 80)
if hasattr(head, "cv4"):
    for i, bnh in enumerate(head.cv4):
        print(f"\ncv4[{i}]: {bnh.__class__.__name__} from {bnh.__class__.__module__}")
        for attr in sorted(dir(bnh)):
            if attr.startswith("_"):
                continue
            val = getattr(bnh, attr, None)
            if callable(val) and not isinstance(val, (torch.nn.Module, torch.Tensor)):
                continue
            if isinstance(val, torch.nn.Module):
                ps = sum(p.numel() for p in val.parameters())
                print(f"  .{attr} = {val.__class__.__name__} (params={ps})")
            elif isinstance(val, torch.Tensor):
                print(f"  .{attr} = Tensor shape={list(val.shape)} dtype={val.dtype}")
            elif isinstance(val, (int, float, bool, str, type(None))):
                print(f"  .{attr} = {repr(val)}")
        # Also print named buffers
        print(f"  --- named_buffers ---")
        for name, buf in bnh.named_buffers():
            print(f"  buffer '{name}': shape={list(buf.shape)} dtype={buf.dtype}")
        print(f"  --- named_parameters ---")
        for name, p in bnh.named_parameters():
            print(f"  param '{name}': shape={list(p.shape)} dtype={p.dtype}")

print("\n" + "=" * 80)
print("3) head forward / _inference source (first 100 lines)")
print("=" * 80)
for cls in type(head).__mro__:
    if "forward" in cls.__dict__:
        print(f"\nforward from {cls.__name__}:")
        try:
            src = inspect.getsource(cls.__dict__["forward"])
            lines = src.split("\n")[:50]
            print("\n".join(lines))
        except:
            print("  (cannot get source)")
        break

for cls in type(head).__mro__:
    if "_inference" in cls.__dict__:
        print(f"\n_inference from {cls.__name__}:")
        try:
            src = inspect.getsource(cls.__dict__["_inference"])
            lines = src.split("\n")[:50]
            print("\n".join(lines))
        except:
            print("  (cannot get source)")
        break

print("\n" + "=" * 80)
print("4) BNContrastiveHead forward source")
print("=" * 80)
if hasattr(head, "cv4") and len(head.cv4) > 0:
    bnh = head.cv4[0]
    for cls in type(bnh).__mro__:
        if "forward" in cls.__dict__:
            print(f"forward from {cls.__module__}.{cls.__name__}:")
            try:
                print(inspect.getsource(cls.__dict__["forward"]))
            except:
                print("  (cannot get source)")
            break

print("\nDone.")
