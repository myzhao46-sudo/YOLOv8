#!/usr/bin/env python
"""Find how your YOLOE codebase generates text prompt embeddings."""
import sys, os, inspect, importlib

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics")]:
    if os.path.isdir(os.path.join(_c, "ultralytics")):
        sys.path.insert(0, _c)
        break

print("=" * 80)
print("1) Search for get_text_pe / build_text / encode_text functions")
print("=" * 80)

# Check YOLOEModel and its bases for any text-related methods
from ultralytics import YOLO
import torch

MODEL = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\ultralytics\best.pt"
model = YOLO(MODEL)
inner = model.model

print(f"inner class: {inner.__class__.__name__}")
print(f"MRO: {[c.__name__ for c in type(inner).__mro__]}")

# Find all text/pe/embed related methods
for cls in type(inner).__mro__:
    for name, method in cls.__dict__.items():
        if any(kw in name.lower() for kw in ["text", "pe", "embed", "prompt", "clip", "class"]):
            if callable(method):
                print(f"\n  {cls.__name__}.{name}():")
                try:
                    src = inspect.getsource(method)
                    # Print first 30 lines
                    for line in src.split("\n")[:30]:
                        print(f"    {line}")
                except:
                    print(f"    (cannot get source)")

print("\n" + "=" * 80)
print("2) Search for text model / CLIP loading utilities")
print("=" * 80)

# Check if there's a text model builder
search_modules = [
    "ultralytics.utils.text",
    "ultralytics.nn.text",
    "ultralytics.models.yolo.yoloe",
    "ultralytics.models.yolo.yoloe.train",
    "ultralytics.models.yolo.yoloe.predict",
    "ultralytics.models.yolo.yoloe.val",
]

for mod_name in search_modules:
    try:
        mod = importlib.import_module(mod_name)
        attrs = [a for a in dir(mod) if any(kw in a.lower() for kw in ["text", "pe", "embed", "clip", "prompt"])]
        if attrs:
            print(f"\n  {mod_name}: {attrs}")
            for a in attrs:
                obj = getattr(mod, a)
                if callable(obj):
                    try:
                        src = inspect.getsource(obj)
                        for line in src.split("\n")[:20]:
                            print(f"    {line}")
                    except:
                        print(f"    (cannot get source)")
    except ImportError:
        pass

print("\n" + "=" * 80)
print("3) Search predict() for how pe is used")
print("=" * 80)

# The predict method in tasks.py line 1197 calls m(x) for head
# But how does pe get passed?
for cls in type(inner).__mro__:
    if "predict" in cls.__dict__:
        print(f"\n  {cls.__name__}.predict():")
        try:
            src = inspect.getsource(cls.__dict__["predict"])
            for line in src.split("\n")[:60]:
                print(f"    {line}")
        except:
            print(f"    (cannot get source)")
        break

print("\n" + "=" * 80)
print("4) Head forward - how does it get text embeddings?")
print("=" * 80)

head = list(inner.model)[-1]
# Look at YOLOEDetect.forward
for cls in type(head).__mro__:
    if cls.__name__ in ("YOLOEDetect", "YOLOESegment"):
        if "forward" in cls.__dict__:
            print(f"\n  {cls.__name__}.forward():")
            try:
                print(inspect.getsource(cls.__dict__["forward"]))
            except:
                print("  (cannot get source)")

# Also look at the parent that has _inference with scores
for cls in type(head).__mro__:
    if cls.__name__ == "YOLOEDetect":
        for method_name in ["forward", "_inference", "_get_scores"]:
            if method_name in cls.__dict__:
                print(f"\n  {cls.__name__}.{method_name}():")
                try:
                    print(inspect.getsource(cls.__dict__[method_name]))
                except:
                    print("  (cannot get source)")

print("\nDone.")