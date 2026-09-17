"""Pin project imports and writable runtime state to this checkout."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def setup_runtime() -> Path:
    for path in (ROOT, ROOT / "ultralytics"):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    settings = Path(os.environ.get("YOLO_CONFIG_DIR", ROOT / "runs" / "settings"))
    settings.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings.resolve())
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return ROOT


def assert_local_imports() -> dict[str, str]:
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name == "ultralytics" or name.startswith(("ultralytics.", "tools.")):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(ROOT):
                    raise RuntimeError(f"Foreign project import: {name}: {path}; expected {ROOT}")
                origins[name] = str(path)
    for name in ("ultralytics", "ultralytics.nn.tasks", "tools.eval.eval_late_fusion_ship_bridge"):
        if name in origins:
            print(f"[local import] {name}: {origins[name]}", flush=True)
    return origins


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
