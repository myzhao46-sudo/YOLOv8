"""Recommended frozen YOLOE box-only CLI.

Uses the existing custom step-integral AP evaluator, with historical defaults
CPU, 640px, candidate confidence .001, NMS IoU .7, and GT match IoU .5.
The original ship/harbor/tank prompts are initialized with real embeddings;
the fused segmentation head and checkpoint stay unchanged.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.clean_pipeline.runtime import setup_runtime

setup_runtime()

from tools.eval.eval_teacher_ship_external import main


if __name__ == "__main__":
    raise SystemExit(main(default_data=PROJECT_ROOT / "configs" / "datasets" / "ship_val.yaml"))
