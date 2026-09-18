from __future__ import annotations

import sys
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))
sys.path.insert(1, str(PROJECT_ROOT))

DEFAULT_PROJECT = "runs/bridge_expert"
DEFAULT_NAME = "yolov8m_total_bridge_split_img1024_ep150"
DEFAULT_DATA = "configs/datasets/total_bridge_trainval.yaml"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_runtime_data_yaml(data_yaml: Path, project_dir: Path) -> Path:
    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {data_yaml}")
    source_path = data.get("path")
    data["path"] = str(resolve_path(source_path).resolve()) if source_path else str(data_yaml.parent.resolve())
    if not data.get("train") or not data.get("val"):
        raise ValueError(f"Dataset YAML must define the existing train and val splits: {data_yaml}")
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): value for key, value in names.items()}
    if names != {0: "bridge"} or int(data.get("nc", 1)) != 1:
        raise ValueError(f"Bridge expert requires nc=1 and names={{0: 'bridge'}}, got {names}")
    project_dir.mkdir(parents=True, exist_ok=True)
    # All new caches belong to this checkout, including formal runs on external data.
    if not Path(data["path"]).is_relative_to(PROJECT_ROOT):
        from tools.clean_pipeline.data import isolate_training_data
        data = isolate_training_data(data, project_dir / "dataset")
    runtime_yaml = project_dir / "runtime_total_bridge_trainval.yaml"
    runtime_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_yaml


class TrainingUpdateAudit:
    """Observe the original trainer and optimizer without changing their computations."""

    def __init__(self, resume_progress=None):
        self.result = {
            "optimizer_steps": 0,
            "batches": 0,
            "finite_loss": True,
            "finite_gradients": True,
            "parameter_in_optimizer": False,
            "parameter_max_abs_diff": 0.0,
            "gradient_max_abs": 0.0,
            "losses": [],
            "passed": False,
        }
        self.parameter = None
        self.before = None
        self.handles = []
        self.started = time.time()
        self.epoch_records = []
        self.resume_progress = resume_progress
        if resume_progress:
            self.epoch_records = resume_progress["epochs"]
            self.result["prior_optimizer_steps"] = resume_progress["latest"]["optimizer_steps"]
            self.result["optimizer_steps"] = self.result["prior_optimizer_steps"]
            self.result["loss_scope"] = "This resumed session only; earlier epoch means retained in training_progress.json"

    def on_train_start(self, trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        if self.resume_progress:
            previous = self.resume_progress["latest"]
            if int(trainer.start_epoch) != previous["epoch"]:
                raise RuntimeError("Checkpoint and saved progress epochs disagree")
            if trainer.ema.updates != previous["optimizer_steps"]:
                raise RuntimeError("Restored EMA update count and optimizer audit disagree")
            # The framework restores optimizer/EMA/fitness, but not the stopper.
            # Restore the observed original validation history, without changing its rule.
            trainer.stopper.best_epoch = previous["best_epoch"]
            trainer.stopper.best_fitness = previous["best_fitness"]

        candidates = ((name, value) for name, value in unwrap_model(trainer.model).named_parameters()
                      if value.requires_grad)
        name, self.parameter = next(candidates)
        self.before = self.parameter.detach().float().cpu().clone()
        self.result["parameter_name"] = name
        self.result["parameter_shape"] = list(self.parameter.shape)
        self.result["parameter_in_optimizer"] = any(
            value is self.parameter for group in trainer.optimizer.param_groups for value in group["params"]
        )
        if not self.result["parameter_in_optimizer"]:
            raise RuntimeError(f"Expected trainable parameter is missing from optimizer: {name}")
        # Optimizer hooks count actual steps, including AMP's decision to skip a step.
        self.handles = [
            trainer.optimizer.register_step_pre_hook(self.before_optimizer_step),
            trainer.optimizer.register_step_post_hook(self.after_optimizer_step),
        ]

    def before_optimizer_step(self, optimizer, args, kwargs):
        import torch

        gradient = self.parameter.grad
        if gradient is None:
            raise RuntimeError("Audited trainable parameter did not receive a gradient")
        finite = bool(torch.isfinite(gradient).all().item())
        self.result["finite_gradients"] &= finite
        if not finite:
            raise FloatingPointError("Non-finite gradient in the original training optimizer")
        self.result["gradient_max_abs"] = max(self.result["gradient_max_abs"], float(gradient.abs().max().item()))

    def after_optimizer_step(self, optimizer, args, kwargs):
        self.result["optimizer_steps"] += 1

    def on_train_batch_end(self, trainer):
        values = trainer.loss_items.detach().float().cpu().reshape(-1).tolist()
        total = float(trainer.loss.detach().float().cpu().item())
        finite = all(math.isfinite(value) for value in [total, *values])
        self.result["finite_loss"] &= finite
        self.result["batches"] += 1
        self.result["losses"].append({
            "epoch": int(trainer.epoch), "total": total,
            **dict(zip(trainer.loss_names, values)),
        })
        if not finite:
            raise FloatingPointError("Non-finite loss in the original bridge expert trainer")

    def on_fit_epoch_end(self, trainer):
        """Persist observation only; checkpoint choice remains the original trainer's."""
        row = {
            "epoch": int(trainer.epoch) + 1,
            "planned_epochs": int(trainer.epochs),
            "best_epoch": int(trainer.stopper.best_epoch),
            "fitness": float(trainer.fitness),
            "best_fitness": float(trainer.best_fitness),
            "elapsed_seconds": time.time() - self.started,
            "session_elapsed_seconds": time.time() - self.started,
            "optimizer_steps": self.result["optimizer_steps"],
            "metrics": {str(k): float(v) for k,v in trainer.metrics.items()},
            "train_loss": dict(zip(trainer.loss_names, trainer.tloss.detach().float().cpu().tolist())),
            "stop": bool(trainer.stop),
        }
        if self.resume_progress:
            row["elapsed_seconds"] += self.resume_progress["latest"]["elapsed_seconds"]
        if self.epoch_records and self.epoch_records[-1]["epoch"] == row["epoch"]:
            self.epoch_records[-1] = row
        else:
            self.epoch_records.append(row)
        target = Path(trainer.save_dir) / "training_progress.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"latest": row, "epochs": self.epoch_records},indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(target)

    def finish(self):
        self.result["session_optimizer_steps"] = self.result["optimizer_steps"] - self.result.get("prior_optimizer_steps", 0)
        if self.parameter is not None:
            self.result["parameter_max_abs_diff"] = float(
                (self.parameter.detach().float().cpu() - self.before).abs().max().item()
            )
        rows = self.result["losses"]
        if rows:
            self.result["mean_loss"] = {
                key: sum(row[key] for row in rows) / len(rows)
                for key in rows[0] if key != "epoch"
            }
        self.result["passed"] = bool(
            self.result["batches"] > 0 and self.result["optimizer_steps"] > 0
            and self.result["finite_loss"] and self.result["finite_gradients"]
            and self.result["parameter_in_optimizer"]
            and self.result["parameter_max_abs_diff"] > 0 and self.result["gradient_max_abs"] > 0
        )
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return self.result


def write_md(path: Path, summary: dict) -> None:
    lines = ["# Bridge Expert Train Summary", ""]
    for key in [
        "model_base",
        "data_yaml",
        "trainval_root",
        "runtime_data_yaml",
        "imgsz",
        "epochs",
        "batch",
        "device",
        "seed",
        "best_checkpoint",
        "last_checkpoint",
        "start_time",
        "end_time",
        "split_summary_path",
        "completed",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ordinary YOLOv8 bridge-only expert on total_bridge_trainval.")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true", default=False)
    parser.add_argument("--verify-updates", action="store_true", help="Assert finite loss, real optimizer steps and parameter updates.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size; use 2 for batch-2 one-epoch smoke.")
    parser.add_argument("--resume-from", type=Path, help="Resume this same interrupted run's last.pt, including optimizer/EMA; requires --exist-ok")
    args = parser.parse_args()

    start = datetime.now().isoformat()
    data_yaml = resolve_path(args.data).resolve()
    project = resolve_path(args.project).resolve()
    run_dir = project / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}; choose a new --name")
    if args.epochs < 1 or args.nbs < 1:
        raise ValueError("epochs and nbs must be positive")
    summary = {
        "purpose": "Bridge-only expert training for late fusion. This is not YOLOE and not a 2-class student.",
        "model_base": args.model,
        "data_yaml_input": args.data,
        "data_yaml": str(data_yaml.resolve()),
        "completed": False,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "amp": args.amp,
        "nbs": args.nbs,
        "close_mosaic": args.close_mosaic,
        "seed": args.seed,
        "project_input": args.project,
        "project": str(project.resolve()),
        "name": args.name,
        "start_time": start,
    }
    resume_progress = None
    if args.resume_from:
        if not args.exist_ok or args.resume_from.resolve() != (run_dir / "weights/last.pt").resolve():
            raise ValueError("Resume must explicitly target this run's last.pt with --exist-ok")
        resume_progress = json.loads((run_dir / "training_progress.json").read_text(encoding="utf-8"))
        summary["resume_from"] = str(args.resume_from.resolve())
        summary["previous_completed_epochs"] = resume_progress["latest"]["epoch"]
    audit = TrainingUpdateAudit(resume_progress) if args.verify_updates else None
    model = None
    try:
        from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports

        setup_runtime()
        from ultralytics import YOLO
        import yaml

        assert_local_imports()
        # Runtime YAML stays isolated per run, while keeping the original path conversion.
        if args.resume_from:
            model = YOLO(str(args.resume_from.resolve()))
            if model.ckpt.get("optimizer") is None or model.ckpt.get("epoch", -1) < 0:
                raise ValueError("Interrupted checkpoint has no resumable optimizer/epoch state")
            runtime_data_yaml = Path(model.ckpt["train_args"]["data"]).resolve()
            if not runtime_data_yaml.is_relative_to(project) or not runtime_data_yaml.is_file():
                raise ValueError("Resume runtime YAML must belong to this run's project")
        else:
            runtime_data_yaml = make_runtime_data_yaml(data_yaml, project / "runtime" / args.name)
        actual_data = yaml.safe_load(runtime_data_yaml.read_text(encoding="utf-8"))
        trainval_root = Path(actual_data["path"])
        summary.update({
            "runtime_data_yaml": str(runtime_data_yaml),
            "trainval_root": str(trainval_root),
            "train_split": actual_data["train"],
            "val_split": actual_data["val"],
            "names": actual_data["names"],
            "split_summary_path": str(trainval_root / "split_summary.json"),
        })
        if model is None:
            model = YOLO(args.model)
        if audit:
            model.add_callback("on_train_start", audit.on_train_start)
            model.add_callback("on_train_batch_end", audit.on_train_batch_end)
            model.add_callback("on_fit_epoch_end", audit.on_fit_epoch_end)
        results = model.train(
            data=str(runtime_data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=args.name,
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            pretrained=True,
            seed=args.seed,
            exist_ok=args.exist_ok,
            amp=args.amp,
            nbs=args.nbs,
            resume=str(args.resume_from.resolve()) if args.resume_from else False,
        )
        run_dir = Path(model.trainer.save_dir).resolve()
        if audit and not audit.finish()["passed"]:
            raise RuntimeError(f"Training update audit failed: {audit.result}")
        if not (run_dir / "weights" / "last.pt").is_file():
            raise RuntimeError(f"Training did not save last.pt under {run_dir}")
        assert_local_imports()
        summary["train_return"] = str(results)
        summary["completed"] = True
        summary["actual_epochs"] = int(model.trainer.epoch) + 1
        summary["best_epoch"] = int(model.trainer.stopper.best_epoch)
        summary["termination_reason"] = "early_stopping" if summary["actual_epochs"] < args.epochs else "epoch_limit"
    except Exception as exc:
        summary["completed"] = False
        summary["error"] = repr(exc)
        raise
    finally:
        trainer = getattr(model, "trainer", None)
        if trainer is not None and getattr(trainer, "save_dir", None):
            run_dir = Path(trainer.save_dir).resolve()
        end = datetime.now().isoformat()
        summary["end_time"] = end
        summary["run_dir"] = str(run_dir)
        summary["best_checkpoint"] = str(run_dir / "weights" / "best.pt")
        summary["last_checkpoint"] = str(run_dir / "weights" / "last.pt")
        run_dir.mkdir(parents=True, exist_ok=True)
        if audit:
            summary["update_audit"] = audit.finish()
            (run_dir / "training_update_audit.json").write_text(
                json.dumps(audit.result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
        (run_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_md(run_dir / "train_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
