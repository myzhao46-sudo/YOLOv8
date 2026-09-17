"""Run real data -> expert training -> new checkpoint -> frozen-base fusion -> custom evaluation."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports, sha256, save_json

setup_runtime()


def compare(reference: Path, actual: Path) -> dict:
    import numpy as np
    ref = json.loads((reference / "result.json").read_text(encoding="utf-8"))
    new = json.loads((actual / "result.json").read_text(encoding="utf-8"))
    assert ref["status"] == new["status"] == "passed"
    assert ref["base_sha256"] == new["base_sha256"]
    assert ref["expert_sha256"] == new["expert_sha256"]
    for key in ("imgsz","batch","device","ship_conf","bridge_conf","nms_iou","report_conf","max_det","fusion_nms"):
        assert ref["arguments"][key] == new["arguments"][key], key
    differences = {}
    for mode in ("ship", "bridge", "fusion"):
        rows = []
        for directory in (reference, actual):
            rows.append([json.loads(line) for line in (directory / f"{mode}_predictions.jsonl").read_text(encoding="utf-8").splitlines()])
        assert len(rows[0]) == len(rows[1])
        max_box = max_score = 0.
        candidates = 0
        for a,b in zip(*rows):
            assert (Path(a["image"]).name,a["source_split"],a["width"],a["height"]) == (Path(b["image"]).name,b["source_split"],b["width"],b["height"])
            assert len(a["predictions"]) == len(b["predictions"])
            for pa,pb in zip(a["predictions"],b["predictions"]):
                assert pa["class_id"] == pb["class_id"]
                assert pa["source_expert"] == pb["source_expert"]
                box = float(np.max(np.abs(np.array(pa["xyxy"])-np.array(pb["xyxy"]))))
                score = abs(pa["score"]-pb["score"])
                assert box <= .001 and score <= 1e-6, (mode, box, score)
                max_box,max_score = max(max_box,box),max(max_score,score)
                candidates += 1
        for split in ("test_ship","test_bridge","combined"):
            for key in ("gt_histogram","prediction_histogram_all","prediction_histogram_at_report_conf","per_class","overall"):
                assert ref["metrics"][mode][split][key] == new["metrics"][mode][split][key], (mode,split,key)
        assert ref["matches"][mode] == new["matches"][mode], mode
        differences[mode] = {"candidates":candidates,"max_box_abs_diff":max_box,"max_score_abs_diff":max_score}
    old_gt = json.loads((reference / "ground_truth.json").read_text(encoding="utf-8"))
    new_gt = json.loads((actual / "ground_truth.json").read_text(encoding="utf-8"))
    assert [(Path(r["image"]).name,r["ground_truth"]) for r in old_gt] == [(Path(r["image"]).name,r["ground_truth"]) for r in new_gt]
    return {"status":"passed","box_atol":.001,"score_atol":1e-6,"metrics_and_matches":"exact","differences":differences}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT.parent / "YOLOv8")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fixtures", type=Path, help="Existing prepare_data result directory, for the fixed migration comparison")
    parser.add_argument("--reference", type=Path, help="Captured source evaluation directory; no source code imported")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="Smoke defaults to FP32, avoiding the framework's unrelated AMP-check model download")
    args = parser.parse_args()
    out = (args.out or ROOT / "runs" / "clean_smoke" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if not out.is_relative_to(ROOT / "runs"):
        raise ValueError("Smoke output must be in this worktree's runs directory")
    out.mkdir(parents=True,exist_ok=False)
    source = args.asset_root.resolve()
    assets = {
        "ship": source / "ultralytics" / "best.pt",
        "expert": source / "runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt",
        "init": source / "yolov8m.pt",
        "text": source / "mobileclip_blt.ts",
    }
    report = {"status":"failed","output":str(out),"steps":[],"kd":"not implemented; only standard expert supervision and late fusion validated"}
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "ultralytics"),str(ROOT)]),PYTHONDONTWRITEBYTECODE="1",PYTHONHASHSEED="42",YOLO_CONFIG_DIR=str(out / "settings"),YOLO_AUTOINSTALL="false",YOLO_OFFLINE="true")
    (out / "settings").mkdir()

    def run(name, command):
        log = out / f"{name}.log"
        print(f"[{name}] {subprocess.list2cmdline(command)}",flush=True)
        step = {"name":name,"argv":command,"cwd":str(ROOT),"log":str(log)}
        report["steps"].append(step)
        save_json(out / "summary.json",report)
        with log.open("w",encoding="utf-8") as handle:
            process = subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        step["exit_code"] = process.returncode
        save_json(out / "summary.json",report)
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}; see {log}")

    try:
        for key,path in assets.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing external asset {key}: {path}")
        hashes = {key:sha256(path) for key,path in assets.items()}
        report["assets"] = {key:{"path":str(path),"sha256":hashes[key]} for key,path in assets.items()}
        # The frozen text encoder is an external read-only asset, never source code.
        text_link = ROOT / "mobileclip_blt.ts"
        if not text_link.exists():
            try:
                os.link(assets["text"],text_link)
            except OSError:
                import shutil
                shutil.copy2(assets["text"],text_link)
        if sha256(text_link) != hashes["text"]:
            raise ValueError(f"Unexpected text encoder asset: {text_link}")
        import torch, ultralytics, importlib.metadata
        report["environment"] = {"python":sys.version,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"ultralytics":ultralytics.__version__,"modules":assert_local_imports(),"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}
        from tools.clean_pipeline.data import prepare_data, validate_prepared
        if args.fixtures:
            data = json.loads((args.fixtures / "prepared.json").read_text(encoding="utf-8"))
        else:
            data = prepare_data(source,out,seed=42,train_count=12,val_count=6,eval_per_group=3)
        validate_prepared(data)
        save_json(out / "prepared.json", data)
        report["data"] = data
        evaluation = [sys.executable,"-B","tools/clean_pipeline/evaluate.py","--dataset-root",data["eval_root"],"--ship-weights",str(assets["ship"]),"--imgsz",str(args.imgsz),"--batch",str(args.batch),"--device",args.device]
        run("existing_weights",evaluation+["--bridge-weights",str(assets["expert"]),"--out",str(out / "existing_weights")])
        if args.reference:
            report["consistency"] = compare(args.reference,out / "existing_weights")
            save_json(out / "consistency.json", report["consistency"])
        else:
            report["consistency"] = {"status":"not requested","note":"Supply --reference for a source migration comparison"}
        train=[sys.executable,"-B","tools/train/train_bridge_expert_yolo.py","--model",str(assets["init"]),"--data",data["train_yaml"],"--imgsz",str(args.imgsz),"--epochs","1","--batch",str(args.batch),"--device",args.device,"--workers","0","--project",str(out / "training"),"--name","expert","--patience","0","--close-mosaic","0","--seed","42","--nbs",str(args.batch),"--verify-updates"]
        if not args.amp:
            train.append("--no-amp")
        run("train",train)
        weight = out / "training/expert/weights/last.pt"
        if not weight.is_file():
            raise FileNotFoundError(f"Training did not save expected checkpoint: {weight}")
        report["new_weight"] = {"path":str(weight),"sha256":sha256(weight)}
        run("new_weights",evaluation+["--bridge-weights",str(weight),"--out",str(out / "new_weights")])
        assert {key:sha256(path) for key,path in assets.items()} == hashes, "External assets changed"
        report.update(status="passed",external_assets_unchanged=True)
        print(f"PASS: {out / 'summary.json'}",flush=True)
        return 0
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        save_json(out / "summary.json",report)


if __name__ == "__main__":
    raise SystemExit(main())
