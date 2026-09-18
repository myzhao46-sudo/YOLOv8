"""Full historical-weight verification and a fresh 150-epoch supervised expert baseline."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.clean_pipeline.runtime import setup_runtime, save_json, sha256
setup_runtime()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root",type=Path,default=ROOT.parent/"YOLOv8")
    parser.add_argument("--source-root",type=Path,default=ROOT.parent/"YOLOv8_source_736cb86")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--reference-only",action="store_true",help="Record source/clean pre-fix equality, including explicitly audited legacy EXIF defects; never train")
    parser.add_argument("--legacy-reference",type=Path,help="Completed pre-fix full source/clean reference, used to audit the EXIF correction")
    parser.add_argument("--resume-run",type=Path,help="Continue an interrupted formal run from its own resumable last.pt")
    args=parser.parse_args()
    if not args.reference_only and args.legacy_reference is None and args.resume_run is None:
        parser.error("Formal runs require --legacy-reference pointing to the recorded pre-fix full comparison")
    out=(args.resume_run or args.out or ROOT/"runs/baseline_verification"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    assert out.is_relative_to(ROOT/"runs")
    if args.resume_run and (args.out or args.reference_only):
        parser.error("--resume-run cannot be combined with --out or --reference-only")
    if not args.resume_run:
        out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    state={"status":"running","started_at":datetime.datetime.now().isoformat(),"output":str(out),"steps":[],
           "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
           "tolerance":{"box_abs":.001,"score_abs":1e-6,"metric_abs":1e-8,"historical_4_decimal_rounding":.00005},
           "training":{"epochs":150,"batch":2,"workers":0,"imgsz":1024,"amp":False,"seed":42,"patience":40,"close_mosaic":20,"nbs":64,
                       "adjustment":"8GB VRAM and 16GB host RAM: batch16/workers8/AMP replaced by proven batch2/workers0/FP32; full data and original epoch/early-stop policy unchanged"}}
    resume_archive=None
    if args.resume_run:
        current_commit=state["code_commit"]
        state=json.loads((out/"state.json").read_text(encoding="utf-8"))
        if state["status"]=="passed" or state.get("phase_one")!="passed":
            raise ValueError("Resume requires an unfinished run with completed full baseline verification")
        # Refuse to launch another trainer while the previously recorded child lives.
        for step in state["steps"]:
            if step.get("pid") and "exit_code" not in step and step.get("status")!="interrupted":
                if os.name=="nt":
                    import ctypes
                    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,step["pid"])
                    if handle:
                        ctypes.windll.kernel32.CloseHandle(handle)
                        raise RuntimeError(f"Recorded process {step['pid']} still exists; do not duplicate it")
                else:
                    try: os.kill(step["pid"],0)
                    except ProcessLookupError: pass
                    else: raise RuntimeError(f"Recorded process {step['pid']} still exists")
        resume_archive=out/"resume_archive"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_archive.mkdir(parents=True,exist_ok=False)
        shutil.copy2(out/"state.json",resume_archive/"state_before_resume.json")
        for name in ("last.pt","best.pt"):
            shutil.copy2(out/"training/expert/weights"/name,resume_archive/name)
        shutil.copy2(out/"training/expert/training_progress.json",resume_archive/"training_progress.json")
        for step in state["steps"]:
            if "exit_code" not in step and step.get("status")!="interrupted":
                step.update(status="interrupted",exit_code=None,note="Process absent on continuation; no normal exit status was available")
        state.setdefault("resume_sessions",[]).append({"started_at":datetime.datetime.now().isoformat(),"code_commit":current_commit,"archive":str(resume_archive),"last_sha256":sha256(resume_archive/"last.pt")})
        state.update(status="running")
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONHASHSEED="42",YOLO_CONFIG_DIR=str(out/"settings"),YOLO_AUTOINSTALL="false",YOLO_OFFLINE="true",PYTHONUTF8="1")
    (out/"settings").mkdir(exist_ok=True)

    def run(name, arguments, cwd=ROOT):
        command=[sys.executable,"-B",*map(str,arguments)]
        step={"name":name,"argv":command,"cwd":str(cwd),"log":str(out/f"{name}.log"),"started_at":datetime.datetime.now().isoformat()}
        state["steps"].append(step);save_json(out/"state.json",state)
        child_env=dict(env,PYTHONPATH=os.pathsep.join([str(cwd/"ultralytics"),str(cwd)]))
        with (out/f"{name}.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,cwd=cwd,env=child_env,stdout=log,stderr=subprocess.STDOUT)
            step["pid"]=process.pid;save_json(out/"state.json",state)
            print(f"START {name} PID={process.pid}",flush=True)
            step["exit_code"]=process.wait()
        step["finished_at"]=datetime.datetime.now().isoformat();save_json(out/"state.json",state)
        print(f"END {name} exit={step['exit_code']}",flush=True)
        if step["exit_code"]:
            raise RuntimeError(f"{name} failed: {step['log']}")

    try:
        from tools.clean_pipeline.baseline import asset_inventory,environment,replay
        observed_environment=environment()
        dataset=asset_inventory(args.asset_root,resume_archive or out)
        if args.resume_run:
            assert dataset==state["dataset"], "Dataset changed since the interrupted run"
            state["resume_sessions"][-1]["environment"]=observed_environment
        else:
            state["environment"]=observed_environment
            state["dataset"]=dataset
        asset={"ship":args.asset_root/"ultralytics/best.pt","expert":args.asset_root/"runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt","init":args.asset_root/"yolov8m.pt","text":args.asset_root/"mobileclip_blt.ts"}
        observed_assets={k:{"path":str(v),"sha256":sha256(v)} for k,v in asset.items()}
        if args.resume_run:
            assert observed_assets==state["assets"], "External weights changed since the interrupted run"
        else:
            state["assets"]=observed_assets
        if args.reference_only:
            source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.source_root,text=True).strip()
            assert source_sha=="736cb86c865fa490e9ae77e3189b8098825897fd",source_sha
            assert not subprocess.check_output(["git","diff","--name-only"],cwd=args.source_root,text=True).strip()
            state["source_commit"]=source_sha
            source_tools=args.source_root/"tools/clean_pipeline";source_tools.mkdir(parents=True,exist_ok=True)
            for name in ("evaluate.py","runtime.py"):
                shutil.copy2(ROOT/"tools/clean_pipeline"/name,source_tools/name)
        for checkout in ((ROOT,args.source_root) if args.reference_only else (ROOT,)):
            text_asset=checkout/"mobileclip_blt.ts"
            if not text_asset.exists():shutil.copy2(asset["text"],text_asset)
            assert sha256(text_asset)==state["assets"]["text"]["sha256"]
        common=["tools/clean_pipeline/evaluate.py","--dataset-root",args.asset_root/"ultralytics/datasets/bridge_ship_distill_2cls_v2","--ship-weights",asset["ship"],"--imgsz","1024","--batch","2","--device","0"]
        if not args.resume_run:
            if args.reference_only:
                common.append("--legacy-coordinate-audit")
                run("source_full",common+["--bridge-weights",asset["expert"],"--out",out/"source_full"],args.source_root)
            run("historical_full",common+["--bridge-weights",asset["expert"],"--out",out/"historical_full"])
            if args.reference_only:
                from tools.run_clean_smoke import compare
                state["same_weight_consistency"]=compare(out/"source_full",out/"historical_full")
                save_json(out/"same_weight_consistency.json",state["same_weight_consistency"])
                state.update(status="reference_recorded",note="Historical equality only; legacy geometry defect remains, no training authorized by this mode")
                return
            from tools.clean_pipeline.baseline import audit_exif_fix
            reference=args.legacy_reference.resolve()
            previous=json.loads((reference/"state.json").read_text(encoding="utf-8"))
            assert previous["status"]=="reference_recorded" and previous["same_weight_consistency"]["status"]=="passed"
            assert previous["assets"]==state["assets"] and previous["dataset"]==state["dataset"]
            state["legacy_reference"]={"path":str(reference),"state_sha256":sha256(reference/"state.json"),"same_weight_consistency":previous["same_weight_consistency"]}
            state["exif_fix_audit"]=audit_exif_fix(reference/"historical_full",out/"historical_full")
            replay(out/"historical_full")
        def external(name,weight,data):
            run(name,["tools/eval/eval_bridge_expert_external_yolov8.py","--weights",weight,"--data",data,"--imgsz","1024","--batch","8","--device","0","--conf","0.25","--iou","0.7","--project",out,"--name",name,"--save-json","--save-md","--save-preds","--max-vis","0","--max-gt-preview","0"])
        if not args.resume_run:
            external("historical_bridge_official",asset["expert"],ROOT/"configs/datasets/total_bridge_test.yaml")
            external("historical_nwpu",asset["expert"],ROOT/"configs/datasets/bridge_external_nwpu.yaml")
            state["phase_one"]="passed";save_json(out/"state.json",state)
        train_arguments=["tools/train/train_bridge_expert_yolo.py","--model",asset["init"],"--data",ROOT/"configs/datasets/total_bridge_trainval.yaml","--imgsz","1024","--epochs","150","--batch","2","--device","0","--workers","0","--project",out/"training","--name","expert","--patience","40","--close-mosaic","20","--seed","42","--nbs","64","--no-amp","--verify-updates"]
        if args.resume_run:
            historical=json.loads((out/"historical_full/result.json").read_text(encoding="utf-8"))
            assert historical["status"]=="passed" and historical["geometry_valid"]
            assert all(sha256(ROOT/name)==digest for name,digest in historical["inference_code"].items()), "Inference changed since phase one"
            train_arguments += ["--resume-from",out/"training/expert/weights/last.pt","--exist-ok"]
        run("train_resume_"+str(len(state["resume_sessions"])) if args.resume_run else "train",train_arguments)
        new=out/"training/expert/weights/best.pt"
        assert new.is_file()
        state["new_weight"]={"path":str(new),"sha256":sha256(new)}
        run("new_full",common+["--bridge-weights",new,"--ship-cache",out/"historical_full","--out",out/"new_full"])
        replay(out/"new_full")
        external("new_bridge_official",new,ROOT/"configs/datasets/total_bridge_test.yaml")
        external("new_nwpu",new,ROOT/"configs/datasets/bridge_external_nwpu.yaml")
        assert {k:sha256(v) for k,v in asset.items()}=={k:v["sha256"] for k,v in state["assets"].items()}
        state.update(status="passed",external_weights_unchanged=True)
    except BaseException as exc:
        state.update(status="failed",error=repr(exc))
        raise
    finally:
        state.update(updated_at=datetime.datetime.now().isoformat(),elapsed_seconds=time.time()-started)
        save_json(out/"state.json",state)


if __name__=="__main__":
    main()
