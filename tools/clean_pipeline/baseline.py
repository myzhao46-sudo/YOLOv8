"""Evidence collection for full baseline runs; inference/metrics stay in existing evaluators."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tools.clean_pipeline.runtime import ROOT, assert_local_imports, save_json, sha256


def environment():
    from tools.clean_pipeline.runtime import DISABLED_EDITABLE_FINDERS
    import torch
    import ultralytics
    distributions = {}
    for dist in importlib.metadata.distributions():
        direct = dist.read_text("direct_url.json")
        if direct:
            distributions[dist.metadata["Name"]] = json.loads(direct)
    return {"python":sys.version, "executable":sys.executable, "platform":platform.platform(),
            "torch":torch.__version__, "cuda":torch.version.cuda, "gpu":torch.cuda.get_device_name(0),
            "gpu_bytes":torch.cuda.get_device_properties(0).total_memory,
            "ultralytics":ultralytics.__version__, "modules":assert_local_imports(),
            "sys_path":sys.path, "direct_urls":distributions,
            "disabled_editable_finders":DISABLED_EDITABLE_FINDERS,
            "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}}


def asset_inventory(asset_root: Path, out: Path):
    """Hash actual image bytes and labels; preserve and validate all original splits."""
    import numpy as np
    from PIL import Image
    base = asset_root / "ultralytics/datasets"
    rows, manifests = [], {}
    for dataset in ("total_bridge_trainval", "total_bridge_test", "bridge_ship_distill_2cls_v2"):
        file = base / dataset / "manifest.csv"
        manifests[dataset] = {"path":str(file), "sha256":sha256(file)}
        with file.open(encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                split, modality = row["split"], row["modality"].lower()
                if dataset == "bridge_ship_distill_2cls_v2" and split not in {"test_ship","test_bridge"}:
                    continue
                evaluation = dataset == "bridge_ship_distill_2cls_v2"
                name = row["target_image" if evaluation else "image_path_new"].replace("\\", "/").rsplit("/",1)[-1]
                folder = split if evaluation else f"{split}_{modality}"
                image = base / dataset / "images" / folder / name
                label = base / dataset / "labels" / folder / Path(name).with_suffix(".txt").name
                digest = sha256(image)
                assert digest == row["hash_sha256" if evaluation else "sha256"], image
                rows.append({"dataset":dataset,"split":split,"modality":modality,"image":str(image),"label":str(label),"image_sha256":digest,"expected_class":1 if split=="test_bridge" else 0})
    external = base / "external_bridge_nwpu_cls0_from1"
    for image in sorted((external / "images/test").iterdir()):
        if image.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
            continue
        rows.append({"dataset":external.name,"split":"external","modality":"rgb","image":str(image),"label":str(external / "labels/test" / image.with_suffix(".txt").name),"image_sha256":sha256(image),"expected_class":0})
    counts, boxes, hashes = Counter(), Counter(), {}
    for row in rows:
        image, label = Path(row["image"]), Path(row["label"])
        with Image.open(image) as opened:
            row["width"], row["height"] = opened.size
            opened.verify()
        parsed = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = [float(v) for v in line.split()]
            assert len(fields)==5 and all(math.isfinite(v) for v in fields), label
            assert fields[0]==row["expected_class"] and all(0<=v<=1 for v in fields[1:]) and fields[3]>0 and fields[4]>0, label
            parsed.append(fields)
        assert parsed, label
        row.update(label_sha256=sha256(label),boxes=len(parsed),labels=parsed)
        key=f"{row['split']}/{row['modality']}"
        counts[key]+=1; boxes[key]+=len(parsed)
        hashes.setdefault(row["split"],set()).add(row["image_sha256"])
    for folder in {Path(row["image"]).parent for row in rows}:
        listed={Path(row["image"]) for row in rows if Path(row["image"]).parent==folder}
        actual={p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}}
        assert listed==actual, f"Manifest/directory mismatch: {folder}"
    assert len(hashes["train"])==2481 and len(hashes["val"])==532 and len(hashes["test"])==532
    assert not (hashes["train"] & hashes["val"] or hashes["train"] & hashes["test"] or hashes["val"] & hashes["test"])
    assert hashes["test"]==hashes["test_bridge"]
    assert not hashes["test_ship"] & hashes["test_bridge"]
    assert len(hashes["test_ship"])==300 and len(hashes["external"])==67
    assert not hashes["external"] & (hashes["train"]|hashes["val"]|hashes["test"])
    by_hash={r["image_sha256"]:r for r in rows if r["split"]=="test"}
    for row in [r for r in rows if r["split"]=="test_bridge"]:
        a=np.asarray(row["labels"]); b=np.asarray(by_hash[row["image_sha256"]]["labels"])
        assert a.shape==b.shape and np.array_equal(a[:,1:],b[:,1:]), row["label"]
    totals={split:{"images":sum(r["split"]==split for r in rows),"boxes":sum(r["boxes"] for r in rows if r["split"]==split)} for split in hashes}
    assert totals["test_ship"]=={"images":300,"boxes":694}
    assert totals["test"]==totals["test_bridge"]=={"images":532,"boxes":798}
    assert totals["external"]=={"images":67,"boxes":124}
    master={"images":sum(totals[s]["images"] for s in ("train","val","test")),"boxes":sum(totals[s]["boxes"] for s in ("train","val","test"))}
    assert master=={"images":3545,"boxes":5152},master
    result={"status":"passed","master":master,"splits":totals,"images_by_modality":dict(counts),"boxes_by_modality":dict(boxes),"source_manifests":manifests,
            "hash_scope":"All selected full train/val/test/combined/external image bytes and labels rehashed; fixed source manifests checked",
            "overlap":"train/val/test disjoint; test_bridge equals held-out test by image bytes and remapped labels; ship/bridge disjoint; NWPU disjoint from bridge splits","records":rows}
    save_json(out / "input_manifest.json",result)
    return {k:v for k,v in result.items() if k!="records"}


def replay(directory: Path):
    """Re-evaluate existing per-expert post-NMS candidates with unchanged fusion functions."""
    from tools.eval import eval_late_fusion_ship_bridge as fusion
    result=json.loads((directory / "result.json").read_text(encoding="utf-8"))
    records=json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))
    predictions={}
    for mode in ("ship","bridge"):
        predictions[mode]={r["image"]:r["predictions"] for r in [json.loads(line) for line in (directory / f"{mode}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]}
    args=SimpleNamespace(**result["arguments"])
    output={"protocol":result["cache_protocol"],"candidate_stage":"post per-expert NMS; before cross-expert fusion NMS", "weight_sha256":{"ship":result["base_sha256"],"bridge":result["expert_sha256"]},"modes":{}}
    groups={"test_ship":[r for r in records if r["source_split"]=="test_ship"],"test_bridge":[r for r in records if r["source_split"]=="test_bridge"],"combined":records}
    for split in ("test_ship","test_bridge"):
        for modality in ("rgb","sar","ir"):
            groups[f"{split}/{modality}"]=[r for r in groups[split] if f"_{modality}_" in Path(r["image"]).name]
    for nms_mode in ("class-aware","class-agnostic"):
        args.fusion_nms=nms_mode
        fused=fusion.fuse_predictions(records,predictions["ship"],predictions["bridge"],args)
        evaluated={name:fusion.evaluate_subset(name,selected,fused,args.report_conf) for name,selected in groups.items()}
        output["modes"][nms_mode]=evaluated
        fusion.write_jsonl(directory / f"fusion_{nms_mode}_replayed.jsonl",[fusion.prediction_record(r,fused[r["image_key"]]) for r in records])
    save_json(directory / "replayed_metrics.json",output)
    with (directory / "replayed_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["mode","split","class","ap50","map50_95","precision","recall","tp","fp","fn"])
        for mode,splits in output["modes"].items():
            for split,metrics in splits.items():
                for cls,m in {"overall":metrics["overall"],**metrics["per_class"]}.items():
                    writer.writerow([mode,split,cls,*[m[k] for k in ("ap50","map50_95","precision","recall","tp","fp","fn")]])
    return output


def audit_exif_fix(before: Path, after: Path):
    """Require unchanged ship/all unaffected expert predictions, with fixed tolerances."""
    import numpy as np
    from PIL import Image
    old=json.loads((before/"result.json").read_text(encoding="utf-8"))
    new=json.loads((after/"result.json").read_text(encoding="utf-8"))
    assert old["label_fingerprint"]==new["label_fingerprint"] and old["cache_protocol"]==new["cache_protocol"]
    assert old["base_sha256"]==new["base_sha256"] and old["expert_sha256"]==new["expert_sha256"]
    assert new["geometry_valid"] and not new["diagnostic_legacy_mode"]
    images=new["image_ids"]; tagged={}; allowed=set()
    for index,path in enumerate(images):
        with Image.open(path) as opened:
            orientation=opened.getexif().get(274,1)
            if orientation!=1:
                tagged[path]=orientation
                # Rotated dimensions can change rectangular padding for the batch.
                start=index//new["arguments"]["batch"]*new["arguments"]["batch"]
                allowed.update(images[start:start+new["arguments"]["batch"]])
    changes={}
    for mode in ("ship","bridge","fusion"):
        left=[json.loads(line) for line in (before/f"{mode}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        right=[json.loads(line) for line in (after/f"{mode}_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["image"] for r in left]==[r["image"] for r in right]==images
        changed=[]
        for a,b in zip(left,right):
            ap,bp=a["predictions"],b["predictions"]
            equal=len(ap)==len(bp)
            if equal:
                equal=all(x["class_id"]==y["class_id"] and abs(x["score"]-y["score"])<=1e-6 and np.allclose(x["xyxy"],y["xyxy"],atol=.001,rtol=0) for x,y in zip(ap,bp))
            if not equal:
                assert mode!="ship" and a["image"] in allowed, f"Unexpected prediction change: {mode}: {a['image']}"
                changed.append({"image":a["image"],"before_candidates":len(ap),"after_candidates":len(bp)})
        changes[mode]=changed
    result={"status":"passed","before":str(before),"after":str(after),"tagged_images":tagged,"allowed_batch_images":sorted(allowed),"changed_images":changes,
            "unaffected_predictions":"same counts, classes, order; boxes atol=.001px, scores atol=1e-6", "old_geometry_valid":old["geometry_valid"],"new_geometry_valid":new["geometry_valid"],
            "before_metrics":old["metrics"],"after_metrics":new["metrics"]}
    save_json(after/"exif_fix_audit.json",result)
    return {k:v for k,v in result.items() if k not in ("before_metrics","after_metrics")}
