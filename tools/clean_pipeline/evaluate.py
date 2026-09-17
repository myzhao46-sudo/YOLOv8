"""Acceptance adapter using the existing fusion inference and custom metrics unchanged.

The identical adapter is copied to the detached source worktree for the migration
reference. Every project import must then originate in that worktree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.clean_pipeline.runtime import setup_runtime, assert_local_imports, save_json, sha256

setup_runtime()
import numpy as np
import torch
from tools.eval import eval_late_fusion_ship_bridge as fusion


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ship-weights", type=Path, required=True)
    parser.add_argument("--bridge-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--ship-conf", type=float, default=.001)
    parser.add_argument("--bridge-conf", type=float, default=.001)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--report-conf", type=float, default=.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--fusion-nms", choices=["class-aware", "class-agnostic"], default="class-aware")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    np.random.seed(42)
    result = {"status": "failed", "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    try:
        records = []
        stats = {}
        for split, cls in [("test_ship", 0), ("test_bridge", 1)]:
            images = fusion.list_images(args.dataset_root / "images" / split)
            rows, stat = fusion.read_ground_truth(images, split, cls, fusion.load_manifest_modalities(args.dataset_root))
            if stat["missing_labels"] or stat["bad_rows"] or stat["empty_labels"]:
                raise ValueError(f"Invalid acceptance labels: {stat}")
            records.extend(rows)
            stats[split] = stat
        before_file = sha256(args.ship_weights)
        ship_wrapper, ship_inner, ship_meta = fusion.prepare_ship_expert(args.ship_weights)
        names = fusion.normalize_names(ship_inner.names)
        assert names == {0: "ship", 1: "harbor", 2: "tank"}, names
        assert ship_inner.model[-1].nc == 3
        assert ship_meta["text_embedding_shape"] == [1, 3, 512]
        before_params = state_hash(ship_inner)
        ship, ship_counts = fusion.infer_ship(ship_inner, records, args, fusion.parse_device(args.device))
        assert state_hash(ship_inner) == before_params, "Frozen base parameters changed"
        assert all(not p.requires_grad for p in ship_inner.parameters())
        del ship_wrapper, ship_inner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bridge_wrapper, _, bridge_meta = fusion.prepare_bridge_expert(args.bridge_weights)
        bridge, bridge_counts = fusion.infer_bridge(bridge_wrapper, records, args)
        fused = fusion.fuse_predictions(records, ship, bridge, args)
        # Both experts must exercise nonempty inference; low smoke AP is allowed.
        assert sum(map(len, ship.values())) > 0, "Base produced no candidates"
        assert sum(map(len, bridge.values())) > 0, "Expert produced no candidates; inspect threshold/training"
        outputs = {"ship": ship, "bridge": bridge, "fusion": fused}
        metrics = {}
        matches = {}
        for mode, predictions in outputs.items():
            for record in records:
                for box in predictions[record["image_key"]]:
                    values = np.asarray(box["xyxy"] + [box["score"]], dtype=float)
                    assert np.isfinite(values).all()
                    x1,y1,x2,y2,score = values
                    assert 0 <= score <= 1 and 0 <= x1 <= x2 <= record["width"] + .001
                    assert 0 <= y1 <= y2 <= record["height"] + .001
                    assert box["class_id"] in (0,1)
            fusion.write_jsonl(args.out / f"{mode}_predictions.jsonl", [fusion.prediction_record(r,predictions[r["image_key"]]) for r in records])
            metrics[mode] = {}
            for split in ("test_ship", "test_bridge", "combined"):
                selected = records if split == "combined" else [r for r in records if r["source_split"] == split]
                metrics[mode][split] = fusion.evaluate_subset(split, selected, predictions, args.report_conf)
            matches[mode] = {}
            for cls in (0,1):
                match = fusion.match_class(records, predictions, cls, .5, minimum_score=args.report_conf)
                matches[mode][str(cls)] = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in match.items()}
        result.update(status="passed", dataset=stats, image_ids=[r["image_key"] for r in records],
                      ship=ship_meta, bridge=bridge_meta, ship_inference=ship_counts, bridge_inference=bridge_counts,
                      metrics=metrics, matches=matches, modules=assert_local_imports(),
                      base_sha256=before_file, base_parameters_sha256=before_params,
                      base_file_unchanged=sha256(args.ship_weights)==before_file,
                      expert_sha256=sha256(args.bridge_weights))
        assert result["base_file_unchanged"]
        save_json(args.out / "metrics.json", metrics)
        save_json(args.out / "matches.json", matches)
        save_json(args.out / "ground_truth.json", records)
        with (args.out / "metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["mode","split","ap50","map50_95","precision","recall","tp","fp","fn"])
            writer.writeheader()
            for mode, splits in metrics.items():
                for split, m in splits.items():
                    writer.writerow({"mode":mode,"split":split,**{k:m["overall"][k] for k in writer.fieldnames[2:]}})
        print({mode: metrics[mode]["combined"]["overall"] for mode in metrics},flush=True)
    except Exception as exc:
        result.update(error=repr(exc))
        raise
    finally:
        save_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
