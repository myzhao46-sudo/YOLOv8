# tools/data/make_global4_yaml.py
from pathlib import Path
import argparse


GLOBAL4_NAMES = {
    0: "ship",
    1: "harbor",
    2: "tank",
    3: "bridge",
}


def p(path: Path) -> str:
    """Use forward slashes for Ultralytics YAML on Windows."""
    return path.resolve().as_posix()


def write_yaml(path: Path, train_paths=None, val_paths=None):
    lines = []

    if train_paths is not None:
        lines.append("train:")
        for x in train_paths:
            lines.append(f"  - {p(x)}")
        lines.append("")

    if val_paths is not None:
        lines.append("val:")
        for x in val_paths:
            lines.append(f"  - {p(x)}")
        lines.append("")

    lines.append("nc: 4")
    lines.append("")
    lines.append("names:")
    for k, v in GLOBAL4_NAMES.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="YOLOv8 repo root. Default: auto infer from this script path.",
    )
    parser.add_argument(
        "--ship-name",
        type=str,
        default="ship_small",
        help="ship dataset folder name, e.g. ship_small or ship_small_split",
    )
    parser.add_argument(
        "--sar-name",
        type=str,
        default="MASR_bridge_sar30",
        help="SAR bridge folder name. Change to MSAR_bridge_sar30 if that is your real folder.",
    )
    args = parser.parse_args()

    if args.repo_root is None:
        # tools/data/make_global4_yaml.py -> repo root is parents[2]
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path(args.repo_root).resolve()

    datasets_root = repo_root / "ultralytics" / "datasets"
    configs_root = repo_root / "configs"

    ship = datasets_root / args.ship_name

    bridge_root = datasets_root / "bridge_small"
    bridge_rgb = bridge_root / "DIOR_bridge_rgb30"
    bridge_sar = bridge_root / args.sar_name
    bridge_infr = bridge_root / "MassMIND_bridge_infr30"

    train_paths = [
        ship / "images" / "train",
        bridge_rgb / "images" / "train",
        bridge_sar / "images" / "train",
        bridge_infr / "images" / "train",
    ]

    val_paths = [
        ship / "images" / "val",
        bridge_rgb / "images" / "val",
        bridge_sar / "images" / "val",
        bridge_infr / "images" / "val",
    ]

    write_yaml(
        configs_root / "global4_train_ship_bridge.yaml",
        train_paths=train_paths,
        val_paths=val_paths,
    )

    write_yaml(
        configs_root / "global4_eval_ship.yaml",
        val_paths=[ship / "images" / "val"],
    )

    write_yaml(
        configs_root / "global4_eval_bridge_rgb.yaml",
        val_paths=[bridge_rgb / "images" / "val"],
    )

    write_yaml(
        configs_root / "global4_eval_bridge_sar.yaml",
        val_paths=[bridge_sar / "images" / "val"],
    )

    write_yaml(
        configs_root / "global4_eval_bridge_infr.yaml",
        val_paths=[bridge_infr / "images" / "val"],
    )

    write_yaml(
        configs_root / "global4_eval_bridge_all.yaml",
        val_paths=[
            bridge_rgb / "images" / "val",
            bridge_sar / "images" / "val",
            bridge_infr / "images" / "val",
        ],
    )

    print("\nDone.")
    print(f"Repo root:     {repo_root}")
    print(f"Datasets root: {datasets_root}")
    print(f"Configs root:  {configs_root}")


if __name__ == "__main__":
    main()