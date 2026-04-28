#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


PHASE3_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PHASE3_ROOT / "src" / "phase3_bridge"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from expected_gait import DEFAULT_SEGMENT_OFFSETS_MS, render_expected_gait_video


def _default_mapping_csv(phase3_root: Path) -> Path:
    return phase3_root / "data" / "inputs" / "mappings" / "mn_to_actuator_mapping_rebuilt.csv"


def _default_out_dir(phase3_root: Path, save_tag: str, run_name: str) -> Path:
    return phase3_root / "data" / "derived" / save_tag / run_name / "expected_gait"


def _default_run_dir(workspace_root: Path, hemi_tag: str, run_name: str) -> Path:
    return workspace_root / "Hemilineage Simulations" / hemi_tag / "runs" / run_name


def _default_added_ids_csv(workspace_root: Path, hemi_tag: str) -> Path:
    return workspace_root / "Hemilineage Simulations" / hemi_tag / "metadata" / "added_motor_neuron_ids.csv"


def main() -> None:
    workspace_root = PHASE3_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Render a synthetic expected-gait comparison video using motor-neuron action labels."
    )
    parser.add_argument("--hemi-tag", default="Hemi_09A")
    parser.add_argument("--run-name", default="hemi_09a_baseline")
    parser.add_argument("--save-tag", default=None)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--added-motor-ids-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--profile-name", default="expected_gait_compare")
    parser.add_argument("--focus-mode", choices=["active", "added", "all"], default="active")
    parser.add_argument("--duration-ms", type=float, default=18000.0)
    parser.add_argument("--dt-ms", type=float, default=5.0)
    parser.add_argument("--stride-period-ms", type=float, default=1200.0)
    parser.add_argument("--swing-fraction", type=float, default=0.32)
    parser.add_argument("--t1-offset-ms", type=float, default=DEFAULT_SEGMENT_OFFSETS_MS["T1"])
    parser.add_argument("--t2-offset-ms", type=float, default=DEFAULT_SEGMENT_OFFSETS_MS["T2"])
    parser.add_argument("--t3-offset-ms", type=float, default=DEFAULT_SEGMENT_OFFSETS_MS["T3"])
    args = parser.parse_args()

    save_tag = args.save_tag or args.hemi_tag
    mapping_csv = (args.mapping_csv or _default_mapping_csv(PHASE3_ROOT)).expanduser().resolve()
    run_dir = (args.run_dir or _default_run_dir(workspace_root, args.hemi_tag, args.run_name)).expanduser().resolve()
    added_motor_ids_csv = (
        args.added_motor_ids_csv or _default_added_ids_csv(workspace_root, args.hemi_tag)
    ).expanduser().resolve()
    out_dir = (args.out_dir or _default_out_dir(PHASE3_ROOT, save_tag, args.run_name)).expanduser().resolve()

    profile_yaml = PHASE3_ROOT / "configs" / "phase3_video_profiles.yaml"
    with open(profile_yaml, "r", encoding="utf-8") as f:
        profile_doc = yaml.safe_load(f)

    summary = render_expected_gait_video(
        phase3_root=PHASE3_ROOT,
        mapping_csv=mapping_csv,
        out_dir=out_dir,
        run_dir=run_dir if run_dir.exists() else None,
        added_motor_ids_csv=added_motor_ids_csv if added_motor_ids_csv.exists() else None,
        save_tag=save_tag,
        run_name=args.run_name,
        profile_doc=profile_doc,
        profile_name=args.profile_name,
        focus_mode=args.focus_mode,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        stride_period_ms=args.stride_period_ms,
        swing_fraction=args.swing_fraction,
        segment_offsets_ms={
            "T1": float(args.t1_offset_ms),
            "T2": float(args.t2_offset_ms),
            "T3": float(args.t3_offset_ms),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
