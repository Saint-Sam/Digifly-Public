#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PHASE3_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PHASE3_ROOT / "src" / "phase3_bridge"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gait_compare import compare_gait_to_expected


def _default_sim_dir(phase3_root: Path, save_tag: str, run_name: str) -> Path:
    return phase3_root / "data" / "derived" / "gait_expectation" / save_tag / run_name


def _default_expected_dir(phase3_root: Path, save_tag: str, run_name: str) -> Path:
    return phase3_root / "data" / "derived" / save_tag / run_name / "expected_gait"


def _default_out_dir(phase3_root: Path, save_tag: str, run_name: str) -> Path:
    return phase3_root / "data" / "derived" / "gait_compare" / save_tag / run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a hemilineage run against the synthetic expected-gait baseline.")
    parser.add_argument("--save-tag", default="Hemi_09A")
    parser.add_argument("--run-name", default="hemi_09a_baseline")
    parser.add_argument("--sim-audit-dir", type=Path, default=None)
    parser.add_argument("--expected-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sim_dir = (args.sim_audit_dir or _default_sim_dir(PHASE3_ROOT, args.save_tag, args.run_name)).expanduser().resolve()
    expected_dir = (args.expected_dir or _default_expected_dir(PHASE3_ROOT, args.save_tag, args.run_name)).expanduser().resolve()
    out_dir = (args.out_dir or _default_out_dir(PHASE3_ROOT, args.save_tag, args.run_name)).expanduser().resolve()

    report = compare_gait_to_expected(
        sim_audit_dir=sim_dir,
        expected_dir=expected_dir,
        out_dir=out_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
