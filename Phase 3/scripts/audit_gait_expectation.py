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

from gait_audit import run_gait_expectation_audit


def _default_run_dir(workspace_root: Path) -> Path:
    return workspace_root / "Hemilineage Simulations" / "Hemi_09A" / "runs" / "hemi_09a_baseline"


def _default_mapping_csv(phase3_root: Path) -> Path:
    return phase3_root / "data" / "inputs" / "mappings" / "mn_to_actuator_mapping_rebuilt.csv"


def _default_out_dir(phase3_root: Path) -> Path:
    return phase3_root / "data" / "derived" / "gait_expectation" / "Hemi_09A" / "hemi_09a_baseline"


def main() -> None:
    workspace_root = PHASE3_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Audit a Phase 2 run against a heuristic gait baseline derived from motor neuron labels."
    )
    parser.add_argument("--run-dir", type=Path, default=_default_run_dir(workspace_root))
    parser.add_argument("--mapping-csv", type=Path, default=_default_mapping_csv(PHASE3_ROOT))
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir(PHASE3_ROOT))
    parser.add_argument("--tau-rise-ms", type=float, default=1.0)
    parser.add_argument("--tau-decay-ms", type=float, default=6.0)
    args = parser.parse_args()

    report = run_gait_expectation_audit(
        run_dir=args.run_dir,
        mapping_csv=args.mapping_csv,
        out_dir=args.out_dir,
        tau_rise_ms=args.tau_rise_ms,
        tau_decay_ms=args.tau_decay_ms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
