#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


PHASE3_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PHASE3_ROOT / "src" / "phase3_bridge"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mapping_rebuild import run_mapping_rebuild


def _find_world_xml(phase3_root: Path) -> Path | None:
    candidates = [
        phase3_root / "data" / "inputs" / "flybody" / "floor.xml",
        Path.home() / "Desktop" / "Digifly" / "flybody-main" / "flybody" / "fruitfly" / "assets" / "floor.xml",
    ]
    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path
    return None


def main() -> None:
    phase3_root = PHASE3_ROOT
    workspace_root = phase3_root.parent

    phase1_mn_csv = workspace_root / "Phase 1" / "motor_neuron_query" / "outputs" / "all_motor_neurons_instance_contains_MN.csv"
    phase2_template_csv = workspace_root / "Phase 2" / "data" / "all_neurons_neuroncriteria_template.csv"
    out_mapping_csv = phase3_root / "data" / "inputs" / "mappings" / "mn_to_actuator_mapping_rebuilt.csv"
    out_dir = phase3_root / "data" / "derived" / "mapping_rebuild"
    hemi_added_motor_csv = (
        workspace_root / "Hemilineage Simulations" / "Hemi_09A" / "metadata" / "added_motor_neuron_ids.csv"
    )
    hemi_spike_csv = workspace_root / "Hemilineage Simulations" / "Hemi_09A" / "runs" / "hemi_09a_baseline" / "spike_times.csv"

    summary = run_mapping_rebuild(
        phase1_mn_csv=phase1_mn_csv,
        phase2_template_csv=phase2_template_csv if phase2_template_csv.exists() else None,
        out_mapping_csv=out_mapping_csv,
        out_dir=out_dir,
        mjcf_xml=_find_world_xml(phase3_root),
        hemi_added_motor_csv=hemi_added_motor_csv if hemi_added_motor_csv.exists() else None,
        hemi_spike_csv=hemi_spike_csv if hemi_spike_csv.exists() else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
