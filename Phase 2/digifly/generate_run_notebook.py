from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

NOTEBOOK_PATH = Path("notebooks/run_simulation.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

cells = []

# ---------------------------------------------------------------------
cells.append(new_markdown_cell(
"""# Digifly Phase 2 – Unified Simulation Runner

This notebook runs **all simulation modes**:
- single neuron
- custom network
- hemilineage network (e.g. 09A)

You only edit the configuration cells and run top-to-bottom.
"""
))

# ---------------------------------------------------------------------
cells.append(new_code_cell(
"""from __future__ import annotations

import os
import sys
from pathlib import Path
import json

REPO_ROOT = Path.cwd()
if (REPO_ROOT / "digifly").exists():
    sys.path.insert(0, str(REPO_ROOT))

from digifly.phase2.api import build_config, run_walking_simulation
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Required paths (Phase 2 Master compatible)"))

cells.append(new_code_cell(
"""# REQUIRED
SWC_DIR = r"/path/to/export_swc"

# Optional override (default is SWC_DIR/../all_neurons_neuroncriteria_template.csv)
MASTER_CSV = None

if not Path(SWC_DIR).expanduser().resolve().exists():
    raise FileNotFoundError(f"SWC_DIR does not exist: {SWC_DIR}")

if MASTER_CSV is not None and not Path(MASTER_CSV).expanduser().resolve().exists():
    raise FileNotFoundError(f"MASTER_CSV does not exist: {MASTER_CSV}")

os.environ["DIGIFLY_SWC_DIR"] = SWC_DIR
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Select simulation mode"))

cells.append(new_code_cell(
"""# Choose ONE: "hemilineage", "single", "custom"
MODE = "hemilineage"
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("### Hemilineage configuration"))

cells.append(new_code_cell(
"""HEMI_LABEL = "09A"
SEEDS = None  # None = all seeds
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("### Single-neuron configuration"))

cells.append(new_code_cell(
"""NEURON_ID = 10000
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("### Custom-network configuration"))

cells.append(new_code_cell(
"""NEURON_IDS = [10000, 10002]
EDGES_PATH = r"/path/to/custom_edges.parquet"
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Simulation timing"))

cells.append(new_code_cell(
"""RUN_ID = "run_001"
TSTOP_MS = 4000.0
DT_MS = 0.1
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Biophysics knobs"))

cells.append(new_code_cell(
"""E_NA_MV = 50.0
E_K_MV  = -77.0
E_L_MV  = -54.3
V_REST_MV = -65.0

HH_GLOBAL = {
    "gnabar": 0.12,
    "gkbar": 0.036,
    "gl": 0.0003,
    "el": E_L_MV,
}

PASSIVE_GLOBAL = {
    "cm": 1.0,
    "Ra": 100.0,
}
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Stimulation (Daniel-style IClamp)"))

cells.append(new_code_cell(
"""ICLAMP = {
    "amp_nA": 2.5,
    "delay_ms": 100.0,
    "dur_ms": 200.0,
    "location": "soma",
}

NEG_PULSE = {
    "enabled": False,
    "amp_nA": -1.0,
    "delay_ms": 150.0,
    "dur_ms": 50.0,
    "location": "soma",
}
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Recording controls"))

cells.append(new_code_cell(
"""RECORD = {
    "soma_v": "seeds",
    "spikes": "seeds",
    "spike_thresh_mV": 0.0,
}
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Assemble CONFIG overrides"))

cells.append(new_code_cell(
"""if MODE == "hemilineage":
    SELECTION = {"mode": "hemilineage", "label": HEMI_LABEL}
    EDGES = None
elif MODE == "single":
    SELECTION = {"mode": "single", "neuron_id": int(NEURON_ID)}
    EDGES = None
elif MODE == "custom":
    SELECTION = {"mode": "custom", "neuron_ids": [int(x) for x in NEURON_IDS]}
    EDGES = EDGES_PATH
else:
    raise ValueError(f"Unknown MODE: {MODE}")

USER_OVERRIDES = {
    "swc_dir": SWC_DIR,
    "master_csv": MASTER_CSV,
    "selection": SELECTION,
    "edges_path": EDGES,
    "seeds": SEEDS,
    "run_id": RUN_ID,
    "tstop_ms": TSTOP_MS,
    "dt_ms": DT_MS,
    "ena_mV": E_NA_MV,
    "ek_mV": E_K_MV,
    "el_mV": E_L_MV,
    "v_rest_mV": V_REST_MV,
    "hh_global": HH_GLOBAL,
    "passive_global": PASSIVE_GLOBAL,
    "stim": {
        "iclamp": ICLAMP,
        "neg_pulse": NEG_PULSE,
    },
    "record": RECORD,
}

CONFIG = build_config(USER_OVERRIDES)

print("CONFIG ready:")
print("  swc_dir   =", CONFIG["swc_dir"])
print("  run_id    =", CONFIG["run_id"])
"""
))

# ---------------------------------------------------------------------
cells.append(new_markdown_cell("## Run simulation"))

cells.append(new_code_cell(
"""out_dir = run_walking_simulation(CONFIG)
print("Results saved to:", out_dir)

out_dir = Path(out_dir)
(out_dir / "config_used.json").write_text(
    json.dumps(CONFIG, indent=2, default=str)
)
"""
))

# ---------------------------------------------------------------------
nb = new_notebook(cells=cells)
nbformat.write(nb, NOTEBOOK_PATH)

print(f"Notebook written to: {NOTEBOOK_PATH.resolve()}")
