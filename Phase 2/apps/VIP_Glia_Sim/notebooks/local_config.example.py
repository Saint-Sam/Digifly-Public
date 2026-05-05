"""Copy to local_config.py for machine-specific VIP glia launcher settings."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

# Public cache used by the launch notebooks.
SWC_DIR = REPO_ROOT / "Phase 1" / "manc_v1.2.1" / "export_swc"
FLOW_RUNS_ROOT = SWC_DIR / "hemi_runs"

# Optional source tree used to seed the public cache when SWCs are missing.
# FALLBACK_SWC_DIR = Path("/path/to/source/export_swc")

# Optional explicit finished run to load into the flow visualizer.
# FLOW_RUN_DIR = FLOW_RUNS_ROOT / "single_neuron_debug"

# Optional overrides. Leave commented unless the installation uses different paths.
# PHASE2_ROOT = REPO_ROOT / "Phase 2"
# APP_ROOT = PHASE2_ROOT / "apps" / "VIP_Glia_Sim"
# PYTHON_BIN = Path("/path/to/python")
