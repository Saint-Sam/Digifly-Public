from __future__ import annotations

from typing import Any, Dict
import os
from pathlib import Path

from digifly.phase2.neuron_build.config import DEFAULT_CONFIG as _NEURON_DEFAULTS
from digifly.phase2.neuron_build.config import DEFAULT_GLOBAL_TIMING as _TIMING_DEFAULTS


def get_default_config() -> Dict[str, Any]:
    """
    Phase-2 defaults WITHOUT touching the filesystem.
    Validation happens in loader.build_config().

    Priority rule:
      user_overrides > defaults
    """
    cfg: Dict[str, Any] = dict(_NEURON_DEFAULTS)
    # Phase-2 runner default wall-clock span for quick debug runs.
    # Notebook/user overrides still take priority via loader.build_config().
    cfg["tstop_ms"] = 10.0

    # Optional env default (still just a default; user overrides win)
    env_swc = os.environ.get("DIGIFLY_SWC_DIR", "").strip()
    if env_swc:
        cfg["swc_dir"] = env_swc

    # Runner-level controls (no-ops unless user sets them)
    cfg.update(
        {
            "selection": {
                "mode": None,       # 'single' | 'custom' | 'hemilineage'
                "label": None,      # hemilineage label
                "neuron_id": None,  # single neuron id
                "neuron_ids": None, # custom list[int]
            },
            "seeds": None,          # None => all seeds
            "edges_path": None,     # optional explicit edges file
            "morph_swc_dir": None,  # optional SWC root used only for morphology loading

            "run_id": None,
            "run_notes": "",

            # derived from swc_dir when running (Option A)
            "edges_root": None,
            "runs_root": None,
            "master_csv": None,

            # allow future extension
            "threads": None,
            "parallel": {
                "threads": None,
                "build_backend": "single_host",     # planned: 'single_host' | 'distributed_gid'
                "ownership_strategy": "round_robin",
                "owner_by_gid": None,
                "maxstep_ms": 10.0,
            },
            "cache": {
                "ais": True,
                "edges": True,
                "synapse_sites": True,
            },
            # Canonical edge cache for fast custom subgraph extraction.
            "edge_cache": {
                "enabled": False,
                "db_path": None,           # default: <edges_root>/master_edges_cache.sqlite
                "build_if_missing": True,
                "force_rebuild": False,
                "build_mode": "from_edges_files",  # 'from_edges_files' | 'from_synapses_csv'
                "source_paths": None,      # list[str]; None => auto-discover *_from_synapses* under edges_root
                "overlay_dir": None,       # default: <runs_root>/_edge_cache_overlays
                "query": {
                    "mode": "loaded_subgraph",      # 'loaded_subgraph' | 'seed_io_1hop'
                    "include_loaded": True,         # when seed_io_1hop, keep user-loaded IDs too
                    "max_nodes": None,              # optional safety cap
                    "max_rows": None,               # optional safety cap
                },
            },

            # electrical coupling (disabled by default)
            "gap": {
                "enabled": False,
                "mechanisms_dir": None,
                "default_site": "ais",
                "default_g_uS": 0.001,
                "pairs": [],
            },

            # variable-step integration (must not be combined with CoreNEURON)
            "cvode": {
                "enabled": False,
                "atol": 1e-3,
                "rtol": None,
                "maxstep_ms": None,
            },

            # edge-column schema (your current edges columns)
            "global_timing": dict(_TIMING_DEFAULTS),

            "record": {
                "soma_v": "seeds",          # 'none' | 'seeds' | 'all' | list[int]
                "spikes": "seeds",
                "spike_thresh_mV": 0.0,
            },

            # silencing: present but no-op unless enabled
            "silence": {
                "enable": False,
                "mode": "set_gnabar",       # 'set_gnabar' | 'clamp_rest'
                "targets": [],
                "gnabar": 0.0,
                "rest_mV": -65.0,
            },

            # neurotransmitter mapping: future extension
            "nt_mapping": {
                "enabled": False,
                "rules": {
                    "acetylcholine": "exc",
                    "glutamate": "inh",
                    "gaba": "inh",
                    "glycine": "inh",
                },
            },

            # v_init controls h.finitialize()
            "v_init_mV": -65.0,

            # equilibrium potentials (optional)
            "ena_mV": None,
            "ek_mV": None,
        }
    )

    env_threads = os.environ.get("DIGIFLY_NEURON_THREADS", "").strip()
    if env_threads:
        try:
            cfg["threads"] = max(1, int(env_threads))
        except Exception:
            pass

    return cfg


# -----------------------------------------------------------------------------
# Backward-compat legacy exports (for edges_from_synapses.py import compatibility)
# IMPORTANT: These MUST NOT touch the filesystem or raise at import time.
# They are just "best-effort defaults" derived from DIGIFLY_SWC_DIR if set.
# -----------------------------------------------------------------------------
def _maybe_resolved_path(p: str | None) -> Path | None:
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return None


_ENV_SWC = os.environ.get("DIGIFLY_SWC_DIR", "").strip()
HEMI_SWC_ROOT: Path | None = _maybe_resolved_path(_ENV_SWC)

HEMI_MASTER_CSV: Path | None = (
    (HEMI_SWC_ROOT.parent / "all_neurons_neuroncriteria_template.csv").resolve()
    if HEMI_SWC_ROOT is not None
    else None
)

HEMI_EDGES_ROOT: Path | None = (
    (HEMI_SWC_ROOT / "edges").resolve()
    if HEMI_SWC_ROOT is not None
    else None
)

HEMI_RUN_ROOT: Path | None = (
    (HEMI_SWC_ROOT / "hemi_runs").resolve()
    if HEMI_SWC_ROOT is not None
    else None
)

# Legacy name expected by older modules
CONFIG: Dict[str, Any] = get_default_config()
