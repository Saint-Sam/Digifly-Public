from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

# Defaults only. Users override at runtime via merge_cfg(DEFAULT_CONFIG, {...}).

DEFAULT_CONFIG: Dict[str, Any] = {
    # paths
    "swc_dir": str(Path.home()),
    "morph_swc_dir": None,
    "edges_csv": "",

    # ids (optional)
    "pre_id": None,
    "post_id": None,

    # sim
    "dt_ms": 0.1,
    "tstop_ms": 10.0,
    "iclamp_amp_nA": 2.2,
    "iclamp_delay_ms": 10.0,
    "iclamp_dur_ms": 900.0,

    # progress
    "progress": True,
    "progress_chunk_ms": 0.5,
    "use_tqdm": True,
    "threads": None,
    "parallel": {
        "threads": None,
        "build_backend": "single_host",
        "ownership_strategy": "round_robin",
        "owner_by_gid": None,
        "maxstep_ms": 10.0,
    },

    # CoreNEURON
    "enable_coreneuron": False,
    "coreneuron_gpu": False,
    "coreneuron_verbose": False,
    "coreneuron_nthread": None,
    "io_workers": None,
    "cvode": {
        "enabled": False,
        "atol": 1e-3,
        "rtol": None,
        "maxstep_ms": None,
    },

    # syn filtering / defaults
    "epsilon_um": 6.0,
    "syn_e_rev_mV": 0.0,
    "syn_tau1_ms": 0.5,
    "syn_tau2_ms": 3.0,
    "default_weight_uS": 0.000003,
    "default_delay_ms": 1.0,

    # biophys
    "Ra": 100.0,
    "cm": 1.0,
    "pre_soma_hh": dict(gnabar=0.12, gkbar=0.036, gl=3e-4, el=-65.0),
    "pre_branch_hh": dict(gnabar=0.02, gkbar=0.01, gl=1e-4, el=-65.0),
    "passive_g": 1e-4,
    "passive_e": -65.0,

    # POST can be active
    "post_active": True,
    "post_soma_hh": dict(gnabar=0.12, gkbar=0.036, gl=3e-4, el=-65.0),
    "post_branch_hh": dict(gnabar=0.02, gkbar=0.01, gl=1e-4, el=-65.0),

    # AIS mapping
    "ais_min_dist_um": 1.0,
    "ais_min_xloc": 0.05,
    "ais_cache_csv": str(Path.home() / "_ais_cache.csv"),
    "ais_strict_axon_map": True,
    "ais_override_filename": "ais_overrides.csv",

    # runtime flags
    "run_ais_prompt": False,
    "run_strict_viz": True,
    "use_geom_delay": True,

    # strict viz
    "viz_strict_enforce_swc_soma": True,
    "viz_tol_um": 0.75,
    "viz_report_csv": None,
}

SYN_PRESETS = {
    "cholinergic_fast": (0.15, 1.40, 0.0, 1.0),
    "ampa_fast":        (0.20, 1.60, 0.0, 1.0),
    "gabaa_fast":       (0.25, 6.00, -70.0, 1.0),
}

NT_TO_CLASS = {
    "acetylcholine": "cholinergic_fast",
    "ach": "cholinergic_fast",
    "glutamate": "ampa_fast",
    "gaba": "gabaa_fast",
    "glycine": "gabaa_fast",
}

DEFAULT_GLOBAL_TIMING = {
    "base_release_delay_ms": 0.40,
    "vel_um_per_ms": 1500.0,
    "global_weight_scale": 1.0,
    "default_class": "cholinergic_fast",
    "pathlen_col": "path_length_um",
    "nt_col": "predicted_nt",
    "weight_col": "weight_uS",
    "delay_col": "delay_ms",
    "tau1_col": "tau1_ms",
    "tau2_col": "tau2_ms",
    "erev_col": "erev_mV",
}

def merge_cfg(base: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Shallow merge dicts; nested dicts are not deep-merged."""
    out = dict(base)
    if overrides:
        out.update(overrides)
    return out
