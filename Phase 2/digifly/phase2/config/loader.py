from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
import copy
import uuid

from .defaults import get_default_config


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = copy.deepcopy(v)
    return out


def _require_existing_dir(p: Union[str, Path], name: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} must exist and be a directory: {path}")
    return path


def _require_existing_file(p: Union[str, Path], name: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} must exist and be a file: {path}")
    return path


def _apply_user_friendly_overrides(cfg: Dict[str, Any]) -> None:
    """
    Support the notebook's "knobs" while still producing the EXACT keys
    the simulator already uses (Phase 2 Master-compatible).
    """
    # Map stim -> iclamp_* if provided
    stim = cfg.get("stim") or {}
    if isinstance(stim, dict):
        ic = stim.get("iclamp") or {}
        if isinstance(ic, dict):
            if "amp_nA" in ic:
                cfg["iclamp_amp_nA"] = float(ic["amp_nA"])
            if "delay_ms" in ic:
                cfg["iclamp_delay_ms"] = float(ic["delay_ms"])
            if "dur_ms" in ic:
                cfg["iclamp_dur_ms"] = float(ic["dur_ms"])
            # location is consumed by walking.runner via cfg['iclamp_location']
            if "location" in ic:
                cfg["iclamp_location"] = str(ic["location"])

        neg = stim.get("neg_pulse") or {}
        if isinstance(neg, dict) and bool(neg.get("enabled", False)):
            neg_cfg = dict(neg)
            if "amp_nA" in neg_cfg:
                neg_cfg["amp_nA"] = float(neg_cfg["amp_nA"])
            if "delay_ms" in neg_cfg:
                neg_cfg["delay_ms"] = float(neg_cfg["delay_ms"])
            if "dur_ms" in neg_cfg:
                neg_cfg["dur_ms"] = float(neg_cfg["dur_ms"])
            if "location" in neg_cfg and neg_cfg["location"] is not None:
                neg_cfg["location"] = str(neg_cfg["location"])
            cfg["neg_pulse"] = neg_cfg

        # Repeated pulse train stimulation (optional).
        # Canonical location is stim.pulse_train, but we also accept
        # frequency fields inside stim.iclamp for compatibility.
        pulse_train = stim.get("pulse_train")
        if not isinstance(pulse_train, dict) and isinstance(ic, dict):
            hz_alias = ic.get("freq_hz", ic.get("frequency_hz", None))
            if hz_alias is not None:
                pulse_train = {
                    "enabled": bool(ic.get("train_enabled", True)),
                    "freq_hz": hz_alias,
                    "amp_nA": ic.get("amp_nA"),
                    "delay_ms": ic.get("delay_ms"),
                    "dur_ms": ic.get("dur_ms"),
                    "location": ic.get("location"),
                    "stop_ms": ic.get("stop_ms"),
                    "max_pulses": ic.get("max_pulses"),
                    "include_base_iclamp": bool(ic.get("include_base_iclamp", False)),
                }

        if isinstance(pulse_train, dict) and bool(pulse_train.get("enabled", False)):
            tr_cfg = dict(pulse_train)
            if "freq_hz" not in tr_cfg and "frequency_hz" in tr_cfg:
                tr_cfg["freq_hz"] = tr_cfg.get("frequency_hz")
            if "freq_hz" in tr_cfg:
                tr_cfg["freq_hz"] = float(tr_cfg["freq_hz"])
            if "amp_nA" in tr_cfg and tr_cfg["amp_nA"] is not None:
                tr_cfg["amp_nA"] = float(tr_cfg["amp_nA"])
            if "delay_ms" in tr_cfg and tr_cfg["delay_ms"] is not None:
                tr_cfg["delay_ms"] = float(tr_cfg["delay_ms"])
            if "dur_ms" in tr_cfg and tr_cfg["dur_ms"] is not None:
                tr_cfg["dur_ms"] = float(tr_cfg["dur_ms"])
            if "stop_ms" in tr_cfg and tr_cfg["stop_ms"] is not None:
                tr_cfg["stop_ms"] = float(tr_cfg["stop_ms"])
            if "max_pulses" in tr_cfg and tr_cfg["max_pulses"] is not None:
                tr_cfg["max_pulses"] = int(tr_cfg["max_pulses"])
            if "location" in tr_cfg and tr_cfg["location"] is not None:
                tr_cfg["location"] = str(tr_cfg["location"])
            tr_cfg["include_base_iclamp"] = bool(tr_cfg.get("include_base_iclamp", False))
            cfg["pulse_train"] = tr_cfg

    # Map HH_GLOBAL -> existing hh dicts (pre/post soma/branch)
    hh_global = cfg.get("hh_global")
    if isinstance(hh_global, dict) and hh_global:
        # Update all HH dicts but preserve keys if user already set them explicitly
        for k in ("pre_soma_hh", "pre_branch_hh", "post_soma_hh", "post_branch_hh"):
            if isinstance(cfg.get(k), dict):
                merged = dict(cfg[k])
                for kk, vv in hh_global.items():
                    merged[kk] = vv
                cfg[k] = merged

    # Map PASSIVE_GLOBAL -> Ra/cm (Phase 2 keys)
    passive_global = cfg.get("passive_global")
    if isinstance(passive_global, dict) and passive_global:
        if "Ra" in passive_global:
            cfg["Ra"] = float(passive_global["Ra"])
        if "cm" in passive_global:
            cfg["cm"] = float(passive_global["cm"])
        if "g_pas" in passive_global:
            cfg["passive_g"] = float(passive_global["g_pas"])
        if "e_pas" in passive_global:
            cfg["passive_e"] = float(passive_global["e_pas"])

    # If user sets V_REST / E_L globally, propagate to existing expected keys
    vrest = cfg.get("v_rest_mV")
    if vrest is not None:
        cfg["passive_e"] = float(vrest)
        cfg["v_init_mV"] = float(cfg.get("v_init_mV", vrest))

        # Also set el inside the hh dicts unless user already specified el there
        for k in ("pre_soma_hh", "pre_branch_hh", "post_soma_hh", "post_branch_hh"):
            d = cfg.get(k)
            if isinstance(d, dict) and "el" not in d:
                d = dict(d)
                d["el"] = float(vrest)
                cfg[k] = d

    # If el_mV is given explicitly, treat it as canonical leak reversal for HH dicts.
    el_mV = cfg.get("el_mV")
    if el_mV is not None:
        for k in ("pre_soma_hh", "pre_branch_hh", "post_soma_hh", "post_branch_hh"):
            d = cfg.get(k)
            if isinstance(d, dict):
                d = dict(d)
                d["el"] = float(el_mV)
                cfg[k] = d


def build_config(user_overrides: Optional[Mapping[str, Any]] = None, *, strict: bool = True) -> Dict[str, Any]:
    """
    Build CONFIG:
      priority: user_overrides > defaults
      derives Phase-2 folder conventions from swc_dir:
        edges_root = <swc_dir>/edges
        runs_root  = <swc_dir>/hemi_runs
        master_csv = <swc_dir>/../all_neurons_neuroncriteria_template.csv

    STRICT rules (as you requested):
      - swc_dir MUST exist
      - selection.mode must be explicitly set (no unknown/default)
      - if master_csv is needed, it must exist
      - if custom edges_path is needed, it must exist
    """
    base = get_default_config()
    cfg = _deep_merge(base, user_overrides or {})

    _apply_user_friendly_overrides(cfg)

    # swc_dir is mandatory and must exist
    if not cfg.get("swc_dir"):
        raise ValueError("CONFIG['swc_dir'] is required")
    swc_root = _require_existing_dir(cfg["swc_dir"], "swc_dir")
    cfg["swc_dir"] = str(swc_root)

    # Optional morphology-only SWC root (for reduced datasets)
    if cfg.get("morph_swc_dir"):
        morph_root = _require_existing_dir(cfg["morph_swc_dir"], "morph_swc_dir")
        cfg["morph_swc_dir"] = str(morph_root)

    # Optional edge cache config (used by custom mode for on-demand subgraph edges).
    edge_cache = cfg.get("edge_cache") or {}
    if not isinstance(edge_cache, Mapping):
        raise ValueError("CONFIG['edge_cache'] must be a dictionary when provided.")
    edge_cache = dict(edge_cache)
    if edge_cache.get("db_path"):
        edge_cache["db_path"] = str(Path(edge_cache["db_path"]).expanduser().resolve())
    srcs = edge_cache.get("source_paths")
    if srcs is not None:
        if not isinstance(srcs, (list, tuple)):
            raise ValueError("CONFIG['edge_cache']['source_paths'] must be a list of file paths or None.")
        norm_srcs = []
        for p in srcs:
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            norm_srcs.append(str(Path(s).expanduser().resolve()))
        edge_cache["source_paths"] = norm_srcs
    q = edge_cache.get("query")
    if q is not None:
        if not isinstance(q, Mapping):
            raise ValueError("CONFIG['edge_cache']['query'] must be a dictionary when provided.")
        edge_cache["query"] = dict(q)
    cfg["edge_cache"] = edge_cache

    parallel_cfg = cfg.get("parallel") or {}
    if not isinstance(parallel_cfg, Mapping):
        raise ValueError("CONFIG['parallel'] must be a dictionary when provided.")
    parallel_cfg = dict(parallel_cfg)
    if parallel_cfg.get("owner_by_gid") is not None and not isinstance(parallel_cfg.get("owner_by_gid"), Mapping):
        raise ValueError("CONFIG['parallel']['owner_by_gid'] must be a mapping when provided.")
    if isinstance(parallel_cfg.get("owner_by_gid"), Mapping):
        owner_map_raw = dict(parallel_cfg.get("owner_by_gid") or {})
        owner_map_norm = {}
        for raw_gid, raw_owner in owner_map_raw.items():
            gid = int(raw_gid)
            owner = int(raw_owner)
            owner_map_norm[gid] = owner
        parallel_cfg["owner_by_gid"] = owner_map_norm
    cfg["parallel"] = parallel_cfg

    # Phase-2 conventions (Option A)
    edges_root = Path(cfg.get("edges_root") or (swc_root / "edges")).resolve()
    runs_root = Path(cfg.get("runs_root") or (swc_root / "hemi_runs")).resolve()
    master_csv = Path(cfg.get("master_csv") or (swc_root.parent / "all_neurons_neuroncriteria_template.csv")).resolve()

    edges_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    cfg["edges_root"] = str(edges_root)
    cfg["runs_root"] = str(runs_root)
    cfg["master_csv"] = str(master_csv)

    # Enforce explicit selection.mode
    sel = cfg.get("selection") or {}
    mode = sel.get("mode")
    if strict and not mode:
        raise ValueError("CONFIG['selection']['mode'] must be set: 'single'|'custom'|'hemilineage'")

    # Master CSV required for hemilineage selection (and typically for ID lookup)
    if strict and mode == "hemilineage":
        _require_existing_file(master_csv, "master_csv")

    # Custom mode requires explicit edges_path
    if strict and mode == "custom":
        ep = cfg.get("edges_path")
        edge_cache_enabled = bool((cfg.get("edge_cache") or {}).get("enabled", False))
        if not ep and not edge_cache_enabled:
            raise ValueError("CONFIG['edges_path'] is required for mode='custom' when edge_cache.enabled=False")
        if ep:
            _require_existing_file(ep, "edges_path")
        if edge_cache_enabled:
            ec = cfg.get("edge_cache") or {}
            if str(ec.get("build_mode", "from_edges_files")).strip().lower() == "from_edges_files":
                for p in (ec.get("source_paths") or []):
                    _require_existing_file(p, "edge_cache.source_paths[]")

    # run_id
    if not cfg.get("run_id"):
        cfg["run_id"] = f"run_{uuid.uuid4().hex[:10]}"

    # Force canonical columns you specified (your edges schema)
    gt = cfg.get("global_timing") or {}
    gt.update(
        {
            "weight_col": "weight_uS",
            "delay_col": "delay_ms",
            "tau1_col": "tau1_ms",
            "tau2_col": "tau2_ms",
            "erev_col": "syn_e_rev_mV",
        }
    )
    cfg["global_timing"] = gt

    return cfg
