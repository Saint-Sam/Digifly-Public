from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at import time for notebook portability
    plt = None

from digifly.phase2.api import build_config, run_walking_simulation
from digifly.phase2.graph.custom_circuit_workflow import apply_recording_policy
from digifly.phase2.graph.requested_edge_sets import (
    DEFAULT_NEUPRINT_DATASET,
    EDGE_SET_COLUMNS,
    default_edges_registry_root,
    ensure_named_edge_set,
    expand_requested_network,
    normalize_neuprint_dataset,
)
from digifly.phase2.neuron_build.builders import _first_present_col, _prepare_wiring_df
from digifly.phase2.neuron_build.gaps import gap_pair_ohmic, gap_pair_rectifying
from digifly.tools import reduce_swc_dataset as rsd


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "first_spike_ms": 0.20,
    "latency_ms": 0.20,
    "peak_mV": 8.0,
    "rmse_mV": 5.0,
}

DEFAULT_REDUCTION_PROFILES: List[Dict[str, Any]] = [
    {"name": "mild_tight", "max_path_um": 4.5, "max_turn_deg": 22.5, "max_diam_rel": 0.22},
    {"name": "mild", "max_path_um": 5.0, "max_turn_deg": 25.0, "max_diam_rel": 0.25},
    {"name": "mild_plus", "max_path_um": 6.0, "max_turn_deg": 28.0, "max_diam_rel": 0.28},
    {"name": "bridge_7_32_32", "max_path_um": 7.0, "max_turn_deg": 32.0, "max_diam_rel": 0.32},
    {"name": "default_v1", "max_path_um": 8.0, "max_turn_deg": 35.0, "max_diam_rel": 0.35},
    {"name": "default_plus", "max_path_um": 9.0, "max_turn_deg": 38.0, "max_diam_rel": 0.38},
    {"name": "default_relax", "max_path_um": 10.0, "max_turn_deg": 40.0, "max_diam_rel": 0.40},
]

DEFAULT_COALESCE_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "site_default",
        "coalesce_syns": True,
        "coalesce_mode": "site",
        "use_geom_delay": True,
        "enable_coreneuron": True,
        "coreneuron_gpu": True,
        "cvode": {"enabled": False},
    },
    {
        "name": "pair_default",
        "coalesce_syns": True,
        "coalesce_mode": "pair",
        "use_geom_delay": True,
        "enable_coreneuron": True,
        "coreneuron_gpu": True,
        "cvode": {"enabled": False},
    },
    {
        "name": "pair_no_geom_delay",
        "coalesce_syns": True,
        "coalesce_mode": "pair",
        "use_geom_delay": False,
        "enable_coreneuron": True,
        "coreneuron_gpu": True,
        "cvode": {"enabled": False},
    },
]

RUN_SIMULATION_GAP_PAIRS: List[Tuple[int, int]] = [
    (10000, 10110),
    (10002, 10068),
    (10000, 11446),
    (10000, 11654),
    (10002, 11446),
    (10002, 11654),
]

HEMI_EDGE_COLUMNS: List[str] = list(EDGE_SET_COLUMNS)


def _can_plot() -> bool:
    return plt is not None


def _maybe_tqdm(iterable, *, total: int | None = None, desc: str = "", enabled: bool = True, leave: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=leave)
    except Exception:
        return iterable


def _make_tqdm(*, total: int, desc: str, enabled: bool = True, leave: bool = True):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, leave=leave)
    except Exception:
        return None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return out


def _run_reuse_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    global_timing = dict(cfg.get("global_timing") or {})
    parallel = dict(cfg.get("parallel") or {})
    record = dict(cfg.get("record") or {})
    gap = dict(cfg.get("gap") or {})
    passive_global = dict(cfg.get("passive_global") or {})
    return {
        "edges_path": str(cfg.get("edges_path") or ""),
        "master_csv": str(cfg.get("master_csv") or ""),
        "seeds": [int(x) for x in cfg.get("seeds", [])] if cfg.get("seeds") is not None else None,
        "tstop_ms": cfg.get("tstop_ms"),
        "dt_ms": cfg.get("dt_ms"),
        "celsius_C": cfg.get("celsius_C"),
        "enable_coreneuron": bool(cfg.get("enable_coreneuron", False)),
        "coreneuron_gpu": bool(cfg.get("coreneuron_gpu", False)),
        "coreneuron_nthread": cfg.get("coreneuron_nthread"),
        "io_workers": cfg.get("io_workers"),
        "default_weight_uS": cfg.get("default_weight_uS"),
        "default_delay_ms": cfg.get("default_delay_ms"),
        "syn_tau1_ms": cfg.get("syn_tau1_ms"),
        "syn_tau2_ms": cfg.get("syn_tau2_ms"),
        "syn_e_rev_mV": cfg.get("syn_e_rev_mV"),
        "use_geom_delay": cfg.get("use_geom_delay"),
        "global_timing": {
            "nt_col": global_timing.get("nt_col"),
            "weight_col": global_timing.get("weight_col"),
            "delay_col": global_timing.get("delay_col"),
            "tau1_col": global_timing.get("tau1_col"),
            "tau2_col": global_timing.get("tau2_col"),
            "erev_col": global_timing.get("erev_col"),
            "global_weight_scale": global_timing.get("global_weight_scale"),
            "base_release_delay_ms": global_timing.get("base_release_delay_ms"),
            "vel_um_per_ms": global_timing.get("vel_um_per_ms"),
        },
        "v_rest_mV": cfg.get("v_rest_mV"),
        "v_init_mV": cfg.get("v_init_mV"),
        "ena_mV": cfg.get("ena_mV"),
        "ek_mV": cfg.get("ek_mV"),
        "el_mV": cfg.get("el_mV"),
        "passive_global": passive_global,
        "pre_soma_hh": dict(cfg.get("pre_soma_hh") or {}),
        "pre_branch_hh": dict(cfg.get("pre_branch_hh") or {}),
        "post_soma_hh": dict(cfg.get("post_soma_hh") or {}),
        "post_branch_hh": dict(cfg.get("post_branch_hh") or {}),
        "post_active": bool(cfg.get("post_active", False)),
        "iclamp_amp_nA": cfg.get("iclamp_amp_nA"),
        "iclamp_delay_ms": cfg.get("iclamp_delay_ms"),
        "iclamp_dur_ms": cfg.get("iclamp_dur_ms"),
        "iclamp_location": cfg.get("iclamp_location"),
        "record": record,
        "gap": gap,
        "parallel": {
            "build_backend": parallel.get("build_backend"),
            "ownership_strategy": parallel.get("ownership_strategy"),
            "maxstep_ms": parallel.get("maxstep_ms"),
        },
    }


def _reuse_is_compatible(existing_cfg: Mapping[str, Any], desired_cfg: Mapping[str, Any]) -> bool:
    return _run_reuse_signature(existing_cfg) == _run_reuse_signature(desired_cfg)


def _ensure_edge_columns(df: pd.DataFrame, *, columns: Sequence[str] = HEMI_EDGE_COLUMNS) -> pd.DataFrame:
    df_use = df.copy()
    for col in columns:
        if col in df_use.columns:
            continue
        if col in {"pre_id", "post_id", "syn_index"}:
            df_use[col] = pd.Series(dtype="Int64")
        else:
            df_use[col] = pd.Series(dtype=float)
    ordered = [str(col) for col in columns]
    extras = [str(col) for col in df_use.columns if str(col) not in ordered]
    return df_use[ordered + extras]


def _read_csv_allow_empty(path: str | Path, *, columns: Sequence[str] | None = None, repair: bool = False) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=list(columns or []))
        if repair:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        return df


def normalize_hemilineage_label(label: str) -> str:
    cleaned = re.sub(r"\s+", "", str(label or "")).upper()
    if not cleaned:
        raise ValueError("hemilineage label must be non-empty")
    return cleaned


def hemilineage_project_name(label: str) -> str:
    return f"Hemi_{normalize_hemilineage_label(label)}"


def _suffix_run_id(base_run_id: str, run_id_suffix: str | None = None) -> str:
    suffix_raw = str(run_id_suffix or "").strip()
    if not suffix_raw:
        return str(base_run_id)
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix_raw).strip("-_.")
    if not suffix:
        return str(base_run_id)
    return f"{base_run_id}_{suffix}"


def coerce_int_ids(values: Iterable[Any], *, name: str = "ids") -> List[int]:
    out: List[int] = []
    seen = set()
    for raw in values:
        val = int(raw)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    if not out:
        raise ValueError(f"{name} must contain at least one neuron id")
    return out


def infer_track_ids(neuron_ids: Sequence[int], seed_ids: Sequence[int], *, limit: int = 8) -> List[int]:
    merged = coerce_int_ids(list(seed_ids) + list(neuron_ids), name="track ids")
    return merged[: max(1, int(limit))]


def infer_latency_pairs(neuron_ids: Sequence[int], seed_ids: Sequence[int], *, limit: int = 8) -> List[Tuple[int, int]]:
    seeds = coerce_int_ids(seed_ids or neuron_ids[:1], name="seed ids")
    targets = [int(x) for x in coerce_int_ids(neuron_ids, name="neuron ids") if int(x) not in set(seeds)]
    pairs: List[Tuple[int, int]] = []
    for seed in seeds:
        for target in targets:
            if seed == target:
                continue
            pairs.append((int(seed), int(target)))
            if len(pairs) >= max(1, int(limit)):
                return pairs
    return pairs


def project_paths(projects_root: str | Path, hemilineage_label: str) -> Dict[str, Path]:
    root = Path(projects_root).expanduser().resolve() / hemilineage_project_name(hemilineage_label)
    paths = {
        "project_root": root,
        "metadata_dir": root / "metadata",
        "edges_dir": root / "edges",
        "runs_root": root / "runs",
        "analysis_dir": root / "analysis",
        "reduction_root": root / "reduction_pipeline",
        "reduction_outputs": root / "reduction_pipeline" / "outputs",
        "reduction_datasets": root / "reduction_pipeline" / "reduced_datasets",
        "coalescing_root": root / "coalescing_pipeline",
        "coalescing_outputs": root / "coalescing_pipeline" / "outputs",
        "optimized_root": root / "optimized_pipeline",
        "optimized_outputs": root / "optimized_pipeline" / "outputs",
        "benchmark_root": root / "simulation_benchmarking",
        "benchmark_projects": root / "simulation_benchmarking" / "projects",
        "benchmark_outputs": root / "simulation_benchmarking" / "outputs",
        "benchmark_figures": root / "simulation_benchmarking" / "outputs" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_master_csv_for_swc_dir(swc_dir: str | Path) -> Path:
    return (Path(swc_dir).expanduser().resolve().parent / "all_neurons_neuroncriteria_template.csv").resolve()


def _load_master_table(master_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(master_csv).expanduser().resolve(), low_memory=False)


def _bodyid_column(df_master: pd.DataFrame) -> str:
    cols = {str(col).lower(): str(col) for col in df_master.columns}
    if "bodyid" not in cols:
        raise ValueError("Master CSV must contain a bodyId column.")
    return cols["bodyid"]


def _motor_neuron_id_set_from_master(master_csv: str | Path) -> set[int]:
    df_master = _load_master_table(master_csv)
    id_col = _bodyid_column(df_master)
    mask = pd.Series(False, index=df_master.index)

    class_like_cols = [col for col in df_master.columns if str(col).lower() in {"class", "class_", "class.1", "superclass"}]
    for col in class_like_cols:
        vals = df_master[col].astype(str).str.strip().str.lower()
        mask = mask | vals.eq("motor neuron")

    if not bool(mask.any()):
        fallback_cols = [col for col in df_master.columns if str(col).lower() in {"instance", "type", "label"}]
        for col in fallback_cols:
            vals = df_master[col].astype(str).str.strip().str.lower()
            mask = mask | vals.str.startswith("mn") | vals.str.contains(" motor neuron", regex=False)

    ids = pd.to_numeric(df_master.loc[mask, id_col], errors="coerce").dropna().astype(int).tolist()
    return set(int(x) for x in ids)


def expand_core_network_to_immediate_motor_postsynaptic(
    core_ids: Sequence[int],
    raw_edges_df: pd.DataFrame,
    *,
    master_csv: str | Path,
) -> Dict[str, Any]:
    expanded = expand_requested_network(
        core_ids,
        raw_edges_df,
        expansion_mode="immediate_motor_postsynaptic",
        master_csv=master_csv,
    )
    core_ids_use = coerce_int_ids(core_ids, name="core neuron ids")
    added_motor_ids = [int(x) for x in expanded["added_ids"]]
    final_network_ids = coerce_int_ids(expanded["final_network_ids"], name="final network ids")
    final_edges_df = _ensure_edge_columns(expanded["final_edges_df"])
    report = {
        "selection_rule": str(expanded["selection_rule"]),
        "core_ids": [int(x) for x in core_ids_use],
        "core_count": int(len(core_ids_use)),
        "added_motor_neuron_ids": [int(x) for x in added_motor_ids],
        "added_motor_neuron_count": int(len(added_motor_ids)),
        "final_network_ids": [int(x) for x in final_network_ids],
        "final_network_count": int(len(final_network_ids)),
        "raw_edge_rows": int(len(raw_edges_df)),
        "final_edge_rows": int(len(final_edges_df)),
    }
    return {
        "core_ids": core_ids_use,
        "added_motor_neuron_ids": added_motor_ids,
        "final_network_ids": final_network_ids,
        "final_edges_df": final_edges_df,
        "report": report,
    }


def build_gap_config_from_run_simulation_preset(
    *,
    swc_dir: str | Path,
    enabled: bool = False,
    mode: str = "rectifying",
    rectify_direction: str = "a_to_b",
    g_uS: float = 0.001,
    default_site: str = "ais",
    all_synapses: bool = True,
    max_synapses: int = 1,
    mechanisms_dir: str | Path | None = None,
    gap_pairs: Sequence[Tuple[int, int]] | None = None,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "mechanisms_dir": str(Path(mechanisms_dir).expanduser().resolve()) if mechanisms_dir else str(Path(swc_dir).expanduser().resolve().parent),
            "default_site": str(default_site),
            "default_g_uS": float(g_uS),
            "pairs": [],
        }

    pairs = list(gap_pairs or RUN_SIMULATION_GAP_PAIRS)
    mode_norm = str(mode or "rectifying").strip().lower()
    if mode_norm not in {"ohmic", "rectifying"}:
        raise ValueError("gap mode must be 'ohmic' or 'rectifying'")

    cfg_pairs: List[Dict[str, Any]] = []
    for a_id, b_id in pairs:
        if mode_norm == "ohmic":
            entry = gap_pair_ohmic(int(a_id), int(b_id), g_uS=float(g_uS), site_a=str(default_site), site_b=str(default_site))
        else:
            entry = gap_pair_rectifying(
                int(a_id),
                int(b_id),
                direction=str(rectify_direction),
                g_uS=float(g_uS),
                site_a=str(default_site),
                site_b=str(default_site),
            )
        entry["placement"] = "synapse"
        entry["all_synapses"] = bool(all_synapses)
        entry["max_synapses"] = int(max_synapses)
        cfg_pairs.append(entry)

    return {
        "enabled": True,
        "mechanisms_dir": str(Path(mechanisms_dir).expanduser().resolve()) if mechanisms_dir else str(Path(swc_dir).expanduser().resolve().parent),
        "default_site": str(default_site),
        "default_g_uS": float(g_uS),
        "pairs": cfg_pairs,
    }


def build_edges_for_hemilineage_ids(
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    *,
    swc_dir: str | Path,
    project_root: str | Path,
    edge_set_name: str | None = None,
    edges_registry_root: str | Path | None = None,
    master_csv: str | Path | None = None,
    neuprint_dataset: str | None = None,
    seeds: Sequence[int] | None = None,
    default_weight_uS: float = 6e-6,
    one_row_per_synapse: bool = True,
    workers: int = 16,
    force_rebuild: bool = False,
    phase1_fallback_enabled: bool = True,
    phase1_upsample_nm: float = 2000.0,
    phase1_min_conf: float = 0.4,
    phase1_batch_size: int = 10000,
    phase1_export_workers: int = 1,
    phase1_progress_every: int = 25,
) -> Tuple[Path, pd.DataFrame]:
    label = normalize_hemilineage_label(hemilineage_label)
    neuprint_dataset_use = normalize_neuprint_dataset(neuprint_dataset or DEFAULT_NEUPRINT_DATASET)
    project_root = Path(project_root).expanduser().resolve()
    edges_dir = (project_root / "edges").resolve()
    metadata_dir = (project_root / "metadata").resolve()
    edges_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_edges_path = (edges_dir / f"{hemilineage_project_name(label).lower()}_core_presyn_edges.csv").resolve()
    edges_path = (edges_dir / f"{hemilineage_project_name(label).lower()}_final_network_edges.csv").resolve()
    report_json = (metadata_dir / "network_build_report.json").resolve()
    final_ids_csv = (metadata_dir / "final_network_ids.csv").resolve()
    added_motor_csv = (metadata_dir / "added_motor_neuron_ids.csv").resolve()

    if edges_path.exists() and report_json.exists() and not force_rebuild:
        report_existing = _read_json(report_json)
        report_dataset = normalize_neuprint_dataset(report_existing.get("neuprint_dataset") or DEFAULT_NEUPRINT_DATASET)
        if report_dataset == neuprint_dataset_use:
            canonical_edges_path = Path(report_existing.get("final_edges_path", edges_path)).expanduser().resolve()
            if not canonical_edges_path.exists():
                canonical_edges_path = edges_path
            return canonical_edges_path, _read_csv_allow_empty(canonical_edges_path, columns=HEMI_EDGE_COLUMNS, repair=True)

    ids = coerce_int_ids(neuron_ids, name="neuron ids")
    master_csv_use = Path(master_csv).expanduser().resolve() if master_csv is not None else default_master_csv_for_swc_dir(swc_dir)
    edge_request = ensure_named_edge_set(
        edge_set_name=edge_set_name or hemilineage_project_name(label),
        requested_ids=ids,
        swc_dir=swc_dir,
        selection_mode="hemilineage",
        selection_label=label,
        expansion_mode="immediate_motor_postsynaptic",
        master_csv=master_csv_use,
        neuprint_dataset=neuprint_dataset_use,
        registry_root=edges_registry_root or default_edges_registry_root(),
        force_rebuild=bool(force_rebuild),
        default_weight_uS=float(default_weight_uS),
        one_row_per_synapse=bool(one_row_per_synapse),
        workers=max(1, int(workers)),
        phase1_fallback_enabled=bool(phase1_fallback_enabled),
        phase1_upsample_nm=float(phase1_upsample_nm),
        phase1_min_conf=float(phase1_min_conf),
        phase1_batch_size=int(phase1_batch_size),
        phase1_export_workers=int(phase1_export_workers),
        phase1_progress_every=int(phase1_progress_every),
    )

    df_edges_raw = _ensure_edge_columns(edge_request["raw_edges_df"])
    df_edges_final = _ensure_edge_columns(edge_request["final_edges_df"])
    raw_edges_path.write_bytes(Path(edge_request["raw_edges_path"]).read_bytes())
    edges_path.write_bytes(Path(edge_request["final_edges_path"]).read_bytes())

    expansion = {
        "report": {
            "selection_rule": str(edge_request["selection_rule"]),
            "core_ids": [int(x) for x in ids],
            "core_count": int(len(ids)),
            "added_motor_neuron_ids": [int(x) for x in edge_request["added_ids"]],
            "added_motor_neuron_count": int(len(edge_request["added_ids"])),
            "final_network_ids": [int(x) for x in edge_request["final_network_ids"]],
            "final_network_count": int(len(edge_request["final_network_ids"])),
            "raw_edge_rows": int(len(df_edges_raw)),
            "final_edge_rows": int(len(df_edges_final)),
            "edge_set_name": str(edge_request["edge_set_name"]),
            "edge_set_slug": str(edge_request["edge_set_slug"]),
            "edge_signature": str(edge_request["edge_signature"]),
            "neuprint_dataset": str(edge_request.get("neuprint_dataset", "")),
            "edges_registry_root": str(edge_request["registry_root"]),
            "phase1_report": edge_request.get("phase1_report", {}),
            "reused_existing_edge_set": bool(edge_request.get("reused_existing", False)),
        },
        "final_network_ids": [int(x) for x in edge_request["final_network_ids"]],
        "added_motor_neuron_ids": [int(x) for x in edge_request["added_ids"]],
    }
    _write_json(
        report_json,
        {
            **expansion["report"],
            "hemilineage": label,
            "master_csv": str(master_csv_use),
            "project_raw_edges_path": str(raw_edges_path),
            "project_final_edges_path": str(edges_path),
            "raw_edges_path": str(Path(edge_request["raw_edges_path"]).expanduser().resolve()),
            "final_edges_path": str(Path(edge_request["final_edges_path"]).expanduser().resolve()),
            "edge_metadata_json_path": str(Path(edge_request["metadata_json_path"]).expanduser().resolve()),
        },
    )
    pd.DataFrame({"neuron_id": expansion["final_network_ids"]}).to_csv(final_ids_csv, index=False)
    pd.DataFrame({"neuron_id": expansion["added_motor_neuron_ids"]}).to_csv(added_motor_csv, index=False)
    return Path(edge_request["final_edges_path"]).expanduser().resolve(), df_edges_final


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def load_phase_timing_summary(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    phase_json = out_dir / "_phase_timings.json"
    if phase_json.exists():
        payload = _read_json(phase_json)
        payload.setdefault("build_wall_s", payload.get("pre_sim_wall_s"))
        payload.setdefault("pre_sim_wall_s", payload.get("build_wall_s"))
        payload.setdefault("sim_wall_s", float("nan"))
        payload.setdefault("post_sim_save_wall_s", float("nan"))
        payload.setdefault("total_wall_s", float("nan"))
        payload.setdefault("phase_rows", [])
        payload["out_dir"] = str(out_dir)
        return payload

    fallback = float("nan")
    for name in sorted(out_dir.glob("*timing*.json")):
        try:
            fallback = float(_read_json(name).get("total_wall_s", float("nan")))
            break
        except Exception:
            continue
    return {
        "out_dir": str(out_dir),
        "build_wall_s": float("nan"),
        "pre_sim_wall_s": float("nan"),
        "sim_wall_s": float("nan"),
        "post_sim_save_wall_s": float("nan"),
        "total_wall_s": float(fallback),
        "backend": None,
        "integrator": None,
        "phase_rows": [],
    }


def _core_connectivity_edges(raw_edges_df: pd.DataFrame, core_ids: Sequence[int]) -> pd.DataFrame:
    if raw_edges_df.empty:
        return pd.DataFrame(columns=list(raw_edges_df.columns))
    if not {"pre_id", "post_id"}.issubset(raw_edges_df.columns):
        raise ValueError("raw edges table must include pre_id and post_id columns")
    core_set = set(coerce_int_ids(core_ids, name="core neuron ids"))
    pre = pd.to_numeric(raw_edges_df["pre_id"], errors="coerce")
    post = pd.to_numeric(raw_edges_df["post_id"], errors="coerce")
    mask = pre.isin(list(core_set)) & post.isin(list(core_set)) & pre.notna() & post.notna() & (pre != post)
    df = raw_edges_df.loc[mask].copy()
    if not df.empty:
        df["pre_id"] = pd.to_numeric(df["pre_id"], errors="coerce").astype(int)
        df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce").astype(int)
    return df


def build_connected_core_benchmark_subsets(
    core_ids: Sequence[int],
    raw_edges_df: pd.DataFrame,
    *,
    min_size: int = 2,
    max_size: int = 20,
) -> Dict[str, Any]:
    core_use = coerce_int_ids(core_ids, name="core neuron ids")
    min_size_use = max(2, int(min_size))
    max_size_use = max(min_size_use, int(max_size))
    df_core = _core_connectivity_edges(raw_edges_df, core_use)
    if df_core.empty:
        raise RuntimeError("No core-to-core connectivity edges were found for benchmark subset selection.")

    adjacency: Dict[int, set[int]] = {int(nid): set() for nid in core_use}
    for row in df_core[["pre_id", "post_id"]].itertuples(index=False):
        a = int(row.pre_id)
        b = int(row.post_id)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    components: List[List[int]] = []
    seen: set[int] = set()
    candidate_nodes = [nid for nid in core_use if adjacency.get(int(nid))]
    for nid in candidate_nodes:
        if int(nid) in seen:
            continue
        stack = [int(nid)]
        component: List[int] = []
        seen.add(int(nid))
        while stack:
            cur = stack.pop()
            component.append(int(cur))
            neighbors = sorted(adjacency.get(int(cur), set()), key=lambda x: (-len(adjacency.get(int(x), set())), int(x)))
            for nxt in neighbors:
                if int(nxt) in seen:
                    continue
                seen.add(int(nxt))
                stack.append(int(nxt))
        components.append(sorted(component))

    if not components:
        raise RuntimeError("Could not find any connected benchmark subset inside the supplied core ids.")

    components.sort(key=lambda comp: (-len(comp), -sum(len(adjacency.get(int(nid), set())) for nid in comp), min(comp)))
    largest_component = list(components[0])
    if len(largest_component) < min_size_use:
        raise RuntimeError(
            f"Largest connected core component has only {len(largest_component)} neurons; need at least {min_size_use}."
        )

    start = min(
        largest_component,
        key=lambda nid: (-len(adjacency.get(int(nid), set())), int(nid)),
    )
    ordered: List[int] = []
    queue: List[int] = [int(start)]
    visited: set[int] = {int(start)}
    while queue:
        cur = queue.pop(0)
        ordered.append(int(cur))
        neighbors = sorted(
            [int(nid) for nid in adjacency.get(int(cur), set()) if int(nid) in set(largest_component)],
            key=lambda nid: (-len(adjacency.get(int(nid), set())), int(nid)),
        )
        for nxt in neighbors:
            if int(nxt) in visited:
                continue
            visited.add(int(nxt))
            queue.append(int(nxt))

    size_stop = min(max_size_use, len(ordered))
    subsets = [ordered[:size] for size in range(min_size_use, size_stop + 1)]
    if not subsets:
        raise RuntimeError("No benchmark subsets could be generated with the requested size range.")

    subset_rows = []
    for subset in subsets:
        subset_set = set(int(x) for x in subset)
        edge_rows = int(
            ((df_core["pre_id"].astype(int).isin(list(subset_set))) & (df_core["post_id"].astype(int).isin(list(subset_set)))).sum()
        )
        subset_rows.append(
            {
                "core_subset_size": int(len(subset)),
                "selected_core_ids": [int(x) for x in subset],
                "core_internal_edge_rows": edge_rows,
            }
        )

    return {
        "ordered_core_ids": [int(x) for x in ordered],
        "largest_component_ids": [int(x) for x in largest_component],
        "largest_component_size": int(len(largest_component)),
        "core_connectivity_edge_rows": int(len(df_core)),
        "subset_rows": subset_rows,
    }


def _load_template_config(template_config_path: str | Path | None) -> Dict[str, Any]:
    if template_config_path is None:
        return {}
    path = Path(template_config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"template config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_master_config(
    *,
    swc_dir: str | Path,
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    edges_path: str | Path,
    project_root: str | Path,
    seeds: Sequence[int] | None = None,
    run_id: str,
    master_csv: str | Path | None = None,
    template_config_path: str | Path | None = None,
    morph_swc_dir: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    biophysics: Mapping[str, Any] | None = None,
    stim: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
    gap: Mapping[str, Any] | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
    record_soma_for_all: bool = True,
    run_notes: str = "",
) -> Dict[str, Any]:
    base = _load_template_config(template_config_path)
    ids = coerce_int_ids(neuron_ids, name="neuron ids")
    seed_ids = coerce_int_ids(seeds or ids, name="seed ids")
    project_root = Path(project_root).expanduser().resolve()
    swc_dir = Path(swc_dir).expanduser().resolve()

    runtime_cfg = dict(runtime or {})
    timing_cfg = dict(timing or {})
    biophys_cfg = dict(biophysics or {})
    stim_cfg = dict(stim or {})
    record_cfg = dict(record or {})
    gap_cfg = dict(gap or {})

    cfg = copy.deepcopy(base)
    cfg["swc_dir"] = str(swc_dir)
    cfg["edges_root"] = str((project_root / "edges").resolve())
    cfg["runs_root"] = str((project_root / "runs").resolve())
    cfg["run_id"] = str(run_id)
    cfg["run_notes"] = str(run_notes or "")
    cfg["master_csv"] = str(Path(master_csv).expanduser().resolve()) if master_csv else None
    cfg["morph_swc_dir"] = str(Path(morph_swc_dir).expanduser().resolve()) if morph_swc_dir else None
    cfg["selection"] = {
        "mode": "custom",
        "label": normalize_hemilineage_label(hemilineage_label),
        "neuron_id": None,
        "neuron_ids": ids,
    }
    cfg["seeds"] = seed_ids
    cfg["edges_path"] = str(Path(edges_path).expanduser().resolve())

    record_final = apply_recording_policy(record_cfg, ids, seed_ids, record_soma_for_all=bool(record_soma_for_all))
    if "soma_v" not in record_final:
        record_final["soma_v"] = list(ids) if bool(record_soma_for_all) else list(seed_ids)
    if "spike_thresh_mV" not in record_final:
        record_final["spike_thresh_mV"] = 20.0
    cfg["record"] = record_final

    cfg["tstop_ms"] = float(runtime_cfg.get("tstop_ms", cfg.get("tstop_ms", 10.0)))
    cfg["dt_ms"] = float(runtime_cfg.get("dt_ms", cfg.get("dt_ms", 0.025)))
    cfg["celsius_C"] = float(runtime_cfg.get("celsius_C", cfg.get("celsius_C", 22.0)))
    cfg["progress"] = bool(runtime_cfg.get("progress", cfg.get("progress", True)))
    cfg["use_tqdm"] = bool(runtime_cfg.get("use_tqdm", cfg.get("use_tqdm", True)))
    cfg["progress_chunk_ms"] = float(runtime_cfg.get("progress_chunk_ms", cfg.get("progress_chunk_ms", 0.5)))
    cfg["enable_coreneuron"] = bool(runtime_cfg.get("enable_coreneuron", cfg.get("enable_coreneuron", True)))
    cfg["coreneuron_gpu"] = bool(runtime_cfg.get("coreneuron_gpu", cfg.get("coreneuron_gpu", True)))
    if runtime_cfg.get("threads", None) is not None:
        cfg["threads"] = int(runtime_cfg["threads"])
    if runtime_cfg.get("coreneuron_nthread", None) is not None:
        cfg["coreneuron_nthread"] = int(runtime_cfg["coreneuron_nthread"])
    if runtime_cfg.get("io_workers", None) is not None:
        cfg["io_workers"] = int(runtime_cfg["io_workers"])
    cfg["cvode"] = {
        "enabled": bool(runtime_cfg.get("cvode_enabled", (cfg.get("cvode") or {}).get("enabled", False))),
        "atol": float(runtime_cfg.get("cvode_atol", (cfg.get("cvode") or {}).get("atol", 1e-3))),
        "rtol": runtime_cfg.get("cvode_rtol", (cfg.get("cvode") or {}).get("rtol")),
        "maxstep_ms": runtime_cfg.get("cvode_maxstep_ms", (cfg.get("cvode") or {}).get("maxstep_ms")),
    }
    if cfg["enable_coreneuron"] and bool((cfg.get("cvode") or {}).get("enabled", False)):
        cfg["cvode"] = {"enabled": False}

    cfg["wire_force_soma"] = bool(timing_cfg.get("wire_force_soma", cfg.get("wire_force_soma", False)))
    cfg["coalesce_syns"] = bool(timing_cfg.get("coalesce_syns", cfg.get("coalesce_syns", False)))
    cfg["coalesce_mode"] = str(timing_cfg.get("coalesce_mode", cfg.get("coalesce_mode", "none")))
    cfg["min_weight_uS"] = float(timing_cfg.get("min_weight_uS", cfg.get("min_weight_uS", 0.0)))
    cfg["default_weight_uS"] = float(timing_cfg.get("default_weight_uS", cfg.get("default_weight_uS", 6e-6)))
    cfg["default_delay_ms"] = float(timing_cfg.get("default_delay_ms", cfg.get("default_delay_ms", 1.0)))
    cfg["global_weight_scale"] = float(timing_cfg.get("global_weight_scale", cfg.get("global_weight_scale", 1.0)))
    cfg["global_base_release_delay_ms"] = float(
        timing_cfg.get("global_base_release_delay_ms", cfg.get("global_base_release_delay_ms", 0.40))
    )
    cfg["global_vel_um_per_ms"] = float(
        timing_cfg.get("global_vel_um_per_ms", cfg.get("global_vel_um_per_ms", 1500.0))
    )
    cfg["use_geom_delay"] = bool(timing_cfg.get("use_geom_delay", cfg.get("use_geom_delay", True)))
    cfg["syn_tau1_ms"] = float(timing_cfg.get("syn_tau1_ms", cfg.get("syn_tau1_ms", 0.5)))
    cfg["syn_tau2_ms"] = float(timing_cfg.get("syn_tau2_ms", cfg.get("syn_tau2_ms", 3.0)))
    cfg["syn_e_rev_mV"] = float(timing_cfg.get("syn_e_rev_mV", cfg.get("syn_e_rev_mV", 0.0)))
    cfg["epsilon_um"] = timing_cfg.get("epsilon_um", cfg.get("epsilon_um"))
    if timing_cfg.get("coalesce_guard_w_med_uS") is not None:
        cfg["coalesce_guard_w_med_uS"] = float(timing_cfg["coalesce_guard_w_med_uS"])
    if timing_cfg.get("coalesce_guard_drop") is not None:
        cfg["coalesce_guard_drop"] = float(timing_cfg["coalesce_guard_drop"])

    cfg["v_rest_mV"] = float(biophys_cfg.get("v_rest_mV", cfg.get("v_rest_mV", -65.0)))
    cfg["v_init_mV"] = float(biophys_cfg.get("v_init_mV", cfg.get("v_init_mV", cfg["v_rest_mV"])))
    cfg["ena_mV"] = float(biophys_cfg.get("ena_mV", cfg.get("ena_mV", 65.0)))
    cfg["ek_mV"] = float(biophys_cfg.get("ek_mV", cfg.get("ek_mV", -74.0)))
    cfg["el_mV"] = float(biophys_cfg.get("el_mV", cfg.get("el_mV", cfg["v_rest_mV"])))
    if biophys_cfg.get("passive_global") is not None:
        cfg["passive_global"] = copy.deepcopy(dict(biophys_cfg["passive_global"]))
    if biophys_cfg.get("hh_global") is not None:
        cfg["hh_global"] = copy.deepcopy(biophys_cfg["hh_global"])
    if biophys_cfg.get("pre_soma_hh") is not None:
        cfg["pre_soma_hh"] = copy.deepcopy(dict(biophys_cfg["pre_soma_hh"]))
    if biophys_cfg.get("pre_branch_hh") is not None:
        cfg["pre_branch_hh"] = copy.deepcopy(dict(biophys_cfg["pre_branch_hh"]))
    if biophys_cfg.get("post_soma_hh") is not None:
        cfg["post_soma_hh"] = copy.deepcopy(dict(biophys_cfg["post_soma_hh"]))
    if biophys_cfg.get("post_branch_hh") is not None:
        cfg["post_branch_hh"] = copy.deepcopy(dict(biophys_cfg["post_branch_hh"]))
    if biophys_cfg.get("post_active") is not None:
        cfg["post_active"] = bool(biophys_cfg["post_active"])

    cfg["stim"] = {
        "iclamp": copy.deepcopy(dict(stim_cfg.get("iclamp") or {})),
        "neg_pulse": copy.deepcopy(dict(stim_cfg.get("neg_pulse") or {})),
        "pulse_train": copy.deepcopy(dict(stim_cfg.get("pulse_train") or {})),
    }

    cfg["gap"] = copy.deepcopy(gap_cfg)
    if extra_overrides:
        cfg = _deep_merge(cfg, extra_overrides)

    return build_config(cfg, strict=True)


def _load_records(out_dir: str | Path) -> pd.DataFrame:
    path = Path(out_dir).expanduser().resolve() / "records.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing records.csv: {path}")
    return pd.read_csv(path)


def _load_spike_times(out_dir: str | Path) -> pd.DataFrame:
    base = Path(out_dir).expanduser().resolve()
    master = base / "spike_times.csv"
    legacy = base / "spikes.csv"
    if master.exists():
        df = pd.read_csv(master)
        if {"neuron_id", "spike_time_ms"}.issubset(df.columns):
            return df[["neuron_id", "spike_time_ms"]].copy()
    if legacy.exists():
        df = pd.read_csv(legacy)
        if {"nid", "t_ms"}.issubset(df.columns):
            return df.rename(columns={"nid": "neuron_id", "t_ms": "spike_time_ms"})[["neuron_id", "spike_time_ms"]]
    return pd.DataFrame(columns=["neuron_id", "spike_time_ms"])


def _soma_col(df: pd.DataFrame, nid: int) -> str | None:
    exact = f"{int(nid)}_soma_v"
    if exact in df.columns:
        return exact
    pat = re.compile(rf"^{int(nid)}_soma_v(__\d+)?$")
    for col in df.columns:
        if pat.match(col):
            return col
    return None


def _first_spike(df_sp: pd.DataFrame, nid: int) -> float:
    if df_sp.empty:
        return float("nan")
    sub = df_sp.loc[pd.to_numeric(df_sp["neuron_id"], errors="coerce") == int(nid), "spike_time_ms"]
    arr = pd.to_numeric(sub, errors="coerce").dropna().to_numpy(float)
    return float(np.min(arr)) if arr.size else float("nan")


def _trace_rmse(base_t: Sequence[Any], base_v: Sequence[Any], test_t: Sequence[Any], test_v: Sequence[Any]) -> float:
    if len(base_t) < 2 or len(test_t) < 2:
        return float("nan")
    vb = pd.to_numeric(pd.Series(base_v), errors="coerce").to_numpy(float)
    vt = pd.to_numeric(pd.Series(test_v), errors="coerce").to_numpy(float)
    tb = pd.to_numeric(pd.Series(base_t), errors="coerce").to_numpy(float)
    tt = pd.to_numeric(pd.Series(test_t), errors="coerce").to_numpy(float)
    okb = np.isfinite(tb) & np.isfinite(vb)
    okt = np.isfinite(tt) & np.isfinite(vt)
    if okb.sum() < 2 or okt.sum() < 2:
        return float("nan")
    vt_interp = np.interp(tb[okb], tt[okt], vt[okt])
    diff = vt_interp - vb[okb]
    return float(np.sqrt(np.mean(diff * diff)))


def summarize_run(
    out_dir: str | Path,
    *,
    track_ids: Sequence[int],
    latency_pairs: Sequence[Tuple[int, int]],
    prefix: str = "_run",
) -> Dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    records = _load_records(out_dir)
    spikes = _load_spike_times(out_dir)
    t = pd.to_numeric(records.get("t_ms"), errors="coerce").to_numpy(float)

    rows = []
    for nid in coerce_int_ids(track_ids, name="track ids"):
        col = _soma_col(records, int(nid))
        if not col:
            rows.append({"nid": int(nid), "status": "missing_trace"})
            continue
        v = pd.to_numeric(records[col], errors="coerce").to_numpy(float)
        finite = np.isfinite(v)
        first_spike_ms = _first_spike(spikes, int(nid))
        rows.append(
            {
                "nid": int(nid),
                "status": "ok",
                "trace_col": col,
                "peak_mV": float(np.nanmax(v)) if finite.any() else float("nan"),
                "min_mV": float(np.nanmin(v)) if finite.any() else float("nan"),
                "mean_mV": float(np.nanmean(v)) if finite.any() else float("nan"),
                "first_spike_ms": first_spike_ms,
                "n_spikes": int((pd.to_numeric(spikes.get("neuron_id"), errors="coerce") == int(nid)).sum()),
                "trace_samples": int(np.isfinite(t).sum()),
            }
        )
    neuron_df = pd.DataFrame(rows)

    lat_rows = []
    for pre_id, post_id in latency_pairs:
        pre_t = _first_spike(spikes, int(pre_id))
        post_t = _first_spike(spikes, int(post_id))
        lat_rows.append(
            {
                "pair": f"{int(pre_id)}->{int(post_id)}",
                "pre_first_spike_ms": pre_t,
                "post_first_spike_ms": post_t,
                "latency_ms": float(post_t - pre_t) if np.isfinite(pre_t) and np.isfinite(post_t) else float("nan"),
            }
        )
    lat_df = pd.DataFrame(lat_rows)

    neuron_csv = out_dir / f"{prefix}_neuron_summary.csv"
    latency_csv = out_dir / f"{prefix}_latency_summary.csv"
    summary_json = out_dir / f"{prefix}_summary.json"

    neuron_df.to_csv(neuron_csv, index=False)
    lat_df.to_csv(latency_csv, index=False)
    summary = {
        "run_dir": str(out_dir),
        "track_ids": [int(x) for x in track_ids],
        "latency_pairs": [[int(a), int(b)] for a, b in latency_pairs],
        "peak_mV_max": float(pd.to_numeric(neuron_df.get("peak_mV"), errors="coerce").max()) if not neuron_df.empty else float("nan"),
        "first_spike_min_ms": float(pd.to_numeric(neuron_df.get("first_spike_ms"), errors="coerce").min()) if not neuron_df.empty else float("nan"),
        "max_latency_ms": float(pd.to_numeric(lat_df.get("latency_ms"), errors="coerce").max()) if not lat_df.empty else float("nan"),
    }
    _write_json(summary_json, summary)
    return {
        "neuron_df": neuron_df,
        "latency_df": lat_df,
        "summary": summary,
        "neuron_csv": neuron_csv,
        "latency_csv": latency_csv,
        "summary_json": summary_json,
    }


def compare_runs(
    baseline_dir: str | Path,
    test_dir: str | Path,
    *,
    track_ids: Sequence[int],
    latency_pairs: Sequence[Tuple[int, int]],
    thresholds: Mapping[str, float] | None = None,
    output_dir: str | Path | None = None,
    stem: str = "comparison",
) -> Dict[str, Any]:
    thresholds_use = dict(DEFAULT_THRESHOLDS)
    thresholds_use.update(dict(thresholds or {}))

    baseline_dir = Path(baseline_dir).expanduser().resolve()
    test_dir = Path(test_dir).expanduser().resolve()

    base_rec = _load_records(baseline_dir)
    test_rec = _load_records(test_dir)
    base_sp = _load_spike_times(baseline_dir)
    test_sp = _load_spike_times(test_dir)

    base_t = pd.to_numeric(base_rec.get("t_ms"), errors="coerce").to_numpy(float)
    test_t = pd.to_numeric(test_rec.get("t_ms"), errors="coerce").to_numpy(float)

    rows = []
    for nid in coerce_int_ids(track_ids, name="track ids"):
        cb = _soma_col(base_rec, int(nid))
        ct = _soma_col(test_rec, int(nid))
        if not cb or not ct:
            rows.append({"nid": int(nid), "status": "missing_trace", "base_col": cb, "test_col": ct})
            continue
        vb = pd.to_numeric(base_rec[cb], errors="coerce").to_numpy(float)
        vt = pd.to_numeric(test_rec[ct], errors="coerce").to_numpy(float)
        peak_b = float(np.nanmax(vb)) if np.isfinite(vb).any() else float("nan")
        peak_t = float(np.nanmax(vt)) if np.isfinite(vt).any() else float("nan")
        fs_b = _first_spike(base_sp, int(nid))
        fs_t = _first_spike(test_sp, int(nid))
        rows.append(
            {
                "nid": int(nid),
                "status": "ok",
                "peak_base_mV": peak_b,
                "peak_test_mV": peak_t,
                "peak_abs_diff_mV": float(abs(peak_t - peak_b)) if np.isfinite(peak_b) and np.isfinite(peak_t) else float("nan"),
                "first_spike_base_ms": fs_b,
                "first_spike_test_ms": fs_t,
                "first_spike_abs_diff_ms": float(abs(fs_t - fs_b)) if np.isfinite(fs_b) and np.isfinite(fs_t) else float("nan"),
                "trace_rmse_mV": _trace_rmse(base_t, vb, test_t, vt),
            }
        )
    neuron_cmp = pd.DataFrame(rows)

    lat_rows = []
    for pre_id, post_id in latency_pairs:
        b_pre = _first_spike(base_sp, int(pre_id))
        b_post = _first_spike(base_sp, int(post_id))
        t_pre = _first_spike(test_sp, int(pre_id))
        t_post = _first_spike(test_sp, int(post_id))
        b_lat = float(b_post - b_pre) if np.isfinite(b_pre) and np.isfinite(b_post) else float("nan")
        t_lat = float(t_post - t_pre) if np.isfinite(t_pre) and np.isfinite(t_post) else float("nan")
        lat_rows.append(
            {
                "pair": f"{int(pre_id)}->{int(post_id)}",
                "latency_base_ms": b_lat,
                "latency_test_ms": t_lat,
                "latency_abs_diff_ms": float(abs(t_lat - b_lat)) if np.isfinite(b_lat) and np.isfinite(t_lat) else float("nan"),
            }
        )
    latency_cmp = pd.DataFrame(lat_rows)

    metrics = {
        "fail_first_spike": int((pd.to_numeric(neuron_cmp.get("first_spike_abs_diff_ms", pd.Series(dtype=float)), errors="coerce") > thresholds_use["first_spike_ms"]).fillna(False).sum()),
        "fail_peak": int((pd.to_numeric(neuron_cmp.get("peak_abs_diff_mV", pd.Series(dtype=float)), errors="coerce") > thresholds_use["peak_mV"]).fillna(False).sum()),
        "fail_rmse": int((pd.to_numeric(neuron_cmp.get("trace_rmse_mV", pd.Series(dtype=float)), errors="coerce") > thresholds_use["rmse_mV"]).fillna(False).sum()),
        "fail_latency": int((pd.to_numeric(latency_cmp.get("latency_abs_diff_ms", pd.Series(dtype=float)), errors="coerce") > thresholds_use["latency_ms"]).fillna(False).sum()),
        "max_peak_abs_diff_mV": float(pd.to_numeric(neuron_cmp.get("peak_abs_diff_mV", pd.Series(dtype=float)), errors="coerce").max()) if not neuron_cmp.empty else float("nan"),
        "max_trace_rmse_mV": float(pd.to_numeric(neuron_cmp.get("trace_rmse_mV", pd.Series(dtype=float)), errors="coerce").max()) if not neuron_cmp.empty else float("nan"),
        "max_first_spike_abs_diff_ms": float(pd.to_numeric(neuron_cmp.get("first_spike_abs_diff_ms", pd.Series(dtype=float)), errors="coerce").max()) if not neuron_cmp.empty else float("nan"),
        "max_latency_abs_diff_ms": float(pd.to_numeric(latency_cmp.get("latency_abs_diff_ms", pd.Series(dtype=float)), errors="coerce").max()) if not latency_cmp.empty else float("nan"),
    }
    metrics["fail_total"] = int(metrics["fail_first_spike"] + metrics["fail_peak"] + metrics["fail_rmse"] + metrics["fail_latency"])

    out_payload = {
        "neuron_cmp": neuron_cmp,
        "latency_cmp": latency_cmp,
        "metrics": metrics,
    }
    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        neuron_csv = output_dir / f"{stem}_neuron_cmp.csv"
        latency_csv = output_dir / f"{stem}_latency_cmp.csv"
        summary_json = output_dir / f"{stem}_summary.json"
        neuron_cmp.to_csv(neuron_csv, index=False)
        latency_cmp.to_csv(latency_csv, index=False)
        _write_json(summary_json, metrics)
        out_payload.update(
            {
                "neuron_csv": neuron_csv,
                "latency_csv": latency_csv,
                "summary_json": summary_json,
            }
        )
    return out_payload


def _run_or_reuse(cfg: Mapping[str, Any], *, timing_filename: str, force: bool = False) -> Tuple[Path, float]:
    built = build_config(copy.deepcopy(dict(cfg)), strict=True)
    out_dir = (Path(built["runs_root"]).expanduser().resolve() / str(built["run_id"])).resolve()
    records = out_dir / "records.csv"
    spikes = out_dir / "spike_times.csv"
    config_json = out_dir / "config.json"

    if (not force) and records.exists() and spikes.exists():
        if config_json.exists():
            try:
                existing_cfg = json.loads(config_json.read_text(encoding="utf-8"))
                if not _reuse_is_compatible(existing_cfg, built):
                    print(
                        f"[hemi] existing run '{built['run_id']}' does not match current config; rerunning instead of reusing"
                    )
                    force = True
            except Exception as exc:
                print(
                    f"[hemi] could not validate existing run config for '{built['run_id']}' ({exc}); rerunning"
                )
                force = True
        if not force:
            wall_s = float("nan")
            timing_json = out_dir / timing_filename
            if timing_json.exists():
                try:
                    wall_s = float(json.loads(timing_json.read_text(encoding="utf-8")).get("total_wall_s", float("nan")))
                except Exception:
                    wall_s = float("nan")
            return out_dir, wall_s

    t0 = time.perf_counter()
    out = Path(run_walking_simulation(built)).expanduser().resolve()
    wall_s = time.perf_counter() - t0
    _write_json(out / timing_filename, {"total_wall_s": float(wall_s)})
    return out, float(wall_s)


def _read_edges_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return _read_csv_allow_empty(path, columns=HEMI_EDGE_COLUMNS, repair=False)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if sfx in {".feather", ".ftr"}:
        return pd.read_feather(path)
    return _read_csv_allow_empty(path, columns=HEMI_EDGE_COLUMNS, repair=False)


def estimate_wiring_rows(cfg_in: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = build_config(copy.deepcopy(dict(cfg_in)), strict=True)
    edges_path = Path(cfg.get("edges_path", "")).expanduser().resolve()
    if not edges_path.exists():
        raise FileNotFoundError(f"edges_path not found for estimate: {edges_path}")

    df = _read_edges_table(edges_path)
    for col in (
        "pre_id",
        "post_id",
        "pre_syn_index",
        "post_syn_index",
        "pre_match_um",
        "post_match_um",
        "weight_uS",
        "weight",
        "delay_ms",
        "e_rev_mV",
        "tau1_ms",
        "tau2_ms",
        "post_x",
        "post_y",
        "post_z",
        "x_post",
        "y_post",
        "z_post",
        "pre_x",
        "pre_y",
        "pre_z",
        "x_pre",
        "y_pre",
        "z_pre",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"pre_match_um", "post_match_um"}.issubset(df.columns) and cfg.get("epsilon_um") is not None:
        eps = float(cfg["epsilon_um"])
        df = df[(df.pre_match_um <= eps) & (df.post_match_um <= eps)].copy()

    selection_ids = [int(x) for x in ((cfg.get("selection") or {}).get("neuron_ids") or [])]
    seeds = [int(x) for x in (cfg.get("seeds") or [])]
    nodes = set(selection_ids) | set(seeds)

    pre_filter = str(cfg.get("custom_pre_filter", "all_loaded")).strip().lower()
    allowed_pres = set(seeds) if pre_filter in {"drivers", "seed_only", "seeds", "drivers_only"} else set(nodes)

    df_sub = df[df["pre_id"].astype(int).isin(allowed_pres) & df["post_id"].astype(int).isin(nodes)].copy()

    gt = cfg.get("global_timing", {}) or {}
    allow_legacy_weight = bool(cfg.get("allow_legacy_weight_column", False) or gt.get("allow_legacy_weight_column", False))
    weight_candidates = [gt.get("weight_col"), "weight_uS", "w_uS"]
    if allow_legacy_weight:
        weight_candidates.append("weight")
    delay_candidates = [gt.get("delay_col"), "delay_ms", "delay", "d_ms"]
    erev_candidates = [gt.get("erev_col"), "syn_e_rev_mV", "erev_mV", "e_rev_mV", "syn_erev_mV"]
    tau1_candidates = [gt.get("tau1_col"), "tau1_ms", "tau1"]
    tau2_candidates = [gt.get("tau2_col"), "tau2_ms", "tau2"]

    df_wire = _prepare_wiring_df(
        df_sub,
        cfg,
        force_soma=bool(cfg.get("wire_force_soma", False)),
        w_col=_first_present_col(df_sub, weight_candidates),
        d_col=_first_present_col(df_sub, delay_candidates),
        e_col=_first_present_col(df_sub, erev_candidates),
        t1_col=_first_present_col(df_sub, tau1_candidates),
        t2_col=_first_present_col(df_sub, tau2_candidates),
        w_default=cfg.get("default_weight_uS", None),
        d_default=cfg.get("default_delay_ms", None),
        e_default=cfg.get("syn_e_rev_mV", None),
        t1_default=cfg.get("syn_tau1_ms", None),
        t2_default=cfg.get("syn_tau2_ms", None),
        w_scale=float((cfg.get("global_timing", {}) or {}).get("global_weight_scale", 1.0)),
    )

    raw_rows = int(len(df_sub))
    effective_rows = int(len(df_wire))
    drop_pct = 100.0 * (raw_rows - effective_rows) / max(raw_rows, 1)
    return {
        "raw_rows": raw_rows,
        "effective_rows": effective_rows,
        "wiring_row_drop_pct": float(drop_pct),
    }


_SYN_IDX_FULL_CACHE: Dict[str, Dict[int, str]] = {}


def _find_swc_for_id(swc_root: str | Path, nid: int) -> Path:
    swc_root = Path(swc_root).expanduser().resolve()
    patterns = [
        f"**/{int(nid)}/{int(nid)}_axodendro_with_synapses.swc",
        f"**/{int(nid)}_axodendro_with_synapses.swc",
        f"**/*{int(nid)}*with_synapses*.swc",
        f"**/{int(nid)}/{int(nid)}_healed_final.swc",
        f"**/{int(nid)}_healed_final.swc",
        f"**/{int(nid)}/{int(nid)}_healed.swc",
        f"**/{int(nid)}_healed.swc",
        f"**/{int(nid)}.swc",
        f"**/*{int(nid)}*.swc",
    ]
    for pattern in patterns:
        hits = sorted(swc_root.glob(pattern), key=lambda p: len(str(p)))
        if hits:
            return hits[0].resolve()
    raise FileNotFoundError(f"no SWC found for id={int(nid)} under {swc_root}")


def _collect_swc_paths_for_ids(swc_root: str | Path, neuron_ids: Sequence[int]) -> List[Path]:
    found: List[Path] = []
    missing: List[int] = []
    for nid in coerce_int_ids(neuron_ids, name="neuron ids"):
        try:
            found.append(_find_swc_for_id(swc_root, int(nid)))
        except FileNotFoundError:
            missing.append(int(nid))
    if missing:
        preview = missing[:20]
        suffix = " ..." if len(missing) > 20 else ""
        print(f"[reduce] warning: missing SWCs for {len(missing)} ids: {preview}{suffix}")
    return found


def _reduce_selected_swcs(
    *,
    input_root: str | Path,
    output_root: str | Path,
    swc_paths: Sequence[Path],
    profile: Mapping[str, Any],
    workers: int = 32,
    write_map: bool = False,
    protect_synapses: bool = True,
    max_syn_points: int = 2000,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    summary_path = output_root / "_swc_reduction_summary.csv"
    manifest_path = output_root / "_swc_reduction_manifest.json"

    if force_rebuild and output_root.exists():
        import shutil

        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    if (not force_rebuild) and summary_path.exists() and manifest_path.exists():
        return pd.read_csv(summary_path), json.loads(manifest_path.read_text(encoding="utf-8"))

    params_dict = {
        "max_path_um": float(profile["max_path_um"]),
        "max_turn_deg": float(profile["max_turn_deg"]),
        "max_diam_rel": float(profile["max_diam_rel"]),
        "protect_synapses": bool(protect_synapses),
        "max_syn_points": int(max_syn_points),
        "write_map": bool(write_map),
        "overwrite": True,
        "dry_run": False,
    }

    target_ids = set(int(rsd._extract_neuron_id_from_swc_path(Path(path)) or -1) for path in swc_paths)
    syn_idx: Dict[int, str] = {}
    if protect_synapses:
        cache_key = str(input_root)
        if cache_key not in _SYN_IDX_FULL_CACHE:
            _SYN_IDX_FULL_CACHE[cache_key] = rsd._build_syn_csv_index(input_root)
        full_syn_idx = _SYN_IDX_FULL_CACHE[cache_key]
        syn_idx = {int(key): value for key, value in full_syn_idx.items() if int(key) in target_ids}

    jobs = [str(Path(path).resolve()) for path in swc_paths]
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    max_workers = max(1, int(workers))
    progress_enabled = True
    profile_name = str(profile.get("name", "reduction"))

    if max_workers == 1:
        for swc_path in _maybe_tqdm(
            jobs,
            total=len(jobs),
            desc=f"Reduce SWCs:{profile_name}",
            enabled=progress_enabled,
            leave=True,
        ):
            results.append(rsd._reduce_one(swc_path, str(input_root), str(output_root), params_dict, syn_idx))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(rsd._reduce_one, swc_path, str(input_root), str(output_root), params_dict, syn_idx)
                for swc_path in jobs
            ]
            for future in _maybe_tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Reduce SWCs:{profile_name}",
                enabled=progress_enabled,
                leave=True,
            ):
                results.append(future.result())

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(results)
    if "status" not in df.columns:
        df["status"] = "error"

    ok_df = df[df["status"] == "ok"].copy()
    old_sum = int(pd.to_numeric(ok_df.get("old_nodes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    new_sum = int(pd.to_numeric(ok_df.get("new_nodes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    ratio = float(new_sum) / float(old_sum) if old_sum > 0 else float("nan")
    reduction_pct = float(100.0 * (1.0 - ratio)) if np.isfinite(ratio) else float("nan")

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "elapsed_s": float(elapsed),
        "profile": dict(profile),
        "target_file_count": int(len(jobs)),
        "node_totals": {
            "old": int(old_sum),
            "new": int(new_sum),
            "ratio": float(ratio),
            "reduction_pct": float(reduction_pct),
        },
    }

    df.to_csv(summary_path, index=False)
    _write_json(manifest_path, manifest)
    return df, manifest


def _choose_best_row(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    if "passes_thresholds" in data.columns:
        passing = data[data["passes_thresholds"].astype(bool)].copy()
        if not passing.empty:
            data = passing
    elif "fail_total" in data.columns:
        passing = data[pd.to_numeric(data["fail_total"], errors="coerce").fillna(9999).astype(float) == 0].copy()
        if not passing.empty:
            data = passing
    if "speedup_vs_baseline" in data.columns:
        data = data.sort_values(["speedup_vs_baseline"], ascending=[False])
    return data.iloc[0]


def run_reduction_sweep(
    *,
    project_info: Mapping[str, Path],
    base_config: Mapping[str, Any],
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    track_ids: Sequence[int],
    latency_pairs: Sequence[Tuple[int, int]],
    thresholds: Mapping[str, float] | None = None,
    reduction_profiles: Sequence[Mapping[str, Any]] | None = None,
    baseline_out_dir: str | Path | None = None,
    baseline_wall_s: float | None = None,
    force_baseline_rerun: bool = False,
    force_reduced_rerun: bool = False,
    force_reduce_rebuild: bool = False,
    workers: int = 32,
    write_map: bool = False,
    protect_synapses: bool = True,
    max_syn_points: int = 2000,
) -> Dict[str, Any]:
    project_info = dict(project_info)
    base_cfg = build_config(copy.deepcopy(dict(base_config)), strict=True)
    reduction_profiles_use = [dict(profile) for profile in (reduction_profiles or DEFAULT_REDUCTION_PROFILES)]
    thresholds_use = dict(DEFAULT_THRESHOLDS)
    thresholds_use.update(dict(thresholds or {}))

    if baseline_out_dir is None or baseline_wall_s is None or not np.isfinite(float(baseline_wall_s)):
        baseline_cfg = copy.deepcopy(base_cfg)
        baseline_cfg["run_id"] = _suffix_run_id(
            f"{hemilineage_project_name(hemilineage_label).lower()}_baseline",
            run_id_suffix,
        )
        baseline_out_dir, baseline_wall_s = _run_or_reuse(
            baseline_cfg,
            timing_filename="_hemilineage_baseline_timing.json",
            force=bool(force_baseline_rerun),
        )
    baseline_out_dir = Path(baseline_out_dir).expanduser().resolve()
    baseline_wall_s = float(baseline_wall_s)

    swc_paths = _collect_swc_paths_for_ids(base_cfg["swc_dir"], neuron_ids)
    if not swc_paths:
        raise RuntimeError("no SWC paths found for the supplied neuron ids")

    outputs_root = Path(project_info["reduction_outputs"]).expanduser().resolve()
    datasets_root = Path(project_info["reduction_datasets"]).expanduser().resolve()
    figures_dir = outputs_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    progress_enabled = bool(base_cfg.get("use_tqdm", True))

    rows: List[Dict[str, Any]] = []
    all_cmp: Dict[str, pd.DataFrame] = {}
    all_lat: Dict[str, pd.DataFrame] = {}
    profile_runs: Dict[str, Path] = {}

    for profile in _maybe_tqdm(
        reduction_profiles_use,
        total=len(reduction_profiles_use),
        desc=f"{hemilineage_project_name(hemilineage_label)} CRO profiles",
        enabled=progress_enabled,
        leave=True,
    ):
        profile_name = str(profile["name"])
        reduced_root = (datasets_root / profile_name).resolve()
        summary_df, manifest = _reduce_selected_swcs(
            input_root=base_cfg["swc_dir"],
            output_root=reduced_root,
            swc_paths=swc_paths,
            profile=profile,
            workers=int(workers),
            write_map=bool(write_map),
            protect_synapses=bool(protect_synapses),
            max_syn_points=int(max_syn_points),
            force_rebuild=bool(force_reduce_rebuild),
        )

        cfg_red = copy.deepcopy(base_cfg)
        cfg_red["run_id"] = _suffix_run_id(
            f"{hemilineage_project_name(hemilineage_label).lower()}_reduction_{profile_name}",
            run_id_suffix,
        )
        cfg_red["morph_swc_dir"] = str(reduced_root)

        red_out_dir, red_wall_s = _run_or_reuse(
            cfg_red,
            timing_filename="_hemilineage_reduction_timing.json",
            force=bool(force_reduced_rerun),
        )
        profile_runs[profile_name] = red_out_dir

        summarize_run(red_out_dir, track_ids=track_ids, latency_pairs=latency_pairs, prefix="_reduction_run")
        cmp_payload = compare_runs(
            baseline_out_dir,
            red_out_dir,
            track_ids=track_ids,
            latency_pairs=latency_pairs,
            thresholds=thresholds_use,
        )

        all_cmp[profile_name] = cmp_payload["neuron_cmp"]
        all_lat[profile_name] = cmp_payload["latency_cmp"]

        node_ratio = float(manifest.get("node_totals", {}).get("ratio", float("nan")))
        reduction_pct = float(manifest.get("node_totals", {}).get("reduction_pct", float("nan")))
        speedup = float(baseline_wall_s) / float(red_wall_s) if np.isfinite(baseline_wall_s) and np.isfinite(red_wall_s) and float(red_wall_s) > 0 else float("nan")

        rows.append(
            {
                "profile": profile_name,
                "reduced_root": str(reduced_root),
                "reduction_elapsed_s": float(manifest.get("elapsed_s", float("nan"))),
                "target_ids_count": int(len(coerce_int_ids(neuron_ids, name="neuron ids"))),
                "target_swc_count": int(len(swc_paths)),
                "ok_reduced_files": int((summary_df["status"] == "ok").sum()) if "status" in summary_df else 0,
                "node_ratio_new_over_old": node_ratio,
                "node_reduction_pct": reduction_pct,
                "baseline_total_wall_s": float(baseline_wall_s),
                "reduced_run_id": str(Path(red_out_dir).name),
                "reduced_total_wall_s": float(red_wall_s),
                "speedup_vs_baseline": speedup,
                **cmp_payload["metrics"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["fail_total", "speedup_vs_baseline"], ascending=[True, False]).reset_index(drop=True)
    results_df["passes_thresholds"] = results_df["fail_total"] == 0

    results_csv = outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_reduction_results.csv"
    results_df.to_csv(results_csv, index=False)
    for name, df in all_cmp.items():
        df.to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_{name}_neuron_cmp.csv", index=False)
    for name, df in all_lat.items():
        df.to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_{name}_latency_cmp.csv", index=False)

    plot_df = results_df.copy()
    plot_df["label"] = plot_df["profile"].astype(str)
    figures: List[Path] = []

    pass_df = results_df[results_df["passes_thresholds"]].copy()
    if not pass_df.empty:
        best_row = pass_df.sort_values(["speedup_vs_baseline", "node_reduction_pct"], ascending=[False, False]).iloc[0]
    else:
        best_row = results_df.sort_values(["fail_total", "speedup_vs_baseline"], ascending=[True, False]).iloc[0]

    best_profile = str(best_row["profile"])
    best_run_dir = profile_runs[best_profile]
    if _can_plot():
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        x = np.arange(len(plot_df))
        ax1.bar(x - 0.18, plot_df["node_reduction_pct"], width=0.36, label="Node reduction %")
        ax2.bar(x + 0.18, plot_df["speedup_vs_baseline"], width=0.36, color="tab:orange", label="Speedup x")
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df["label"], rotation=20)
        ax1.set_ylabel("Node reduction (%)")
        ax2.set_ylabel("Speedup vs baseline (x)")
        ax1.set_title(f"{hemilineage_project_name(hemilineage_label)} reduction sweep")
        ax1.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig1 = figures_dir / "reduction_vs_speedup.pdf"
        fig.savefig(fig1, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig1)

        fig, ax = plt.subplots(figsize=(6, 5))
        passing = plot_df["passes_thresholds"]
        ax.scatter(plot_df.loc[passing, "speedup_vs_baseline"], plot_df.loc[passing, "max_trace_rmse_mV"], c="tab:green", label="Pass", s=70)
        ax.scatter(plot_df.loc[~passing, "speedup_vs_baseline"], plot_df.loc[~passing, "max_trace_rmse_mV"], c="tab:red", label="Fail", s=70)
        for _, row in plot_df.iterrows():
            ax.annotate(str(row["profile"]), (row["speedup_vs_baseline"], row["max_trace_rmse_mV"]), fontsize=8)
        ax.set_xlabel("Speedup vs baseline (x)")
        ax.set_ylabel("Max trace RMSE (mV)")
        ax.set_title("Reduction speed vs fidelity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig2 = figures_dir / "speed_vs_fidelity.pdf"
        fig.savefig(fig2, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig2)

        best_records = pd.read_csv(best_run_dir / "records.csv")
        base_records = pd.read_csv(baseline_out_dir / "records.csv")

        fig, axes = plt.subplots(len(track_ids), 1, figsize=(10, 2.2 * len(track_ids)), sharex=True)
        if len(track_ids) == 1:
            axes = [axes]
        for ax, nid in zip(axes, track_ids):
            base_col = _soma_col(base_records, int(nid))
            best_col = _soma_col(best_records, int(nid))
            if not base_col or not best_col:
                ax.text(0.5, 0.5, f"missing trace for {int(nid)}", ha="center", va="center")
                ax.set_ylabel(str(int(nid)))
                continue
            tb = pd.to_numeric(base_records["t_ms"], errors="coerce").to_numpy(float)
            vb = pd.to_numeric(base_records[base_col], errors="coerce").to_numpy(float)
            tr = pd.to_numeric(best_records["t_ms"], errors="coerce").to_numpy(float)
            vr = pd.to_numeric(best_records[best_col], errors="coerce").to_numpy(float)
            ax.plot(tb, vb, label="baseline", lw=1.2)
            ax.plot(tr, vr, label=f"reduced:{best_profile}", lw=1.0, alpha=0.9)
            ax.set_ylabel(f"{int(nid)} mV")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="upper right", ncol=2)
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"{hemilineage_project_name(hemilineage_label)} baseline vs best reduced traces", y=1.02)
        fig.tight_layout()
        fig3 = figures_dir / "baseline_vs_best_traces.pdf"
        fig.savefig(fig3, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig3)
    else:
        (outputs_root / "_plotting_skipped.txt").write_text(
            "matplotlib is not available in this Python environment, so sweep figures were skipped.\n",
            encoding="utf-8",
        )

    best_payload = {
        "profile": best_profile,
        "reduced_root": str(best_row["reduced_root"]),
        "run_id": str(best_row["reduced_run_id"]),
        "results_csv": str(results_csv),
        "figures": [str(fig) for fig in figures],
        "metrics": {key: best_row[key] for key in best_row.index if key != "profile"},
    }
    best_json = outputs_root / "best_reduction.json"
    _write_json(best_json, best_payload)

    return {
        "results_df": results_df,
        "results_csv": results_csv,
        "best_row": best_row,
        "best_profile": best_profile,
        "best_json": best_json,
        "best_reduced_root": Path(str(best_row["reduced_root"])).expanduser().resolve(),
        "baseline_out_dir": baseline_out_dir,
        "baseline_wall_s": baseline_wall_s,
        "figures": figures,
    }


def run_coalescing_sweep(
    *,
    project_info: Mapping[str, Path],
    base_config: Mapping[str, Any],
    hemilineage_label: str,
    track_ids: Sequence[int],
    latency_pairs: Sequence[Tuple[int, int]],
    thresholds: Mapping[str, float] | None = None,
    coalesce_profiles: Sequence[Mapping[str, Any]] | None = None,
    baseline_out_dir: str | Path | None = None,
    baseline_wall_s: float | None = None,
    force_baseline_rerun: bool = False,
    force_profile_rerun: bool = False,
) -> Dict[str, Any]:
    project_info = dict(project_info)
    base_cfg = build_config(copy.deepcopy(dict(base_config)), strict=True)
    thresholds_use = dict(DEFAULT_THRESHOLDS)
    thresholds_use.update(dict(thresholds or {}))
    profiles_use = [dict(profile) for profile in (coalesce_profiles or DEFAULT_COALESCE_PROFILES)]

    if baseline_out_dir is None or baseline_wall_s is None or not np.isfinite(float(baseline_wall_s)):
        baseline_cfg = copy.deepcopy(base_cfg)
        baseline_cfg["run_id"] = _suffix_run_id(
            f"{hemilineage_project_name(hemilineage_label).lower()}_coalesce_baseline",
            run_id_suffix,
        )
        baseline_cfg["coalesce_syns"] = False
        baseline_cfg["coalesce_mode"] = "none"
        baseline_cfg["use_geom_delay"] = True
        baseline_out_dir, baseline_wall_s = _run_or_reuse(
            baseline_cfg,
            timing_filename="_hemilineage_coalesce_baseline_timing.json",
            force=bool(force_baseline_rerun),
        )
    baseline_out_dir = Path(baseline_out_dir).expanduser().resolve()
    baseline_wall_s = float(baseline_wall_s)

    outputs_root = Path(project_info["coalescing_outputs"]).expanduser().resolve()
    figures_dir = outputs_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    progress_enabled = bool(base_cfg.get("use_tqdm", True))

    rows: List[Dict[str, Any]] = []
    all_cmp: Dict[str, pd.DataFrame] = {}
    all_lat: Dict[str, pd.DataFrame] = {}

    for profile in _maybe_tqdm(
        profiles_use,
        total=len(profiles_use),
        desc=f"{hemilineage_project_name(hemilineage_label)} SC profiles",
        enabled=progress_enabled,
        leave=True,
    ):
        name = str(profile["name"])
        cfg = copy.deepcopy(base_cfg)
        cfg["run_id"] = _suffix_run_id(
            f"{hemilineage_project_name(hemilineage_label).lower()}_coalesce_{name}",
            run_id_suffix,
        )
        cfg["coalesce_syns"] = bool(profile.get("coalesce_syns", False))
        cfg["coalesce_mode"] = str(profile.get("coalesce_mode", "none"))
        cfg["use_geom_delay"] = bool(profile.get("use_geom_delay", True))
        cfg["enable_coreneuron"] = bool(profile.get("enable_coreneuron", cfg.get("enable_coreneuron", True)))
        cfg["coreneuron_gpu"] = bool(profile.get("coreneuron_gpu", cfg.get("coreneuron_gpu", True)))
        cfg["cvode"] = copy.deepcopy(dict(profile.get("cvode", cfg.get("cvode", {"enabled": False}))))
        if profile.get("coalesce_guard_drop") is not None:
            cfg["coalesce_guard_drop"] = float(profile["coalesce_guard_drop"])
        if profile.get("coalesce_guard_w_med_uS") is not None:
            cfg["coalesce_guard_w_med_uS"] = float(profile["coalesce_guard_w_med_uS"])

        wiring_stats = estimate_wiring_rows(cfg)
        out_dir, wall_s = _run_or_reuse(
            cfg,
            timing_filename="_hemilineage_coalescing_timing.json",
            force=bool(force_profile_rerun),
        )
        summarize_run(out_dir, track_ids=track_ids, latency_pairs=latency_pairs, prefix="_coalescing_run")
        cmp_payload = compare_runs(
            baseline_out_dir,
            out_dir,
            track_ids=track_ids,
            latency_pairs=latency_pairs,
            thresholds=thresholds_use,
        )
        all_cmp[name] = cmp_payload["neuron_cmp"]
        all_lat[name] = cmp_payload["latency_cmp"]

        speedup = float(baseline_wall_s) / float(wall_s) if np.isfinite(baseline_wall_s) and np.isfinite(wall_s) and float(wall_s) > 0 else float("nan")
        rows.append(
            {
                "profile": name,
                "run_id": str(Path(out_dir).name),
                "out_dir": str(out_dir),
                "baseline_wall_s": float(baseline_wall_s),
                "test_wall_s": float(wall_s),
                "speedup_vs_baseline": speedup,
                "raw_rows": int(wiring_stats["raw_rows"]),
                "effective_rows": int(wiring_stats["effective_rows"]),
                "wiring_row_drop_pct": float(wiring_stats["wiring_row_drop_pct"]),
                **cmp_payload["metrics"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["fail_total", "speedup_vs_baseline"], ascending=[True, False]).reset_index(drop=True)
    results_df["passes_thresholds"] = results_df["fail_total"] == 0
    results_csv = outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_coalescing_results.csv"
    results_df.to_csv(results_csv, index=False)
    for name, df in all_cmp.items():
        df.to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_{name}_neuron_cmp.csv", index=False)
    for name, df in all_lat.items():
        df.to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_{name}_latency_cmp.csv", index=False)

    plot_df = results_df.copy()
    plot_df["label"] = plot_df["profile"].astype(str)

    figures: List[Path] = []
    if _can_plot():
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        x = np.arange(len(plot_df))
        ax1.bar(x - 0.18, plot_df["wiring_row_drop_pct"], width=0.36, label="Wiring row drop %")
        ax2.bar(x + 0.18, plot_df["speedup_vs_baseline"], width=0.36, color="tab:orange", label="Speedup x")
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df["label"], rotation=20)
        ax1.set_ylabel("Wiring-row reduction (%)")
        ax2.set_ylabel("Speedup vs baseline (x)")
        ax1.set_title(f"{hemilineage_project_name(hemilineage_label)} coalescing sweep")
        ax1.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig1 = figures_dir / "coalescing_rows_vs_speedup.pdf"
        fig.savefig(fig1, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig1)

        fig, ax = plt.subplots(figsize=(6, 5))
        passing = plot_df["passes_thresholds"]
        ax.scatter(plot_df.loc[passing, "speedup_vs_baseline"], plot_df.loc[passing, "max_trace_rmse_mV"], c="tab:green", label="Pass", s=70)
        ax.scatter(plot_df.loc[~passing, "speedup_vs_baseline"], plot_df.loc[~passing, "max_trace_rmse_mV"], c="tab:red", label="Fail", s=70)
        for _, row in plot_df.iterrows():
            ax.annotate(str(row["profile"]), (row["speedup_vs_baseline"], row["max_trace_rmse_mV"]), fontsize=8)
        ax.set_xlabel("Speedup vs baseline (x)")
        ax.set_ylabel("Max trace RMSE (mV)")
        ax.set_title("Coalescing speed vs fidelity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig2 = figures_dir / "coalescing_speed_vs_fidelity.pdf"
        fig.savefig(fig2, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig2)
    else:
        (outputs_root / "_plotting_skipped.txt").write_text(
            "matplotlib is not available in this Python environment, so sweep figures were skipped.\n",
            encoding="utf-8",
        )

    pass_df = results_df[results_df["passes_thresholds"]].copy()
    if not pass_df.empty:
        best_row = pass_df.sort_values(["speedup_vs_baseline", "wiring_row_drop_pct"], ascending=[False, False]).iloc[0]
    else:
        best_row = results_df.sort_values(["fail_total", "speedup_vs_baseline"], ascending=[True, False]).iloc[0]

    best_payload = {
        "profile": str(best_row["profile"]),
        "run_id": str(best_row["run_id"]),
        "results_csv": str(results_csv),
        "coalesce_syns": bool(str(best_row["profile"]).strip().lower() != "baseline_none"),
        "coalesce_mode": "pair" if "pair" in str(best_row["profile"]).strip().lower() else ("site" if "site" in str(best_row["profile"]).strip().lower() else "none"),
        "use_geom_delay": bool("no_geom_delay" not in str(best_row["profile"]).strip().lower() and "nogeom" not in str(best_row["profile"]).strip().lower()),
        "metrics": {key: best_row[key] for key in best_row.index if key != "profile"},
        "figures": [str(fig) for fig in figures],
    }
    best_json = outputs_root / "best_coalescing.json"
    _write_json(best_json, best_payload)

    return {
        "results_df": results_df,
        "results_csv": results_csv,
        "best_row": best_row,
        "best_json": best_json,
        "best_profile": str(best_row["profile"]),
        "best_coalesce_params": {
            "coalesce_syns": bool(best_payload["coalesce_syns"]),
            "coalesce_mode": str(best_payload["coalesce_mode"]),
            "use_geom_delay": bool(best_payload["use_geom_delay"]),
        },
        "baseline_out_dir": baseline_out_dir,
        "baseline_wall_s": baseline_wall_s,
        "figures": figures,
    }


def run_combined_optimization(
    *,
    project_info: Mapping[str, Path],
    base_config: Mapping[str, Any],
    hemilineage_label: str,
    track_ids: Sequence[int],
    latency_pairs: Sequence[Tuple[int, int]],
    thresholds: Mapping[str, float] | None = None,
    best_reduction_root: str | Path,
    best_reduction_profile: str,
    best_coalesce_params: Mapping[str, Any],
    baseline_out_dir: str | Path | None = None,
    baseline_wall_s: float | None = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    project_info = dict(project_info)
    base_cfg = build_config(copy.deepcopy(dict(base_config)), strict=True)
    thresholds_use = dict(DEFAULT_THRESHOLDS)
    thresholds_use.update(dict(thresholds or {}))

    if baseline_out_dir is None or baseline_wall_s is None or not np.isfinite(float(baseline_wall_s)):
        baseline_cfg = copy.deepcopy(base_cfg)
        baseline_cfg["run_id"] = _suffix_run_id(
            f"{hemilineage_project_name(hemilineage_label).lower()}_optimized_baseline",
            run_id_suffix,
        )
        baseline_cfg["coalesce_syns"] = False
        baseline_cfg["coalesce_mode"] = "none"
        baseline_cfg["use_geom_delay"] = True
        baseline_cfg["morph_swc_dir"] = None
        baseline_out_dir, baseline_wall_s = _run_or_reuse(
            baseline_cfg,
            timing_filename="_hemilineage_optimized_baseline_timing.json",
            force=bool(force_rerun),
        )
    baseline_out_dir = Path(baseline_out_dir).expanduser().resolve()
    baseline_wall_s = float(baseline_wall_s)

    opt_cfg = copy.deepcopy(base_cfg)
    opt_cfg["run_id"] = _suffix_run_id(
        f"{hemilineage_project_name(hemilineage_label).lower()}_optimized_combo",
        run_id_suffix,
    )
    opt_cfg["morph_swc_dir"] = str(Path(best_reduction_root).expanduser().resolve())
    opt_cfg["coalesce_syns"] = bool(best_coalesce_params.get("coalesce_syns", False))
    opt_cfg["coalesce_mode"] = str(best_coalesce_params.get("coalesce_mode", "none"))
    opt_cfg["use_geom_delay"] = bool(best_coalesce_params.get("use_geom_delay", True))

    opt_out_dir, opt_wall_s = _run_or_reuse(
        opt_cfg,
        timing_filename="_hemilineage_optimized_timing.json",
        force=bool(force_rerun),
    )
    summarize_run(opt_out_dir, track_ids=track_ids, latency_pairs=latency_pairs, prefix="_optimized_run")
    cmp_payload = compare_runs(
        baseline_out_dir,
        opt_out_dir,
        track_ids=track_ids,
        latency_pairs=latency_pairs,
        thresholds=thresholds_use,
    )

    outputs_root = Path(project_info["optimized_outputs"]).expanduser().resolve()
    figures_dir = outputs_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    speedup = float(baseline_wall_s) / float(opt_wall_s) if np.isfinite(baseline_wall_s) and np.isfinite(opt_wall_s) and float(opt_wall_s) > 0 else float("nan")
    summary = {
        "hemilineage": normalize_hemilineage_label(hemilineage_label),
        "baseline_run_id": str(Path(baseline_out_dir).name),
        "optimized_run_id": str(Path(opt_out_dir).name),
        "baseline_out_dir": str(baseline_out_dir),
        "optimized_out_dir": str(opt_out_dir),
        "baseline_wall_s": float(baseline_wall_s),
        "optimized_wall_s": float(opt_wall_s),
        "optimized_speedup_vs_baseline": float(speedup) if np.isfinite(speedup) else float("nan"),
        "best_reduction_profile": str(best_reduction_profile),
        "best_reduction_root": str(Path(best_reduction_root).expanduser().resolve()),
        "best_coalescing_params": dict(best_coalesce_params),
        **cmp_payload["metrics"],
    }

    summary_json = outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_optimized_summary.json"
    cmp_payload["neuron_cmp"].to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_optimized_neuron_cmp.csv", index=False)
    cmp_payload["latency_cmp"].to_csv(outputs_root / f"{hemilineage_project_name(hemilineage_label).lower()}_optimized_latency_cmp.csv", index=False)
    _write_json(summary_json, summary)

    base_records = _load_records(baseline_out_dir)
    opt_records = _load_records(opt_out_dir)

    figures: List[Path] = []
    if _can_plot():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["baseline", "optimized"], [float(baseline_wall_s), float(opt_wall_s)], color=["tab:blue", "tab:orange"])
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"{hemilineage_project_name(hemilineage_label)} runtime")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig1 = figures_dir / "runtime_comparison.pdf"
        fig.savefig(fig1, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig1)

        fig, axes = plt.subplots(len(track_ids), 1, figsize=(10, 2.2 * len(track_ids)), sharex=True)
        if len(track_ids) == 1:
            axes = [axes]
        for ax, nid in zip(axes, track_ids):
            base_col = _soma_col(base_records, int(nid))
            opt_col = _soma_col(opt_records, int(nid))
            if not base_col or not opt_col:
                ax.text(0.5, 0.5, f"missing trace for {int(nid)}", ha="center", va="center")
                ax.set_ylabel(str(int(nid)))
                continue
            tb = pd.to_numeric(base_records["t_ms"], errors="coerce").to_numpy(float)
            vb = pd.to_numeric(base_records[base_col], errors="coerce").to_numpy(float)
            to = pd.to_numeric(opt_records["t_ms"], errors="coerce").to_numpy(float)
            vo = pd.to_numeric(opt_records[opt_col], errors="coerce").to_numpy(float)
            ax.plot(tb, vb, lw=1.2, label="baseline")
            ax.plot(to, vo, lw=1.0, alpha=0.9, label="optimized")
            ax.set_ylabel(f"{int(nid)} mV")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="upper right", ncol=2)
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"{hemilineage_project_name(hemilineage_label)} baseline vs optimized traces", y=1.02)
        fig.tight_layout()
        fig2 = figures_dir / "trace_overlay_baseline_vs_optimized.pdf"
        fig.savefig(fig2, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig2)
    else:
        (outputs_root / "_plotting_skipped.txt").write_text(
            "matplotlib is not available in this Python environment, so optimized figures were skipped.\n",
            encoding="utf-8",
        )

    optimized_cfg_json = outputs_root / "best_optimized_config.json"
    _write_json(optimized_cfg_json, opt_cfg)

    return {
        "summary": summary,
        "summary_json": summary_json,
        "optimized_config_json": optimized_cfg_json,
        "baseline_out_dir": baseline_out_dir,
        "optimized_out_dir": opt_out_dir,
        "figures": figures,
    }


def run_full_hemilineage_project(
    *,
    projects_root: str | Path,
    swc_dir: str | Path,
    morph_swc_dir: str | Path | None = None,
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    edge_set_name: str | None = None,
    edges_registry_root: str | Path | None = None,
    neuprint_dataset: str | None = None,
    seeds: Sequence[int] | None = None,
    template_config_path: str | Path | None = None,
    master_csv: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    biophysics: Mapping[str, Any] | None = None,
    stim: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
    gap: Mapping[str, Any] | None = None,
    track_ids: Sequence[int] | None = None,
    latency_pairs: Sequence[Tuple[int, int]] | None = None,
    thresholds: Mapping[str, float] | None = None,
    reduction_profiles: Sequence[Mapping[str, Any]] | None = None,
    coalesce_profiles: Sequence[Mapping[str, Any]] | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
    force_rebuild_edges: bool = False,
    force_baseline_rerun: bool = False,
    force_reduction_rerun: bool = False,
    force_coalescing_rerun: bool = False,
    force_reduce_rebuild: bool = False,
    force_optimized_rerun: bool = False,
    reduction_workers: int = 32,
    reduction_write_map: bool = False,
    reduction_protect_synapses: bool = True,
    reduction_max_syn_points: int = 2000,
    build_edges_workers: int = 16,
    phase1_fallback_enabled: bool = True,
    phase1_upsample_nm: float = 2000.0,
    phase1_min_conf: float = 0.4,
    phase1_batch_size: int = 10000,
    phase1_export_workers: int = 1,
    phase1_progress_every: int = 25,
    run_id_suffix: str | None = None,
    run_reduction_pipeline: bool = True,
    run_coalescing_pipeline: bool = True,
    run_combined_pipeline: bool = True,
    run_notes: str = "",
) -> Dict[str, Any]:
    label = normalize_hemilineage_label(hemilineage_label)
    ids = coerce_int_ids(neuron_ids, name="core hemilineage neuron ids")
    seed_ids = coerce_int_ids(seeds or ids, name="seed ids")
    master_csv_use = Path(master_csv).expanduser().resolve() if master_csv is not None else default_master_csv_for_swc_dir(swc_dir)
    thresholds_use = dict(DEFAULT_THRESHOLDS)
    thresholds_use.update(dict(thresholds or {}))

    paths = project_paths(projects_root, label)
    edges_path, edges_df = build_edges_for_hemilineage_ids(
        label,
        ids,
        swc_dir=swc_dir,
        project_root=paths["project_root"],
        edge_set_name=edge_set_name,
        edges_registry_root=edges_registry_root,
        master_csv=master_csv_use,
        neuprint_dataset=neuprint_dataset,
        seeds=seed_ids,
        default_weight_uS=float((timing or {}).get("default_weight_uS", 6e-6)),
        workers=int(build_edges_workers),
        force_rebuild=bool(force_rebuild_edges),
        phase1_fallback_enabled=bool(phase1_fallback_enabled),
        phase1_upsample_nm=float(phase1_upsample_nm),
        phase1_min_conf=float(phase1_min_conf),
        phase1_batch_size=int(phase1_batch_size),
        phase1_export_workers=int(phase1_export_workers),
        phase1_progress_every=int(phase1_progress_every),
    )
    network_build_report = json.loads((paths["metadata_dir"] / "network_build_report.json").read_text(encoding="utf-8"))
    final_network_ids = coerce_int_ids(network_build_report["final_network_ids"], name="final network ids")
    added_motor_ids = [int(x) for x in network_build_report.get("added_motor_neuron_ids", [])]
    if track_ids is None:
        default_track_order = list(seed_ids) + list(added_motor_ids) + list(final_network_ids)
        track_ids_use = coerce_int_ids(default_track_order, name="track ids")[:8]
    else:
        track_ids_use = coerce_int_ids(track_ids, name="track ids")
    if latency_pairs is None:
        latency_pairs_pref: List[Tuple[int, int]] = []
        for seed in seed_ids:
            for motor_id in added_motor_ids:
                if int(seed) == int(motor_id):
                    continue
                latency_pairs_pref.append((int(seed), int(motor_id)))
                if len(latency_pairs_pref) >= 8:
                    break
            if len(latency_pairs_pref) >= 8:
                break
        latency_pairs_use = latency_pairs_pref or list(infer_latency_pairs(final_network_ids, seed_ids))
    else:
        latency_pairs_use = list(latency_pairs)

    metadata = {
        "hemilineage": label,
        "project_root": str(paths["project_root"]),
        "swc_dir": str(Path(swc_dir).expanduser().resolve()),
        "morph_swc_dir": (str(Path(morph_swc_dir).expanduser().resolve()) if morph_swc_dir else None),
        "template_config_path": str(Path(template_config_path).expanduser().resolve()) if template_config_path else None,
        "master_csv": str(master_csv_use),
        "core_hemilineage_ids": ids,
        "seeds": seed_ids,
        "final_network_ids": final_network_ids,
        "track_ids": track_ids_use,
        "latency_pairs": [[int(a), int(b)] for a, b in latency_pairs_use],
        "edges_path": str(edges_path),
        "edges_rows": int(len(edges_df)),
        "network_build_report": network_build_report,
        "run_id_suffix": str(run_id_suffix or ""),
        "performance_flags": {
            "enable_coreneuron": bool(base_cfg.get("enable_coreneuron", False)) if "base_cfg" in locals() else bool((runtime or {}).get("enable_coreneuron", True)),
            "coreneuron_gpu": bool(base_cfg.get("coreneuron_gpu", False)) if "base_cfg" in locals() else bool((runtime or {}).get("coreneuron_gpu", True)),
            "threads": (runtime or {}).get("threads"),
            "coreneuron_nthread": (runtime or {}).get("coreneuron_nthread"),
            "io_workers": (runtime or {}).get("io_workers"),
        },
    }
    base_cfg = build_master_config(
        swc_dir=swc_dir,
        hemilineage_label=label,
        neuron_ids=final_network_ids,
        edges_path=edges_path,
        project_root=paths["project_root"],
        seeds=seed_ids,
        run_id=_suffix_run_id(f"{hemilineage_project_name(label).lower()}_baseline", run_id_suffix),
        master_csv=master_csv_use,
        template_config_path=template_config_path,
        morph_swc_dir=morph_swc_dir,
        runtime=runtime,
        timing=timing,
        biophysics=biophysics,
        stim=stim,
        record=record,
        gap=gap,
        extra_overrides=extra_overrides,
        run_notes=run_notes,
        record_soma_for_all=True,
    )
    metadata["performance_flags"] = {
        "enable_coreneuron": bool(base_cfg.get("enable_coreneuron", False)),
        "coreneuron_gpu": bool(base_cfg.get("coreneuron_gpu", False)),
        "threads": base_cfg.get("threads"),
        "coreneuron_nthread": base_cfg.get("coreneuron_nthread"),
        "io_workers": base_cfg.get("io_workers"),
        "cvode_enabled": bool((base_cfg.get("cvode") or {}).get("enabled", False)),
    }
    metadata_json = _write_json(paths["metadata_dir"] / "project_inputs.json", metadata)
    pd.DataFrame({"neuron_id": ids}).to_csv(paths["metadata_dir"] / "core_hemilineage_ids.csv", index=False)
    stage_count = 1
    if run_reduction_pipeline:
        stage_count += 1
    if run_coalescing_pipeline:
        stage_count += 1
    if run_combined_pipeline and run_reduction_pipeline and run_coalescing_pipeline:
        stage_count += 1
    stage_bar = _make_tqdm(
        total=stage_count,
        desc=f"{hemilineage_project_name(label)} pipeline",
        enabled=bool(base_cfg.get("use_tqdm", True)),
        leave=True,
    )

    base_out_dir, base_wall_s = _run_or_reuse(
        base_cfg,
        timing_filename="_hemilineage_baseline_timing.json",
        force=bool(force_baseline_rerun),
    )
    base_summary = summarize_run(base_out_dir, track_ids=track_ids_use, latency_pairs=latency_pairs_use, prefix="_baseline_run")
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.set_postfix_str("baseline complete")

    output: Dict[str, Any] = {
        "project_paths": paths,
        "project_inputs_json": metadata_json,
        "network_build_report": network_build_report,
        "core_hemilineage_ids": ids,
        "final_network_ids": final_network_ids,
        "edges_path": edges_path,
        "edges_rows": int(len(edges_df)),
        "baseline_out_dir": base_out_dir,
        "baseline_wall_s": float(base_wall_s),
        "baseline_summary": base_summary["summary"],
        "base_config": base_cfg,
    }

    if run_reduction_pipeline:
        reduction_payload = run_reduction_sweep(
            project_info=paths,
            base_config=base_cfg,
            hemilineage_label=label,
            neuron_ids=final_network_ids,
            track_ids=track_ids_use,
            latency_pairs=latency_pairs_use,
            thresholds=thresholds_use,
            reduction_profiles=reduction_profiles,
            baseline_out_dir=base_out_dir,
            baseline_wall_s=base_wall_s,
            force_baseline_rerun=False,
            force_reduced_rerun=bool(force_reduction_rerun),
            force_reduce_rebuild=bool(force_reduce_rebuild),
            workers=int(reduction_workers),
            write_map=bool(reduction_write_map),
            protect_synapses=bool(reduction_protect_synapses),
            max_syn_points=int(reduction_max_syn_points),
        )
        output["reduction"] = reduction_payload
        if stage_bar is not None:
            stage_bar.update(1)
            stage_bar.set_postfix_str("CRO complete")

    if run_coalescing_pipeline:
        coalesce_payload = run_coalescing_sweep(
            project_info=paths,
            base_config=base_cfg,
            hemilineage_label=label,
            track_ids=track_ids_use,
            latency_pairs=latency_pairs_use,
            thresholds=thresholds_use,
            coalesce_profiles=coalesce_profiles,
            baseline_out_dir=base_out_dir,
            baseline_wall_s=base_wall_s,
            force_profile_rerun=bool(force_coalescing_rerun),
        )
        output["coalescing"] = coalesce_payload
        if stage_bar is not None:
            stage_bar.update(1)
            stage_bar.set_postfix_str("SC complete")

    if run_combined_pipeline and ("reduction" in output) and ("coalescing" in output):
        optimized_payload = run_combined_optimization(
            project_info=paths,
            base_config=base_cfg,
            hemilineage_label=label,
            track_ids=track_ids_use,
            latency_pairs=latency_pairs_use,
            thresholds=thresholds_use,
            best_reduction_root=output["reduction"]["best_reduced_root"],
            best_reduction_profile=output["reduction"]["best_profile"],
            best_coalesce_params=output["coalescing"]["best_coalesce_params"],
            baseline_out_dir=base_out_dir,
            baseline_wall_s=base_wall_s,
            force_rerun=bool(force_optimized_rerun),
        )
        output["optimized"] = optimized_payload
        if stage_bar is not None:
            stage_bar.update(1)
            stage_bar.set_postfix_str("optimized complete")

    project_summary = {
        "hemilineage": label,
        "project_root": str(paths["project_root"]),
        "core_hemilineage_count": int(len(ids)),
        "added_motor_neuron_count": int(network_build_report.get("added_motor_neuron_count", 0)),
        "final_network_count": int(len(final_network_ids)),
        "network_build_report_json": str((paths["metadata_dir"] / "network_build_report.json").resolve()),
        "edges_path": str(edges_path),
        "edge_set_name": network_build_report.get("edge_set_name"),
        "edge_signature": network_build_report.get("edge_signature"),
        "neuprint_dataset": network_build_report.get("neuprint_dataset"),
        "edges_registry_root": network_build_report.get("edges_registry_root"),
        "baseline_out_dir": str(base_out_dir),
        "baseline_wall_s": float(base_wall_s),
        "has_reduction": bool("reduction" in output),
        "has_coalescing": bool("coalescing" in output),
        "has_optimized": bool("optimized" in output),
    }
    if "reduction" in output:
        project_summary["best_reduction_profile"] = str(output["reduction"]["best_profile"])
        project_summary["best_reduction_root"] = str(output["reduction"]["best_reduced_root"])
    if "coalescing" in output:
        project_summary["best_coalescing_profile"] = str(output["coalescing"]["best_profile"])
        project_summary["best_coalescing_params"] = dict(output["coalescing"]["best_coalesce_params"])
    if "optimized" in output:
        project_summary["optimized_summary_json"] = str(output["optimized"]["summary_json"])

    output["project_summary_json"] = _write_json(paths["metadata_dir"] / "project_summary.json", project_summary)
    if stage_bar is not None:
        stage_bar.close()
    return output


def preview_hemilineage_benchmark_plan(
    *,
    projects_root: str | Path,
    swc_dir: str | Path,
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    edge_set_name: str | None = None,
    edges_registry_root: str | Path | None = None,
    neuprint_dataset: str | None = None,
    seeds: Sequence[int] | None = None,
    master_csv: str | Path | None = None,
    timing: Mapping[str, Any] | None = None,
    build_edges_workers: int = 16,
    reduction_profiles: Sequence[Mapping[str, Any]] | None = None,
    coalesce_profiles: Sequence[Mapping[str, Any]] | None = None,
    min_core_size: int = 2,
    max_core_size: int = 20,
    repeats: int = 1,
    force_rebuild_edges: bool = False,
    phase1_fallback_enabled: bool = True,
    phase1_upsample_nm: float = 2000.0,
    phase1_min_conf: float = 0.4,
    phase1_batch_size: int = 10000,
    phase1_export_workers: int = 1,
    phase1_progress_every: int = 25,
    run_reduction_pipeline: bool = True,
    run_coalescing_pipeline: bool = True,
    run_combined_pipeline: bool = True,
) -> Dict[str, Any]:
    label = normalize_hemilineage_label(hemilineage_label)
    ids = coerce_int_ids(neuron_ids, name="core hemilineage neuron ids")
    seed_ids = coerce_int_ids(seeds or ids, name="seed ids")
    master_csv_use = Path(master_csv).expanduser().resolve() if master_csv is not None else default_master_csv_for_swc_dir(swc_dir)
    reduction_profiles_use = [dict(profile) for profile in (reduction_profiles or DEFAULT_REDUCTION_PROFILES)]
    coalesce_profiles_use = [dict(profile) for profile in (coalesce_profiles or DEFAULT_COALESCE_PROFILES)]
    paths = project_paths(projects_root, label)

    edges_path, _ = build_edges_for_hemilineage_ids(
        label,
        ids,
        swc_dir=swc_dir,
        project_root=paths["project_root"],
        edge_set_name=edge_set_name,
        edges_registry_root=edges_registry_root,
        master_csv=master_csv_use,
        neuprint_dataset=neuprint_dataset,
        seeds=seed_ids,
        default_weight_uS=float((timing or {}).get("default_weight_uS", 6e-6)),
        workers=int(build_edges_workers),
        force_rebuild=bool(force_rebuild_edges),
        phase1_fallback_enabled=bool(phase1_fallback_enabled),
        phase1_upsample_nm=float(phase1_upsample_nm),
        phase1_min_conf=float(phase1_min_conf),
        phase1_batch_size=int(phase1_batch_size),
        phase1_export_workers=int(phase1_export_workers),
        phase1_progress_every=int(phase1_progress_every),
    )
    network_build_report = _read_json(paths["metadata_dir"] / "network_build_report.json")
    raw_edges_path = Path(network_build_report["raw_edges_path"]).expanduser().resolve()
    raw_edges_df = _read_csv_allow_empty(raw_edges_path, columns=HEMI_EDGE_COLUMNS, repair=True)
    if raw_edges_df.empty:
        raise RuntimeError(
            "Benchmark preview cannot continue because the raw hemilineage edge build produced 0 rows. "
            "This usually means the supplied SWC/synapse directory does not contain matching "
            f"'<id>_synapses_new.csv' files for the provided core IDs, or those files contain no outbound 'pre' rows. "
            f"Current SWC_DIR: {Path(swc_dir).expanduser().resolve()}"
        )

    subset_plan = build_connected_core_benchmark_subsets(
        ids,
        raw_edges_df,
        min_size=int(min_core_size),
        max_size=int(max_core_size),
    )
    subset_rows = list(subset_plan["subset_rows"])
    subset_df = pd.DataFrame(
        [
            {
                "core_subset_size": int(row["core_subset_size"]),
                "selected_core_ids": json.dumps([int(x) for x in row["selected_core_ids"]]),
                "core_internal_edge_rows": int(row["core_internal_edge_rows"]),
            }
            for row in subset_rows
        ]
    )
    subset_csv = paths["benchmark_outputs"] / "selected_core_subsets.csv"
    subset_df.to_csv(subset_csv, index=False)

    run_combined_use = bool(run_combined_pipeline and run_reduction_pipeline and run_coalescing_pipeline)
    simulation_runs_per_variant = int(
        1
        + (len(reduction_profiles_use) if run_reduction_pipeline else 0)
        + (len(coalesce_profiles_use) if run_coalescing_pipeline else 0)
        + (1 if run_combined_use else 0)
    )
    reduction_builds_per_variant = int(len(reduction_profiles_use) if run_reduction_pipeline else 0)
    variant_count = int(len(subset_rows) * max(1, int(repeats)))

    plan = {
        "hemilineage": label,
        "selection_rule": "connected core subset prefixes + immediate postsynaptic motor neurons",
        "full_core_count": int(len(ids)),
        "full_seed_count": int(len(seed_ids)),
        "full_added_motor_neuron_count": int(network_build_report.get("added_motor_neuron_count", 0)),
        "full_final_network_count": int(network_build_report.get("final_network_count", 0)),
        "neuprint_dataset": network_build_report.get("neuprint_dataset"),
        "benchmark_core_sizes": [int(row["core_subset_size"]) for row in subset_rows],
        "benchmark_variant_count": int(variant_count),
        "repeats": int(max(1, int(repeats))),
        "largest_connected_core_component_size": int(subset_plan["largest_component_size"]),
        "core_connectivity_edge_rows": int(subset_plan["core_connectivity_edge_rows"]),
        "reduction_profile_count": int(len(reduction_profiles_use)) if run_reduction_pipeline else 0,
        "coalescing_profile_count": int(len(coalesce_profiles_use)) if run_coalescing_pipeline else 0,
        "simulation_runs_per_variant": int(simulation_runs_per_variant),
        "reduction_builds_per_variant": int(reduction_builds_per_variant),
        "total_expected_simulation_runs": int(variant_count * simulation_runs_per_variant),
        "total_expected_reduction_builds": int(variant_count * reduction_builds_per_variant),
        "raw_edges_path": str(raw_edges_path),
        "final_edges_path": str(edges_path),
        "subset_rows_csv": str(subset_csv),
        "selected_core_order": [int(x) for x in subset_plan["ordered_core_ids"]],
    }
    plan_json = _write_json(paths["benchmark_outputs"] / "benchmark_plan.json", plan)
    return {
        "project_paths": paths,
        "plan": plan,
        "plan_json": plan_json,
        "subset_rows": subset_rows,
        "subset_csv": subset_csv,
        "network_build_report": network_build_report,
    }


def _fit_linear_runtime_metric(x: Sequence[Any], y: Sequence[Any], target_x: float) -> Dict[str, Any]:
    x_arr = pd.to_numeric(pd.Series(list(x)), errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy(float)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(ok.sum()) < 2:
        return {
            "n_points": int(ok.sum()),
            "slope_s_per_unit": float("nan"),
            "intercept_s": float("nan"),
            "estimate_at_target_s": float("nan"),
            "r2": float("nan"),
        }
    x_ok = x_arr[ok]
    y_ok = y_arr[ok]
    slope, intercept = np.polyfit(x_ok, y_ok, 1)
    y_hat = slope * x_ok + intercept
    ss_res = float(np.sum((y_ok - y_hat) ** 2))
    ss_tot = float(np.sum((y_ok - np.mean(y_ok)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    return {
        "n_points": int(len(x_ok)),
        "slope_s_per_unit": float(slope),
        "intercept_s": float(intercept),
        "estimate_at_target_s": float((slope * float(target_x)) + intercept),
        "r2": float(r2),
    }


def _safe_total(df: pd.DataFrame, cols: Sequence[str]) -> float:
    total = 0.0
    any_finite = False
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(vals).any():
            total += float(vals.fillna(0.0).sum())
            any_finite = True
    return float(total) if any_finite else float("nan")


def _save_benchmark_plots(
    agg_df: pd.DataFrame,
    *,
    figures_dir: str | Path,
    target_core_count: int,
    overall_means: Mapping[str, float],
) -> List[Path]:
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: List[Path] = []
    if agg_df.empty:
        return figures
    if not _can_plot():
        (figures_dir / "_plotting_skipped.txt").write_text(
            "matplotlib is not available in this Python environment, so benchmark figures were skipped.\n",
            encoding="utf-8",
        )
        return figures

    def _line_plot(y_col: str, title: str, ylabel: str, stem: str) -> None:
        if y_col not in agg_df.columns:
            return
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(agg_df["core_subset_size"], agg_df[y_col], marker="o", lw=1.8)
        ax.set_xlabel("Core hemilineage neurons in benchmark subset")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.axvline(float(target_core_count), ls="--", lw=1.0, color="tab:red", alpha=0.5)
        fig.tight_layout()
        out = figures_dir / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        figures.append(out)

    _line_plot(
        "baseline_total_wall_s_mean",
        "Baseline runtime across benchmark variants",
        "Mean wall time (s)",
        "baseline_total_wall_by_variant",
    )
    _line_plot(
        "cro_best_total_wall_s_mean",
        "Best CRO runtime across benchmark variants",
        "Mean wall time (s)",
        "cro_total_wall_by_variant",
    )
    _line_plot(
        "cro_sc_total_wall_s_mean",
        "Best CRO + SC runtime across benchmark variants",
        "Mean wall time (s)",
        "cro_sc_total_wall_by_variant",
    )
    _line_plot(
        "cro_search_total_wall_s_mean",
        "CRO search completion time across benchmark variants",
        "Mean wall time (s)",
        "cro_search_total_wall_by_variant",
    )
    _line_plot(
        "experiment_total_wall_s_mean",
        "Full benchmark experiment completion time across benchmark variants",
        "Mean wall time (s)",
        "full_experiment_total_wall_by_variant",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for col, label in (
        ("baseline_total_wall_s_mean", "Baseline"),
        ("cro_best_total_wall_s_mean", "CRO"),
        ("cro_sc_total_wall_s_mean", "CRO + SC"),
    ):
        if col in agg_df.columns:
            ax.plot(agg_df["core_subset_size"], agg_df[col], marker="o", lw=1.6, label=label)
    ax.set_xlabel("Core hemilineage neurons in benchmark subset")
    ax.set_ylabel("Mean wall time (s)")
    ax.set_title("Baseline vs CRO vs CRO + SC across variants")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out = figures_dir / "baseline_vs_cro_vs_cro_sc_across_variants.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    figures.append(out)

    compare_labels = ["Baseline", "CRO", "CRO + SC"]
    compare_vals = [
        float(overall_means.get("baseline_total_wall_s_mean", float("nan"))),
        float(overall_means.get("cro_best_total_wall_s_mean", float("nan"))),
        float(overall_means.get("cro_sc_total_wall_s_mean", float("nan"))),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(compare_labels, compare_vals, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("Average wall time across benchmark variants (s)")
    ax.set_title("Average completion time comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "average_completion_time_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    figures.append(out)
    return figures


def run_hemilineage_benchmark(
    *,
    projects_root: str | Path,
    swc_dir: str | Path,
    hemilineage_label: str,
    neuron_ids: Sequence[int],
    edge_set_name: str | None = None,
    edges_registry_root: str | Path | None = None,
    neuprint_dataset: str | None = None,
    seeds: Sequence[int] | None = None,
    template_config_path: str | Path | None = None,
    master_csv: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    biophysics: Mapping[str, Any] | None = None,
    stim: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
    gap: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    reduction_profiles: Sequence[Mapping[str, Any]] | None = None,
    coalesce_profiles: Sequence[Mapping[str, Any]] | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
    build_edges_workers: int = 16,
    reduction_workers: int = 32,
    reduction_write_map: bool = False,
    reduction_protect_synapses: bool = True,
    reduction_max_syn_points: int = 2000,
    min_core_size: int = 2,
    max_core_size: int = 20,
    repeats: int = 1,
    force_rebuild_edges: bool = False,
    phase1_fallback_enabled: bool = True,
    phase1_upsample_nm: float = 2000.0,
    phase1_min_conf: float = 0.4,
    phase1_batch_size: int = 10000,
    phase1_export_workers: int = 1,
    phase1_progress_every: int = 25,
    force_baseline_rerun: bool = False,
    force_reduction_rerun: bool = False,
    force_coalescing_rerun: bool = False,
    force_reduce_rebuild: bool = False,
    force_optimized_rerun: bool = False,
    run_reduction_pipeline: bool = True,
    run_coalescing_pipeline: bool = True,
    run_combined_pipeline: bool = True,
    run_notes: str = "",
) -> Dict[str, Any]:
    label = normalize_hemilineage_label(hemilineage_label)
    ids = coerce_int_ids(neuron_ids, name="core hemilineage neuron ids")
    seed_ids = coerce_int_ids(seeds or ids, name="seed ids")

    plan_payload = preview_hemilineage_benchmark_plan(
        projects_root=projects_root,
        swc_dir=swc_dir,
        hemilineage_label=label,
        neuron_ids=ids,
        edge_set_name=edge_set_name,
        edges_registry_root=edges_registry_root,
        neuprint_dataset=neuprint_dataset,
        seeds=seed_ids,
        master_csv=master_csv,
        timing=timing,
        build_edges_workers=build_edges_workers,
        reduction_profiles=reduction_profiles,
        coalesce_profiles=coalesce_profiles,
        min_core_size=min_core_size,
        max_core_size=max_core_size,
        repeats=repeats,
        force_rebuild_edges=force_rebuild_edges,
        phase1_fallback_enabled=phase1_fallback_enabled,
        phase1_upsample_nm=phase1_upsample_nm,
        phase1_min_conf=phase1_min_conf,
        phase1_batch_size=phase1_batch_size,
        phase1_export_workers=phase1_export_workers,
        phase1_progress_every=phase1_progress_every,
        run_reduction_pipeline=run_reduction_pipeline,
        run_coalescing_pipeline=run_coalescing_pipeline,
        run_combined_pipeline=run_combined_pipeline,
    )
    paths = dict(plan_payload["project_paths"])
    subset_rows = list(plan_payload["subset_rows"])
    threshold_use = dict(DEFAULT_THRESHOLDS)
    threshold_use.update(dict(thresholds or {}))
    progress_enabled = bool((runtime or {}).get("use_tqdm", True))

    raw_rows: List[Dict[str, Any]] = []
    per_variant_jsons: List[str] = []

    variant_tasks = [
        (subset_row, repeat_idx)
        for subset_row in subset_rows
        for repeat_idx in range(1, max(1, int(repeats)) + 1)
    ]

    for subset_row, repeat_idx in _maybe_tqdm(
        variant_tasks,
        total=len(variant_tasks),
        desc=f"{hemilineage_project_name(label)} benchmark",
        enabled=progress_enabled,
        leave=True,
    ):
        size = int(subset_row["core_subset_size"])
        subset_ids = coerce_int_ids(subset_row["selected_core_ids"], name="benchmark subset ids")
        bench_label = f"{label}_BENCH_N{size:02d}_R{repeat_idx:02d}"
        variant_t0 = time.perf_counter()
        project_run = run_full_hemilineage_project(
            projects_root=paths["benchmark_projects"],
            swc_dir=swc_dir,
            hemilineage_label=bench_label,
            neuron_ids=subset_ids,
            edge_set_name=(f"{edge_set_name}_n{size:02d}_r{repeat_idx:02d}" if edge_set_name else None),
            edges_registry_root=edges_registry_root,
            neuprint_dataset=neuprint_dataset,
            seeds=subset_ids,
            template_config_path=template_config_path,
            master_csv=master_csv,
            runtime=runtime,
            timing=timing,
            biophysics=biophysics,
            stim=stim,
            record=record,
            gap=gap,
            thresholds=threshold_use,
            reduction_profiles=reduction_profiles,
            coalesce_profiles=coalesce_profiles,
            extra_overrides=extra_overrides,
            force_rebuild_edges=bool(force_rebuild_edges),
            force_baseline_rerun=bool(force_baseline_rerun),
            force_reduction_rerun=bool(force_reduction_rerun),
            force_coalescing_rerun=bool(force_coalescing_rerun),
            force_reduce_rebuild=bool(force_reduce_rebuild),
            force_optimized_rerun=bool(force_optimized_rerun),
            reduction_workers=int(reduction_workers),
            reduction_write_map=bool(reduction_write_map),
            reduction_protect_synapses=bool(reduction_protect_synapses),
            reduction_max_syn_points=int(reduction_max_syn_points),
            build_edges_workers=int(build_edges_workers),
            phase1_fallback_enabled=bool(phase1_fallback_enabled),
            phase1_upsample_nm=float(phase1_upsample_nm),
            phase1_min_conf=float(phase1_min_conf),
            phase1_batch_size=int(phase1_batch_size),
            phase1_export_workers=int(phase1_export_workers),
            phase1_progress_every=int(phase1_progress_every),
            run_reduction_pipeline=bool(run_reduction_pipeline),
            run_coalescing_pipeline=bool(run_coalescing_pipeline),
            run_combined_pipeline=bool(run_combined_pipeline),
            run_notes=str(run_notes or f"benchmark subset size={size} repeat={repeat_idx}"),
        )
        variant_total_wall_s = float(time.perf_counter() - variant_t0)
        base_runs_root = Path(project_run["base_config"]["runs_root"]).expanduser().resolve()
        baseline_timing = load_phase_timing_summary(project_run["baseline_out_dir"])

        row: Dict[str, Any] = {
            "benchmark_label": bench_label,
            "repeat_index": int(repeat_idx),
            "core_subset_size": int(size),
            "selected_core_ids": json.dumps([int(x) for x in subset_ids]),
            "core_internal_edge_rows": int(subset_row["core_internal_edge_rows"]),
            "added_motor_neuron_count": int(project_run["network_build_report"].get("added_motor_neuron_count", 0)),
            "final_network_count": int(len(project_run["final_network_ids"])),
            "project_root": str(project_run["project_paths"]["project_root"]),
            "baseline_run_id": str(Path(project_run["baseline_out_dir"]).name),
            "baseline_build_wall_s": float(baseline_timing.get("build_wall_s", float("nan"))),
            "baseline_sim_wall_s": float(baseline_timing.get("sim_wall_s", float("nan"))),
            "baseline_total_wall_s": float(baseline_timing.get("total_wall_s", float(project_run["baseline_wall_s"]))),
            "baseline_backend": baseline_timing.get("backend"),
            "baseline_integrator": baseline_timing.get("integrator"),
            "variant_total_wall_s": float(variant_total_wall_s),
            "experiment_total_wall_s": float(variant_total_wall_s),
        }

        reduction_search_total = float("nan")
        coalescing_search_total = float("nan")
        cro_timing = {}
        cro_sc_timing = {}
        if "reduction" in project_run:
            reduction_df = project_run["reduction"]["results_df"].copy()
            reduction_search_total = _safe_total(reduction_df, ["reduction_elapsed_s", "reduced_total_wall_s"])
            best_red_row = project_run["reduction"]["best_row"]
            best_red_out_dir = (base_runs_root / str(best_red_row["reduced_run_id"])).resolve()
            cro_timing = load_phase_timing_summary(best_red_out_dir)
            row.update(
                {
                    "best_reduction_profile": str(project_run["reduction"]["best_profile"]),
                    "cro_search_total_wall_s": float(reduction_search_total),
                    "cro_best_run_id": str(best_red_row["reduced_run_id"]),
                    "cro_best_build_wall_s": float(cro_timing.get("build_wall_s", float("nan"))),
                    "cro_best_sim_wall_s": float(cro_timing.get("sim_wall_s", float("nan"))),
                    "cro_best_total_wall_s": float(cro_timing.get("total_wall_s", float("nan"))),
                }
            )
        else:
            row.update(
                {
                    "best_reduction_profile": None,
                    "cro_search_total_wall_s": float("nan"),
                    "cro_best_run_id": None,
                    "cro_best_build_wall_s": float("nan"),
                    "cro_best_sim_wall_s": float("nan"),
                    "cro_best_total_wall_s": float("nan"),
                }
            )

        if "coalescing" in project_run:
            coalesce_df = project_run["coalescing"]["results_df"].copy()
            coalescing_search_total = _safe_total(coalesce_df, ["test_wall_s"])
            best_coal_row = project_run["coalescing"]["best_row"]
            row.update(
                {
                    "best_coalescing_profile": str(project_run["coalescing"]["best_profile"]),
                    "coalescing_search_total_wall_s": float(coalescing_search_total),
                    "best_coalescing_run_id": str(best_coal_row["run_id"]),
                }
            )
        else:
            row.update(
                {
                    "best_coalescing_profile": None,
                    "coalescing_search_total_wall_s": float("nan"),
                    "best_coalescing_run_id": None,
                }
            )

        if "optimized" in project_run:
            cro_sc_timing = load_phase_timing_summary(project_run["optimized"]["optimized_out_dir"])
            row.update(
                {
                    "cro_sc_run_id": str(Path(project_run["optimized"]["optimized_out_dir"]).name),
                    "cro_sc_build_wall_s": float(cro_sc_timing.get("build_wall_s", float("nan"))),
                    "cro_sc_sim_wall_s": float(cro_sc_timing.get("sim_wall_s", float("nan"))),
                    "cro_sc_total_wall_s": float(cro_sc_timing.get("total_wall_s", float("nan"))),
                }
            )
        else:
            row.update(
                {
                    "cro_sc_run_id": None,
                    "cro_sc_build_wall_s": float("nan"),
                    "cro_sc_sim_wall_s": float("nan"),
                    "cro_sc_total_wall_s": float("nan"),
                }
            )

        known_runtime = 0.0
        any_known = False
        for key in (
            "baseline_total_wall_s",
            "cro_search_total_wall_s",
            "coalescing_search_total_wall_s",
            "cro_sc_total_wall_s",
        ):
            val = float(row.get(key, float("nan")))
            if np.isfinite(val):
                known_runtime += val
                any_known = True
        row["known_runtime_components_wall_s"] = float(known_runtime) if any_known else float("nan")
        row["non_run_overhead_wall_s"] = (
            float(variant_total_wall_s - known_runtime) if any_known else float("nan")
        )

        raw_rows.append(row)
        variant_json = _write_json(
            paths["benchmark_outputs"] / f"{bench_label.lower()}_summary.json",
            row,
        )
        per_variant_jsons.append(str(variant_json))

    raw_df = pd.DataFrame(raw_rows).sort_values(["core_subset_size", "repeat_index"]).reset_index(drop=True)
    raw_csv = paths["benchmark_outputs"] / "benchmark_raw_runs.csv"
    raw_df.to_csv(raw_csv, index=False)

    numeric_cols = [
        col
        for col in raw_df.columns
        if col not in {"repeat_index", "baseline_backend", "baseline_integrator"} and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    agg_df = raw_df.groupby("core_subset_size", as_index=False)[numeric_cols].mean(numeric_only=True)
    repeat_counts = raw_df.groupby("core_subset_size").size().rename("repeat_count").reset_index()
    agg_df = agg_df.merge(repeat_counts, on="core_subset_size", how="left")
    agg_df = agg_df.rename(columns={col: f"{col}_mean" for col in numeric_cols if col != "core_subset_size"})
    if "core_subset_size_mean" in agg_df.columns:
        agg_df = agg_df.drop(columns=["core_subset_size_mean"])

    for metric in (
        "baseline_total_wall_s",
        "cro_search_total_wall_s",
        "cro_best_total_wall_s",
        "cro_sc_total_wall_s",
        "experiment_total_wall_s",
        "variant_total_wall_s",
        "baseline_build_wall_s",
        "baseline_sim_wall_s",
        "cro_best_build_wall_s",
        "cro_best_sim_wall_s",
        "cro_sc_build_wall_s",
        "cro_sc_sim_wall_s",
        "final_network_count",
    ):
        if metric in raw_df.columns:
            std_series = raw_df.groupby("core_subset_size")[metric].std(ddof=0).rename(f"{metric}_std").reset_index()
            agg_df = agg_df.merge(std_series, on="core_subset_size", how="left")

    agg_csv = paths["benchmark_outputs"] / "benchmark_by_size.csv"
    agg_df.to_csv(agg_csv, index=False)

    growth_rows: List[Dict[str, Any]] = []
    for idx in range(1, len(agg_df)):
        prev = agg_df.iloc[idx - 1]
        cur = agg_df.iloc[idx]
        delta_core = float(cur["core_subset_size"] - prev["core_subset_size"])
        delta_final = float(cur.get("final_network_count_mean", float("nan")) - prev.get("final_network_count_mean", float("nan")))
        growth_rows.append(
            {
                "from_core_subset_size": int(prev["core_subset_size"]),
                "to_core_subset_size": int(cur["core_subset_size"]),
                "delta_core_neurons": float(delta_core),
                "delta_final_network_neurons": float(delta_final),
                "baseline_total_delta_s_per_core_neuron": float(cur.get("baseline_total_wall_s_mean", float("nan")) - prev.get("baseline_total_wall_s_mean", float("nan"))) / delta_core,
                "cro_search_delta_s_per_core_neuron": float(cur.get("cro_search_total_wall_s_mean", float("nan")) - prev.get("cro_search_total_wall_s_mean", float("nan"))) / delta_core,
                "cro_sc_total_delta_s_per_core_neuron": float(cur.get("cro_sc_total_wall_s_mean", float("nan")) - prev.get("cro_sc_total_wall_s_mean", float("nan"))) / delta_core,
                "experiment_total_delta_s_per_core_neuron": float(cur.get("variant_total_wall_s_mean", float("nan")) - prev.get("variant_total_wall_s_mean", float("nan"))) / delta_core,
            }
        )
    growth_df = pd.DataFrame(growth_rows)
    growth_csv = paths["benchmark_outputs"] / "benchmark_growth_rates.csv"
    growth_df.to_csv(growth_csv, index=False)

    target_final_count = float(plan_payload["plan"]["full_final_network_count"])
    target_core_count = int(plan_payload["plan"]["full_core_count"])
    estimate_rows: List[Dict[str, Any]] = []
    for metric, metric_label in (
        ("baseline_total_wall_s_mean", "baseline"),
        ("cro_search_total_wall_s_mean", "cro_search"),
        ("cro_best_total_wall_s_mean", "cro_best"),
        ("cro_sc_total_wall_s_mean", "cro_sc"),
        ("experiment_total_wall_s_mean", "full_experiment"),
    ):
        if metric not in agg_df.columns:
            continue
        fit = _fit_linear_runtime_metric(agg_df["final_network_count_mean"], agg_df[metric], target_final_count)
        estimate_rows.append(
            {
                "metric": metric_label,
                "target_final_network_count": float(target_final_count),
                **fit,
            }
        )
    estimate_df = pd.DataFrame(estimate_rows)
    estimate_csv = paths["benchmark_outputs"] / "benchmark_estimates.csv"
    estimate_df.to_csv(estimate_csv, index=False)

    overall_means = {}
    for metric in (
        "baseline_total_wall_s_mean",
        "cro_best_total_wall_s_mean",
        "cro_sc_total_wall_s_mean",
        "cro_search_total_wall_s_mean",
        "experiment_total_wall_s_mean",
    ):
        if metric in agg_df.columns:
            overall_means[metric] = float(pd.to_numeric(agg_df[metric], errors="coerce").mean())

    figures = _save_benchmark_plots(
        agg_df,
        figures_dir=paths["benchmark_figures"],
        target_core_count=target_core_count,
        overall_means=overall_means,
    )

    summary = {
        "hemilineage": label,
        "plan_json": str(plan_payload["plan_json"]),
        "raw_csv": str(raw_csv),
        "agg_csv": str(agg_csv),
        "growth_csv": str(growth_csv),
        "estimate_csv": str(estimate_csv),
        "figure_paths": [str(fig) for fig in figures],
        "variant_jsons": per_variant_jsons,
        "target_core_count": int(target_core_count),
        "target_final_network_count": float(target_final_count),
        "overall_mean_wall_s": overall_means,
    }
    summary_json = _write_json(paths["benchmark_outputs"] / "benchmark_summary.json", summary)

    summary_lines = [
        f"Hemilineage benchmark: {label}",
        f"Target core count: {target_core_count}",
        f"Target final network count (with motor expansion): {int(target_final_count)}",
        f"Benchmark variants completed: {int(len(raw_df))}",
    ]
    if "baseline_total_wall_s_mean" in overall_means:
        summary_lines.append(f"Average baseline wall time across variants: {overall_means['baseline_total_wall_s_mean']:.2f} s")
    if "cro_search_total_wall_s_mean" in overall_means:
        summary_lines.append(f"Average CRO search wall time across variants: {overall_means['cro_search_total_wall_s_mean']:.2f} s")
    if "cro_sc_total_wall_s_mean" in overall_means:
        summary_lines.append(f"Average CRO + SC wall time across variants: {overall_means['cro_sc_total_wall_s_mean']:.2f} s")
    if "experiment_total_wall_s_mean" in overall_means:
        summary_lines.append(f"Average full benchmark experiment wall time across variants: {overall_means['experiment_total_wall_s_mean']:.2f} s")
    if not estimate_df.empty:
        for row in estimate_df.itertuples(index=False):
            if np.isfinite(float(row.estimate_at_target_s)):
                summary_lines.append(
                    f"Linear estimate for {row.metric} at final network size: {float(row.estimate_at_target_s):.2f} s "
                    f"(slope={float(row.slope_s_per_unit):.4f} s/neuron, R^2={float(row.r2):.3f})"
                )
    summary_txt = paths["benchmark_outputs"] / "benchmark_summary.txt"
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "plan": plan_payload["plan"],
        "plan_json": plan_payload["plan_json"],
        "raw_df": raw_df,
        "raw_csv": raw_csv,
        "agg_df": agg_df,
        "agg_csv": agg_csv,
        "growth_df": growth_df,
        "growth_csv": growth_csv,
        "estimate_df": estimate_df,
        "estimate_csv": estimate_csv,
        "summary_json": summary_json,
        "summary_txt": summary_txt,
        "figures": figures,
    }
