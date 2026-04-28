from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .artifacts import build_failure_report, prepare_workbench_bundle, record_execution_status, write_plan_copy
from .cache_identity import summarize_impacts
from .presets import get_preset
from .validation import clean_optional_text, parse_int_list, parse_json_block, parse_path_list, validate_state


@dataclass
class ExecutionPlan:
    preset_slug: str
    preset_label: str
    runner_kind: str
    description: str
    run_id: str
    payload: Dict[str, Any]
    impact_summary: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_execution_plan(state: Mapping[str, Any], *, preset_slug: str) -> ExecutionPlan:
    report = validate_state(state)
    if not report.ok:
        raise ValueError(report.format())

    preset = get_preset(preset_slug)
    runner_kind = str(state.get("runner_kind", "shared_runner")).strip()
    run_id = clean_optional_text(state.get("run_id")) or preset_slug.replace("-", "_")
    if runner_kind == "shared_runner":
        payload = _build_shared_payload(state, run_id=run_id)
        description = f"{preset.label}: shared runner ({payload['selection']['mode']})"
    elif runner_kind == "hemilineage_project":
        payload = _build_hemi_project_payload(state, run_id=run_id)
        description = f"{preset.label}: hemilineage project pipeline"
    else:
        raise ValueError(f"Unknown runner kind: {runner_kind}")
    return ExecutionPlan(
        preset_slug=preset_slug,
        preset_label=preset.label,
        runner_kind=runner_kind,
        description=description,
        run_id=run_id,
        payload=payload,
        impact_summary=summarize_impacts(state, preset_slug=preset_slug),
    )


def execute_plan(state: Mapping[str, Any], *, preset_slug: str, phase2_root: str | Path) -> Dict[str, Any]:
    plan = build_execution_plan(state, preset_slug=preset_slug)
    bundle_dir = prepare_workbench_bundle(
        phase2_root,
        run_id=plan.run_id,
        preset_slug=plan.preset_slug,
        state=state,
        plan=plan.to_dict(),
    )
    record_execution_status(bundle_dir, status="planned")
    try:
        if plan.runner_kind == "shared_runner":
            from digifly.phase2.walking.runner import run_walking_simulation

            out_dir = Path(run_walking_simulation(plan.payload)).expanduser().resolve()
            result = {
                "runner_kind": "shared_runner",
                "output_dir": str(out_dir),
            }
            write_plan_copy(out_dir, state=state, plan=plan.to_dict())
            record_execution_status(out_dir, status="completed", result=result)
        else:
            from digifly.phase2.hemi.sim_project import run_full_hemilineage_project

            output = run_full_hemilineage_project(**plan.payload)
            project_paths = output.get("project_paths") or {}
            metadata_dir = Path(project_paths.get("metadata_dir")).expanduser().resolve()
            result = {
                "runner_kind": "hemilineage_project",
                "metadata_dir": str(metadata_dir),
                "project_root": str(project_paths.get("project_root")),
                "baseline_out_dir": str(output.get("baseline_out_dir")),
            }
            write_plan_copy(metadata_dir, state=state, plan=plan.to_dict())
            record_execution_status(metadata_dir, status="completed", result=result)

        record_execution_status(bundle_dir, status="completed", result=result)
        return {
            "bundle_dir": str(bundle_dir),
            "plan": plan.to_dict(),
            "result": result,
        }
    except Exception as exc:
        failure = build_failure_report(exc)
        record_execution_status(bundle_dir, status="failed", failure=failure)
        raise


def _build_shared_payload(state: Mapping[str, Any], *, run_id: str) -> Dict[str, Any]:
    mode = str(state.get("mode", "hemilineage")).strip()
    selection: Dict[str, Any]
    edges_path: Optional[str] = None
    if mode == "single":
        selection = {"mode": "single", "neuron_id": int(state.get("neuron_id", 0))}
    elif mode == "custom":
        selection = {
            "mode": "custom",
            "neuron_ids": parse_int_list(state.get("neuron_ids_text", ""), allow_empty=False),
        }
        edges_path = clean_optional_text(state.get("edges_path"))
    elif mode == "hemilineage":
        selection = {"mode": "hemilineage", "label": clean_optional_text(state.get("hemi_label"))}
    else:
        raise ValueError(f"Unknown shared runner mode: {mode}")

    stim = {
        "iclamp": {
            "amp_nA": float(state.get("iclamp_amp_nA", 0.0)),
            "delay_ms": float(state.get("iclamp_delay_ms", 0.0)),
            "dur_ms": float(state.get("iclamp_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
        },
    }
    if bool(state.get("neg_pulse_enabled", False)):
        stim["neg_pulse"] = {
            "enabled": True,
            "amp_nA": float(state.get("neg_pulse_amp_nA", -1.0)),
            "delay_ms": float(state.get("neg_pulse_delay_ms", 0.0)),
            "dur_ms": float(state.get("neg_pulse_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
        }
    if bool(state.get("pulse_train_enabled", False)):
        pulse_train = {
            "enabled": True,
            "freq_hz": float(state.get("pulse_train_freq_hz", 0.0)),
            "amp_nA": float(state.get("pulse_train_amp_nA", 0.0)),
            "delay_ms": float(state.get("pulse_train_delay_ms", 0.0)),
            "dur_ms": float(state.get("pulse_train_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
            "include_base_iclamp": bool(state.get("pulse_train_include_base", False)),
        }
        stop_ms = _none_if_zero(state.get("pulse_train_stop_ms"))
        max_pulses = _none_if_zero(state.get("pulse_train_max_pulses"))
        if stop_ms is not None:
            pulse_train["stop_ms"] = float(stop_ms)
        if max_pulses is not None:
            pulse_train["max_pulses"] = int(max_pulses)
        stim["pulse_train"] = pulse_train

    gap_pairs = parse_json_block(state.get("gap_pairs_json", "[]"), expect=(list, dict), label="gap_pairs_json")
    if isinstance(gap_pairs, dict):
        gap_pairs = [gap_pairs]

    payload: Dict[str, Any] = {
        "swc_dir": clean_optional_text(state.get("swc_dir")),
        "morph_swc_dir": clean_optional_text(state.get("morph_swc_dir")),
        "master_csv": clean_optional_text(state.get("master_csv")),
        "edges_root": clean_optional_text(state.get("edges_root")),
        "runs_root": clean_optional_text(state.get("runs_root")),
        "selection": selection,
        "seeds": parse_int_list(state.get("seeds_text", "")),
        "edges_path": edges_path,
        "run_id": run_id,
        "run_notes": str(state.get("run_notes", "")),
        "tstop_ms": float(state.get("tstop_ms", 0.0)),
        "dt_ms": float(state.get("dt_ms", 0.1)),
        "threads": _none_if_zero(state.get("threads")),
        "enable_coreneuron": bool(state.get("enable_coreneuron", False)),
        "coreneuron_gpu": bool(state.get("coreneuron_gpu", False)),
        "coreneuron_nthread": _none_if_zero(state.get("coreneuron_nthread")),
        "io_workers": _none_if_zero(state.get("io_workers")),
        "progress": bool(state.get("progress", True)),
        "parallel": {
            "build_backend": str(state.get("build_backend", "single_host")),
            "ownership_strategy": str(state.get("ownership_strategy", "round_robin")),
            "maxstep_ms": float(state.get("maxstep_ms", 10.0)),
        },
        "default_weight_uS": float(state.get("default_weight_uS", 0.000003)),
        "default_delay_ms": float(state.get("default_delay_ms", 1.0)),
        "syn_tau1_ms": float(state.get("syn_tau1_ms", 0.5)),
        "syn_tau2_ms": float(state.get("syn_tau2_ms", 3.0)),
        "syn_e_rev_mV": float(state.get("syn_e_rev_mV", 0.0)),
        "post_active": bool(state.get("post_active", True)),
        "active_posts_mode": str(state.get("active_posts_mode", "all_selected")),
        "stim": stim,
        "record": {
            "soma_v": str(state.get("record_soma_v", "seeds")),
            "spikes": str(state.get("record_spikes", "seeds")),
            "spike_thresh_mV": float(state.get("spike_thresh_mV", 0.0)),
        },
        "gap": {
            "enabled": bool(state.get("gap_enabled", False)),
            "mechanisms_dir": clean_optional_text(state.get("gap_mechanisms_dir")),
            "default_site": str(state.get("gap_default_site", "ais")),
            "default_g_uS": float(state.get("gap_default_g_uS", 0.001)),
            "pairs": gap_pairs,
        },
    }

    if mode == "custom" and bool(state.get("edge_cache_enabled", False)):
        payload["edge_cache"] = {
            "enabled": True,
            "build_if_missing": bool(state.get("edge_cache_build_if_missing", True)),
            "force_rebuild": bool(state.get("edge_cache_force_rebuild", False)),
            "build_mode": str(state.get("edge_cache_build_mode", "from_edges_files")),
            "source_paths": parse_path_list(state.get("edge_cache_source_paths_text", "")),
            "query": {
                "mode": str(state.get("edge_cache_query_mode", "loaded_subgraph")),
            },
        }

    extra = parse_json_block(state.get("shared_overrides_json", "{}"), expect=dict, label="shared_overrides_json")
    payload = _deep_merge(payload, extra)
    return _drop_nones(payload)


def _build_hemi_project_payload(state: Mapping[str, Any], *, run_id: str) -> Dict[str, Any]:
    hemi_label = clean_optional_text(state.get("hemi_label")) or ""
    neuron_ids = parse_int_list(state.get("hemi_core_ids_text", ""))
    if not neuron_ids:
        neuron_ids = _load_core_ids_from_master(
            clean_optional_text(state.get("master_csv")),
            clean_optional_text(state.get("swc_dir")),
            hemi_label,
        )

    runtime = {
        "tstop_ms": float(state.get("tstop_ms", 0.0)),
        "dt_ms": float(state.get("dt_ms", 0.1)),
        "threads": _none_if_zero(state.get("threads")),
        "enable_coreneuron": bool(state.get("enable_coreneuron", False)),
        "coreneuron_gpu": bool(state.get("coreneuron_gpu", False)),
        "coreneuron_nthread": _none_if_zero(state.get("coreneuron_nthread")),
        "io_workers": _none_if_zero(state.get("io_workers")),
        "progress": bool(state.get("progress", True)),
    }
    timing = {
        "default_weight_uS": float(state.get("default_weight_uS", 0.000003)),
        "default_delay_ms": float(state.get("default_delay_ms", 1.0)),
        "syn_tau1_ms": float(state.get("syn_tau1_ms", 0.5)),
        "syn_tau2_ms": float(state.get("syn_tau2_ms", 3.0)),
        "syn_e_rev_mV": float(state.get("syn_e_rev_mV", 0.0)),
    }
    biophysics = {
        "post_active": bool(state.get("post_active", True)),
    }
    stim = {
        "iclamp": {
            "amp_nA": float(state.get("iclamp_amp_nA", 0.0)),
            "delay_ms": float(state.get("iclamp_delay_ms", 0.0)),
            "dur_ms": float(state.get("iclamp_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
        }
    }
    if bool(state.get("neg_pulse_enabled", False)):
        stim["neg_pulse"] = {
            "enabled": True,
            "amp_nA": float(state.get("neg_pulse_amp_nA", -1.0)),
            "delay_ms": float(state.get("neg_pulse_delay_ms", 0.0)),
            "dur_ms": float(state.get("neg_pulse_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
        }
    if bool(state.get("pulse_train_enabled", False)):
        stim["pulse_train"] = {
            "enabled": True,
            "freq_hz": float(state.get("pulse_train_freq_hz", 0.0)),
            "amp_nA": float(state.get("pulse_train_amp_nA", 0.0)),
            "delay_ms": float(state.get("pulse_train_delay_ms", 0.0)),
            "dur_ms": float(state.get("pulse_train_dur_ms", 0.0)),
            "location": str(state.get("iclamp_location", "soma")),
            "include_base_iclamp": bool(state.get("pulse_train_include_base", False)),
        }
    record = {
        "soma_v": str(state.get("record_soma_v", "seeds")),
        "spikes": str(state.get("record_spikes", "seeds")),
        "spike_thresh_mV": float(state.get("spike_thresh_mV", 0.0)),
    }
    gap_pairs = parse_json_block(state.get("gap_pairs_json", "[]"), expect=(list, dict), label="gap_pairs_json")
    if isinstance(gap_pairs, dict):
        gap_pairs = [gap_pairs]
    gap = {
        "enabled": bool(state.get("gap_enabled", False)),
        "mechanisms_dir": clean_optional_text(state.get("gap_mechanisms_dir")),
        "default_site": str(state.get("gap_default_site", "ais")),
        "default_g_uS": float(state.get("gap_default_g_uS", 0.001)),
        "pairs": gap_pairs,
    }

    runtime = _deep_merge(runtime, parse_json_block(state.get("runtime_json", "{}"), expect=dict, label="runtime_json"))
    timing = _deep_merge(timing, parse_json_block(state.get("timing_json", "{}"), expect=dict, label="timing_json"))
    biophysics = _deep_merge(biophysics, parse_json_block(state.get("biophysics_json", "{}"), expect=dict, label="biophysics_json"))
    stim = _deep_merge(stim, parse_json_block(state.get("stim_json", "{}"), expect=dict, label="stim_json"))
    record = _deep_merge(record, parse_json_block(state.get("record_json", "{}"), expect=dict, label="record_json"))
    gap = _deep_merge(gap, parse_json_block(state.get("gap_json", "{}"), expect=dict, label="gap_json"))

    payload = {
        "projects_root": clean_optional_text(state.get("projects_root")),
        "swc_dir": clean_optional_text(state.get("swc_dir")),
        "morph_swc_dir": clean_optional_text(state.get("morph_swc_dir")),
        "hemilineage_label": hemi_label,
        "neuron_ids": neuron_ids,
        "edge_set_name": clean_optional_text(state.get("edge_set_name")),
        "edges_registry_root": clean_optional_text(state.get("edges_registry_root")),
        "neuprint_dataset": clean_optional_text(state.get("neuprint_dataset")),
        "seeds": parse_int_list(state.get("seeds_text", "")),
        "template_config_path": clean_optional_text(state.get("template_config_path")),
        "master_csv": clean_optional_text(state.get("master_csv")),
        "runtime": runtime,
        "timing": timing,
        "biophysics": biophysics,
        "stim": stim,
        "record": record,
        "gap": gap,
        "thresholds": parse_json_block(state.get("thresholds_json", "{}"), expect=dict, label="thresholds_json"),
        "reduction_profiles": parse_json_block(state.get("reduction_profiles_json", "[]"), expect=list, label="reduction_profiles_json"),
        "coalesce_profiles": parse_json_block(state.get("coalesce_profiles_json", "[]"), expect=list, label="coalesce_profiles_json"),
        "extra_overrides": parse_json_block(state.get("extra_overrides_json", "{}"), expect=dict, label="extra_overrides_json"),
        "force_rebuild_edges": bool(state.get("force_rebuild_edges", False)),
        "reduction_workers": int(state.get("reduction_workers", 32)),
        "build_edges_workers": int(state.get("build_edges_workers", 16)),
        "run_reduction_pipeline": bool(state.get("run_reduction_pipeline", False)),
        "run_coalescing_pipeline": bool(state.get("run_coalescing_pipeline", False)),
        "run_combined_pipeline": bool(state.get("run_combined_pipeline", False)),
        "run_notes": str(state.get("run_notes", "")),
        "run_id_suffix": run_id,
    }
    return _drop_nones(payload)


def _load_core_ids_from_master(master_csv: str | None, swc_dir: str | None, hemi_label: str) -> List[int]:
    if not hemi_label:
        raise ValueError("Cannot derive hemilineage IDs without a hemilineage label.")
    if master_csv:
        master_path = Path(master_csv).expanduser().resolve()
    elif swc_dir:
        master_path = (Path(swc_dir).expanduser().resolve().parent / "all_neurons_neuroncriteria_template.csv").resolve()
    else:
        raise ValueError("Cannot derive hemilineage IDs without a master CSV or SWC root.")
    df = pd.read_csv(master_path)
    cols = {str(col).lower(): str(col) for col in df.columns}
    if "hemilineage" not in cols or "bodyid" not in cols:
        raise ValueError("Master CSV must contain hemilineage and bodyId columns.")
    mask = df[cols["hemilineage"]].astype(str).str.strip().str.lower() == hemi_label.strip().lower()
    ids = pd.to_numeric(df.loc[mask, cols["bodyid"]], errors="coerce").dropna().astype(int).tolist()
    if not ids:
        raise ValueError(f"No core IDs found in master CSV for hemilineage '{hemi_label}'.")
    return [int(x) for x in ids]


def _none_if_zero(value: Any) -> Any:
    if value in (None, "", 0, 0.0, False):
        return None
    return value


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), dict):
            payload[key] = _deep_merge(payload[key], value)
        else:
            payload[key] = value
    return payload


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = _drop_nones(item)
            if cleaned is not None:
                result[key] = cleaned
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            cleaned = _drop_nones(item)
            if cleaned is not None:
                result.append(cleaned)
        return result
    if value in (None, "", []):
        return None
    return value
