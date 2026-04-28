from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines: List[str] = []
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {msg}" for msg in self.errors)
        if self.warnings:
            if lines:
                lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {msg}" for msg in self.warnings)
        if not lines:
            return "Validation OK."
        return "\n".join(lines)


def parse_json_block(raw: Any, *, expect: type | tuple[type, ...] = dict, label: str = "JSON block") -> Any:
    if raw in (None, "", {}):
        return {} if expect in (dict, (dict,)) else []
    if isinstance(raw, expect):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be {expect} or a JSON string.")
    text = raw.strip()
    if not text:
        return {} if expect in (dict, (dict,)) else []
    value = json.loads(text)
    if not isinstance(value, expect):
        raise ValueError(f"{label} must decode to {expect}, got {type(value).__name__}.")
    return value


def parse_int_list(raw: Any, *, allow_empty: bool = True) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        text = str(raw).strip()
        if not text:
            return [] if allow_empty else _raise_value_error("Expected at least one integer.")
        text = text.replace("\n", ",")
        values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    result: List[int] = []
    for value in values:
        result.append(int(value))
    return result


def parse_path_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    text = text.replace("\n", ",")
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def clean_optional_text(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def validate_state(state: Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    runner_kind = str(state.get("runner_kind", "shared_runner")).strip()
    mode = str(state.get("mode", "hemilineage")).strip()

    swc_dir_text = clean_optional_text(state.get("swc_dir"))
    if not swc_dir_text:
        report.errors.append("SWC root is required.")
    else:
        swc_path = Path(swc_dir_text).expanduser().resolve()
        if not swc_path.exists() or not swc_path.is_dir():
            report.errors.append(f"SWC root does not exist: {swc_path}")

    for key in (
        "shared_overrides_json",
        "runtime_json",
        "timing_json",
        "biophysics_json",
        "stim_json",
        "record_json",
        "gap_json",
        "thresholds_json",
        "extra_overrides_json",
    ):
        try:
            parse_json_block(state.get(key, "{}"), expect=dict, label=key)
        except Exception as exc:
            report.errors.append(f"{key}: {exc}")

    for key in ("gap_pairs_json",):
        try:
            parse_json_block(state.get(key, "[]"), expect=(list, dict), label=key)
        except Exception as exc:
            report.errors.append(f"{key}: {exc}")

    for key in ("reduction_profiles_json", "coalesce_profiles_json"):
        try:
            parse_json_block(state.get(key, "[]"), expect=list, label=key)
        except Exception as exc:
            report.errors.append(f"{key}: {exc}")

    try:
        parse_int_list(state.get("seeds_text", ""))
    except Exception as exc:
        report.errors.append(f"Seed IDs: {exc}")

    if runner_kind == "shared_runner":
        if mode == "single":
            try:
                int(state.get("neuron_id", 0))
            except Exception as exc:
                report.errors.append(f"Single neuron ID: {exc}")
        elif mode == "custom":
            try:
                ids = parse_int_list(state.get("neuron_ids_text", ""), allow_empty=False)
                if not ids:
                    report.errors.append("Custom mode needs at least one neuron ID.")
            except Exception as exc:
                report.errors.append(f"Custom neuron IDs: {exc}")

            edge_cache_enabled = bool(state.get("edge_cache_enabled", False))
            edges_path = clean_optional_text(state.get("edges_path"))
            if not edge_cache_enabled and not edges_path:
                report.errors.append("Custom mode needs either an explicit edges file or edge cache enabled.")
            if edges_path:
                edge_path = Path(edges_path).expanduser().resolve()
                if not edge_path.exists():
                    report.errors.append(f"Explicit edges file does not exist: {edge_path}")
            if edge_cache_enabled:
                for path_text in parse_path_list(state.get("edge_cache_source_paths_text", "")):
                    edge_path = Path(path_text).expanduser().resolve()
                    if not edge_path.exists():
                        report.errors.append(f"Edge cache source path does not exist: {edge_path}")
        elif mode == "hemilineage":
            hemi_label = clean_optional_text(state.get("hemi_label"))
            if not hemi_label:
                report.errors.append("Hemilineage label is required for hemilineage mode.")
            master_path = _resolved_master_csv_path(state)
            if master_path is not None and not master_path.exists():
                report.errors.append(f"Master CSV does not exist: {master_path}")
    elif runner_kind == "hemilineage_project":
        hemi_label = clean_optional_text(state.get("hemi_label"))
        if not hemi_label:
            report.errors.append("Hemilineage project runs require a hemilineage label.")
        projects_root = clean_optional_text(state.get("projects_root"))
        if not projects_root:
            report.errors.append("Projects root is required for the hemilineage project runner.")
        try:
            parse_int_list(state.get("hemi_core_ids_text", ""))
        except Exception as exc:
            report.errors.append(f"Core hemilineage IDs: {exc}")
        master_path = _resolved_master_csv_path(state)
        if master_path is not None and not master_path.exists():
            report.errors.append(f"Master CSV does not exist: {master_path}")
        elif master_path is not None:
            ids = parse_int_list(state.get("hemi_core_ids_text", ""))
            if not ids and not hemi_label:
                report.errors.append("Need either core hemilineage IDs or a hemilineage label that can be resolved from the master CSV.")
        else:
            report.warnings.append("Master CSV could not be resolved. Core IDs will need to be supplied explicitly.")
    else:
        report.errors.append(f"Unknown runner kind: {runner_kind}")

    if bool(state.get("enable_coreneuron", False)) and bool(_json_flag_enabled(state.get("runtime_json"), "cvode", "enabled")):
        report.warnings.append("CoreNEURON and CVODE do not mix well. Double-check the runtime JSON before launching.")

    return report


def _resolved_master_csv_path(state: Mapping[str, Any]) -> Path | None:
    master_override = clean_optional_text(state.get("master_csv"))
    if master_override:
        return Path(master_override).expanduser().resolve()
    swc_dir_text = clean_optional_text(state.get("swc_dir"))
    if not swc_dir_text:
        return None
    return (Path(swc_dir_text).expanduser().resolve().parent / "all_neurons_neuroncriteria_template.csv").resolve()


def _json_flag_enabled(raw_json: Any, block_key: str, nested_key: str) -> bool:
    try:
        payload = parse_json_block(raw_json, expect=dict, label="runtime_json")
    except Exception:
        return False
    block = payload.get(block_key)
    if isinstance(block, Mapping):
        return bool(block.get(nested_key, False))
    return False


def _raise_value_error(msg: str):
    raise ValueError(msg)


def compact_state(state: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: state.get(key) for key in keys}
