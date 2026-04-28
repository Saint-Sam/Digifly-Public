from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SIDE_RE = re.compile(r"(?:^|_)(L|R)(?:$|_)")
LEG_PREFIX_TO_THORAX = {"MNfl": "T1", "MNml": "T2", "MNhl": "T3"}
NERVE_TO_THORAX = {
    "Pro": "T1",
    "Meso": "T2",
    "Meta": "T3",
    "Ab": "Ab",
    "Cv": "Head",
    "DMeta": "T2",
    "PDM": "T2",
    "ADM": "Wing",
}
TARGET_BY_THORAX = {"T1": "front_leg", "T2": "middle_leg", "T3": "hind_leg"}


@dataclass(frozen=True)
class MappingSpec:
    family: str
    joint: str
    dof: str
    sign: float
    action: str
    weight: float = 1.0
    confidence: str = "high"
    notes: str = ""
    mapping_basis: str = "explicit_type_rule"


RULES_BY_TYPE: Dict[str, List[MappingSpec]] = {
    "Acc. ti flexor MN": [
        MappingSpec("tibia", "tibia", "extend", -1.0, "flex", 1.0, "high", "Explicit tibia flexor rule."),
    ],
    "Ti flexor MN": [
        MappingSpec("tibia", "tibia", "extend", -1.0, "flex", 1.0, "high", "Explicit tibia flexor rule."),
    ],
    "Ti extensor MN": [
        MappingSpec("tibia", "tibia", "extend", 1.0, "extend", 1.0, "high", "Explicit tibia extensor rule."),
    ],
    "ltm1-tibia MN": [
        MappingSpec(
            "tibia",
            "tibia",
            "extend",
            -1.0,
            "flex_like",
            1.0,
            "medium",
            "Mapped to tibia using the explicit tibia-target hint in the neuron type.",
        ),
    ],
    "Acc. tr flexor MN": [
        MappingSpec("femur", "femur", "extend", -1.0, "flex", 1.0, "high", "Explicit trochanter flexor rule."),
    ],
    "Tr flexor MN": [
        MappingSpec("femur", "femur", "extend", -1.0, "flex", 1.0, "high", "Explicit trochanter flexor rule."),
    ],
    "Tr extensor MN": [
        MappingSpec(
            "femur",
            "femur",
            "extend",
            1.0,
            "extend",
            1.0,
            "high",
            "Explicit trochanter extensor rule.",
        ),
    ],
    "Sternotrochanter MN": [
        MappingSpec(
            "femur",
            "femur",
            "extend",
            1.0,
            "extend_like",
            1.0,
            "medium",
            "Mapped to femur extension as the closest flybody DoF for sternotrochanter drive.",
        ),
    ],
    "Tergotr. MN": [
        MappingSpec(
            "femur",
            "femur",
            "extend",
            -1.0,
            "flex_like",
            1.0,
            "medium",
            "Mapped as the antagonistic partner to sternotrochanter/femur extension.",
        ),
    ],
    "Fe reductor MN": [
        MappingSpec(
            "femur_twist",
            "femur",
            "twist",
            -1.0,
            "reduce",
            1.0,
            "medium",
            "Mapped to femur twist because flybody does not expose a dedicated femur reductor actuator.",
        ),
    ],
    "ltm2-femur MN": [
        MappingSpec(
            "femur_twist",
            "femur",
            "twist",
            -1.0,
            "reduce_like",
            1.0,
            "medium",
            "Mapped to femur twist using the explicit femur-target hint in the neuron type.",
        ),
    ],
    "ltm MN": [
        MappingSpec(
            "femur",
            "femur",
            "extend",
            1.0,
            "femur_drive",
            1.0,
            "medium",
            "Generic long-tendon femur rule; keep under review when better anatomy arrives.",
        ),
    ],
    "Sternal posterior rotator MN": [
        MappingSpec(
            "coxa_twist",
            "coxa",
            "twist",
            -1.0,
            "rotate_posterior",
            1.0,
            "high",
            "Explicit posterior coxa rotation rule.",
        ),
    ],
    "Sternal anterior rotator MN": [
        MappingSpec(
            "coxa_twist",
            "coxa",
            "twist",
            1.0,
            "rotate_anterior",
            1.0,
            "high",
            "Explicit anterior coxa rotation rule.",
        ),
    ],
    "Pleural remotor/abductor MN": [
        MappingSpec(
            "coxa",
            "coxa",
            "extend",
            -1.0,
            "remotor",
            0.65,
            "medium",
            "Compound rule splits remotor drive across coxa extension and abduction.",
        ),
        MappingSpec(
            "coxa_abduct",
            "coxa",
            "abduct",
            1.0,
            "abductor",
            0.35,
            "medium",
            "Compound rule splits remotor drive across coxa extension and abduction.",
        ),
    ],
    "Tergopleural/Pleural promotor MN": [
        MappingSpec(
            "coxa",
            "coxa",
            "extend",
            1.0,
            "promotor",
            1.0,
            "medium",
            "Mapped to coxa extension as the closest flybody promotor axis.",
        ),
    ],
    "Sternal adductor MN": [
        MappingSpec(
            "coxa_abduct",
            "coxa",
            "abduct",
            -1.0,
            "adductor",
            1.0,
            "high",
            "Explicit coxa adduction rule.",
        ),
    ],
    "Ta depressor MN": [
        MappingSpec(
            "tarsus",
            "tarsus",
            "extend",
            -1.0,
            "depress",
            0.7,
            "medium",
            "Split across the two flybody tarsus DoFs to keep distal motion visible.",
        ),
        MappingSpec(
            "tarsus2",
            "tarsus",
            "extend",
            -1.0,
            "depress",
            0.3,
            "medium",
            "Split across the two flybody tarsus DoFs to keep distal motion visible.",
        ),
    ],
    "Ta levator MN": [
        MappingSpec(
            "tarsus",
            "tarsus",
            "extend",
            1.0,
            "elevate",
            0.7,
            "medium",
            "Split across the two flybody tarsus DoFs to keep distal motion visible.",
        ),
        MappingSpec(
            "tarsus2",
            "tarsus",
            "extend",
            1.0,
            "elevate",
            0.3,
            "medium",
            "Split across the two flybody tarsus DoFs to keep distal motion visible.",
        ),
    ],
    "TTMn": [
        MappingSpec(
            "coxa",
            "coxa",
            "extend",
            1.0,
            "jump_extensor_chain",
            0.35,
            "medium",
            "Special-case jump chain based on the old Phase 3 TTMn override.",
        ),
        MappingSpec(
            "femur",
            "femur",
            "extend",
            1.0,
            "jump_extensor_chain",
            0.6,
            "medium",
            "Special-case jump chain based on the old Phase 3 TTMn override.",
        ),
        MappingSpec(
            "tibia",
            "tibia",
            "extend",
            1.0,
            "jump_extensor_chain",
            1.0,
            "medium",
            "Special-case jump chain based on the old Phase 3 TTMn override.",
        ),
    ],
    "STTMm": [
        MappingSpec(
            "femur",
            "femur",
            "extend",
            1.0,
            "jump_assist",
            1.0,
            "low",
            "Coarse T2 jump-muscle rule; confirm against richer thoracic metadata later.",
        ),
    ],
    "DLMn a, b": [
        MappingSpec(
            "wing_pitch",
            "wing",
            "pitch",
            1.0,
            "wing_drive",
            1.0,
            "low",
            "Coarse DLMn wing fallback restored from the older bridge behavior; treat as a pitch proxy until a richer wing actuator mapping is curated.",
        ),
    ],
    "DLMn c-f": [
        MappingSpec(
            "wing_pitch",
            "wing",
            "pitch",
            1.0,
            "wing_drive",
            1.0,
            "low",
            "Coarse DLMn wing fallback restored from the older bridge behavior; treat as a pitch proxy until a richer wing actuator mapping is curated.",
        ),
    ],
}


def _register_phase1_connectivity_fallback_rules() -> None:
    tibia_flexor_fallback = MappingSpec(
        "tibia",
        "tibia",
        "extend",
        -1.0,
        "flex_like",
        1.0,
        "low",
        "Phase 1 connectivity fallback: premotor-input similarity clustered this generic family with tibia flexor references in the same thoracic segment.",
        "phase1_connectivity_fallback",
    )
    posterior_rotator_fallback = MappingSpec(
        "coxa_twist",
        "coxa",
        "twist",
        -1.0,
        "rotate_posterior_like",
        1.0,
        "low",
        "Phase 1 connectivity fallback: premotor-input similarity clustered this generic family with posterior coxa rotators in the same thoracic segment.",
        "phase1_connectivity_fallback",
    )

    for mn_type in {
        "MNhl61",
        "MNhl63",
        "MNhl64",
        "MNhl66",
        "MNhl67",
        "MNhl68",
        "MNhl69",
        "MNhl70",
        "MNhl73",
        "MNhl74",
        "MNhl75",
        "MNml76",
        "MNml77",
        "MNml78",
        "MNml80",
        "MNml82",
        "MNml84",
        "MNml85",
        "MNml86",
    }:
        RULES_BY_TYPE[mn_type] = [tibia_flexor_fallback]

    for mn_type in {"MNhl29", "MNml29"}:
        RULES_BY_TYPE[mn_type] = [posterior_rotator_fallback]


_register_phase1_connectivity_fallback_rules()


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _as_int(value: object) -> int | None:
    text = _norm_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _extract_side(instance: str, phase2_row: Dict[str, str]) -> str:
    soma_side = _norm_text(phase2_row.get("somaSide", "")).lower()
    if soma_side in {"left", "right"}:
        return soma_side
    base = _norm_text(instance).split()[0]
    match = SIDE_RE.search(base)
    if not match:
        return ""
    return "left" if match.group(1).upper() == "L" else "right"


def _extract_exit_nerve(instance: str, phase2_row: Dict[str, str]) -> str:
    nerve = _norm_text(phase2_row.get("exitNerve", ""))
    if nerve:
        return nerve
    base = _norm_text(instance).split()[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def _extract_thorax(instance: str, phase2_row: Dict[str, str]) -> str:
    soma = _norm_text(phase2_row.get("somaNeuromere", ""))
    if soma in {"T1", "T2", "T3", "Ab"}:
        return soma
    target = _norm_text(phase2_row.get("target", ""))
    target_lc = target.lower()
    if target_lc == "front leg":
        return "T1"
    if target_lc == "middle leg":
        return "T2"
    if target_lc == "hind leg":
        return "T3"
    nerve = _extract_exit_nerve(instance, phase2_row)
    nerve_key = nerve.split("_")[0]
    for prefix, thorax in NERVE_TO_THORAX.items():
        if nerve_key.startswith(prefix):
            return thorax
    base = _norm_text(instance).split()[0]
    for prefix, thorax in LEG_PREFIX_TO_THORAX.items():
        if base.startswith(prefix):
            return thorax
    return ""


def _infer_subsystem(instance: str, mn_type: str, thorax: str, exit_nerve: str) -> str:
    base = _norm_text(instance).split()[0]
    mn_type_lc = _norm_text(mn_type).lower()
    if base.startswith("MNad") or thorax == "Ab" or exit_nerve.startswith("Ab"):
        return "abdomen"
    if mn_type in RULES_BY_TYPE:
        return "leg"
    if base.startswith(("MNfl", "MNml", "MNhl")):
        return "leg"
    if base.startswith("MNwm"):
        return "wing"
    if base.startswith(("MNnm", "MNhm")):
        return "head"
    if base.startswith("MNxm"):
        return "mixed"
    if "wing" in mn_type_lc:
        return "wing"
    if "antenna" in mn_type_lc or "head" in mn_type_lc:
        return "head"
    return "unknown"


def _target_body(subsystem: str, thorax: str) -> str:
    if subsystem == "leg":
        return TARGET_BY_THORAX.get(thorax, "leg")
    if subsystem == "abdomen":
        return "abdomen"
    if subsystem == "wing":
        return "wing"
    if subsystem == "head":
        return "head"
    return subsystem or "unknown"


def _actuator_name(family: str, thorax: str, side: str) -> str:
    if family == "abdomen":
        return "abdomen"
    if family == "abdomen_abduct":
        return "abdomen_abduct"
    if family in {"head", "head_abduct", "head_twist", "rostrum", "haustellum", "haustellum_abduct"}:
        return family
    if family in {"wing_pitch", "wing_roll", "wing_yaw", "antenna", "antenna_abduct", "antenna_twist"}:
        if not side:
            return ""
        return f"{family}_{side}"
    if family == "coxa":
        return f"coxa_{thorax}_{side}"
    if family == "coxa_abduct":
        return f"coxa_abduct_{thorax}_{side}"
    if family == "coxa_twist":
        return f"coxa_twist_{thorax}_{side}"
    if family == "femur":
        return f"femur_{thorax}_{side}"
    if family == "femur_twist":
        return f"femur_twist_{thorax}_{side}"
    if family == "tibia":
        return f"tibia_{thorax}_{side}"
    if family == "tarsus":
        return f"tarsus_{thorax}_{side}"
    if family == "tarsus2":
        return f"tarsus2_{thorax}_{side}"
    return ""


def _load_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_phase2_template(csv_path: Path | str | None) -> Dict[int, Dict[str, str]]:
    if csv_path is None:
        return {}
    rows = _load_csv_rows(csv_path)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        nid = _as_int(row.get("bodyId"))
        if nid is None:
            continue
        if _norm_text(row.get("class", "")) != "motor neuron":
            continue
        out[nid] = row
    return out


def _load_mjcf_actuator_names(mjcf_xml: Path | str | None) -> List[str]:
    if mjcf_xml is None:
        return []
    path = Path(mjcf_xml).expanduser().resolve()
    if not path.exists():
        return []
    visited: set[Path] = set()
    names: set[str] = set()

    def walk(xml_path: Path) -> None:
        xml_path = xml_path.resolve()
        if xml_path in visited or not xml_path.exists():
            return
        visited.add(xml_path)
        root = ET.parse(xml_path).getroot()
        for elem in root.iter():
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
            if tag in {"motor", "position", "velocity", "general", "adhesion"}:
                name = _norm_text(elem.attrib.get("name", ""))
                if name:
                    names.add(name)
            if tag == "include":
                inc = _norm_text(elem.attrib.get("file", ""))
                if inc:
                    walk((xml_path.parent / inc).resolve())

    walk(path)
    return sorted(names)


def _row_meta(base_row: Dict[str, str], phase2_row: Dict[str, str]) -> Dict[str, str]:
    mn_id = _as_int(base_row.get("bodyId"))
    if mn_id is None:
        raise ValueError(f"Invalid bodyId in Phase 1 row: {base_row}")
    instance = _norm_text(base_row.get("instance", "")) or _norm_text(phase2_row.get("instance", ""))
    mn_type = _norm_text(base_row.get("type", "")) or _norm_text(phase2_row.get("type", ""))
    exit_nerve = _extract_exit_nerve(instance, phase2_row)
    thorax = _extract_thorax(instance, phase2_row)
    side = _extract_side(instance, phase2_row)
    subsystem = _infer_subsystem(instance, mn_type, thorax, exit_nerve)
    return {
        "mn_id": str(mn_id),
        "instance": instance,
        "mn_type": mn_type,
        "class": _norm_text(base_row.get("class", "")) or _norm_text(phase2_row.get("class", "")),
        "status": _norm_text(base_row.get("status", "")) or _norm_text(phase2_row.get("status", "")),
        "predicted_nt": _norm_text(phase2_row.get("predictedNt", "")) or _norm_text(phase2_row.get("consensusNt", "")),
        "exit_nerve": exit_nerve,
        "hemilineage": _norm_text(phase2_row.get("hemilineage", "")),
        "target": _norm_text(phase2_row.get("target", "")),
        "description": _norm_text(phase2_row.get("description", "")),
        "thorax": thorax,
        "side": side,
        "subsystem": subsystem,
        "label": _norm_text(phase2_row.get("label", "")),
    }


def _build_mapping_rows(meta: Dict[str, str]) -> Tuple[List[Dict[str, str]], str | None]:
    mn_type = meta["mn_type"]
    subsystem = meta["subsystem"]
    thorax = meta["thorax"]
    side = meta["side"]
    instance = meta["instance"]

    if mn_type in RULES_BY_TYPE:
        specs = RULES_BY_TYPE[mn_type]
        required = thorax and side
        if not required:
            return [], "missing_leg_side_or_thorax"
        rows = []
        for spec in specs:
            actuator_name = _actuator_name(spec.family, thorax, side)
            if not actuator_name:
                return [], "failed_to_build_actuator_name"
            rows.append(
                {
                    "mn_id": meta["mn_id"],
                    "instance": instance,
                    "mn_type": mn_type,
                    "class": meta["class"],
                    "status": meta["status"],
                    "predicted_nt": meta["predicted_nt"],
                    "exit_nerve": meta["exit_nerve"],
                    "hemilineage": meta["hemilineage"],
                    "target": meta["target"],
                    "subsystem": subsystem,
                    "thorax": thorax,
                    "side": side,
                    "joint": spec.joint,
                    "dof": spec.dof,
                    "actuator_name": actuator_name,
                    "sign": f"{spec.sign:.1f}",
                    "action": spec.action,
                    "gain": "1.0",
                    "bias": "0.0",
                    "weight": f"{spec.weight:.3f}",
                    "confidence": spec.confidence,
                    "mapping_basis": spec.mapping_basis,
                    "notes": spec.notes,
                    "target_body": _target_body(subsystem, thorax),
                    "target_joint": spec.family,
                }
            )
        return rows, None

    if instance.startswith("MNad") or mn_type.startswith("MNad") or subsystem == "abdomen":
        return [
            {
                "mn_id": meta["mn_id"],
                "instance": instance,
                "mn_type": mn_type,
                "class": meta["class"],
                "status": meta["status"],
                "predicted_nt": meta["predicted_nt"],
                "exit_nerve": meta["exit_nerve"],
                "hemilineage": meta["hemilineage"],
                "target": meta["target"],
                "subsystem": "abdomen",
                "thorax": thorax,
                "side": side,
                "joint": "abdomen",
                "dof": "main",
                "actuator_name": "abdomen",
                "sign": "1.0",
                "action": "abdomen_drive",
                "gain": "1.0",
                "bias": "0.0",
                "weight": "1.000",
                "confidence": "medium",
                "mapping_basis": "generic_abdomen_rule",
                "notes": "Generic abdomen placeholder retained from the old bridge, now isolated as an explicit coarse rule.",
                "target_body": "abdomen",
                "target_joint": "abdomen",
            }
        ], None

    if subsystem == "leg":
        return [], "generic_leg_type_needs_joint_assignment"
    if subsystem == "wing":
        return [], "wing_type_needs_query_or_manual_rule"
    if subsystem == "head":
        return [], "head_type_needs_query_or_manual_rule"
    if subsystem == "mixed":
        return [], "mixed_target_type_needs_query_or_manual_rule"
    if mn_type:
        return [], "no_rule_for_type"
    return [], "missing_type"


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _hemi_audit(
    mapping_rows: List[Dict[str, str]],
    meta_by_id: Dict[int, Dict[str, str]],
    added_motor_csv: Path | str | None,
    spike_csv: Path | str | None,
    out_dir: Path,
) -> Dict[str, object]:
    if added_motor_csv is None or spike_csv is None:
        return {}
    added_rows = _load_csv_rows(added_motor_csv)
    spike_rows = _load_csv_rows(spike_csv)

    added_ids = {_as_int(row.get("neuron_id")) for row in added_rows}
    added_ids.discard(None)
    spiking_added_ids = set()
    for row in spike_rows:
        nid = _as_int(row.get("neuron_id"))
        if nid is not None and nid in added_ids:
            spiking_added_ids.add(nid)

    mapping_by_id: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in mapping_rows:
        nid = _as_int(row.get("mn_id"))
        if nid is not None:
            mapping_by_id[nid].append(row)

    audit_rows: List[Dict[str, str]] = []
    for nid in sorted(added_ids):
        meta = meta_by_id.get(nid, {})
        mapped_rows = mapping_by_id.get(nid, [])
        actuators = sorted({_norm_text(row.get("actuator_name", "")) for row in mapped_rows if _norm_text(row.get("actuator_name", ""))})
        audit_rows.append(
            {
                "neuron_id": str(nid),
                "instance": meta.get("instance", ""),
                "mn_type": meta.get("mn_type", ""),
                "thorax": meta.get("thorax", ""),
                "side": meta.get("side", ""),
                "subsystem": meta.get("subsystem", ""),
                "hemilineage": meta.get("hemilineage", ""),
                "target": meta.get("target", ""),
                "is_added_motor": "true",
                "is_spiking_in_run": "true" if nid in spiking_added_ids else "false",
                "is_mapped": "true" if actuators else "false",
                "mapping_row_count": str(len(mapped_rows)),
                "mapped_actuators": "; ".join(actuators),
            }
        )

    audit_csv = out_dir / "hemi_09a_baseline_audit.csv"
    _write_csv(
        audit_csv,
        [
            "neuron_id",
            "instance",
            "mn_type",
            "thorax",
            "side",
            "subsystem",
            "hemilineage",
            "target",
            "is_added_motor",
            "is_spiking_in_run",
            "is_mapped",
            "mapping_row_count",
            "mapped_actuators",
        ],
        audit_rows,
    )

    mapped_spiking = sum(1 for row in audit_rows if row["is_spiking_in_run"] == "true" and row["is_mapped"] == "true")
    summary = {
        "added_motor_count": len(added_ids),
        "spiking_added_motor_count": len(spiking_added_ids),
        "mapped_spiking_added_motor_count": mapped_spiking,
        "unmapped_spiking_added_motor_count": len(spiking_added_ids) - mapped_spiking,
        "audit_csv": str(audit_csv),
    }
    summary_path = out_dir / "hemi_09a_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def run_mapping_rebuild(
    phase1_mn_csv: Path | str,
    out_mapping_csv: Path | str,
    out_dir: Path | str,
    phase2_template_csv: Path | str | None = None,
    mjcf_xml: Path | str | None = None,
    hemi_added_motor_csv: Path | str | None = None,
    hemi_spike_csv: Path | str | None = None,
) -> Dict[str, object]:
    phase1_rows = _load_csv_rows(phase1_mn_csv)
    phase2_by_id = _load_phase2_template(phase2_template_csv)
    actuator_names = set(_load_mjcf_actuator_names(mjcf_xml))

    out_mapping_csv = Path(out_mapping_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows: List[Dict[str, str]] = []
    unresolved_rows: List[Dict[str, str]] = []
    meta_by_id: Dict[int, Dict[str, str]] = {}

    for base_row in phase1_rows:
        if _norm_text(base_row.get("class", "")) != "motor neuron":
            continue
        nid = _as_int(base_row.get("bodyId"))
        if nid is None:
            continue
        phase2_row = phase2_by_id.get(nid, {})
        meta = _row_meta(base_row, phase2_row)
        meta_by_id[nid] = meta
        rows, reason = _build_mapping_rows(meta)
        if rows:
            mapping_rows.extend(rows)
        else:
            unresolved_rows.append(
                {
                    "mn_id": meta["mn_id"],
                    "instance": meta["instance"],
                    "mn_type": meta["mn_type"],
                    "subsystem": meta["subsystem"],
                    "thorax": meta["thorax"],
                    "side": meta["side"],
                    "exit_nerve": meta["exit_nerve"],
                    "hemilineage": meta["hemilineage"],
                    "target": meta["target"],
                    "reason": reason or "unmapped",
                    "description": meta["description"],
                }
            )

    invalid_rows = []
    if actuator_names:
        for row in mapping_rows:
            if row["actuator_name"] not in actuator_names:
                invalid_rows.append(row)
        if invalid_rows:
            bad = ", ".join(sorted({row["actuator_name"] for row in invalid_rows}))
            raise ValueError(f"Rebuilt mapping produced actuator names missing from the MJCF: {bad}")

    mapping_fieldnames = [
        "mn_id",
        "instance",
        "mn_type",
        "class",
        "status",
        "predicted_nt",
        "exit_nerve",
        "hemilineage",
        "target",
        "subsystem",
        "thorax",
        "side",
        "joint",
        "dof",
        "actuator_name",
        "sign",
        "action",
        "gain",
        "bias",
        "weight",
        "confidence",
        "mapping_basis",
        "notes",
        "target_body",
        "target_joint",
    ]
    unresolved_fieldnames = [
        "mn_id",
        "instance",
        "mn_type",
        "subsystem",
        "thorax",
        "side",
        "exit_nerve",
        "hemilineage",
        "target",
        "reason",
        "description",
    ]

    _write_csv(out_mapping_csv, mapping_fieldnames, mapping_rows)
    unresolved_csv = out_dir / "mn_to_actuator_mapping_rebuilt_unresolved.csv"
    _write_csv(unresolved_csv, unresolved_fieldnames, unresolved_rows)

    type_totals: Counter[str] = Counter()
    type_mapped: Counter[str] = Counter()
    type_rows: Counter[str] = Counter()
    for meta in meta_by_id.values():
        type_totals[meta["mn_type"]] += 1
    for row in mapping_rows:
        type_rows[row["mn_type"]] += 1
    for row in mapping_rows:
        type_mapped[row["mn_type"]] += 1

    coverage_rows = []
    for mn_type in sorted(type_totals):
        coverage_rows.append(
            {
                "mn_type": mn_type,
                "total_neuron_count": str(type_totals[mn_type]),
                "mapped_neuron_count": str(type_mapped[mn_type]),
                "unmapped_neuron_count": str(type_totals[mn_type] - type_mapped[mn_type]),
                "mapping_row_count": str(type_rows[mn_type]),
            }
        )
    coverage_csv = out_dir / "mapping_rebuild_type_coverage.csv"
    _write_csv(
        coverage_csv,
        ["mn_type", "total_neuron_count", "mapped_neuron_count", "unmapped_neuron_count", "mapping_row_count"],
        coverage_rows,
    )

    mapped_ids = {_as_int(row["mn_id"]) for row in mapping_rows}
    mapped_ids.discard(None)
    confidence_counts = Counter(row["confidence"] for row in mapping_rows)
    subsystem_counts = Counter(row["subsystem"] for row in mapping_rows)
    unresolved_reason_counts = Counter(row["reason"] for row in unresolved_rows)

    hemi_summary = _hemi_audit(mapping_rows, meta_by_id, hemi_added_motor_csv, hemi_spike_csv, out_dir)

    summary = {
        "phase1_motor_neuron_count": len(meta_by_id),
        "mapped_motor_neuron_count": len(mapped_ids),
        "unresolved_motor_neuron_count": len(meta_by_id) - len(mapped_ids),
        "mapping_row_count": len(mapping_rows),
        "unresolved_row_count": len(unresolved_rows),
        "confidence_counts": dict(confidence_counts),
        "subsystem_counts": dict(subsystem_counts),
        "unresolved_reason_counts": dict(unresolved_reason_counts),
        "mapping_csv": str(out_mapping_csv),
        "unresolved_csv": str(unresolved_csv),
        "type_coverage_csv": str(coverage_csv),
        "mjcf_actuator_count": len(actuator_names),
    }
    if hemi_summary:
        summary["hemi_09a_baseline"] = hemi_summary

    summary_path = out_dir / "mapping_rebuild_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
