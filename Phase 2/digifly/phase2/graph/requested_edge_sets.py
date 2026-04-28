from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from digifly.phase2.data.paths import (
    _find_swc,
    _syn_csv_path,
    export_index_entry,
    refresh_export_path_index,
)
from digifly.phase2.graph.edges_from_synapses import _hemi_fetch_edges_from_synapse_csvs
from digifly.phase2.neuron_build.config import DEFAULT_GLOBAL_TIMING, NT_TO_CLASS, SYN_PRESETS


EDGE_SET_SCHEMA_VERSION = "nt_enriched_v2"
EDGE_SET_COLUMNS: List[str] = [
    "pre_id",
    "post_id",
    "weight_uS",
    "delay_ms",
    "tau1_ms",
    "tau2_ms",
    "syn_e_rev_mV",
    "post_x",
    "post_y",
    "post_z",
    "syn_index",
    "predicted_nt",
    "predicted_nt_prob",
    "syn_class",
    "nt_source",
]
EDGE_SET_INT_COLUMNS = {"pre_id", "post_id", "syn_index"}
EDGE_SET_FLOAT_COLUMNS = {
    "weight_uS",
    "delay_ms",
    "tau1_ms",
    "tau2_ms",
    "syn_e_rev_mV",
    "post_x",
    "post_y",
    "post_z",
    "predicted_nt_prob",
}
EDGE_SET_TEXT_COLUMNS = {"predicted_nt", "syn_class", "nt_source"}

DEFAULT_NEUPRINT_DATASET = "manc:v1.2.1"
DEFAULT_MALE_CNS_DATASET = "male-cns:v0.9"
KNOWN_NEUPRINT_DATASET_VERSIONS: Dict[str, List[str]] = {
    "manc": ["v1.0", "v1.2.1", "v1.2.3"],
    "male-cns": ["v0.9"],
}
DATASET_FAMILY_ALIASES = {
    "manc": "manc",
    "male-cns": "male-cns",
    "male_cns": "male-cns",
    "malecns": "male-cns",
}

EDGE_REGISTRY_COLUMNS: List[str] = [
    "edge_set_name",
    "edge_set_slug",
    "edge_signature",
    "neuprint_dataset",
    "selection_mode",
    "selection_label",
    "selection_rule",
    "expansion_mode",
    "requested_count",
    "final_network_count",
    "requested_ids_json",
    "final_network_ids_json",
    "added_ids_json",
    "swc_dir",
    "master_csv",
    "raw_edges_path",
    "final_edges_path",
    "metadata_json_path",
    "created_at_utc",
    "updated_at_utc",
]

_PHASE1_BRIDGE = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def coerce_int_ids(values: Iterable[Any], *, name: str = "ids") -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for raw in values:
        val = int(raw)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    if not out:
        raise ValueError(f"{name} must contain at least one id")
    return out


def _json_dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def normalize_neuprint_dataset_family(dataset_family: str | None = None) -> str:
    text = str(dataset_family or DEFAULT_NEUPRINT_DATASET).strip().lower()
    if not text:
        return "manc"
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return DATASET_FAMILY_ALIASES.get(text, text)


def known_neuprint_dataset_versions(dataset_family: str | None = None) -> List[str]:
    family = normalize_neuprint_dataset_family(dataset_family)
    return list(KNOWN_NEUPRINT_DATASET_VERSIONS.get(family, []))


def default_neuprint_dataset_version(dataset_family: str | None = None) -> str:
    family = normalize_neuprint_dataset_family(dataset_family)
    if family == "male-cns":
        return DEFAULT_MALE_CNS_DATASET.split(":", 1)[1]
    return DEFAULT_NEUPRINT_DATASET.split(":", 1)[1]


def build_neuprint_dataset_name(dataset_family: str | None, version: str | None = None) -> str:
    family = normalize_neuprint_dataset_family(dataset_family)
    version_text = str(version or "").strip()
    if not version_text:
        version_text = default_neuprint_dataset_version(family)
    if ":" in version_text:
        version_text = version_text.split(":", 1)[1].strip()
    if not version_text.lower().startswith("v"):
        version_text = f"v{version_text}"
    return f"{family}:{version_text}"


def resolve_neuprint_dataset_choice(dataset: str | None = None, version: str | None = None) -> str:
    text = str(dataset or "").strip()
    if not text:
        return build_neuprint_dataset_name("manc", version)
    if ":" in text:
        family, raw_version = text.split(":", 1)
        return build_neuprint_dataset_name(family, version or raw_version)
    family = normalize_neuprint_dataset_family(text)
    return build_neuprint_dataset_name(family, version)


def normalize_neuprint_dataset(dataset: str | None = None) -> str:
    return resolve_neuprint_dataset_choice(dataset)


def canonicalize_edge_df(df: pd.DataFrame, *, columns: Sequence[str] = EDGE_SET_COLUMNS) -> pd.DataFrame:
    df_use = df.copy()
    for col in columns:
        if col in df_use.columns:
            continue
        if col in EDGE_SET_INT_COLUMNS:
            df_use[col] = pd.Series(dtype="Int64")
        elif col in EDGE_SET_TEXT_COLUMNS:
            df_use[col] = pd.Series(dtype="object")
        else:
            df_use[col] = pd.Series(dtype=float)
    ordered = [str(col) for col in columns]
    extras = [str(col) for col in df_use.columns if str(col) not in ordered]
    return df_use[ordered + extras]


def normalize_edge_set_name(name: str | None, *, fallback: str | None = None) -> str:
    text = str(name or fallback or "").strip()
    if not text:
        raise ValueError("edge set name must be non-empty")
    return text


def edge_set_slug(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    base = re.sub(r"_+", "_", base).strip("._-")
    if not base:
        raise ValueError("could not derive a usable edge-set slug from the provided name")
    return base


def default_edges_registry_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    return (Path.home() / "Desktop" / "Edges_all").resolve()


def edges_registry_paths(root: str | Path | None = None) -> Dict[str, Path]:
    base = default_edges_registry_root(root)
    return {
        "root": base,
        "edge_sets_dir": (base / "edge_sets").resolve(),
        "master_edges_dir": (base / "master_edges").resolve(),
        "registry_csv": (base / "edge_registry.csv").resolve(),
        "master_edges_csv": (base / "master_edges" / "master_edges.csv").resolve(),
    }


def ensure_edges_registry_layout(root: str | Path | None = None) -> Dict[str, Path]:
    paths = edges_registry_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["edge_sets_dir"].mkdir(parents=True, exist_ok=True)
    paths["master_edges_dir"].mkdir(parents=True, exist_ok=True)
    if not paths["registry_csv"].exists():
        pd.DataFrame(columns=EDGE_REGISTRY_COLUMNS).to_csv(paths["registry_csv"], index=False)
    if not paths["master_edges_csv"].exists():
        pd.DataFrame(columns=["edge_set_name", "edge_signature", "neuprint_dataset"] + EDGE_SET_COLUMNS).to_csv(paths["master_edges_csv"], index=False)
    return paths


def edge_request_signature(
    requested_ids: Sequence[int],
    *,
    selection_mode: str,
    expansion_mode: str,
    selection_label: str | None = None,
    neuprint_dataset: str | None = None,
    master_csv: str | Path | None = None,
    default_weight_uS: float | None = None,
    default_delay_ms: float | None = None,
) -> str:
    payload = {
        "edge_schema_version": EDGE_SET_SCHEMA_VERSION,
        "requested_ids": [int(x) for x in sorted(set(int(v) for v in requested_ids))],
        "selection_mode": str(selection_mode).strip().lower(),
        "expansion_mode": str(expansion_mode).strip().lower(),
        "selection_label": str(selection_label or "").strip(),
        "neuprint_dataset": normalize_neuprint_dataset(neuprint_dataset),
        "master_csv_fingerprint": _file_fingerprint(master_csv),
        "default_weight_uS": None if default_weight_uS is None else float(default_weight_uS),
        "default_delay_ms": None if default_delay_ms is None else float(default_delay_ms),
    }
    return hashlib.sha256(_json_dump(payload).encode("utf-8")).hexdigest()


def _load_registry_df(root: str | Path | None = None) -> pd.DataFrame:
    paths = ensure_edges_registry_layout(root)
    try:
        df = pd.read_csv(paths["registry_csv"])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=EDGE_REGISTRY_COLUMNS)
    for col in EDGE_REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[EDGE_REGISTRY_COLUMNS]


def _save_registry_df(df: pd.DataFrame, root: str | Path | None = None) -> Path:
    paths = ensure_edges_registry_layout(root)
    out = df.copy()
    for col in EDGE_REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[EDGE_REGISTRY_COLUMNS]
    out.to_csv(paths["registry_csv"], index=False)
    return paths["registry_csv"]


def find_registered_edge_set(
    requested_ids: Sequence[int],
    *,
    selection_mode: str,
    expansion_mode: str,
    selection_label: str | None = None,
    neuprint_dataset: str | None = None,
    master_csv: str | Path | None = None,
    default_weight_uS: float | None = None,
    default_delay_ms: float | None = None,
    root: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    signature = edge_request_signature(
        requested_ids,
        selection_mode=selection_mode,
        expansion_mode=expansion_mode,
        selection_label=selection_label,
        neuprint_dataset=neuprint_dataset,
        master_csv=master_csv,
        default_weight_uS=default_weight_uS,
        default_delay_ms=default_delay_ms,
    )
    df = _load_registry_df(root)
    hits = df[df["edge_signature"].astype(str) == signature]
    if hits.empty:
        return None
    row = hits.iloc[-1].to_dict()
    final_path = Path(str(row["final_edges_path"])).expanduser().resolve()
    if not final_path.exists():
        return None
    row["edge_signature"] = signature
    row["final_edges_path"] = str(final_path)
    row["raw_edges_path"] = str(Path(str(row["raw_edges_path"])).expanduser().resolve())
    row["metadata_json_path"] = str(Path(str(row["metadata_json_path"])).expanduser().resolve())
    return row


def _find_by_edge_set_name(edge_set_name: str, *, root: str | Path | None = None) -> Optional[Dict[str, Any]]:
    df = _load_registry_df(root)
    hits = df[df["edge_set_name"].astype(str) == str(edge_set_name)]
    if hits.empty:
        return None
    return hits.iloc[-1].to_dict()


def refresh_master_edges_csv(root: str | Path | None = None) -> Path:
    paths = ensure_edges_registry_layout(root)
    reg = _load_registry_df(root)
    frames: List[pd.DataFrame] = []
    for row in reg.to_dict(orient="records"):
        final_path = Path(str(row.get("final_edges_path") or "")).expanduser().resolve()
        if not final_path.exists():
            continue
        try:
            df = pd.read_csv(final_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=EDGE_SET_COLUMNS)
        df = canonicalize_edge_df(df)
        df.insert(0, "neuprint_dataset", str(row.get("neuprint_dataset", DEFAULT_NEUPRINT_DATASET)))
        df.insert(0, "edge_signature", str(row.get("edge_signature", "")))
        df.insert(0, "edge_set_name", str(row.get("edge_set_name", "")))
        frames.append(df)
    master_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["edge_set_name", "edge_signature", "neuprint_dataset"] + EDGE_SET_COLUMNS)
    master_df.to_csv(paths["master_edges_csv"], index=False)
    return paths["master_edges_csv"]


def _upsert_registry_row(row: Mapping[str, Any], *, root: str | Path | None = None) -> Path:
    df = _load_registry_df(root)
    row_df = pd.DataFrame([dict(row)])
    for col in EDGE_REGISTRY_COLUMNS:
        if col not in row_df.columns:
            row_df[col] = pd.NA
    edge_signature = str(row_df.iloc[0]["edge_signature"])
    edge_set_name = str(row_df.iloc[0]["edge_set_name"])
    drop_mask = (df["edge_signature"].astype(str) == edge_signature) | (df["edge_set_name"].astype(str) == edge_set_name)
    df = df.loc[~drop_mask].copy()
    df = pd.concat([df, row_df[EDGE_REGISTRY_COLUMNS]], ignore_index=True)
    return _save_registry_df(df, root=root)


def missing_local_export_ids(
    body_ids: Sequence[int],
    *,
    swc_dir: str | Path,
    neuprint_dataset: str | None = None,
    require_synapses: bool = True,
    require_swc: bool = True,
) -> List[int]:
    swc_root = Path(swc_dir).expanduser().resolve()
    dataset_use = normalize_neuprint_dataset(neuprint_dataset)
    out: List[int] = []
    for body_id in coerce_int_ids(body_ids, name="body ids"):
        entry = export_index_entry(swc_root, int(body_id))
        missing = False
        if require_synapses:
            syn_path = entry.get("syn_csv")
            if syn_path is None or not Path(str(syn_path)).exists():
                syn_path = _syn_csv_path(swc_root, int(body_id))
            if syn_path is None or not Path(syn_path).exists():
                missing = True
        if require_swc:
            swc_path = entry.get("swc")
            if swc_path is None or not Path(str(swc_path)).exists():
                try:
                    swc_path = _find_swc(str(swc_root), int(body_id))
                except Exception:
                    swc_path = None
            if swc_path is None or not Path(str(swc_path)).exists():
                missing = True
        meta_path_raw = entry.get("meta_path")
        meta_path = Path(str(meta_path_raw)).expanduser().resolve() if meta_path_raw else None
        if meta_path is not None and meta_path.exists():
            try:
                raw_dataset = entry.get("neuprint_dataset")
                if raw_dataset in (None, "") or bool(entry.get("meta_parse_error")):
                    raise ValueError("missing or unreadable export metadata")
                meta_dataset = normalize_neuprint_dataset(raw_dataset)
                if meta_dataset != dataset_use:
                    missing = True
            except Exception:
                missing = True
        elif dataset_use != DEFAULT_NEUPRINT_DATASET:
            # Older exports may not have metadata; assume they are legacy MANC only.
            missing = True
        if missing:
            out.append(int(body_id))
    return out


def _load_phase1_bridge():
    global _PHASE1_BRIDGE
    if _PHASE1_BRIDGE is not None:
        return _PHASE1_BRIDGE
    repo_root = Path(__file__).resolve().parents[4]
    phase1_path = (repo_root / "Phase 1" / "phase1_bridge.py").resolve()
    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 bridge module not found: {phase1_path}")
    spec = importlib.util.spec_from_file_location("digifly_phase1_bridge", phase1_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase 1 bridge spec from {phase1_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PHASE1_BRIDGE = module
    return module


def ensure_phase1_exports_if_needed(
    body_ids: Sequence[int],
    *,
    swc_dir: str | Path,
    enabled: bool = True,
    neuprint_dataset: str | None = None,
    upsample_nm: float = 2000.0,
    min_conf: float = 0.4,
    batch_size: int = 10000,
    export_workers: int = 1,
    progress_every: int = 25,
    require_synapses: bool = True,
    require_swc: bool = True,
) -> Dict[str, Any]:
    ids = coerce_int_ids(body_ids, name="body ids")
    dataset_use = normalize_neuprint_dataset(neuprint_dataset)
    missing_ids = missing_local_export_ids(
        ids,
        swc_dir=swc_dir,
        neuprint_dataset=dataset_use,
        require_synapses=require_synapses,
        require_swc=require_swc,
    )
    if not missing_ids:
        return {"requested_ids": ids, "missing_ids": [], "status": "already_available", "neuprint_dataset": dataset_use}
    if not enabled:
        return {"requested_ids": ids, "missing_ids": missing_ids, "status": "missing_but_disabled", "neuprint_dataset": dataset_use}

    bridge = _load_phase1_bridge()
    report = bridge.ensure_phase2_neuron_exports(
        body_ids=missing_ids,
        export_root=swc_dir,
        dataset=dataset_use,
        upsample_nm=float(upsample_nm),
        min_conf=float(min_conf),
        batch_size=int(batch_size),
        workers=int(export_workers),
        progress_every=int(progress_every),
        skip_existing=False,
    )
    refresh_export_path_index(swc_dir)
    report["requested_ids"] = ids
    report["missing_ids"] = missing_ids
    report["status"] = "exported_missing"
    report["neuprint_dataset"] = dataset_use
    return report


def _load_master_table(master_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(master_csv).expanduser().resolve(), low_memory=False)


def _bodyid_column(df_master: pd.DataFrame) -> str:
    cols = {str(col).lower(): str(col) for col in df_master.columns}
    if "bodyid" not in cols:
        raise ValueError("Master CSV must contain a bodyId column.")
    return cols["bodyid"]


def _file_fingerprint(path: str | Path | None) -> str:
    if path is None:
        return ""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"missing:{p}"
    stat = p.stat()
    return f"{p}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}"


def _normalize_nt_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    aliases = {
        "ach": "acetylcholine",
        "acetylcholine": "acetylcholine",
        "glut": "glutamate",
        "glutamate": "glutamate",
        "gaba": "gaba",
        "gly": "glycine",
        "glycine": "glycine",
    }
    return aliases.get(text, text)


def _master_nt_lookup(master_csv: str | Path | None) -> pd.DataFrame:
    if master_csv is None:
        return pd.DataFrame(columns=["pre_id", "predicted_nt", "predicted_nt_prob"])
    df_master = _load_master_table(master_csv)
    id_col = _bodyid_column(df_master)
    nt_cols = [c for c in ("predictedNt", "consensusNt", "celltypePredictedNt", "otherNt") if c in df_master.columns]
    prob_col = next((c for c in ("predictedNtProb", "celltypeTotalNtPredictions") if c in df_master.columns), None)
    if not nt_cols:
        return pd.DataFrame(columns=["pre_id", "predicted_nt", "predicted_nt_prob"])

    nt_df = df_master[[id_col] + nt_cols + ([prob_col] if prob_col else [])].copy()
    nt_df[id_col] = pd.to_numeric(nt_df[id_col], errors="coerce")
    nt_df = nt_df[nt_df[id_col].notna()].copy()
    nt_df["pre_id"] = nt_df[id_col].astype(int)

    text_df = nt_df[nt_cols].copy().astype("string")
    for col in nt_cols:
        text_df[col] = text_df[col].replace(r"^\s*$", pd.NA, regex=True)
    nt_series = text_df.bfill(axis=1).iloc[:, 0].fillna("").astype(str).map(_normalize_nt_label)
    out = pd.DataFrame(
        {
            "pre_id": nt_df["pre_id"].astype(int),
            "predicted_nt": nt_series,
            "predicted_nt_prob": (
                pd.to_numeric(nt_df[prob_col], errors="coerce")
                if prob_col
                else pd.Series(float("nan"), index=nt_df.index)
            ),
        }
    )
    return out.drop_duplicates(subset=["pre_id"], keep="first").reset_index(drop=True)


def _synapse_defaults_for_nt(predicted_nt: Any, *, default_class: str | None = None) -> Dict[str, Any]:
    default_class_use = str(default_class or DEFAULT_GLOBAL_TIMING.get("default_class", "cholinergic_fast"))
    nt = _normalize_nt_label(predicted_nt)
    syn_class = NT_TO_CLASS.get(nt, default_class_use)
    tau1_ms, tau2_ms, erev_mV, _ = SYN_PRESETS.get(syn_class, SYN_PRESETS[default_class_use])
    return {
        "predicted_nt": nt,
        "syn_class": str(syn_class),
        "tau1_ms": float(tau1_ms),
        "tau2_ms": float(tau2_ms),
        "syn_e_rev_mV": float(erev_mV),
    }


def enrich_edge_df_from_master_nt(
    df_edges: pd.DataFrame,
    *,
    master_csv: str | Path | None,
    default_delay_ms: float = 1.0,
    default_class: str | None = None,
) -> pd.DataFrame:
    df_use = canonicalize_edge_df(df_edges).copy()
    nt_lookup = _master_nt_lookup(master_csv).set_index("pre_id") if master_csv else pd.DataFrame()
    if not nt_lookup.empty:
        master_nt = pd.to_numeric(df_use["pre_id"], errors="coerce").map(nt_lookup["predicted_nt"])
        master_prob = pd.to_numeric(df_use["pre_id"], errors="coerce").map(nt_lookup["predicted_nt_prob"])
    else:
        master_nt = pd.Series("", index=df_use.index, dtype="object")
        master_prob = pd.Series(float("nan"), index=df_use.index)

    existing_nt = df_use.get("predicted_nt", pd.Series("", index=df_use.index)).fillna("").map(_normalize_nt_label)
    final_nt = existing_nt.where(existing_nt.ne(""), master_nt.fillna("").map(_normalize_nt_label))
    df_use["predicted_nt"] = final_nt

    df_use["predicted_nt_prob"] = pd.to_numeric(df_use.get("predicted_nt_prob"), errors="coerce")
    df_use["predicted_nt_prob"] = df_use["predicted_nt_prob"].where(df_use["predicted_nt_prob"].notna(), master_prob)

    defaults = pd.DataFrame([_synapse_defaults_for_nt(nt, default_class=default_class) for nt in final_nt.tolist()])
    existing_syn_class = df_use.get("syn_class", pd.Series("", index=df_use.index)).fillna("").astype(str).str.strip()
    df_use["syn_class"] = existing_syn_class.where(existing_syn_class.ne(""), defaults["syn_class"].astype(str))

    for col in ("tau1_ms", "tau2_ms", "syn_e_rev_mV"):
        existing = pd.to_numeric(df_use.get(col), errors="coerce")
        df_use[col] = existing.where(existing.notna(), pd.to_numeric(defaults[col], errors="coerce"))

    delay_series = pd.to_numeric(df_use.get("delay_ms"), errors="coerce")
    df_use["delay_ms"] = delay_series.where(delay_series.notna(), float(default_delay_ms))

    nt_source = pd.Series("default_class", index=df_use.index, dtype="object")
    nt_source = nt_source.where(final_nt.eq(""), "master_csv")
    nt_source = nt_source.where(existing_nt.eq(""), "edge_csv")
    df_use["nt_source"] = nt_source
    return canonicalize_edge_df(df_use)


def motor_neuron_id_set_from_master(master_csv: str | Path) -> set[int]:
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


def expand_requested_network(
    requested_ids: Sequence[int],
    raw_edges_df: pd.DataFrame,
    *,
    expansion_mode: str = "none",
    master_csv: str | Path | None = None,
) -> Dict[str, Any]:
    req_ids = coerce_int_ids(requested_ids, name="requested ids")
    req_set = set(int(x) for x in req_ids)
    expansion_mode_use = str(expansion_mode or "none").strip().lower()
    raw_df = canonicalize_edge_df(raw_edges_df)

    if expansion_mode_use == "none":
        pre = pd.to_numeric(raw_df["pre_id"], errors="coerce")
        post = pd.to_numeric(raw_df["post_id"], errors="coerce")
        keep_mask = pre.isin(list(req_set)) & post.isin(list(req_set))
        final_edges_df = raw_df.loc[keep_mask].copy()
        return {
            "selection_rule": "requested ids only",
            "requested_ids": req_ids,
            "added_ids": [],
            "final_network_ids": req_ids,
            "final_edges_df": canonicalize_edge_df(final_edges_df),
        }

    if expansion_mode_use != "immediate_motor_postsynaptic":
        raise ValueError(f"Unsupported expansion_mode: {expansion_mode}")
    if master_csv is None:
        raise ValueError("master_csv is required when expansion_mode='immediate_motor_postsynaptic'")

    motor_ids = motor_neuron_id_set_from_master(master_csv)
    if raw_df.empty or "post_id" not in raw_df.columns:
        added_ids: List[int] = []
    else:
        posts = pd.to_numeric(raw_df["post_id"], errors="coerce").dropna().astype(int).tolist()
        added_ids = sorted((set(posts) & motor_ids) - req_set)

    final_network_ids = coerce_int_ids(list(req_ids) + list(added_ids), name="final network ids")
    final_set = set(final_network_ids)
    pre = pd.to_numeric(raw_df["pre_id"], errors="coerce")
    post = pd.to_numeric(raw_df["post_id"], errors="coerce")
    keep_mask = pre.isin(list(req_set)) & post.isin(list(final_set))
    final_edges_df = raw_df.loc[keep_mask].copy()
    return {
        "selection_rule": "requested ids + immediate postsynaptic motor neurons",
        "requested_ids": req_ids,
        "added_ids": [int(x) for x in added_ids],
        "final_network_ids": [int(x) for x in final_network_ids],
        "final_edges_df": canonicalize_edge_df(final_edges_df),
    }


def ensure_named_edge_set(
    *,
    edge_set_name: str | None,
    requested_ids: Sequence[int],
    swc_dir: str | Path,
    selection_mode: str,
    selection_label: str | None = None,
    expansion_mode: str = "none",
    master_csv: str | Path | None = None,
    neuprint_dataset: str | None = None,
    registry_root: str | Path | None = None,
    force_rebuild: bool = False,
    default_weight_uS: float = 6e-6,
    default_delay_ms: float = 1.0,
    one_row_per_synapse: bool = True,
    workers: int = 16,
    phase1_fallback_enabled: bool = True,
    phase1_upsample_nm: float = 2000.0,
    phase1_min_conf: float = 0.4,
    phase1_batch_size: int = 10000,
    phase1_export_workers: int = 1,
    phase1_progress_every: int = 25,
) -> Dict[str, Any]:
    requested_use = coerce_int_ids(requested_ids, name="requested ids")
    selection_mode_use = str(selection_mode).strip().lower()
    expansion_mode_use = str(expansion_mode or "none").strip().lower()
    dataset_use = normalize_neuprint_dataset(neuprint_dataset)
    signature = edge_request_signature(
        requested_use,
        selection_mode=selection_mode_use,
        expansion_mode=expansion_mode_use,
        selection_label=selection_label,
        neuprint_dataset=dataset_use,
        master_csv=master_csv,
        default_weight_uS=default_weight_uS,
        default_delay_ms=default_delay_ms,
    )

    existing = find_registered_edge_set(
        requested_use,
        selection_mode=selection_mode_use,
        expansion_mode=expansion_mode_use,
        selection_label=selection_label,
        neuprint_dataset=dataset_use,
        master_csv=master_csv,
        default_weight_uS=default_weight_uS,
        default_delay_ms=default_delay_ms,
        root=registry_root,
    )
    if existing is not None and not force_rebuild:
        final_path = Path(existing["final_edges_path"]).expanduser().resolve()
        raw_path = Path(existing["raw_edges_path"]).expanduser().resolve()
        metadata_path = Path(existing["metadata_json_path"]).expanduser().resolve()
        try:
            final_df = pd.read_csv(final_path)
        except pd.errors.EmptyDataError:
            final_df = pd.DataFrame(columns=EDGE_SET_COLUMNS)
        try:
            raw_df = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(columns=EDGE_SET_COLUMNS)
        except pd.errors.EmptyDataError:
            raw_df = pd.DataFrame(columns=EDGE_SET_COLUMNS)
        payload = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {
            "edge_set_name": str(existing["edge_set_name"]),
            "edge_set_slug": str(existing["edge_set_slug"]),
            "edge_signature": signature,
            "neuprint_dataset": str(existing.get("neuprint_dataset", dataset_use) or dataset_use),
            "registry_root": str(default_edges_registry_root(registry_root)),
            "requested_ids": requested_use,
            "final_network_ids": payload.get("final_network_ids", requested_use),
            "added_ids": payload.get("added_ids", []),
            "selection_rule": payload.get("selection_rule", "requested ids only"),
            "raw_edges_path": raw_path,
            "final_edges_path": final_path,
            "metadata_json_path": metadata_path,
            "raw_edges_df": canonicalize_edge_df(raw_df),
            "final_edges_df": canonicalize_edge_df(final_df),
            "phase1_report": {"status": "not_needed"},
            "reused_existing": True,
        }

    named_existing = _find_by_edge_set_name(str(edge_set_name), root=registry_root) if edge_set_name else None
    if (
        named_existing is not None
        and str(named_existing.get("edge_signature") or "") != signature
        and not force_rebuild
    ):
        raise ValueError(
            f"Edge-set name '{edge_set_name}' is already registered for a different neuron list/signature. "
            "Pick a different edge-set name or reuse the existing registered set."
        )

    fallback_name = f"{selection_mode_use}_{selection_label or 'set'}_{signature[:12]}"
    edge_set_name_use = normalize_edge_set_name(edge_set_name, fallback=fallback_name)
    edge_set_slug_use = edge_set_slug(edge_set_name_use)
    paths = ensure_edges_registry_layout(registry_root)
    edge_dir = (paths["edge_sets_dir"] / edge_set_slug_use).resolve()
    edge_dir.mkdir(parents=True, exist_ok=True)

    phase1_report = ensure_phase1_exports_if_needed(
        requested_use,
        swc_dir=swc_dir,
        enabled=bool(phase1_fallback_enabled),
        neuprint_dataset=dataset_use,
        upsample_nm=float(phase1_upsample_nm),
        min_conf=float(phase1_min_conf),
        batch_size=int(phase1_batch_size),
        export_workers=int(phase1_export_workers),
        progress_every=int(phase1_progress_every),
        require_synapses=True,
        require_swc=True,
    )

    raw_edges_df = _hemi_fetch_edges_from_synapse_csvs(
        seeds=requested_use,
        hemi_ids=requested_use,
        label=edge_set_name_use,
        swc_root=Path(swc_dir).expanduser().resolve(),
        default_weight_uS=float(default_weight_uS),
        one_row_per_synapse=bool(one_row_per_synapse),
        smoke_test=False,
        pres_limit=None,
        workers=max(1, int(workers)),
        smoke_seeds_only=False,
    )
    raw_edges_df = enrich_edge_df_from_master_nt(
        raw_edges_df,
        master_csv=master_csv,
        default_delay_ms=float(default_delay_ms),
        default_class=str(DEFAULT_GLOBAL_TIMING.get("default_class", "cholinergic_fast")),
    )
    raw_edges_df = canonicalize_edge_df(raw_edges_df)

    expanded = expand_requested_network(
        requested_use,
        raw_edges_df,
        expansion_mode=expansion_mode_use,
        master_csv=master_csv,
    )
    final_network_ids = coerce_int_ids(expanded["final_network_ids"], name="final network ids")
    final_edges_df = canonicalize_edge_df(expanded["final_edges_df"])

    phase1_final_report = ensure_phase1_exports_if_needed(
        final_network_ids,
        swc_dir=swc_dir,
        enabled=bool(phase1_fallback_enabled),
        neuprint_dataset=dataset_use,
        upsample_nm=float(phase1_upsample_nm),
        min_conf=float(phase1_min_conf),
        batch_size=int(phase1_batch_size),
        export_workers=int(phase1_export_workers),
        progress_every=int(phase1_progress_every),
        require_synapses=False,
        require_swc=True,
    )

    raw_edges_path = (edge_dir / f"{edge_set_slug_use}_raw_edges.csv").resolve()
    final_edges_path = (edge_dir / f"{edge_set_slug_use}_edges.csv").resolve()
    metadata_json_path = (edge_dir / "edge_set_metadata.json").resolve()
    requested_ids_csv = (edge_dir / "requested_ids.csv").resolve()
    final_network_ids_csv = (edge_dir / "final_network_ids.csv").resolve()
    added_ids_csv = (edge_dir / "added_ids.csv").resolve()

    raw_edges_df.to_csv(raw_edges_path, index=False)
    final_edges_df.to_csv(final_edges_path, index=False)
    pd.DataFrame({"neuron_id": requested_use}).to_csv(requested_ids_csv, index=False)
    pd.DataFrame({"neuron_id": final_network_ids}).to_csv(final_network_ids_csv, index=False)
    pd.DataFrame({"neuron_id": [int(x) for x in expanded["added_ids"]]}).to_csv(added_ids_csv, index=False)

    metadata = {
        "edge_set_name": edge_set_name_use,
        "edge_set_slug": edge_set_slug_use,
        "edge_signature": signature,
        "edge_schema_version": EDGE_SET_SCHEMA_VERSION,
        "neuprint_dataset": dataset_use,
        "selection_mode": selection_mode_use,
        "selection_label": str(selection_label or ""),
        "selection_rule": expanded["selection_rule"],
        "expansion_mode": expansion_mode_use,
        "requested_ids": [int(x) for x in requested_use],
        "requested_count": int(len(requested_use)),
        "added_ids": [int(x) for x in expanded["added_ids"]],
        "final_network_ids": [int(x) for x in final_network_ids],
        "final_network_count": int(len(final_network_ids)),
        "raw_edge_rows": int(len(raw_edges_df)),
        "final_edge_rows": int(len(final_edges_df)),
        "swc_dir": str(Path(swc_dir).expanduser().resolve()),
        "master_csv": str(Path(master_csv).expanduser().resolve()) if master_csv else None,
        "default_weight_uS": float(default_weight_uS),
        "default_delay_ms": float(default_delay_ms),
        "raw_edges_path": str(raw_edges_path),
        "final_edges_path": str(final_edges_path),
        "requested_ids_csv": str(requested_ids_csv),
        "final_network_ids_csv": str(final_network_ids_csv),
        "added_ids_csv": str(added_ids_csv),
        "phase1_core_report": phase1_report,
        "phase1_final_report": phase1_final_report,
        "created_at_utc": _utc_now_iso(),
    }
    metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_row = {
        "edge_set_name": edge_set_name_use,
        "edge_set_slug": edge_set_slug_use,
        "edge_signature": signature,
        "neuprint_dataset": dataset_use,
        "selection_mode": selection_mode_use,
        "selection_label": str(selection_label or ""),
        "selection_rule": expanded["selection_rule"],
        "expansion_mode": expansion_mode_use,
        "requested_count": int(len(requested_use)),
        "final_network_count": int(len(final_network_ids)),
        "requested_ids_json": json.dumps([int(x) for x in requested_use]),
        "final_network_ids_json": json.dumps([int(x) for x in final_network_ids]),
        "added_ids_json": json.dumps([int(x) for x in expanded["added_ids"]]),
        "swc_dir": str(Path(swc_dir).expanduser().resolve()),
        "master_csv": str(Path(master_csv).expanduser().resolve()) if master_csv else "",
        "raw_edges_path": str(raw_edges_path),
        "final_edges_path": str(final_edges_path),
        "metadata_json_path": str(metadata_json_path),
        "created_at_utc": metadata["created_at_utc"],
        "updated_at_utc": metadata["created_at_utc"],
    }
    _upsert_registry_row(registry_row, root=registry_root)
    refresh_master_edges_csv(root=registry_root)

    return {
        "edge_set_name": edge_set_name_use,
        "edge_set_slug": edge_set_slug_use,
        "edge_signature": signature,
        "neuprint_dataset": dataset_use,
        "registry_root": str(paths["root"]),
        "requested_ids": requested_use,
        "final_network_ids": final_network_ids,
        "added_ids": [int(x) for x in expanded["added_ids"]],
        "selection_rule": expanded["selection_rule"],
        "raw_edges_path": raw_edges_path,
        "final_edges_path": final_edges_path,
        "metadata_json_path": metadata_json_path,
        "raw_edges_df": raw_edges_df,
        "final_edges_df": final_edges_df,
        "phase1_report": {
            "core": phase1_report,
            "final": phase1_final_report,
        },
        "reused_existing": False,
    }
