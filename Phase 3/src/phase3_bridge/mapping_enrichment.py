from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


LOW_CONF_WORDS = {
    "low",
    "medium",
    "weak",
    "todo",
    "unknown",
    "placeholder",
    "needs review",
}


def _norm_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", ""}:
        return ""
    return s


def _as_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _extract_side(instance: str) -> str:
    base = _norm_text(instance).split()[0] if _norm_text(instance) else ""
    if not base:
        return ""
    m = re.search(r"(?:^|_)(L|R)(?:$|_)", base, flags=re.IGNORECASE)
    if not m:
        return ""
    return "left" if m.group(1).upper() == "L" else "right"


def _extract_mn_type(instance: str, fallback_type: str) -> str:
    if _norm_text(fallback_type):
        return _norm_text(fallback_type)
    base = _norm_text(instance).split()[0] if _norm_text(instance) else ""
    if not base:
        return ""
    return base.split("_")[0]


def _canonicalize_name(name: str, model_names: Iterable[str]) -> str:
    n = _norm_text(name)
    if not n:
        return ""
    names = [str(x) for x in model_names if _norm_text(x)]
    by_lc = {m.lower(): m for m in names}

    hit = by_lc.get(n.lower())
    if hit:
        return hit

    suffix_hits = [m for m in names if m.lower().endswith("/" + n.lower())]
    if len(suffix_hits) == 1:
        return suffix_hits[0]

    rev_hits = [m for m in names if n.lower().endswith("/" + m.lower())]
    if len(rev_hits) == 1:
        return rev_hits[0]
    return ""


def _load_mjcf_actuator_names(mjcf_xml: Path | str | None) -> list[str]:
    if mjcf_xml is None:
        return []
    p = Path(mjcf_xml).expanduser().resolve()
    if not p.exists():
        return []
    visited: set[Path] = set()
    names: set[str] = set()

    def _walk(xml_path: Path) -> None:
        xml_path = xml_path.resolve()
        if xml_path in visited or not xml_path.exists():
            return
        visited.add(xml_path)
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return

        for elem in root.iter():
            tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
            if tag in {"motor", "position", "velocity", "general", "adhesion"}:
                nm = _norm_text(elem.attrib.get("name", ""))
                if nm:
                    names.add(nm)

            if tag == "include":
                inc = _norm_text(elem.attrib.get("file", ""))
                if inc:
                    _walk((xml_path.parent / inc).resolve())

    _walk(p)
    return sorted(names)


def _pick_mode(values: pd.Series) -> str:
    vals = values.dropna().astype(str).str.strip()
    vals = vals[vals.ne("")]
    if vals.empty:
        return ""
    return vals.value_counts().index[0]


def _build_suggestion_tables(df_good: pd.DataFrame) -> dict:
    t1 = (
        df_good.groupby(["type_norm", "side_norm"])["actuator_name_canonical"]
        .agg(_pick_mode)
        .reset_index()
    )
    t2 = df_good.groupby(["type_norm"])["actuator_name_canonical"].agg(_pick_mode).reset_index()
    t3 = df_good.groupby(["mn_type_norm"])["actuator_name_canonical"].agg(_pick_mode).reset_index()

    by_type_side = {(r["type_norm"], r["side_norm"]): r["actuator_name_canonical"] for _, r in t1.iterrows()}
    by_type = {r["type_norm"]: r["actuator_name_canonical"] for _, r in t2.iterrows()}
    by_mn_type = {r["mn_type_norm"]: r["actuator_name_canonical"] for _, r in t3.iterrows()}
    return {"type_side": by_type_side, "type": by_type, "mn_type": by_mn_type}


def _suggest_actuator(type_norm: str, mn_type_norm: str, side_norm: str, tables: dict) -> Tuple[str, str]:
    key = (type_norm, side_norm)
    if key in tables["type_side"]:
        return tables["type_side"][key], "type+side"
    if type_norm in tables["type"]:
        return tables["type"][type_norm], "type"
    if mn_type_norm in tables["mn_type"]:
        return tables["mn_type"][mn_type_norm], "mn_type"
    return "", ""


def run_mapping_enrichment(
    phase1_mn_csv: Path | str,
    mapping_csv: Path | str,
    out_dir: Path | str,
    mjcf_xml: Path | str | None = None,
    low_score_threshold: float = 4.0,
) -> Dict[str, str]:
    """
    Enrich Phase 3 MN->actuator mapping using Phase 1 all-MN export.

    Writes:
    - mapping_existing_review.csv
    - mapping_low_confidence.csv
    - mapping_phase1_coverage.csv
    - mapping_missing_from_phase1_template.csv
    - mapping_enriched_full.csv
    - enrichment_summary.json
    """
    phase1_mn_csv = Path(phase1_mn_csv).expanduser().resolve()
    mapping_csv = Path(mapping_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not phase1_mn_csv.exists():
        raise FileNotFoundError(f"Phase 1 MN export not found: {phase1_mn_csv}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    p1 = pd.read_csv(phase1_mn_csv).copy()
    mp = pd.read_csv(mapping_csv).copy()

    if "bodyId" not in p1.columns:
        raise ValueError("Phase 1 MN export must contain bodyId column.")
    if "mn_id" not in mp.columns:
        raise ValueError("Mapping CSV must contain mn_id column.")

    p1["mn_id"] = _as_int_series(p1["bodyId"])
    p1 = p1.dropna(subset=["mn_id"]).copy()
    p1["mn_id"] = p1["mn_id"].astype(int)
    p1["phase1_instance"] = p1.get("instance", "").map(_norm_text)
    p1["phase1_type"] = p1.get("type", "").map(_norm_text)
    p1["phase1_class"] = p1.get("class", "").map(_norm_text)
    p1["phase1_side"] = p1["phase1_instance"].map(_extract_side)
    p1["phase1_mn_type"] = [
        _extract_mn_type(inst, typ) for inst, typ in zip(p1["phase1_instance"], p1["phase1_type"])
    ]

    mp["mn_id"] = _as_int_series(mp["mn_id"])
    mp = mp.dropna(subset=["mn_id"]).copy()
    mp["mn_id"] = mp["mn_id"].astype(int)

    if "actuator_name" not in mp.columns:
        mp["actuator_name"] = ""
    mp["actuator_name"] = mp["actuator_name"].map(_norm_text)
    if "confidence" not in mp.columns:
        mp["confidence"] = ""
    mp["confidence"] = mp["confidence"].map(_norm_text)
    if "actuator_name_autofill_score" not in mp.columns:
        mp["actuator_name_autofill_score"] = np.nan
    mp["actuator_name_autofill_score"] = pd.to_numeric(mp["actuator_name_autofill_score"], errors="coerce")

    if "type" not in mp.columns:
        mp["type"] = mp.get("mn_type", "")
    if "instance" not in mp.columns:
        mp["instance"] = ""
    if "side" not in mp.columns:
        mp["side"] = mp["instance"].map(_extract_side)

    mp["type_norm"] = mp["type"].map(_norm_text)
    mp["mn_type_norm"] = mp.get("mn_type", "").map(_norm_text)
    mp["side_norm"] = mp["side"].map(_norm_text).str.lower()

    model_actuators = _load_mjcf_actuator_names(mjcf_xml)
    mp["actuator_name_canonical"] = mp["actuator_name"].map(lambda x: _canonicalize_name(x, model_actuators))
    if model_actuators:
        mp["actuator_in_mjcf"] = mp["actuator_name_canonical"].ne("")
    else:
        mp["actuator_in_mjcf"] = True

    low_conf_mask = mp["confidence"].str.lower().isin(LOW_CONF_WORDS)
    low_score_mask = mp["actuator_name_autofill_score"].notna() & (
        mp["actuator_name_autofill_score"] <= float(low_score_threshold)
    )
    missing_act_mask = mp["actuator_name"].eq("")
    invalid_act_mask = ~mp["actuator_in_mjcf"]

    reason_cols = {
        "missing_actuator_name": missing_act_mask,
        "low_confidence_text": low_conf_mask,
        "low_autofill_score": low_score_mask,
        "actuator_not_in_mjcf": invalid_act_mask,
    }

    reason_df = pd.DataFrame(reason_cols)
    mp["review_reasons"] = reason_df.apply(
        lambda row: "; ".join([name for name, flag in row.items() if bool(flag)]), axis=1
    )
    mp["review_needs_attention"] = mp["review_reasons"].ne("")

    # Build suggestion tables from high-confidence rows.
    good = mp[(~mp["review_needs_attention"]) & mp["actuator_name_canonical"].ne("")].copy()
    suggest_tables = _build_suggestion_tables(good if not good.empty else mp[mp["actuator_name_canonical"].ne("")])

    # Coverage by Phase 1 MN export.
    map_counts = mp.groupby("mn_id").size().rename("mapping_row_count")
    map_acts = (
        mp.groupby("mn_id")["actuator_name_canonical"]
        .apply(lambda s: sorted({x for x in s if _norm_text(x)}))
        .rename("existing_actuators")
    )
    cov = p1.merge(map_counts, on="mn_id", how="left").merge(map_acts, on="mn_id", how="left")
    cov["mapping_row_count"] = cov["mapping_row_count"].fillna(0).astype(int)
    cov["existing_actuators"] = cov["existing_actuators"].apply(lambda x: x if isinstance(x, list) else [])
    cov["mapped_actuator_count"] = cov["existing_actuators"].map(len)
    cov["mapping_status"] = np.where(cov["mapping_row_count"] > 0, "mapped", "missing")

    type_norm = cov["phase1_type"].map(_norm_text)
    mn_type_norm = cov["phase1_mn_type"].map(_norm_text)
    side_norm = cov["phase1_side"].map(_norm_text).str.lower()
    sugg = [
        _suggest_actuator(t, mnt, s, suggest_tables) for t, mnt, s in zip(type_norm, mn_type_norm, side_norm)
    ]
    cov["suggested_actuator_name"] = [x[0] for x in sugg]
    cov["suggestion_basis"] = [x[1] for x in sugg]
    cov["existing_actuators_joined"] = cov["existing_actuators"].map(lambda xs: "|".join(xs))

    # Missing template rows (for manual completion).
    missing_cov = cov[cov["mapping_status"].eq("missing")].copy()
    templ = pd.DataFrame(columns=mp.columns)
    templ["mn_id"] = missing_cov["mn_id"].to_numpy(dtype=int)
    if "mn_type" in templ.columns:
        templ["mn_type"] = missing_cov["phase1_mn_type"].to_numpy()
    if "type" in templ.columns:
        templ["type"] = missing_cov["phase1_type"].to_numpy()
    if "instance" in templ.columns:
        templ["instance"] = missing_cov["phase1_instance"].to_numpy()
    if "class" in templ.columns:
        templ["class"] = missing_cov["phase1_class"].to_numpy()
    if "side" in templ.columns:
        templ["side"] = missing_cov["phase1_side"].to_numpy()
    if "actuator_name" in templ.columns:
        templ["actuator_name"] = ""
    templ["suggested_actuator_name"] = missing_cov["suggested_actuator_name"].to_numpy()
    templ["suggestion_basis"] = missing_cov["suggestion_basis"].to_numpy()
    templ["review_needs_attention"] = True
    templ["review_reasons"] = "missing_mapping_row"
    templ["source"] = "phase1_missing"

    mp_out = mp.copy()
    mp_out["source"] = "existing_mapping"
    low_conf_rows = mp_out[mp_out["review_needs_attention"]].copy()
    enriched_full = pd.concat([mp_out, templ], ignore_index=True, sort=False)

    # Write outputs.
    out_existing = out_dir / "mapping_existing_review.csv"
    out_low = out_dir / "mapping_low_confidence.csv"
    out_cov = out_dir / "mapping_phase1_coverage.csv"
    out_missing = out_dir / "mapping_missing_from_phase1_template.csv"
    out_full = out_dir / "mapping_enriched_full.csv"
    out_json = out_dir / "enrichment_summary.json"

    mp_out.to_csv(out_existing, index=False)
    low_conf_rows.to_csv(out_low, index=False)
    cov.to_csv(out_cov, index=False)
    templ.to_csv(out_missing, index=False)
    enriched_full.to_csv(out_full, index=False)

    summary = {
        "phase1_total_mn": int(len(p1)),
        "mapping_rows_in": int(len(mp)),
        "mapped_phase1_mn": int(cov["mapping_status"].eq("mapped").sum()),
        "missing_phase1_mn": int(cov["mapping_status"].eq("missing").sum()),
        "existing_rows_needing_review": int(len(low_conf_rows)),
        "suggestions_available_for_missing": int(
            missing_cov["suggested_actuator_name"].map(_norm_text).ne("").sum()
        ),
        "mjcf_actuator_count": int(len(model_actuators)),
        "outputs": {
            "mapping_existing_review_csv": str(out_existing),
            "mapping_low_confidence_csv": str(out_low),
            "mapping_phase1_coverage_csv": str(out_cov),
            "mapping_missing_template_csv": str(out_missing),
            "mapping_enriched_full_csv": str(out_full),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2))
    return {"summary_json": str(out_json), **summary["outputs"]}
