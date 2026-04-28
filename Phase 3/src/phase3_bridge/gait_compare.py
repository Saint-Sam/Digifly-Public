from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LEG_PHASE_ORDER: List[Tuple[str, str]] = [
    ("T1_left", "swing"),
    ("T1_left", "stance"),
    ("T1_right", "swing"),
    ("T1_right", "stance"),
    ("T2_left", "swing"),
    ("T2_left", "stance"),
    ("T2_right", "swing"),
    ("T2_right", "stance"),
    ("T3_left", "swing"),
    ("T3_left", "stance"),
    ("T3_right", "swing"),
    ("T3_right", "stance"),
]


def _safe_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _expected_leg_phase_from_channels(phase_df: pd.DataFrame) -> pd.DataFrame:
    t = phase_df["t_ms"].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    for leg_id, phase in LEG_PHASE_ORDER:
        col = f"{leg_id}__{phase}"
        if col not in phase_df.columns:
            continue
        sig = phase_df[col].to_numpy(dtype=float)
        peak = float(np.nanmax(sig)) if sig.size else 0.0
        area = _safe_trapezoid(np.maximum(sig, 0.0), t) if sig.size else 0.0
        onset_ms = None
        peak_ms = None
        if peak > 1e-12:
            active = np.flatnonzero(sig >= 0.1 * peak)
            if active.size:
                onset_ms = float(t[int(active[0])])
            peak_ms = float(t[int(np.nanargmax(sig))])
        rows.append(
            {
                "leg_id": leg_id,
                "phase": phase,
                "peak": peak,
                "area": area,
                "onset_ms": onset_ms,
                "peak_ms": peak_ms,
            }
        )
    return pd.DataFrame(rows)


def _expected_weight_by_leg_phase(weight_df: pd.DataFrame) -> pd.DataFrame:
    if weight_df.empty:
        return pd.DataFrame(columns=["leg_id", "phase", "expected_weight"])
    out = (
        weight_df.groupby(["leg_id", "expected_phase"], as_index=False)["base_weight"]
        .sum()
        .rename(columns={"expected_phase": "phase", "base_weight": "expected_weight"})
    )
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
        + 0.5 * np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))
    )


def _load_tripod_gap(tripod_df: pd.DataFrame) -> float | None:
    lookup = tripod_df.set_index(["group", "phase"])
    keys = [("tripod_A", "swing"), ("tripod_B", "swing")]
    if any(key not in lookup.index for key in keys):
        return None
    a = lookup.loc[("tripod_A", "swing"), "onset_ms"]
    b = lookup.loc[("tripod_B", "swing"), "onset_ms"]
    if pd.isna(a) or pd.isna(b):
        return None
    return float(abs(float(a) - float(b)))


def _phase_vector(df: pd.DataFrame, value_col: str) -> np.ndarray:
    lookup = df.set_index(["leg_id", "phase"])
    values: List[float] = []
    for leg_id, phase in LEG_PHASE_ORDER:
        if (leg_id, phase) not in lookup.index:
            values.append(0.0)
            continue
        values.append(float(lookup.loc[(leg_id, phase), value_col]))
    vec = np.asarray(values, dtype=float)
    total = float(vec.sum())
    if total > 1e-12:
        vec = vec / total
    return vec


def _closeness_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if a <= 1e-12 or b <= 1e-12:
        return None
    return float(min(a, b) / max(a, b))


def _classify(score: float) -> str:
    if score >= 0.75:
        return "close"
    if score >= 0.5:
        return "moderately_close"
    if score >= 0.25:
        return "far"
    return "very_far"


def compare_gait_to_expected(
    sim_audit_dir: Path | str,
    expected_dir: Path | str,
    out_dir: Path | str,
) -> Dict[str, object]:
    sim_audit_dir = Path(sim_audit_dir).expanduser().resolve()
    expected_dir = Path(expected_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_report_path = sim_audit_dir / "gait_expectation_report.json"
    sim_leg_path = sim_audit_dir / "leg_phase_summary.csv"
    sim_co_path = sim_audit_dir / "leg_coactivation_summary.csv"
    sim_tripod_path = sim_audit_dir / "tripod_phase_summary.csv"
    expected_phase_path = expected_dir / "expected_gait_phase_channels_active.csv"
    expected_weight_path = expected_dir / "expected_gait_weights_active.csv"

    sim_report = json.loads(sim_report_path.read_text(encoding="utf-8"))
    sim_leg = pd.read_csv(sim_leg_path)
    sim_co = pd.read_csv(sim_co_path)
    sim_tripod = pd.read_csv(sim_tripod_path)
    expected_phase_df = pd.read_csv(expected_phase_path)
    expected_weight_df = pd.read_csv(expected_weight_path)

    expected_leg = _expected_leg_phase_from_channels(expected_phase_df)
    expected_leg.to_csv(out_dir / "expected_leg_phase_summary.csv", index=False)

    expected_weight = _expected_weight_by_leg_phase(expected_weight_df)
    expected_weight.to_csv(out_dir / "expected_leg_phase_weights.csv", index=False)

    sim_area_vec = _phase_vector(sim_leg, "area")
    exp_area_vec = _phase_vector(expected_leg, "area")
    exp_weight_vec = _phase_vector(expected_weight.rename(columns={"expected_weight": "area"}), "area")

    area_cosine = _cosine_similarity(sim_area_vec, exp_area_vec)
    weight_cosine = _cosine_similarity(sim_area_vec, exp_weight_vec)
    area_l1 = float(np.abs(sim_area_vec - exp_area_vec).sum())
    weight_l1 = float(np.abs(sim_area_vec - exp_weight_vec).sum())
    area_js = _js_divergence(sim_area_vec, exp_area_vec)
    weight_js = _js_divergence(sim_area_vec, exp_weight_vec)

    sim_tripod_gap = _load_tripod_gap(sim_tripod)
    expected_tripod_phase = pd.DataFrame(
        [
            {"group": "tripod_A", "phase": "swing", "onset_ms": float(expected_leg[expected_leg["leg_id"].isin(["T1_left", "T2_right", "T3_left"]) & (expected_leg["phase"] == "swing")]["onset_ms"].min())},
            {"group": "tripod_B", "phase": "swing", "onset_ms": float(expected_leg[expected_leg["leg_id"].isin(["T1_right", "T2_left", "T3_right"]) & (expected_leg["phase"] == "swing")]["onset_ms"].min())},
        ]
    )
    expected_tripod_gap = _load_tripod_gap(expected_tripod_phase)

    sim_coactivation = float(sim_co["coactivation_ratio"].dropna().mean()) if not sim_co.empty else None
    expected_coactivation = 0.0

    sim_dominance = float(sim_leg["area"].max() / sim_leg.loc[sim_leg["area"] > 1e-12, "area"].median())
    expected_dominance = float(
        expected_leg["area"].max() / expected_leg.loc[expected_leg["area"] > 1e-12, "area"].median()
    )

    timing_closeness = _closeness_ratio(sim_tripod_gap, expected_tripod_gap)
    coactivation_closeness = None
    if sim_coactivation is not None:
        coactivation_closeness = float(max(0.0, 1.0 - abs(sim_coactivation - expected_coactivation)))
    dominance_closeness = _closeness_ratio(sim_dominance, expected_dominance)

    components = [
        ("phase_area_cosine", area_cosine),
        ("phase_weight_cosine", weight_cosine),
        ("tripod_gap_closeness", timing_closeness if timing_closeness is not None else 0.0),
        ("coactivation_closeness", coactivation_closeness if coactivation_closeness is not None else 0.0),
        ("dominance_closeness", dominance_closeness if dominance_closeness is not None else 0.0),
    ]
    overall_score = float(np.mean([value for _, value in components]))

    merged = (
        sim_leg[["leg_id", "phase", "area"]]
        .merge(expected_weight, on=["leg_id", "phase"], how="left")
        .rename(columns={"area": "sim_area"})
        .fillna({"expected_weight": 0.0})
    )
    sim_total = float(merged["sim_area"].sum()) or 1.0
    exp_total = float(merged["expected_weight"].sum()) or 1.0
    merged["sim_norm"] = merged["sim_area"] / sim_total
    merged["expected_norm"] = merged["expected_weight"] / exp_total
    merged["norm_diff"] = merged["sim_norm"] - merged["expected_norm"]
    merged.to_csv(out_dir / "phase_distribution_comparison.csv", index=False)

    largest_positive = (
        merged.sort_values("norm_diff", ascending=False).head(5)[["leg_id", "phase", "norm_diff"]].to_dict("records")
    )
    largest_negative = (
        merged.sort_values("norm_diff", ascending=True).head(5)[["leg_id", "phase", "norm_diff"]].to_dict("records")
    )

    recommendations: List[str] = []
    if timing_closeness is not None and timing_closeness < 0.1:
        recommendations.append(
            "The simulated motor drive is almost perfectly synchronous across tripod groups; improving premotor phasing or adding more rhythm-generating context should be a top priority."
        )
    if sim_coactivation is not None and sim_coactivation > 0.5:
        recommendations.append(
            "Within-leg swing and stance channels strongly overlap, so antagonistic gating/inhibition is likely missing or too weak."
        )
    if sim_dominance > 3.0:
        recommendations.append(
            "A single leg-phase channel dominates the pattern, so the left-front support/extensor chain is overweight relative to the rest of the network."
        )
    if sim_report.get("cross_phase_neuron_count", 0) > 0:
        recommendations.append(
            "Some active neurons are mapped into contradictory swing and stance categories, so the mapping still contributes part of the mismatch."
        )

    report = {
        "sim_audit_dir": str(sim_audit_dir),
        "expected_dir": str(expected_dir),
        "overall_closeness_score": overall_score,
        "overall_classification": _classify(overall_score),
        "component_scores": {name: float(value) for name, value in components},
        "distribution_similarity": {
            "phase_area_cosine_similarity": area_cosine,
            "phase_area_l1_distance": area_l1,
            "phase_area_js_divergence": area_js,
            "phase_weight_cosine_similarity": weight_cosine,
            "phase_weight_l1_distance": weight_l1,
            "phase_weight_js_divergence": weight_js,
        },
        "timing": {
            "tripod_swing_gap_ms": {
                "sim": sim_tripod_gap,
                "expected": expected_tripod_gap,
                "closeness_ratio": timing_closeness,
            },
            "mean_coactivation": {
                "sim": sim_coactivation,
                "expected": expected_coactivation,
                "closeness_score": coactivation_closeness,
            },
        },
        "balance": {
            "dominance_ratio": {
                "sim": sim_dominance,
                "expected": expected_dominance,
                "closeness_ratio": dominance_closeness,
            }
        },
        "largest_positive_deviations": largest_positive,
        "largest_negative_deviations": largest_negative,
        "sim_report_warnings": sim_report.get("warnings", []),
        "recommendations": recommendations,
    }

    report_path = out_dir / "gait_compare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
