from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from .pipeline import _spikes_to_activation, load_mapping_csv, load_phase2_spike_times, load_phase2_timebase_ms
except ImportError:  # pragma: no cover - allows direct script imports via sys.path injection
    from pipeline import _spikes_to_activation, load_mapping_csv, load_phase2_spike_times, load_phase2_timebase_ms


TRIPOD_GROUPS: Dict[str, Tuple[str, ...]] = {
    "tripod_A": ("T1_left", "T2_right", "T3_left"),
    "tripod_B": ("T1_right", "T2_left", "T3_right"),
}

LEG_ORDER = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
CORE_PHASES = {"swing", "stance"}


@dataclass(frozen=True)
class ActionExpectation:
    expected_phase: str
    confidence: str
    rationale: str


ACTION_EXPECTATIONS: Dict[str, ActionExpectation] = {
    "flex": ActionExpectation("swing", "high", "Flexors are treated as swing-phase effectors."),
    "flex_like": ActionExpectation("swing", "medium", "Flex-like fallback rule treated as swing-phase drive."),
    "promotor": ActionExpectation("swing", "medium", "Promotor action is treated as forward swing initiation."),
    "abductor": ActionExpectation("swing", "medium", "Abduction is used as a swing-support proxy in the bridge."),
    "rotate_anterior": ActionExpectation("swing", "medium", "Anterior rotation is treated as swing-support drive."),
    "elevate": ActionExpectation("swing", "high", "Levators are treated as swing-phase effectors."),
    "extend": ActionExpectation("stance", "high", "Extensors are treated as stance/support effectors."),
    "extend_like": ActionExpectation("stance", "medium", "Extensor-like rule treated as stance/support drive."),
    "femur_drive": ActionExpectation("stance", "low", "Generic femur drive is provisionally treated as stance-like."),
    "reduce": ActionExpectation("stance", "medium", "Reductors are treated as stance/support stabilizers."),
    "reduce_like": ActionExpectation("stance", "low", "Fallback femur reduction treated as stance-like."),
    "remotor": ActionExpectation("stance", "medium", "Remotor action is treated as stance/propulsion support."),
    "adductor": ActionExpectation("stance", "medium", "Adduction is treated as stance/support drive."),
    "depress": ActionExpectation("stance", "high", "Depressors are treated as stance/contact effectors."),
    "rotate_posterior": ActionExpectation("stance", "high", "Posterior rotation is treated as stance-support drive."),
    "rotate_posterior_like": ActionExpectation(
        "stance",
        "low",
        "Fallback posterior rotation treated as stance-support drive.",
    ),
    "jump_extensor_chain": ActionExpectation("special", "low", "Jump-chain activity is excluded from gait scoring."),
    "jump_assist": ActionExpectation("special", "low", "Jump-assist activity is excluded from gait scoring."),
    "abdomen_drive": ActionExpectation("ignore", "high", "Abdomen activity is excluded from leg gait scoring."),
}


def _integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _leg_id(row: pd.Series) -> str:
    thorax = str(row.get("thorax", "")).strip()
    side = str(row.get("side", "")).strip().lower()
    if thorax not in {"T1", "T2", "T3"} or side not in {"left", "right"}:
        return ""
    return f"{thorax}_{side}"


def _channel_stats(t_ms: np.ndarray, y: np.ndarray, onset_fraction: float = 0.1) -> Dict[str, float | None]:
    sig = np.asarray(y, dtype=float)
    peak = float(np.nanmax(sig)) if sig.size else 0.0
    area = _integral(np.maximum(sig, 0.0), t_ms) if sig.size else 0.0
    if peak <= 1e-12:
        return {
            "peak": 0.0,
            "area": 0.0,
            "onset_ms": None,
            "peak_ms": None,
        }
    threshold = max(float(onset_fraction) * peak, 1e-12)
    active = np.flatnonzero(sig >= threshold)
    onset_ms = float(t_ms[active[0]]) if active.size else None
    peak_ms = float(t_ms[int(np.nanargmax(sig))])
    return {
        "peak": peak,
        "area": area,
        "onset_ms": onset_ms,
        "peak_ms": peak_ms,
    }


def _coactivation_ratio(t_ms: np.ndarray, swing: np.ndarray, stance: np.ndarray) -> float | None:
    swing_peak = float(np.nanmax(swing)) if swing.size else 0.0
    stance_peak = float(np.nanmax(stance)) if stance.size else 0.0
    if swing_peak <= 1e-12 or stance_peak <= 1e-12:
        return None
    swing_norm = swing / swing_peak
    stance_norm = stance / stance_peak
    overlap = _integral(np.minimum(swing_norm, stance_norm), t_ms)
    union = _integral(np.maximum(swing_norm, stance_norm), t_ms)
    if union <= 1e-12:
        return None
    return float(overlap / union)


def _action_expectation_reference() -> pd.DataFrame:
    rows = []
    for action, spec in sorted(ACTION_EXPECTATIONS.items()):
        rows.append(
            {
                "action": action,
                "expected_phase": spec.expected_phase,
                "confidence": spec.confidence,
                "rationale": spec.rationale,
            }
        )
    return pd.DataFrame(rows)


def _active_mapping_rows(spikes: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    spike_summary = (
        spikes.groupby("neuron_id")["spike_time_ms"]
        .agg(spike_count="count", first_spike_ms="min", last_spike_ms="max")
        .reset_index()
    )
    active = mapping.merge(spike_summary, left_on="mn_id", right_on="neuron_id", how="inner").copy()
    active["leg_id"] = active.apply(_leg_id, axis=1)
    active["action"] = active.get("action", "").fillna("").astype(str)
    active["expected_phase"] = active["action"].map(lambda a: ACTION_EXPECTATIONS.get(a, ActionExpectation("unknown", "low", "Unrecognized action.")).expected_phase)
    active["phase_confidence"] = active["action"].map(lambda a: ACTION_EXPECTATIONS.get(a, ActionExpectation("unknown", "low", "Unrecognized action.")).confidence)
    active["phase_rationale"] = active["action"].map(lambda a: ACTION_EXPECTATIONS.get(a, ActionExpectation("unknown", "low", "Unrecognized action.")).rationale)
    return active


def build_expected_phase_channels(
    spikes: pd.DataFrame,
    mapping: pd.DataFrame,
    t_ms: np.ndarray,
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    active = _active_mapping_rows(spikes, mapping)
    spikes_by_id = (
        spikes.groupby("neuron_id")["spike_time_ms"].apply(lambda s: s.to_numpy(dtype=float)).to_dict()
    )

    channels: Dict[str, np.ndarray] = {}
    for _, row in active.iterrows():
        leg_id = str(row.get("leg_id", ""))
        phase = str(row.get("expected_phase", ""))
        if leg_id == "" or phase not in CORE_PHASES:
            continue
        spikes_ms = spikes_by_id.get(int(row["mn_id"]))
        if spikes_ms is None or spikes_ms.size == 0:
            continue
        amp = abs(float(row.get("gain", 1.0))) * abs(float(row.get("weight", 1.0)))
        kernel = _spikes_to_activation(
            t_ms=t_ms,
            spikes_ms=spikes_ms,
            tau_rise_ms=tau_rise_ms,
            tau_decay_ms=tau_decay_ms,
        )
        key = f"{leg_id}__{phase}"
        if key not in channels:
            channels[key] = np.zeros_like(t_ms, dtype=float)
        channels[key] += amp * kernel

    phase_df = pd.DataFrame({"t_ms": t_ms})
    for leg_id in LEG_ORDER:
        for phase in ("swing", "stance"):
            key = f"{leg_id}__{phase}"
            phase_df[key] = channels.get(key, np.zeros_like(t_ms, dtype=float))
    return phase_df, active


def summarize_leg_phase_channels(phase_df: pd.DataFrame, active_rows: pd.DataFrame) -> pd.DataFrame:
    t_ms = phase_df["t_ms"].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    grouped_rows = active_rows.groupby(["leg_id", "expected_phase"])

    for col in phase_df.columns:
        if col == "t_ms":
            continue
        leg_id, phase = col.split("__", 1)
        sig = phase_df[col].to_numpy(dtype=float)
        stats = _channel_stats(t_ms, sig)
        phase_rows = grouped_rows.get_group((leg_id, phase)) if (leg_id, phase) in grouped_rows.groups else pd.DataFrame()
        rows.append(
            {
                "leg_id": leg_id,
                "phase": phase,
                "peak": stats["peak"],
                "area": stats["area"],
                "onset_ms": stats["onset_ms"],
                "peak_ms": stats["peak_ms"],
                "mapping_rows": int(len(phase_rows)),
                "unique_mns": int(phase_rows["mn_id"].nunique()) if not phase_rows.empty else 0,
                "total_spikes": int(phase_rows["spike_count"].sum()) if not phase_rows.empty else 0,
                "actions": "|".join(sorted(phase_rows["action"].astype(str).unique())) if not phase_rows.empty else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_tripod_channels(phase_df: pd.DataFrame, leg_phase_summary: pd.DataFrame) -> pd.DataFrame:
    t_ms = phase_df["t_ms"].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []

    for group_name, legs in TRIPOD_GROUPS.items():
        for phase in ("swing", "stance"):
            sig = np.zeros_like(t_ms, dtype=float)
            onset_candidates: List[float] = []
            for leg_id in legs:
                col = f"{leg_id}__{phase}"
                if col in phase_df.columns:
                    sig += phase_df[col].to_numpy(dtype=float)
                leg_row = leg_phase_summary[
                    (leg_phase_summary["leg_id"] == leg_id) & (leg_phase_summary["phase"] == phase)
                ]
                if not leg_row.empty and pd.notna(leg_row.iloc[0]["onset_ms"]):
                    onset_candidates.append(float(leg_row.iloc[0]["onset_ms"]))
            stats = _channel_stats(t_ms, sig)
            rows.append(
                {
                    "group": group_name,
                    "phase": phase,
                    "peak": stats["peak"],
                    "area": stats["area"],
                    "onset_ms": stats["onset_ms"],
                    "peak_ms": stats["peak_ms"],
                    "member_legs": "|".join(legs),
                    "member_onset_spread_ms": (
                        float(max(onset_candidates) - min(onset_candidates)) if len(onset_candidates) >= 2 else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_leg_coactivation(phase_df: pd.DataFrame, leg_phase_summary: pd.DataFrame) -> pd.DataFrame:
    t_ms = phase_df["t_ms"].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    leg_lookup = leg_phase_summary.set_index(["leg_id", "phase"])

    for leg_id in LEG_ORDER:
        swing = phase_df[f"{leg_id}__swing"].to_numpy(dtype=float)
        stance = phase_df[f"{leg_id}__stance"].to_numpy(dtype=float)
        swing_onset = None
        stance_onset = None
        if (leg_id, "swing") in leg_lookup.index:
            swing_onset = leg_lookup.loc[(leg_id, "swing"), "onset_ms"]
        if (leg_id, "stance") in leg_lookup.index:
            stance_onset = leg_lookup.loc[(leg_id, "stance"), "onset_ms"]
        rows.append(
            {
                "leg_id": leg_id,
                "coactivation_ratio": _coactivation_ratio(t_ms, swing, stance),
                "swing_onset_ms": None if pd.isna(swing_onset) else float(swing_onset),
                "stance_onset_ms": None if pd.isna(stance_onset) else float(stance_onset),
                "stance_minus_swing_onset_ms": (
                    None
                    if pd.isna(swing_onset) or pd.isna(stance_onset)
                    else float(stance_onset) - float(swing_onset)
                ),
            }
        )
    return pd.DataFrame(rows)


def _active_phase_rows(active_rows: pd.DataFrame) -> pd.DataFrame:
    return active_rows[active_rows["expected_phase"].isin(CORE_PHASES)].copy()


def build_report(
    run_dir: Path,
    mapping_csv: Path,
    phase_df: pd.DataFrame,
    active_rows: pd.DataFrame,
    leg_phase_summary: pd.DataFrame,
    tripod_summary: pd.DataFrame,
    coactivation_summary: pd.DataFrame,
) -> Dict[str, object]:
    t_ms = phase_df["t_ms"].to_numpy(dtype=float)
    active_phase_rows = _active_phase_rows(active_rows)
    swing_summary = leg_phase_summary[leg_phase_summary["phase"] == "swing"].copy()
    swing_onsets = [float(x) for x in swing_summary["onset_ms"].dropna().tolist()]

    dominant_leg = (
        leg_phase_summary.sort_values(["area", "peak"], ascending=[False, False]).head(1).to_dict("records")
    )
    dominant_ratio = None
    positive_areas = [float(x) for x in leg_phase_summary["area"].tolist() if float(x) > 1e-12]
    if positive_areas:
        max_area = max(positive_areas)
        median_area = float(np.median(positive_areas))
        if median_area > 1e-12:
            dominant_ratio = float(max_area / median_area)

    cross_phase = (
        active_phase_rows.groupby("mn_id")["expected_phase"].nunique().rename("phase_count").reset_index()
    )
    cross_phase_ids = cross_phase[cross_phase["phase_count"] > 1]["mn_id"].astype(int).tolist()
    cross_phase_rows = active_phase_rows[active_phase_rows["mn_id"].isin(cross_phase_ids)].copy()
    cross_phase_types = (
        cross_phase_rows.groupby("mn_type")["mn_id"].nunique().sort_values(ascending=False).head(10).to_dict()
        if "mn_type" in cross_phase_rows.columns and not cross_phase_rows.empty
        else {}
    )

    warnings: List[str] = []
    global_swing_spread_ms = None
    if len(swing_onsets) >= 2:
        global_swing_spread_ms = float(max(swing_onsets) - min(swing_onsets))
        if global_swing_spread_ms <= 5.0:
            warnings.append(
                f"All leg swing channels onset within {global_swing_spread_ms:.2f} ms, which is more synchronous than a clean stepping sequence."
            )

    if dominant_ratio is not None and dominant_ratio >= 3.0 and dominant_leg:
        top = dominant_leg[0]
        warnings.append(
            f"{top['leg_id']} {top['phase']} activity dominates the median leg-phase channel by {dominant_ratio:.2f}x."
        )

    mean_coactivation = None
    valid_coact = [float(x) for x in coactivation_summary["coactivation_ratio"].dropna().tolist()]
    if valid_coact:
        mean_coactivation = float(np.mean(valid_coact))
        if mean_coactivation >= 0.4:
            warnings.append(
                f"Mean within-leg swing/stance overlap is {mean_coactivation:.2f}, suggesting substantial antagonistic co-activation."
            )

    if cross_phase_ids:
        warnings.append(
            f"{len(cross_phase_ids)} active motor neurons are mapped into both swing and stance categories, so mapping ambiguity is contributing contradictory drive."
        )

    tripod_lookup = tripod_summary.set_index(["group", "phase"])
    swing_gap = None
    if ("tripod_A", "swing") in tripod_lookup.index and ("tripod_B", "swing") in tripod_lookup.index:
        a_onset = tripod_lookup.loc[("tripod_A", "swing"), "onset_ms"]
        b_onset = tripod_lookup.loc[("tripod_B", "swing"), "onset_ms"]
        if pd.notna(a_onset) and pd.notna(b_onset):
            swing_gap = float(abs(float(a_onset) - float(b_onset)))
            if swing_gap <= 2.0:
                warnings.append(
                    f"Tripod swing groups onset only {swing_gap:.2f} ms apart, so the run does not show strong A/B alternation."
                )

    diagnosis = {
        "activity_pattern_likely_non_gait": bool(
            (global_swing_spread_ms is not None and global_swing_spread_ms <= 5.0)
            or (dominant_ratio is not None and dominant_ratio >= 3.0)
            or (mean_coactivation is not None and mean_coactivation >= 0.4)
        ),
        "mapping_ambiguity_likely_contributor": bool(cross_phase_ids),
        "mujoco_settings_still_need_review": True,
        "note": (
            "This audit scores the spike/mapping pattern before MuJoCo remapping. A bad score points upstream; "
            "a good score would shift suspicion toward bridge or MuJoCo settings."
        ),
    }

    return {
        "run_dir": str(run_dir),
        "mapping_csv": str(mapping_csv),
        "time_window_ms": [float(t_ms[0]), float(t_ms[-1])],
        "active_mapping_rows": int(len(active_rows)),
        "active_phase_rows": int(len(active_phase_rows)),
        "active_unique_mns": int(active_rows["mn_id"].nunique()),
        "active_phase_unique_mns": int(active_phase_rows["mn_id"].nunique()) if not active_phase_rows.empty else 0,
        "dominant_leg_phase_channel": dominant_leg[0] if dominant_leg else {},
        "dominant_area_ratio_vs_median": dominant_ratio,
        "global_swing_onset_spread_ms": global_swing_spread_ms,
        "tripod_swing_onset_gap_ms": swing_gap,
        "mean_leg_coactivation_ratio": mean_coactivation,
        "cross_phase_neuron_count": int(len(cross_phase_ids)),
        "cross_phase_mn_type_counts": cross_phase_types,
        "warnings": warnings,
        "diagnosis": diagnosis,
    }


def run_gait_expectation_audit(
    run_dir: Path | str,
    mapping_csv: Path | str,
    out_dir: Path | str,
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
) -> Dict[str, object]:
    run_dir = Path(run_dir).expanduser().resolve()
    mapping_csv = Path(mapping_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spikes = load_phase2_spike_times(run_dir)
    mapping = load_mapping_csv(mapping_csv)
    t_ms = load_phase2_timebase_ms(run_dir)

    phase_df, active_rows = build_expected_phase_channels(
        spikes=spikes,
        mapping=mapping,
        t_ms=t_ms,
        tau_rise_ms=tau_rise_ms,
        tau_decay_ms=tau_decay_ms,
    )
    leg_phase_summary = summarize_leg_phase_channels(phase_df, active_rows)
    tripod_summary = summarize_tripod_channels(phase_df, leg_phase_summary)
    coactivation_summary = summarize_leg_coactivation(phase_df, leg_phase_summary)
    report = build_report(
        run_dir=run_dir,
        mapping_csv=mapping_csv,
        phase_df=phase_df,
        active_rows=active_rows,
        leg_phase_summary=leg_phase_summary,
        tripod_summary=tripod_summary,
        coactivation_summary=coactivation_summary,
    )

    _action_expectation_reference().to_csv(out_dir / "action_expectation_reference.csv", index=False)
    active_rows.to_csv(out_dir / "active_motor_expectation_rows.csv", index=False)
    phase_df.to_csv(out_dir / "leg_phase_channels.csv", index=False)
    leg_phase_summary.to_csv(out_dir / "leg_phase_summary.csv", index=False)
    tripod_summary.to_csv(out_dir / "tripod_phase_summary.csv", index=False)
    coactivation_summary.to_csv(out_dir / "leg_coactivation_summary.csv", index=False)
    (out_dir / "gait_expectation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
