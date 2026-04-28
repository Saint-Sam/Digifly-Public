from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

try:
    from .expected_gait import (
        _focus_ids_for_mode,
        build_expected_gait_controls,
        build_tripod_phase_channels,
    )
    from .gait_audit import ACTION_EXPECTATIONS, CORE_PHASES, _channel_stats, _leg_id
    from .pipeline import _spikes_to_activation, load_mapping_csv, load_phase2_spike_times
except ImportError:  # pragma: no cover - allows direct script imports via sys.path injection
    from expected_gait import _focus_ids_for_mode, build_expected_gait_controls, build_tripod_phase_channels
    from gait_audit import ACTION_EXPECTATIONS, CORE_PHASES, _channel_stats, _leg_id
    from pipeline import _spikes_to_activation, load_mapping_csv, load_phase2_spike_times


def _focused_mapping(mapping: pd.DataFrame, focus_ids: Iterable[int]) -> pd.DataFrame:
    focus_set = {int(x) for x in focus_ids}
    return mapping[mapping["mn_id"].isin(list(focus_set))].copy()


def build_expected_mn_phase_prior(
    mapping: pd.DataFrame,
    focus_ids: Iterable[int],
    phase_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mp = _focused_mapping(mapping, focus_ids)
    t_ms = phase_df["t_ms"].to_numpy(dtype=float)
    if mp.empty:
        return pd.DataFrame({"t_ms": t_ms}), pd.DataFrame()

    mp["leg_id"] = mp.apply(_leg_id, axis=1)
    mp["action"] = mp.get("action", "").fillna("").astype(str)
    mp["expected_phase"] = mp["action"].map(
        lambda a: ACTION_EXPECTATIONS.get(a, ACTION_EXPECTATIONS.get("abdomen_drive")).expected_phase
        if a in ACTION_EXPECTATIONS
        else "unknown"
    )
    mp["base_weight"] = mp["gain"].abs() * mp["weight"].abs()
    mp = mp[mp["leg_id"].ne("") & mp["expected_phase"].isin(CORE_PHASES)].copy()

    prior_cols: Dict[str, np.ndarray] = {}
    summary_rows = []
    if mp.empty:
        return pd.DataFrame({"t_ms": t_ms}), pd.DataFrame()

    for mn_id, rows in mp.groupby("mn_id", sort=True):
        signal = np.zeros_like(t_ms, dtype=float)
        base_total = 0.0
        actuator_targets = []
        actions = []
        phases = []
        leg_ids = []
        bases = []
        for _, row in rows.iterrows():
            leg_id = str(row["leg_id"])
            phase = str(row["expected_phase"])
            channel_name = f"{leg_id}__{phase}"
            if channel_name not in phase_df.columns:
                continue
            base = float(row["base_weight"])
            signal += base * phase_df[channel_name].to_numpy(dtype=float)
            base_total += base
            actuator_targets.append(str(row.get("actuator_name", "")))
            actions.append(str(row.get("action", "")))
            phases.append(phase)
            leg_ids.append(leg_id)
            bases.append(base)
        if base_total > 1e-12:
            signal /= base_total

        col_name = f"mn_{int(mn_id)}"
        prior_cols[col_name] = signal
        stats = _channel_stats(t_ms, signal)
        summary_rows.append(
            {
                "mn_id": int(mn_id),
                "column_name": col_name,
                "mapping_rows": int(len(rows)),
                "actuator_targets": "|".join(sorted({x for x in actuator_targets if x})),
                "actions": "|".join(sorted({x for x in actions if x})),
                "expected_phases": "|".join(sorted({x for x in phases if x})),
                "leg_ids": "|".join(sorted({x for x in leg_ids if x})),
                "weighted_mapping_strength": float(base_total),
                "weighted_mapping_strength_mean": float(base_total / max(1, len(bases))),
                "prior_peak": float(stats["peak"]),
                "prior_area": float(stats["area"]),
                "prior_onset_ms": stats["onset_ms"],
                "prior_peak_ms": stats["peak_ms"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["prior_area", "weighted_mapping_strength", "mn_id"],
        ascending=[False, False, True],
    )
    prior_df = pd.DataFrame({"t_ms": t_ms, **prior_cols})
    return prior_df, summary_df.reset_index(drop=True)


def build_mn_actuator_matrix(
    mapping: pd.DataFrame,
    focus_ids: Iterable[int],
) -> tuple[list[str], list[int], np.ndarray, pd.DataFrame]:
    mp = _focused_mapping(mapping, focus_ids)
    if mp.empty:
        return [], [], np.zeros((0, 0), dtype=float), pd.DataFrame()

    mp["signed_strength"] = mp["sign"] * mp["gain"] * mp["weight"]
    grouped = (
        mp.groupby(["actuator_name", "mn_id"], as_index=False)
        .agg(
            signed_strength=("signed_strength", "sum"),
            mapping_rows=("mn_id", "size"),
            actions=("action", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
        )
        .sort_values(["actuator_name", "mn_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if "notes" in mp.columns:
        note_df = (
            mp.groupby(["actuator_name", "mn_id"], as_index=False)["notes"]
            .agg(lambda s: "|".join(sorted({str(x) for x in s if str(x)})))
            .rename(columns={"notes": "notes"})
        )
        grouped = grouped.merge(note_df, on=["actuator_name", "mn_id"], how="left")
    else:
        grouped["notes"] = ""

    actuator_names = sorted(grouped["actuator_name"].astype(str).unique())
    mn_ids = sorted(grouped["mn_id"].astype(int).unique())
    actuator_index = {name: idx for idx, name in enumerate(actuator_names)}
    mn_index = {mn_id: idx for idx, mn_id in enumerate(mn_ids)}
    matrix = np.zeros((len(actuator_names), len(mn_ids)), dtype=float)
    for _, row in grouped.iterrows():
        i = actuator_index[str(row["actuator_name"])]
        j = mn_index[int(row["mn_id"])]
        matrix[i, j] = float(row["signed_strength"])
    return actuator_names, mn_ids, matrix, grouped


def _controls_to_matrix(controls: pd.DataFrame, actuator_names: list[str]) -> np.ndarray:
    if "t_ms" not in controls.columns:
        raise ValueError("controls DataFrame must include t_ms.")
    u = np.zeros((len(actuator_names), len(controls)), dtype=float)
    for idx, actuator_name in enumerate(actuator_names):
        if actuator_name in controls.columns:
            u[idx, :] = controls[actuator_name].to_numpy(dtype=float)
    return u


def solve_inverse_mn_drive(
    expected_controls: pd.DataFrame,
    prior_df: pd.DataFrame,
    actuator_names: list[str],
    mn_ids: list[int],
    matrix: np.ndarray,
    ridge_alpha: float = 0.25,
    nonnegative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | bool]]:
    if len(actuator_names) == 0 or len(mn_ids) == 0:
        empty = pd.DataFrame({"t_ms": expected_controls["t_ms"].to_numpy(dtype=float)})
        report = {
            "mn_count": 0,
            "actuator_count": 0,
            "ridge_alpha": float(ridge_alpha),
            "nonnegative": bool(nonnegative),
        }
        return empty, empty, report

    prior_cols = [f"mn_{mn_id}" for mn_id in mn_ids]
    for col in prior_cols:
        if col not in prior_df.columns:
            prior_df[col] = 0.0

    u = _controls_to_matrix(expected_controls, actuator_names)
    p = prior_df[prior_cols].to_numpy(dtype=float).T
    lhs = matrix.T @ matrix + float(ridge_alpha) * np.eye(len(mn_ids), dtype=float)
    rhs = (matrix.T @ u) + float(ridge_alpha) * p
    drive = np.linalg.solve(lhs, rhs)
    if nonnegative:
        drive = np.maximum(drive, 0.0)
    fit = matrix @ drive

    t_ms = expected_controls["t_ms"].to_numpy(dtype=float)
    drive_df = pd.DataFrame(
        {
            "t_ms": t_ms,
            **{f"mn_{mn_id}": drive[idx, :] for idx, mn_id in enumerate(mn_ids)},
        }
    )

    fit_df = pd.DataFrame(
        {
            "t_ms": t_ms,
            **{actuator_name: fit[idx, :] for idx, actuator_name in enumerate(actuator_names)},
        }
    )

    denom = float(np.linalg.norm(u))
    residual = fit - u
    report = {
        "mn_count": int(len(mn_ids)),
        "actuator_count": int(len(actuator_names)),
        "ridge_alpha": float(ridge_alpha),
        "nonnegative": bool(nonnegative),
        "relative_rmse": float(np.sqrt(np.mean(residual**2)) / max(denom / np.sqrt(u.size), 1e-12)),
        "control_cosine_similarity": float(
            float(np.sum(fit * u)) / max(float(np.linalg.norm(fit) * np.linalg.norm(u)), 1e-12)
        ),
        "control_variance_explained": float(
            1.0 - (float(np.sum(residual**2)) / max(float(np.sum((u - np.mean(u)) ** 2)), 1e-12))
        ),
        "target_control_peak": float(np.max(np.abs(u))) if u.size else 0.0,
        "fit_control_peak": float(np.max(np.abs(fit))) if fit.size else 0.0,
    }
    return drive_df, fit_df, report


def summarize_mn_drive(
    drive_df: pd.DataFrame,
    prior_summary: pd.DataFrame,
) -> pd.DataFrame:
    if drive_df.shape[1] <= 1:
        return pd.DataFrame()
    t_ms = drive_df["t_ms"].to_numpy(dtype=float)
    prior_lookup = (
        prior_summary.set_index("mn_id").to_dict(orient="index") if not prior_summary.empty else {}
    )
    rows = []
    for col in drive_df.columns:
        if col == "t_ms" or not col.startswith("mn_"):
            continue
        mn_id = int(col.split("_", 1)[1])
        y = drive_df[col].to_numpy(dtype=float)
        stats = _channel_stats(t_ms, y)
        prior_row = prior_lookup.get(mn_id, {})
        rows.append(
            {
                "mn_id": mn_id,
                "column_name": col,
                "target_peak": float(stats["peak"]),
                "target_area": float(stats["area"]),
                "target_onset_ms": stats["onset_ms"],
                "target_peak_ms": stats["peak_ms"],
                "actions": prior_row.get("actions", ""),
                "expected_phases": prior_row.get("expected_phases", ""),
                "leg_ids": prior_row.get("leg_ids", ""),
                "actuator_targets": prior_row.get("actuator_targets", ""),
                "mapping_rows": prior_row.get("mapping_rows", 0),
                "weighted_mapping_strength": prior_row.get("weighted_mapping_strength", 0.0),
                "prior_area": prior_row.get("prior_area", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_area", "target_peak", "mn_id"], ascending=[False, False, True]).reset_index(drop=True)


def compare_actual_to_target_drive(
    run_dir: Path | str,
    drive_df: pd.DataFrame,
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    run_dir = Path(run_dir).expanduser().resolve()
    spikes = load_phase2_spike_times(run_dir)
    spikes_by_id = (
        spikes.groupby("neuron_id")["spike_time_ms"].apply(lambda s: s.to_numpy(dtype=float)).to_dict()
    )
    t_ms = drive_df["t_ms"].to_numpy(dtype=float)
    rows = []
    corrs = []
    area_ratios = []
    for col in drive_df.columns:
        if col == "t_ms" or not col.startswith("mn_"):
            continue
        mn_id = int(col.split("_", 1)[1])
        target = drive_df[col].to_numpy(dtype=float)
        spikes_ms = spikes_by_id.get(mn_id, np.array([], dtype=float))
        actual = _spikes_to_activation(
            t_ms=t_ms,
            spikes_ms=spikes_ms,
            tau_rise_ms=tau_rise_ms,
            tau_decay_ms=tau_decay_ms,
        )
        target_stats = _channel_stats(t_ms, target)
        actual_stats = _channel_stats(t_ms, actual)
        corr = None
        if float(np.max(np.abs(target))) > 1e-12 and float(np.max(np.abs(actual))) > 1e-12:
            corr = float(np.corrcoef(target, actual)[0, 1])
            if np.isfinite(corr):
                corrs.append(corr)
        target_area = float(target_stats["area"])
        actual_area = float(actual_stats["area"])
        area_ratio = None if target_area <= 1e-12 else float(actual_area / target_area)
        if area_ratio is not None and np.isfinite(area_ratio):
            area_ratios.append(area_ratio)
        rows.append(
            {
                "mn_id": mn_id,
                "target_peak": float(target_stats["peak"]),
                "target_area": target_area,
                "target_onset_ms": target_stats["onset_ms"],
                "actual_peak": float(actual_stats["peak"]),
                "actual_area": actual_area,
                "actual_onset_ms": actual_stats["onset_ms"],
                "spike_count": int(len(spikes_ms)),
                "target_vs_actual_corr": corr,
                "actual_to_target_area_ratio": area_ratio,
                "onset_delta_ms": (
                    None
                    if target_stats["onset_ms"] is None or actual_stats["onset_ms"] is None
                    else float(actual_stats["onset_ms"]) - float(target_stats["onset_ms"])
                ),
            }
        )
    compare_df = pd.DataFrame(rows).sort_values(
        ["target_area", "actual_area", "mn_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    report = {
        "run_dir": str(run_dir),
        "mn_count": int(len(compare_df)),
        "mean_target_vs_actual_corr": float(np.mean(corrs)) if corrs else None,
        "median_actual_to_target_area_ratio": float(np.median(area_ratios)) if area_ratios else None,
        "target_nonzero_but_silent_count": int(
            len(compare_df[(compare_df["target_area"] > 1e-6) & (compare_df["spike_count"] == 0)])
        ),
        "actual_nonzero_but_target_low_count": int(
            len(compare_df[(compare_df["actual_area"] > 1e-6) & (compare_df["target_area"] <= 1e-6)])
        ),
    }
    return compare_df, report


def derive_expected_mn_drive(
    mapping_csv: Path | str,
    out_dir: Path | str,
    *,
    run_dir: Path | str | None = None,
    added_motor_ids_csv: Path | str | None = None,
    focus_mode: str = "active",
    duration_ms: float = 18000.0,
    dt_ms: float = 5.0,
    stride_period_ms: float = 1200.0,
    swing_fraction: float = 0.32,
    segment_offsets_ms: Dict[str, float] | None = None,
    ridge_alpha: float = 0.25,
    nonnegative: bool = True,
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
) -> Dict[str, object]:
    mapping_csv = Path(mapping_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = None if run_dir is None else Path(run_dir).expanduser().resolve()
    added_motor_ids_csv = None if added_motor_ids_csv is None else Path(added_motor_ids_csv).expanduser().resolve()

    mapping = load_mapping_csv(mapping_csv)
    focus_ids, focus_stats = _focus_ids_for_mode(mapping, run_dir, added_motor_ids_csv, focus_mode)
    phase_df = build_tripod_phase_channels(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        stride_period_ms=stride_period_ms,
        swing_fraction=swing_fraction,
        segment_offsets_ms=segment_offsets_ms,
    )
    expected_controls, _, _ = build_expected_gait_controls(
        mapping=mapping,
        focus_ids=focus_ids,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        stride_period_ms=stride_period_ms,
        swing_fraction=swing_fraction,
        segment_offsets_ms=segment_offsets_ms,
    )
    prior_df, prior_summary = build_expected_mn_phase_prior(mapping, focus_ids, phase_df)
    actuator_names, mn_ids, matrix, matrix_detail = build_mn_actuator_matrix(mapping, focus_ids)
    drive_df, fit_df, fit_report = solve_inverse_mn_drive(
        expected_controls=expected_controls,
        prior_df=prior_df,
        actuator_names=actuator_names,
        mn_ids=mn_ids,
        matrix=matrix,
        ridge_alpha=ridge_alpha,
        nonnegative=nonnegative,
    )
    drive_summary = summarize_mn_drive(drive_df, prior_summary)

    compare_df = pd.DataFrame()
    compare_report = None
    if run_dir is not None and run_dir.exists() and drive_df.shape[1] > 1:
        compare_df, compare_report = compare_actual_to_target_drive(
            run_dir=run_dir,
            drive_df=drive_df,
            tau_rise_ms=tau_rise_ms,
            tau_decay_ms=tau_decay_ms,
        )

    expected_controls_path = out_dir / f"expected_target_controls_{focus_mode}.csv"
    fitted_controls_path = out_dir / f"expected_target_controls_fit_{focus_mode}.csv"
    prior_path = out_dir / f"expected_target_mn_prior_{focus_mode}.csv"
    drive_path = out_dir / f"expected_target_mn_drive_{focus_mode}.csv"
    drive_summary_path = out_dir / f"expected_target_mn_summary_{focus_mode}.csv"
    matrix_detail_path = out_dir / f"expected_target_mn_matrix_{focus_mode}.csv"
    compare_path = out_dir / f"expected_target_mn_compare_{focus_mode}.csv"
    report_path = out_dir / f"expected_target_mn_report_{focus_mode}.json"

    expected_controls.to_csv(expected_controls_path, index=False)
    fit_df.to_csv(fitted_controls_path, index=False)
    prior_df.to_csv(prior_path, index=False)
    drive_df.to_csv(drive_path, index=False)
    drive_summary.to_csv(drive_summary_path, index=False)
    matrix_detail.to_csv(matrix_detail_path, index=False)
    if not compare_df.empty:
        compare_df.to_csv(compare_path, index=False)

    report = {
        "mapping_csv": str(mapping_csv),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "added_motor_ids_csv": str(added_motor_ids_csv) if added_motor_ids_csv is not None else None,
        "focus_mode": str(focus_mode),
        "focus_stats": focus_stats,
        "duration_ms": float(duration_ms),
        "dt_ms": float(dt_ms),
        "stride_period_ms": float(stride_period_ms),
        "swing_fraction": float(swing_fraction),
        "segment_offsets_ms": dict(segment_offsets_ms or {}),
        "ridge_alpha": float(ridge_alpha),
        "nonnegative": bool(nonnegative),
        "fit_report": fit_report,
        "compare_report": compare_report,
        "paths": {
            "expected_controls_csv": str(expected_controls_path),
            "fitted_controls_csv": str(fitted_controls_path),
            "prior_csv": str(prior_path),
            "drive_csv": str(drive_path),
            "drive_summary_csv": str(drive_summary_path),
            "matrix_detail_csv": str(matrix_detail_path),
            "compare_csv": str(compare_path) if not compare_df.empty else None,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["paths"]["report_json"] = str(report_path)
    return report
