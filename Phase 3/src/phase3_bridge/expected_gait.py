from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from .gait_audit import ACTION_EXPECTATIONS, CORE_PHASES, LEG_ORDER, TRIPOD_GROUPS, _leg_id
    from .hemilineage import load_added_motor_neuron_ids
    from .pipeline import load_mapping_csv, load_phase2_spike_times, plot_actuator_controls, save_controls_csv
    from .video_pipeline import apply_profile_transforms, render_controls_mujoco
except ImportError:  # pragma: no cover - allows direct script imports via sys.path injection
    from gait_audit import ACTION_EXPECTATIONS, CORE_PHASES, LEG_ORDER, TRIPOD_GROUPS, _leg_id
    from hemilineage import load_added_motor_neuron_ids
    from pipeline import load_mapping_csv, load_phase2_spike_times, plot_actuator_controls, save_controls_csv
    from video_pipeline import apply_profile_transforms, render_controls_mujoco


DEFAULT_SEGMENT_OFFSETS_MS: Dict[str, float] = {
    "T1": 0.0,
    "T2": 70.0,
    "T3": 140.0,
}


def _pulse_train(
    t_ms: np.ndarray,
    period_ms: float,
    start_ms: float,
    duration_ms: float,
    edge_fraction: float = 0.18,
) -> np.ndarray:
    if period_ms <= 0 or duration_ms <= 0:
        return np.zeros_like(t_ms, dtype=float)

    t = np.asarray(t_ms, dtype=float)
    rel = np.mod(t - float(start_ms), float(period_ms))
    pulse = np.zeros_like(t, dtype=float)
    active = rel < float(duration_ms)
    if not np.any(active):
        return pulse

    pulse[active] = 1.0
    edge_ms = min(float(duration_ms) * float(edge_fraction), 0.5 * float(duration_ms))
    if edge_ms > 0:
        rise = active & (rel < edge_ms)
        if np.any(rise):
            pulse[rise] = 0.5 * (1.0 - np.cos(np.pi * rel[rise] / edge_ms))
        fall = active & (rel > (float(duration_ms) - edge_ms))
        if np.any(fall):
            tail = float(duration_ms) - rel[fall]
            pulse[fall] = 0.5 * (1.0 - np.cos(np.pi * tail / edge_ms))
    return pulse


def build_tripod_phase_channels(
    duration_ms: float = 18000.0,
    dt_ms: float = 5.0,
    stride_period_ms: float = 1200.0,
    swing_fraction: float = 0.32,
    segment_offsets_ms: Dict[str, float] | None = None,
    edge_fraction: float = 0.18,
) -> pd.DataFrame:
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive.")
    if not (0.05 <= swing_fraction <= 0.7):
        raise ValueError("swing_fraction should stay in a sensible range.")

    t_ms = np.arange(0.0, float(duration_ms) + 0.5 * float(dt_ms), float(dt_ms), dtype=float)
    swing_duration_ms = float(stride_period_ms) * float(swing_fraction)
    stance_duration_ms = float(stride_period_ms) - swing_duration_ms
    offsets = dict(DEFAULT_SEGMENT_OFFSETS_MS)
    if segment_offsets_ms:
        offsets.update({str(k): float(v) for k, v in segment_offsets_ms.items()})

    group_lookup = {leg_id: group_name for group_name, legs in TRIPOD_GROUPS.items() for leg_id in legs}
    phase_df = pd.DataFrame({"t_ms": t_ms})

    for leg_id in LEG_ORDER:
        thorax = leg_id.split("_", 1)[0]
        group_name = group_lookup.get(leg_id)
        if group_name is None:
            continue
        group_shift = 0.0 if group_name == "tripod_A" else 0.5 * float(stride_period_ms)
        start_ms = group_shift + float(offsets.get(thorax, 0.0))
        swing = _pulse_train(
            t_ms=t_ms,
            period_ms=float(stride_period_ms),
            start_ms=start_ms,
            duration_ms=swing_duration_ms,
            edge_fraction=edge_fraction,
        )
        stance = _pulse_train(
            t_ms=t_ms,
            period_ms=float(stride_period_ms),
            start_ms=start_ms + swing_duration_ms,
            duration_ms=stance_duration_ms,
            edge_fraction=edge_fraction,
        )
        phase_df[f"{leg_id}__swing"] = swing
        phase_df[f"{leg_id}__stance"] = stance
    return phase_df


def _focus_ids_for_mode(
    mapping: pd.DataFrame,
    run_dir: Path | None,
    added_motor_ids_csv: Path | None,
    focus_mode: str,
) -> tuple[set[int], Dict[str, int | str]]:
    focus_mode = str(focus_mode).strip().lower()
    mapped_ids = set(mapping["mn_id"].astype(int).unique())

    if focus_mode == "all":
        return mapped_ids, {"focus_mode": "all", "focus_neuron_count": int(len(mapped_ids))}

    if focus_mode == "added":
        if added_motor_ids_csv is None or not Path(added_motor_ids_csv).exists():
            raise FileNotFoundError("focus_mode='added' requires added_motor_ids_csv.")
        added_ids = set(load_added_motor_neuron_ids(added_motor_ids_csv).astype(int).tolist())
        return mapped_ids.intersection(added_ids), {"focus_mode": "added", "focus_neuron_count": int(len(mapped_ids.intersection(added_ids)))}

    if focus_mode == "active":
        if run_dir is None:
            raise FileNotFoundError("focus_mode='active' requires run_dir.")
        spikes = load_phase2_spike_times(run_dir)
        active_ids = set(spikes["neuron_id"].astype(int).unique())
        return mapped_ids.intersection(active_ids), {"focus_mode": "active", "focus_neuron_count": int(len(mapped_ids.intersection(active_ids))), "focus_spike_row_count": int(len(spikes))}

    raise ValueError(f"Unsupported focus_mode: {focus_mode}")


def summarize_expected_gait_weights(
    mapping: pd.DataFrame,
    focus_ids: Iterable[int],
) -> pd.DataFrame:
    focus_set = {int(x) for x in focus_ids}
    mp = mapping[mapping["mn_id"].isin(list(focus_set))].copy()
    mp["leg_id"] = mp.apply(_leg_id, axis=1)
    mp["action"] = mp.get("action", "").fillna("").astype(str)
    mp["expected_phase"] = mp["action"].map(
        lambda a: ACTION_EXPECTATIONS.get(a, ACTION_EXPECTATIONS.get("abdomen_drive")).expected_phase
        if a in ACTION_EXPECTATIONS
        else "unknown"
    )
    mp = mp[mp["leg_id"].ne("") & mp["expected_phase"].isin(CORE_PHASES)].copy()
    if mp.empty:
        return pd.DataFrame(
            columns=[
                "leg_id",
                "expected_phase",
                "actuator_name",
                "base_weight",
                "signed_weight_mean",
                "normalized_weight",
                "mapping_rows",
                "unique_mns",
                "actions",
            ]
        )

    mp["base_weight"] = mp["gain"].abs() * mp["weight"].abs()
    mp["signed_weight"] = mp["sign"] * mp["base_weight"]

    grouped = (
        mp.groupby(["leg_id", "expected_phase", "actuator_name"], as_index=False)
        .agg(
            base_weight=("base_weight", "sum"),
            signed_weight=("signed_weight", "sum"),
            mapping_rows=("mn_id", "size"),
            unique_mns=("mn_id", "nunique"),
            actions=("action", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
        )
        .sort_values(["leg_id", "expected_phase", "base_weight", "actuator_name"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    grouped["signed_weight_mean"] = np.where(
        grouped["base_weight"] > 1e-12,
        grouped["signed_weight"] / grouped["base_weight"],
        0.0,
    )
    phase_totals = grouped.groupby(["leg_id", "expected_phase"])["base_weight"].transform("sum")
    grouped["normalized_weight"] = np.where(phase_totals > 1e-12, grouped["base_weight"] / phase_totals, 0.0)
    return grouped


def build_expected_gait_controls(
    mapping: pd.DataFrame,
    focus_ids: Iterable[int],
    duration_ms: float = 18000.0,
    dt_ms: float = 5.0,
    stride_period_ms: float = 1200.0,
    swing_fraction: float = 0.32,
    segment_offsets_ms: Dict[str, float] | None = None,
    edge_fraction: float = 0.18,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_df = build_tripod_phase_channels(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        stride_period_ms=stride_period_ms,
        swing_fraction=swing_fraction,
        segment_offsets_ms=segment_offsets_ms,
        edge_fraction=edge_fraction,
    )
    weight_df = summarize_expected_gait_weights(mapping, focus_ids)
    controls = pd.DataFrame({"t_ms": phase_df["t_ms"].to_numpy(dtype=float)})
    if weight_df.empty:
        return controls, phase_df, weight_df

    actuator_names = sorted(weight_df["actuator_name"].astype(str).unique())
    for actuator_name in actuator_names:
        controls[actuator_name] = 0.0

    for _, row in weight_df.iterrows():
        leg_id = str(row["leg_id"])
        phase = str(row["expected_phase"])
        channel = phase_df[f"{leg_id}__{phase}"].to_numpy(dtype=float)
        amplitude = float(row["normalized_weight"]) * float(row["signed_weight_mean"])
        controls[str(row["actuator_name"])] += amplitude * channel

    return controls, phase_df, weight_df


def _find_world_xml(phase3_root: Path) -> Path | None:
    candidates = [
        phase3_root / "data" / "inputs" / "flybody" / "floor.xml",
        Path.home() / "Desktop" / "Digifly" / "flybody-main" / "flybody" / "fruitfly" / "assets" / "floor.xml",
    ]
    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path
    return None


def render_expected_gait_video(
    phase3_root: Path | str,
    mapping_csv: Path | str,
    out_dir: Path | str,
    *,
    run_dir: Path | str | None = None,
    added_motor_ids_csv: Path | str | None = None,
    save_tag: str = "Hemi_09A",
    run_name: str = "hemi_09a_baseline",
    profile_doc: Dict[str, object],
    profile_name: str = "expected_gait_compare",
    focus_mode: str = "active",
    duration_ms: float = 18000.0,
    dt_ms: float = 5.0,
    stride_period_ms: float = 1200.0,
    swing_fraction: float = 0.32,
    segment_offsets_ms: Dict[str, float] | None = None,
) -> Dict[str, object]:
    phase3_root = Path(phase3_root).expanduser().resolve()
    mapping_csv = Path(mapping_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = None if run_dir is None else Path(run_dir).expanduser().resolve()
    added_motor_ids_csv = None if added_motor_ids_csv is None else Path(added_motor_ids_csv).expanduser().resolve()

    profiles = dict(profile_doc.get("profiles", {}))
    if profile_name not in profiles:
        raise KeyError(f"Profile not found: {profile_name}. Available: {sorted(profiles)}")
    profile = dict(profiles[profile_name])

    mapping = load_mapping_csv(mapping_csv)
    focus_ids, focus_stats = _focus_ids_for_mode(mapping, run_dir, added_motor_ids_csv, focus_mode)
    base_controls_df, phase_df, weight_df = build_expected_gait_controls(
        mapping=mapping,
        focus_ids=focus_ids,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        stride_period_ms=stride_period_ms,
        swing_fraction=swing_fraction,
        segment_offsets_ms=segment_offsets_ms,
    )

    signal_cfg = dict(profile.get("signal", {}))
    control_map_cfg = dict(profile.get("control_map", {}))
    render_cfg = dict(profile.get("render", {}))
    fig_dir = phase3_root / "data" / "outputs" / "figures" / save_tag / run_name / "expected_gait"
    vid_dir = phase3_root / "data" / "outputs" / "videos" / save_tag / run_name / "expected_gait"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    base_act = {
        column: base_controls_df[column].to_numpy(dtype=float)
        for column in base_controls_df.columns
        if column != "t_ms"
    }
    if base_act:
        t_mod, mod_act = apply_profile_transforms(
            base_controls_df["t_ms"].to_numpy(dtype=float),
            base_act,
            signal_cfg,
        )
    else:
        t_mod = base_controls_df["t_ms"].to_numpy(dtype=float)
        mod_act = {}

    profiled_controls_df = pd.DataFrame({"t_ms": t_mod})
    for key in sorted(mod_act):
        profiled_controls_df[key] = mod_act[key]

    run_tag = f"{save_tag}_{run_name}"
    base_csv = save_controls_csv(base_controls_df, out_dir / f"expected_gait_controls_base_{focus_mode}.csv")
    mod_csv = save_controls_csv(profiled_controls_df, out_dir / f"expected_gait_controls_profiled_{focus_mode}.csv")
    phase_csv = save_controls_csv(phase_df, out_dir / f"expected_gait_phase_channels_{focus_mode}.csv")
    weight_csv = save_controls_csv(weight_df, out_dir / f"expected_gait_weights_{focus_mode}.csv")
    base_fig = None
    mod_fig = None
    if base_act:
        base_fig = plot_actuator_controls(
            base_controls_df,
            fig_dir / f"{run_tag}_{profile_name}_{focus_mode}_base_top12.pdf",
            top_n=12,
        )
        mod_fig = plot_actuator_controls(
            profiled_controls_df,
            fig_dir / f"{run_tag}_{profile_name}_{focus_mode}_profiled_top12.pdf",
            top_n=12,
        )

    world_xml = _find_world_xml(phase3_root)
    render_report = None
    render_report_path = out_dir / f"render_report_expected_gait_{focus_mode}_{profile_name}.json"
    video_out = vid_dir / f"{run_tag}_{profile_name}_{focus_mode}.mp4"
    if world_xml is not None and base_act:
        try:
            render_report = render_controls_mujoco(
                mjcf_xml=world_xml,
                t_ms=profiled_controls_df["t_ms"].to_numpy(dtype=float),
                act_controls={
                    column: profiled_controls_df[column].to_numpy(dtype=float)
                    for column in profiled_controls_df.columns
                    if column != "t_ms"
                },
                out_video=video_out,
                camera_name=str(render_cfg.get("camera_name", "track2")),
                camera_distance_factor=float(render_cfg.get("camera_distance_factor", 8.0)),
                camera_fovy_deg=float(render_cfg.get("camera_fovy_deg", 75.0)),
                render_hz=int(render_cfg.get("render_hz", 60)),
                slowmo=float(render_cfg.get("slowmo", 1.0)),
                width=int(render_cfg.get("width", 1280)),
                height=int(render_cfg.get("height", 720)),
                target_ctrl_fraction=float(control_map_cfg.get("target_ctrl_fraction", 0.25)),
                percentile=float(control_map_cfg.get("percentile", 95.0)),
                bias_to_mid=bool(control_map_cfg.get("bias_to_mid", False)),
                clip=bool(control_map_cfg.get("clip", True)),
            )
            render_report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")
        except Exception as exc:
            render_report = {"error": str(exc)}
            render_report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")

    summary = {
        "save_tag": save_tag,
        "run_name": run_name,
        "focus_mode": focus_mode,
        "profile_name": profile_name,
        "mapping_csv": str(mapping_csv),
        "run_dir": str(run_dir) if run_dir else None,
        "added_motor_ids_csv": str(added_motor_ids_csv) if added_motor_ids_csv else None,
        "world_xml": str(world_xml) if world_xml else None,
        "stride_period_ms": float(stride_period_ms),
        "duration_ms": float(duration_ms),
        "dt_ms": float(dt_ms),
        "swing_fraction": float(swing_fraction),
        "segment_offsets_ms": {str(k): float(v) for k, v in (segment_offsets_ms or DEFAULT_SEGMENT_OFFSETS_MS).items()},
        "signal_profile": signal_cfg,
        "control_map": control_map_cfg,
        "base_controls_csv": str(base_csv),
        "profiled_controls_csv": str(mod_csv),
        "phase_channels_csv": str(phase_csv),
        "weights_csv": str(weight_csv),
        "base_plot": str(base_fig) if base_fig else None,
        "profiled_plot": str(mod_fig) if mod_fig else None,
        "video_out": str(video_out) if render_report is not None else None,
        "render_report": str(render_report_path) if render_report is not None else None,
        "render_stats": render_report,
        "actuator_count": int(max(0, profiled_controls_df.shape[1] - 1)),
        "nonzero_weight_rows": int(len(weight_df)),
        **focus_stats,
    }
    summary_path = out_dir / f"expected_gait_summary_{focus_mode}_{profile_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
