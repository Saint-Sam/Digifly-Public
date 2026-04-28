from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import imageio.v2 as imageio
import numpy as np


def _as_array_dict(act_controls: Dict[str, Iterable[float]]) -> Dict[str, np.ndarray]:
    return {str(k): np.asarray(v, dtype=float) for k, v in act_controls.items()}


def loop_signals_simple(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    total_ms: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2:
        return t_ms.copy(), act

    dt = float(np.median(np.diff(t_ms)))
    t0 = float(t_ms[0])
    base_dur = float(t_ms[-1] - t0)
    if base_dur <= 0:
        return t_ms.copy(), act

    total_ms = float(total_ms)
    n_loops = max(1, int(np.ceil(total_ms / base_dur)))
    rel = t_ms - t0
    tiled_t = np.concatenate([rel + k * base_dur for k in range(n_loops)]) + t0
    keep = tiled_t <= (t0 + total_ms + 1e-9)
    out_t = tiled_t[keep]

    out = {}
    for name, sig in act.items():
        if sig.size == 0:
            out[name] = np.zeros_like(out_t, dtype=float)
            continue
        s = np.tile(sig, n_loops)
        out[name] = s[: out_t.size]
    return out_t, out


def stretch_signal_timebase(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    total_ms: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2:
        return t_ms.copy(), act

    t0 = float(t_ms[0])
    base_dur = float(t_ms[-1] - t0)
    if base_dur <= 0:
        return t_ms.copy(), act

    total_ms = float(total_ms)
    if total_ms <= 0:
        return t_ms.copy(), act

    scale = total_ms / base_dur
    out_t = t0 + (t_ms - t0) * scale
    out = {name: sig.copy() for name, sig in act.items()}
    return out_t, out


def prepend_quiet_segment(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    quiet_ms: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2 or quiet_ms <= 0:
        return t_ms.copy(), act

    dt = float(np.median(np.diff(t_ms)))
    n_quiet = int(round(float(quiet_ms) / max(dt, 1e-9)))
    if n_quiet <= 0:
        return t_ms.copy(), act

    t0 = float(t_ms[0])
    out_t = t0 + np.arange(t_ms.size + n_quiet, dtype=float) * dt
    out = {}
    for name, sig in act.items():
        s = np.concatenate([np.zeros(n_quiet, dtype=float), sig])
        out[name] = s[: out_t.size]
    return out_t, out


def append_quiet_segment(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    quiet_ms: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2 or quiet_ms <= 0:
        return t_ms.copy(), act

    dt = float(np.median(np.diff(t_ms)))
    n_quiet = int(round(float(quiet_ms) / max(dt, 1e-9)))
    if n_quiet <= 0:
        return t_ms.copy(), act

    start = float(t_ms[-1]) + dt
    tail_t = start + (np.arange(n_quiet, dtype=float) * dt)
    out_t = np.concatenate([t_ms, tail_t])
    out = {}
    for name, sig in act.items():
        out[name] = np.concatenate([sig, np.zeros(n_quiet, dtype=float)])
    return out_t, out


def ramp_in_controls(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    ramp_ms: float,
) -> Dict[str, np.ndarray]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2 or ramp_ms <= 0:
        return act

    ramp = np.clip((t_ms - t_ms[0]) / float(ramp_ms), 0.0, 1.0)
    out = {}
    for name, sig in act.items():
        out[name] = sig * ramp
    return out


def phase_shift_right_legs(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    delay_ms: float,
) -> Dict[str, np.ndarray]:
    t_ms = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    if t_ms.size < 2 or delay_ms <= 0:
        return act

    dt = float(np.median(np.diff(t_ms)))
    shift = max(1, int(round(float(delay_ms) / max(dt, 1e-9))))
    out = {}
    for name, sig in act.items():
        key = name.lower()
        if "right" in key and ("coxa" in key or "femur" in key or "tibia" in key):
            s = np.concatenate([np.zeros(shift, dtype=float), sig])[: sig.size]
            out[name] = s
        else:
            out[name] = sig.copy()
    return out


def scale_leg_segments(
    act_controls: Dict[str, Iterable[float]],
    front_scale: float = 1.0,
    mid_scale: float = 1.0,
    hind_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    act = _as_array_dict(act_controls)

    def _scale(name: str) -> float:
        k = name.lower()
        if "t1" in k:
            return float(front_scale)
        if "t2" in k:
            return float(mid_scale)
        if "t3" in k:
            return float(hind_scale)
        return 1.0

    out = {}
    for name, sig in act.items():
        out[name] = sig * _scale(name)
    return out


def _normalized_alpha_pulse(
    t_ms: np.ndarray,
    onset_ms: float,
    rise_ms: float,
    decay_ms: float,
) -> np.ndarray:
    t = np.asarray(t_ms, dtype=float)
    rise_ms = max(1e-6, float(rise_ms))
    decay_ms = max(1e-6, float(decay_ms))
    if rise_ms > decay_ms:
        rise_ms, decay_ms = decay_ms, rise_ms

    out = np.zeros_like(t, dtype=float)
    dt = t - float(onset_ms)
    mask = dt >= 0.0
    if not np.any(mask):
        return out

    t_peak = (rise_ms * decay_ms / (decay_ms - rise_ms)) * math.log(decay_ms / rise_ms)
    peak = math.exp(-t_peak / decay_ms) - math.exp(-t_peak / rise_ms)
    if peak <= 0.0:
        peak = 1.0

    kernel = np.exp(-dt[mask] / decay_ms) - np.exp(-dt[mask] / rise_ms)
    out[mask] = kernel / peak
    return out


def _first_threshold_onset_ms(
    t_ms: np.ndarray,
    sig: np.ndarray,
    threshold_frac: float,
) -> float | None:
    arr = np.asarray(sig, dtype=float)
    peak = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(peak) or peak <= 1e-12:
        return None
    threshold = max(0.0, float(threshold_frac)) * peak
    hit = np.flatnonzero(np.abs(arr) >= threshold)
    if hit.size == 0:
        return None
    return float(np.asarray(t_ms, dtype=float)[int(hit[0])])


def shape_middle_leg_jump_impulse(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    rise_ms: float = 1.0,
    decay_ms: float = 8.0,
    threshold_frac: float = 0.08,
    amplitude_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    act = _as_array_dict(act_controls)
    out = {name: sig.copy() for name, sig in act.items()}
    t = np.asarray(t_ms, dtype=float)

    for side in ("left", "right"):
        names = [f"coxa_T2_{side}", f"femur_T2_{side}", f"tibia_T2_{side}"]
        names = [name for name in names if name in act]
        if not names:
            continue

        onsets = []
        for name in names:
            onset_ms = _first_threshold_onset_ms(t, act[name], threshold_frac=threshold_frac)
            if onset_ms is not None:
                onsets.append(onset_ms)
        if not onsets:
            continue

        pulse = _normalized_alpha_pulse(
            t_ms=t,
            onset_ms=min(onsets),
            rise_ms=rise_ms,
            decay_ms=decay_ms,
        )
        for name in names:
            sig = np.asarray(act[name], dtype=float)
            if sig.size == 0:
                continue
            peak_idx = int(np.nanargmax(np.abs(sig)))
            signed_peak = float(sig[peak_idx])
            out[name] = float(amplitude_scale) * signed_peak * pulse
    return out


def synthesize_wingbeat_carrier(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    frequency_hz: float = 180.0,
    onset_threshold_frac: float = 0.08,
    envelope_power: float = 0.65,
    amplitude_scale: float = 1.0,
    left_phase_deg: float = 0.0,
    right_phase_deg: float = 0.0,
    harmonic_mix: float = 0.15,
) -> Dict[str, np.ndarray]:
    act = _as_array_dict(act_controls)
    out = {name: sig.copy() for name, sig in act.items()}
    t = np.asarray(t_ms, dtype=float)
    freq_hz = max(1e-6, float(frequency_hz))
    harmonic_mix = min(max(float(harmonic_mix), 0.0), 1.0)

    for side, phase_deg in (("left", left_phase_deg), ("right", right_phase_deg)):
        name = f"wing_pitch_{side}"
        if name not in act:
            continue

        sig = np.asarray(act[name], dtype=float)
        peak = float(np.nanmax(np.abs(sig))) if sig.size else 0.0
        if not np.isfinite(peak) or peak <= 1e-12:
            out[name] = np.zeros_like(sig, dtype=float)
            continue

        onset_ms = _first_threshold_onset_ms(t, sig, threshold_frac=onset_threshold_frac)
        if onset_ms is None:
            out[name] = np.zeros_like(sig, dtype=float)
            continue

        env = np.clip(sig / peak, 0.0, None)
        power = max(1e-6, float(envelope_power))
        env = np.power(env, power)
        env[np.abs(sig) < (float(onset_threshold_frac) * peak)] = 0.0

        phase_rad = math.radians(float(phase_deg))
        rel_t_s = (t - onset_ms) / 1000.0
        fundamental = np.sin((2.0 * math.pi * freq_hz * rel_t_s) + phase_rad)
        harmonic = np.sin((4.0 * math.pi * freq_hz * rel_t_s) + phase_rad + (0.5 * math.pi))
        carrier = ((1.0 - harmonic_mix) * fundamental) + (harmonic_mix * harmonic)
        out[name] = float(amplitude_scale) * peak * env * carrier

    return out


def zero_selected_actuators(
    act_controls: Dict[str, Iterable[float]],
    names: Iterable[str],
) -> Dict[str, np.ndarray]:
    act = _as_array_dict(act_controls)
    names_lc = {str(n).strip().lower() for n in names}
    out = {}
    for name, sig in act.items():
        if name.lower() in names_lc:
            out[name] = np.zeros_like(sig)
        else:
            out[name] = sig.copy()
    return out


def apply_output_gains(
    act_controls: Dict[str, Iterable[float]],
    actuator_ranges: Dict[str, Tuple[float, float]],
    gains: Dict[str, float],
    clip: bool = True,
) -> Dict[str, np.ndarray]:
    act = _as_array_dict(act_controls)
    gain_lc = {str(k).strip().lower(): float(v) for k, v in dict(gains or {}).items() if str(k).strip()}
    out: Dict[str, np.ndarray] = {}
    for name, sig in act.items():
        gain = gain_lc.get(str(name).strip().lower(), 1.0)
        arr = np.asarray(sig, dtype=float) * float(gain)
        if clip:
            lo_hi = actuator_ranges.get(name)
            if lo_hi is not None:
                lo, hi = float(lo_hi[0]), float(lo_hi[1])
                arr = np.clip(arr, lo, hi)
        out[name] = arr
    return out


def apply_actuator_strength_boosts(
    model,
    model_names: Iterable[str],
    actuator_boosts: Iterable[Dict[str, Any]] | None,
) -> Dict[str, float]:
    boosts = list(actuator_boosts or [])
    if not boosts:
        return {"boost_groups": 0.0, "boosted_actuators": 0.0}

    name_to_id = {str(name): idx for idx, name in enumerate(model_names) if str(name)}
    boosted_ids: set[int] = set()

    for spec in boosts:
        if not isinstance(spec, dict):
            continue
        targets = spec.get("targets", []) or []
        gear_mult = max(0.0, float(spec.get("gear_mult", 1.0)))
        forcerange_mult = max(0.0, float(spec.get("forcerange_mult", 1.0)))

        resolved_targets = []
        for raw_name in targets:
            name = _canonicalize_name(str(raw_name), model_names)
            if name:
                resolved_targets.append(name)

        for name in resolved_targets:
            aid = name_to_id.get(name)
            if aid is None:
                continue
            boosted_ids.add(int(aid))

            if gear_mult not in (0.0, 1.0) and getattr(model, "actuator_gear", None) is not None and model.actuator_gear.shape[1] >= 1:
                model.actuator_gear[aid, 0] *= float(gear_mult)

            if (
                forcerange_mult not in (0.0, 1.0)
                and getattr(model, "actuator_forcerange", None) is not None
                and model.actuator_forcerange.shape[1] >= 2
            ):
                lo = float(model.actuator_forcerange[aid, 0])
                hi = float(model.actuator_forcerange[aid, 1])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    mid = 0.5 * (lo + hi)
                    span = (hi - lo) * float(forcerange_mult)
                    model.actuator_forcerange[aid, 0] = mid - (0.5 * span)
                    model.actuator_forcerange[aid, 1] = mid + (0.5 * span)

    return {
        "boost_groups": float(len(boosts)),
        "boosted_actuators": float(len(boosted_ids)),
    }


def _canonicalize_name(name: str, model_names: Iterable[str]) -> str | None:
    n = str(name).strip()
    if not n:
        return None
    model = [str(x) for x in model_names if str(x)]
    by_lc = {m.lower(): m for m in model}

    # 1) exact case-insensitive
    hit = by_lc.get(n.lower())
    if hit:
        return hit

    # 2) suffix match for prefixed names like walker/coxa_T2_left
    suffix_hits = [m for m in model if m.lower().endswith("/" + n.lower())]
    if len(suffix_hits) == 1:
        return suffix_hits[0]

    # 3) reverse suffix
    rev_hits = [m for m in model if n.lower().endswith("/" + m.lower())]
    if len(rev_hits) == 1:
        return rev_hits[0]

    return None


def canonicalize_actuator_controls_for_model(
    act_controls: Dict[str, Iterable[float]],
    model_names: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], list[str]]:
    act = _as_array_dict(act_controls)
    mapped: Dict[str, np.ndarray] = {}
    rename: Dict[str, str] = {}
    dropped: list[str] = []
    for old_name, sig in act.items():
        new_name = _canonicalize_name(old_name, model_names)
        if new_name is None:
            dropped.append(old_name)
            continue
        rename[old_name] = new_name
        if new_name in mapped:
            mapped[new_name] = mapped[new_name] + sig
        else:
            mapped[new_name] = sig.copy()
    return mapped, rename, dropped


def list_model_cameras(mjcf_xml: Path | str) -> list[dict[str, Any]]:
    import mujoco

    mjcf_xml = Path(mjcf_xml).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(mjcf_xml))
    cameras: list[dict[str, Any]] = []
    for i in range(int(model.ncam)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"
        cameras.append(
            {
                "id": int(i),
                "name": str(name),
                "pos": [float(x) for x in model.cam_pos[i].tolist()],
                "fovy_deg": float(model.cam_fovy[i]),
            }
        )
    return cameras


def remap_to_ctrlrange_auto(
    act_controls: Dict[str, Iterable[float]],
    actuator_ranges: Dict[str, Tuple[float, float]],
    target_fraction: float = 0.7,
    percentile: float = 95.0,
    bias_to_mid: bool = True,
    clip: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    act = _as_array_dict(act_controls)
    out: Dict[str, np.ndarray] = {}
    report: Dict[str, Dict[str, float]] = {}
    tgt = float(target_fraction)
    pct = float(percentile)

    for name, sig in act.items():
        lo_hi = actuator_ranges.get(name)
        if lo_hi is None:
            continue
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            out[name] = sig.copy()
            report[name] = {"scale_used": 1.0, "input_pctl": float(np.nanpercentile(np.abs(sig), pct))}
            continue

        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        abs_sig = np.abs(sig[np.isfinite(sig)])
        pctl = float(np.nanpercentile(abs_sig, pct)) if abs_sig.size else 0.0
        if pctl < 1e-9:
            norm = np.zeros_like(sig)
            scale_used = 0.0
        else:
            norm = sig / pctl
            scale_used = (tgt * half) / pctl

        if bias_to_mid:
            u = mid + (tgt * half) * norm
        else:
            u = (tgt * half) * norm
        if clip:
            u = np.clip(u, lo, hi)
        out[name] = u
        report[name] = {"scale_used": float(scale_used), "input_pctl": float(pctl), "lo": lo, "hi": hi}
    return out, report


def _apply_absolute_camera_pose(
    model,
    mujoco_module,
    *,
    camera_name: str,
    camera_distance_factor: float,
    camera_fovy_deg: float,
) -> int:
    cam_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_CAMERA, str(camera_name))
    if cam_id >= 0:
        center = np.array(model.stat.center, dtype=float)
        vec = model.cam_pos[cam_id].copy() - center
        if np.linalg.norm(vec) < 1e-9:
            vec = np.array([-1.0, 0.0, 1.0], dtype=float)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        dist = float(model.stat.extent) * float(camera_distance_factor)
        model.cam_pos[cam_id] = center + vec * dist
        model.cam_fovy[cam_id] = float(np.clip(camera_fovy_deg, 10.0, 150.0))
    return int(cam_id)


def _prepare_scaled_controls(
    model,
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    *,
    target_ctrl_fraction: float,
    percentile: float,
    bias_to_mid: bool,
    clip: bool,
    output_gains: Dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, str], list[str], dict[str, dict[str, float]]]:
    import mujoco

    model_names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        for i in range(model.nu)
    ]
    act, rename, dropped = canonicalize_actuator_controls_for_model(act_controls, model_names)

    ranges = {
        name: tuple(model.actuator_ctrlrange[i])
        for i, name in enumerate(model_names)
        if name
    }
    scaled, scale_report = remap_to_ctrlrange_auto(
        act,
        ranges,
        target_fraction=float(target_ctrl_fraction),
        percentile=float(percentile),
        bias_to_mid=bool(bias_to_mid),
        clip=bool(clip),
    )
    if output_gains:
        scaled = apply_output_gains(scaled, ranges, output_gains, clip=bool(clip))

    t = np.asarray(t_ms, dtype=float)
    dt_ms = 1000.0 * float(model.opt.timestep)
    if t.size < 2:
        raise ValueError("t_ms must contain at least 2 samples for rendering.")
    steps = int(math.ceil((float(t[-1]) - float(t[0])) / dt_ms)) + 1
    sim_t = float(t[0]) + np.arange(steps, dtype=float) * dt_ms

    name_to_id = {name: i for i, name in enumerate(model_names) if name}
    U = np.zeros((model.nu, steps), dtype=float)
    for name, sig in scaled.items():
        aid = name_to_id.get(name)
        if aid is None:
            continue
        U[aid, :] = np.interp(sim_t, t, sig)

    return t, sim_t, U, rename, dropped, scale_report


def render_camera_previews_mujoco(
    mjcf_xml: Path | str,
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    camera_names: Iterable[str] | None = None,
    preview_time_ms: float | None = None,
    camera_distance_factor: float = 8.0,
    camera_fovy_deg: float = 75.0,
    width: int = 640,
    height: int = 360,
    target_ctrl_fraction: float = 0.7,
    percentile: float = 95.0,
    bias_to_mid: bool = True,
    clip: bool = True,
    output_gains: Dict[str, float] | None = None,
    actuator_boosts: Iterable[Dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    import mujoco

    mjcf_xml = Path(mjcf_xml).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(mjcf_xml))
    data = mujoco.MjData(model)
    model_names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        for i in range(model.nu)
    ]
    apply_actuator_strength_boosts(model, model_names, actuator_boosts)

    t, sim_t, U, _, _, _ = _prepare_scaled_controls(
        model,
        t_ms,
        act_controls,
        target_ctrl_fraction=float(target_ctrl_fraction),
        percentile=float(percentile),
        bias_to_mid=bool(bias_to_mid),
        clip=bool(clip),
        output_gains=output_gains,
    )

    if preview_time_ms is None:
        preview_idx = 0
    else:
        preview_idx = int(np.argmin(np.abs(sim_t - float(preview_time_ms))))
        preview_idx = max(0, min(preview_idx, U.shape[1] - 1))

    for k in range(preview_idx + 1):
        data.ctrl[:] = U[:, k]
        mujoco.mj_step(model, data)

    catalog = list_model_cameras(mjcf_xml)
    all_names = [entry["name"] for entry in catalog]
    target_names = list(camera_names) if camera_names is not None else all_names

    frames: dict[str, np.ndarray] = {}
    renderer = None
    try:
        try:
            renderer = mujoco.Renderer(model, height=int(height), width=int(width))
        except Exception as e:  # pragma: no cover - platform/display dependent
            raise RuntimeError(
                "MuJoCo renderer could not initialize. This usually means the current session has no "
                "graphics context (headless/remote). Run from a local GUI session or adjust MUJOCO_GL."
            ) from e

        for camera_name in target_names:
            cam_id = _apply_absolute_camera_pose(
                model,
                mujoco,
                camera_name=str(camera_name),
                camera_distance_factor=float(camera_distance_factor),
                camera_fovy_deg=float(camera_fovy_deg),
            )
            renderer.update_scene(data, camera=(cam_id if cam_id >= 0 else None))
            frames[str(camera_name)] = renderer.render().copy()
    finally:
        if renderer is not None:
            renderer.close()
    return frames


def apply_profile_transforms(
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    profile: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.asarray(t_ms, dtype=float)
    act = _as_array_dict(act_controls)
    p = dict(profile or {})

    # 1) loop window
    loop_total_ms = p.get("loop_total_ms")
    if loop_total_ms is not None:
        t, act = loop_signals_simple(t, act, total_ms=float(loop_total_ms))

    # 2) stretch biological time without repeating the signal.
    stretch_total_ms = p.get("stretch_total_ms")
    if stretch_total_ms is not None:
        t, act = stretch_signal_timebase(t, act, total_ms=float(stretch_total_ms))

    # 3) settle prepend
    settle_ms = float(p.get("settle_ms", 0.0))
    if settle_ms > 0:
        t, act = prepend_quiet_segment(t, act, quiet_ms=settle_ms)

    # 4) leg phase shift
    phase_ms = float(p.get("right_leg_phase_ms", 0.0))
    if phase_ms > 0:
        act = phase_shift_right_legs(t, act, delay_ms=phase_ms)

    # 5) segment scaling
    act = scale_leg_segments(
        act,
        front_scale=float(p.get("front_scale", 1.0)),
        mid_scale=float(p.get("mid_scale", 1.0)),
        hind_scale=float(p.get("hind_scale", 1.0)),
    )

    # 6) glia-specific jump shaping
    jump_cfg = dict(p.get("jump_pulse", {}) or {})
    if bool(jump_cfg.get("enabled", False)):
        act = shape_middle_leg_jump_impulse(
            t_ms=t,
            act_controls=act,
            rise_ms=float(jump_cfg.get("rise_ms", 1.0)),
            decay_ms=float(jump_cfg.get("decay_ms", 8.0)),
            threshold_frac=float(jump_cfg.get("threshold_frac", 0.08)),
            amplitude_scale=float(jump_cfg.get("amplitude_scale", 1.0)),
        )

    # 7) DLM wingbeat carrier
    wingbeat_cfg = dict(p.get("wingbeat", {}) or {})
    if bool(wingbeat_cfg.get("enabled", False)):
        act = synthesize_wingbeat_carrier(
            t_ms=t,
            act_controls=act,
            frequency_hz=float(wingbeat_cfg.get("frequency_hz", 180.0)),
            onset_threshold_frac=float(wingbeat_cfg.get("onset_threshold_frac", 0.08)),
            envelope_power=float(wingbeat_cfg.get("envelope_power", 0.65)),
            amplitude_scale=float(wingbeat_cfg.get("amplitude_scale", 1.0)),
            left_phase_deg=float(wingbeat_cfg.get("left_phase_deg", 0.0)),
            right_phase_deg=float(wingbeat_cfg.get("right_phase_deg", 0.0)),
            harmonic_mix=float(wingbeat_cfg.get("harmonic_mix", 0.15)),
        )

    # 8) zero selected
    zero_list = p.get("zero_actuators", []) or []
    if zero_list:
        act = zero_selected_actuators(act, zero_list)

    # 9) post-run quiet tail so the body can visibly settle after a transient jump.
    tail_ms = float(p.get("tail_ms", 0.0))
    if tail_ms > 0:
        t, act = append_quiet_segment(t, act, quiet_ms=tail_ms)

    # 10) ramp-in at end
    ramp_ms = float(p.get("ramp_ms", 0.0))
    if ramp_ms > 0:
        act = ramp_in_controls(t, act, ramp_ms=ramp_ms)

    return t, act


def render_controls_mujoco(
    mjcf_xml: Path | str,
    t_ms: np.ndarray,
    act_controls: Dict[str, Iterable[float]],
    out_video: Path | str,
    camera_name: str = "track2",
    camera_distance_factor: float = 8.0,
    camera_fovy_deg: float = 75.0,
    render_hz: int = 120,
    slowmo: float = 20.0,
    width: int = 1280,
    height: int = 720,
    target_ctrl_fraction: float = 0.7,
    percentile: float = 95.0,
    bias_to_mid: bool = True,
    clip: bool = True,
    floor_friction_override: Iterable[float] | None = None,
    output_gains: Dict[str, float] | None = None,
    actuator_boosts: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    import mujoco

    mjcf_xml = Path(mjcf_xml).expanduser().resolve()
    out_video = Path(out_video).expanduser().resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(mjcf_xml))
    data = mujoco.MjData(model)
    floor_override_report: Dict[str, float] = {}

    if floor_friction_override is not None:
        floor_friction = np.asarray(list(floor_friction_override), dtype=float).reshape(-1)
        if floor_friction.size != 3:
            raise ValueError(
                "floor_friction_override must contain exactly 3 values: "
                "(sliding, torsional, rolling)."
            )
        floor_friction = np.maximum(floor_friction, 0.0)
        updated_floor_geoms = []
        for geom_id in range(model.ngeom):
            geom_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            )
            geom_name_l = geom_name.strip().lower()
            if "floor" in geom_name_l or "ground" in geom_name_l:
                model.geom_friction[geom_id, :] = floor_friction
                updated_floor_geoms.append(str(geom_name))
        floor_override_report = {
            "floor_friction_override_applied": float(bool(updated_floor_geoms)),
            "floor_friction_override_geom_count": float(len(updated_floor_geoms)),
            "floor_friction_slide": float(floor_friction[0]),
            "floor_friction_torsion": float(floor_friction[1]),
            "floor_friction_roll": float(floor_friction[2]),
        }

    # Build actuator catalog and canonicalize incoming names.
    model_names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        for i in range(model.nu)
    ]
    boost_report = apply_actuator_strength_boosts(model, model_names, actuator_boosts)
    act, rename, dropped = canonicalize_actuator_controls_for_model(act_controls, model_names)

    ranges = {
        name: tuple(model.actuator_ctrlrange[i])
        for i, name in enumerate(model_names)
        if name
    }
    scaled, scale_report = remap_to_ctrlrange_auto(
        act,
        ranges,
        target_fraction=float(target_ctrl_fraction),
        percentile=float(percentile),
        bias_to_mid=bool(bias_to_mid),
        clip=bool(clip),
    )
    if output_gains:
        scaled = apply_output_gains(scaled, ranges, output_gains, clip=bool(clip))

    # Camera placement (absolute, non-compounding)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera_name))
    if cam_id >= 0:
        center = np.array(model.stat.center, dtype=float)
        vec = model.cam_pos[cam_id].copy() - center
        if np.linalg.norm(vec) < 1e-9:
            vec = np.array([-1.0, 0.0, 1.0], dtype=float)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        dist = float(model.stat.extent) * float(camera_distance_factor)
        model.cam_pos[cam_id] = center + vec * dist
        model.cam_fovy[cam_id] = float(np.clip(camera_fovy_deg, 10.0, 150.0))

    t = np.asarray(t_ms, dtype=float)
    dt_ms = 1000.0 * float(model.opt.timestep)
    if t.size < 2:
        raise ValueError("t_ms must contain at least 2 samples for rendering.")
    steps = int(math.ceil((float(t[-1]) - float(t[0])) / dt_ms)) + 1
    sim_t = float(t[0]) + np.arange(steps, dtype=float) * dt_ms

    name_to_id = {name: i for i, name in enumerate(model_names) if name}
    U = np.zeros((model.nu, steps), dtype=float)
    for name, sig in scaled.items():
        aid = name_to_id.get(name)
        if aid is None:
            continue
        U[aid, :] = np.interp(sim_t, t, sig)

    base_steps_per_frame = max(1, int(round((1000.0 / float(render_hz)) / dt_ms)))
    steps_per_frame = max(1, int(round(base_steps_per_frame / max(float(slowmo), 1.0))))
    out_fps = int(render_hz)

    writer = imageio.get_writer(str(out_video), fps=out_fps)
    frames = 0
    renderer = None
    try:
        try:
            renderer = mujoco.Renderer(model, height=int(height), width=int(width))
        except Exception as e:  # pragma: no cover - platform/display dependent
            raise RuntimeError(
                "MuJoCo renderer could not initialize. This usually means the current session has no "
                "graphics context (headless/remote). Run from a local GUI session or adjust MUJOCO_GL."
            ) from e

        for k in range(steps):
            data.ctrl[:] = U[:, k]
            mujoco.mj_step(model, data)
            if k % steps_per_frame == 0:
                renderer.update_scene(data, camera=(cam_id if cam_id >= 0 else None))
                frame = renderer.render()
                writer.append_data(frame)
                if frames == 0:
                    imageio.imwrite(str(out_video.with_suffix(".preview.png")), frame)
                frames += 1
    finally:
        writer.close()
        if renderer is not None:
            renderer.close()

    sim_window_ms = float(t[-1] - t[0])
    video_duration_s = float(frames) / max(out_fps, 1)
    return {
        "frames": float(frames),
        "video_fps": float(out_fps),
        "video_duration_s": float(video_duration_s),
        "sim_window_ms": float(sim_window_ms),
        "mapped_actuators": float(len(rename)),
        "dropped_actuators": float(len(dropped)),
        "scaled_actuators": float(len(scale_report)),
        "target_ctrl_fraction": float(target_ctrl_fraction),
        "percentile": float(percentile),
        "bias_to_mid": float(bool(bias_to_mid)),
        "clip": float(bool(clip)),
        **floor_override_report,
        "output_gain_count": float(len(dict(output_gains or {}))),
        "actuator_boost_groups": float(boost_report.get("boost_groups", 0.0)),
        "boosted_actuators": float(boost_report.get("boosted_actuators", 0.0)),
    }
