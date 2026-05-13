from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TTMN_IDS = (10068, 10110)


@dataclass(frozen=True)
class BeamParams:
    dt_ms: float = 0.025
    duration_ms: float = 80.0
    stim_time_ms: float = 20.0
    jump_latency_ms: float = 2.0
    resonance_hz: float = 85.0
    damping_ratio: float = 0.18
    horizontal_fraction: float = 0.28
    noise_sd: float = 0.0
    seed: int = 7


def _timebase(duration_ms: float, dt_ms: float) -> np.ndarray:
    return np.arange(0.0, float(duration_ms) + 0.5 * float(dt_ms), float(dt_ms), dtype=float)


def _second_order_impulse(t_ms: np.ndarray, t0_ms: float, resonance_hz: float, damping_ratio: float) -> np.ndarray:
    """Underdamped second-order impulse response, normalized to unit peak."""
    t_s = (t_ms - float(t0_ms)) / 1000.0
    y = np.zeros_like(t_s, dtype=float)
    mask = t_s >= 0.0
    if not np.any(mask):
        return y

    zeta = min(0.98, max(1e-6, float(damping_ratio)))
    omega_n = 2.0 * np.pi * max(1e-6, float(resonance_hz))
    omega_d = omega_n * np.sqrt(max(1e-12, 1.0 - zeta * zeta))
    tt = t_s[mask]
    resp = np.exp(-zeta * omega_n * tt) * np.sin(omega_d * tt)
    peak = float(np.nanmax(np.abs(resp))) if resp.size else 0.0
    if peak > 0.0:
        resp = resp / peak
    y[mask] = resp
    return y


def _smooth_step(t_ms: np.ndarray, start_ms: float, stop_ms: float, edge_ms: float = 20.0) -> np.ndarray:
    edge = max(1e-6, float(edge_ms))
    up = 1.0 / (1.0 + np.exp(-(t_ms - float(start_ms)) / edge))
    down = 1.0 / (1.0 + np.exp((t_ms - float(stop_ms)) / edge))
    return up * down


def _spike_times_from_phase2(run_dir: Path, motor_ids: Iterable[int] = TTMN_IDS) -> list[float]:
    spike_csv = run_dir / "spike_times.csv"
    if not spike_csv.exists():
        return []
    df = pd.read_csv(spike_csv)
    if not {"neuron_id", "spike_time_ms"}.issubset(df.columns):
        return []
    ids = {int(x) for x in motor_ids}
    sub = df[df["neuron_id"].astype(int).isin(ids)].copy()
    if sub.empty:
        return []
    return sorted(float(x) for x in pd.to_numeric(sub["spike_time_ms"], errors="coerce").dropna())


def _jump_trace(
    params: BeamParams,
    amplitude_um: float,
    polarity: float = -1.0,
    phase2_spikes_ms: Iterable[float] | None = None,
    response_probability: float = 1.0,
    unilateral: bool = False,
    requires_motor_spike: bool = False,
    gate_source: str = "preset",
) -> pd.DataFrame:
    t = _timebase(params.duration_ms, params.dt_ms)
    rng = np.random.default_rng(int(params.seed))

    spikes = list(phase2_spikes_ms or [])
    motor_spike_count = len(spikes)
    if requires_motor_spike and not spikes:
        return _standing_trace(
            params,
            gate_source=gate_source,
            phase2_motor_spike_count=motor_spike_count,
        )
    if not spikes:
        spikes = [params.stim_time_ms + params.jump_latency_ms]

    vertical = np.zeros_like(t, dtype=float)
    horizontal = np.zeros_like(t, dtype=float)
    successes = 0

    event_amp = float(amplitude_um) / max(1, len(spikes))
    for i, spike_t in enumerate(spikes):
        if rng.random() > float(response_probability):
            continue
        successes += 1
        amp = event_amp
        if unilateral:
            amp *= 0.5
        jitter = rng.normal(0.0, 0.6)
        resp = _second_order_impulse(
            t,
            float(spike_t) + jitter,
            resonance_hz=params.resonance_hz,
            damping_ratio=params.damping_ratio,
        )
        vertical += float(polarity) * amp * resp
        horizontal += amp * params.horizontal_fraction * rng.choice([-1.0, 1.0]) * resp

    if params.noise_sd > 0.0:
        vertical += rng.normal(0.0, params.noise_sd, size=t.shape)
        horizontal += rng.normal(0.0, params.noise_sd, size=t.shape)

    jump_decision = "jump" if successes > 0 else "no_jump"
    return _waveform_frame(
        t,
        vertical,
        horizontal,
        successes=successes,
        jump_decision=jump_decision,
        gate_source=gate_source,
        phase2_motor_spike_count=motor_spike_count,
    )


def _standing_trace(
    params: BeamParams,
    gate_source: str = "standing_still",
    phase2_motor_spike_count: int = 0,
) -> pd.DataFrame:
    t = _timebase(params.duration_ms, params.dt_ms)
    rng = np.random.default_rng(int(params.seed))
    vertical = np.zeros_like(t, dtype=float)
    horizontal = np.zeros_like(t, dtype=float)
    if params.noise_sd > 0.0:
        vertical += rng.normal(0.0, params.noise_sd, size=t.shape)
        horizontal += rng.normal(0.0, params.noise_sd, size=t.shape)
    return _waveform_frame(
        t,
        vertical,
        horizontal,
        successes=0,
        jump_decision="no_jump",
        gate_source=gate_source,
        phase2_motor_spike_count=int(phase2_motor_spike_count),
    )


def _walking_trace(params: BeamParams, force_uN: float = 100.0) -> pd.DataFrame:
    duration = max(params.duration_ms, 1200.0)
    t = _timebase(duration, params.dt_ms)
    rng = np.random.default_rng(int(params.seed))
    base = np.zeros_like(t)
    freqs = (2.8, 4.3, 7.1)
    for f in freqs:
        base += np.sin(2.0 * np.pi * f * t / 1000.0 + rng.uniform(-np.pi, np.pi))
    base /= max(1e-9, np.nanmax(np.abs(base)))
    horizontal = 0.5 * float(force_uN) * base
    vertical = 0.35 * float(force_uN) * np.sin(2.0 * np.pi * 3.2 * t / 1000.0 + 0.7)

    for release_ms in (420.0, 760.0):
        vertical += 18.0 * _second_order_impulse(t, release_ms, resonance_hz=18.0, damping_ratio=0.16)
        horizontal -= 12.0 * _second_order_impulse(t, release_ms, resonance_hz=16.0, damping_ratio=0.18)

    return _waveform_frame(t, vertical, horizontal)


def _adhesion_trace(params: BeamParams) -> pd.DataFrame:
    duration = max(params.duration_ms, 600.0)
    t = _timebase(duration, params.dt_ms)
    vertical = np.zeros_like(t)
    ramp = np.clip((t - 90.0) / 330.0, 0.0, 1.0)
    vertical -= 120.0 * ramp
    for release_ms, drop in ((420.0, 55.0), (470.0, 45.0)):
        vertical += drop * (t >= release_ms)
        vertical += 18.0 * _second_order_impulse(t, release_ms, resonance_hz=45.0, damping_ratio=0.11)
    horizontal = 8.0 * np.sin(2.0 * np.pi * 2.0 * t / 1000.0)
    return _waveform_frame(t, vertical, horizontal)


def _flight_trace(params: BeamParams) -> pd.DataFrame:
    duration = max(params.duration_ms, 1200.0)
    t = _timebase(duration, params.dt_ms)
    flight = _smooth_step(t, 250.0, 920.0, edge_ms=22.0)
    vertical = -5.0 * flight
    vertical += 3.0 * flight * np.sin(2.0 * np.pi * 14.95 * t / 1000.0)
    horizontal = 2.2 * flight * np.sin(2.0 * np.pi * 10.88 * t / 1000.0 + 0.6)
    vertical += 15.0 * _second_order_impulse(t, 960.0, resonance_hz=18.0, damping_ratio=0.18)
    return _waveform_frame(t, vertical, horizontal)


def _larval_trace(params: BeamParams, contractions_per_min: float, force_uN: float = 420.0) -> pd.DataFrame:
    duration = max(params.duration_ms, 20000.0)
    dt = max(params.dt_ms, 2.0)
    t = _timebase(duration, dt)
    period_ms = 60000.0 / max(1e-6, float(contractions_per_min))
    vertical = np.zeros_like(t)
    for start in np.arange(500.0, duration, period_ms):
        phase = (t - start) / period_ms
        mask = (phase >= 0.0) & (phase <= 1.0)
        pulse = np.zeros_like(t)
        pulse[mask] = np.sin(np.pi * phase[mask]) ** 2
        vertical += float(force_uN) * pulse
    horizontal = np.zeros_like(t)
    return _waveform_frame(t, vertical, horizontal)


def _waveform_frame(t_ms: np.ndarray, vertical: np.ndarray, horizontal: np.ndarray, **meta: object) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "t_ms": t_ms,
            "vertical": vertical,
            "horizontal": horizontal,
        }
    )
    df["vector"] = np.sqrt(np.square(df["vertical"]) + np.square(df["horizontal"]))
    for key, value in meta.items():
        df.attrs[key] = value
    return df


def generate_condition(condition: str, params: BeamParams, phase2_run: Path | None = None) -> pd.DataFrame:
    condition = str(condition).strip().lower()
    phase2_spikes = _spike_times_from_phase2(phase2_run) if phase2_run else []

    if condition in {"standing_still", "fly_stands_still", "still", "no_jump"}:
        return _standing_trace(params, gate_source="standing_still", phase2_motor_spike_count=len(phase2_spikes))
    if condition in {"phase2_gated_jump", "simulation_gated_jump", "digifly_gated_jump", "phase2_escape_jump"}:
        return _jump_trace(
            params,
            amplitude_um=290.0,
            phase2_spikes_ms=phase2_spikes,
            requires_motor_spike=True,
            gate_source="phase2_ttmn_spike",
        )
    if condition in {"phase2_gated_shakb2", "phase2_gated_shak-b2", "simulation_gated_shakb2", "phase2_shakb2_jump", "phase2_shak-b2_jump"}:
        return _jump_trace(
            params,
            amplitude_um=290.0,
            polarity=1.0,
            phase2_spikes_ms=phase2_spikes,
            response_probability=1.0,
            unilateral=True,
            requires_motor_spike=True,
            gate_source="phase2_ttmn_spike_shakb2",
        )
    if condition in {"wildtype_jump", "wildtype_one_leg", "cs_jump"}:
        return _jump_trace(params, amplitude_um=290.0, phase2_spikes_ms=phase2_spikes)
    if condition in {"shakb2_one_leg", "shak-b2_one_leg"}:
        return _jump_trace(params, amplitude_um=290.0, phase2_spikes_ms=phase2_spikes, response_probability=0.5)
    if condition in {"shakb2_six_leg", "shak-b2_six_leg"}:
        return _jump_trace(params, amplitude_um=290.0, polarity=1.0, phase2_spikes_ms=phase2_spikes, response_probability=0.5, unilateral=True)
    if condition in {"amph26_jump", "amph_null_jump"}:
        return _jump_trace(params, amplitude_um=18.0, phase2_spikes_ms=phase2_spikes, response_probability=1.0)
    if condition == "walking":
        return _walking_trace(params)
    if condition in {"adhesion", "adhesion_grip", "grip"}:
        return _adhesion_trace(params)
    if condition in {"flight", "flight_downdraft"}:
        return _flight_trace(params)
    if condition in {"larval_wildtype", "larval_peristalsis_wildtype"}:
        return _larval_trace(params, contractions_per_min=12.0)
    if condition in {"larval_parkin25", "larval_peristalsis_parkin25", "parkin25"}:
        return _larval_trace(params, contractions_per_min=5.0)

    raise ValueError(f"Unknown condition: {condition}")


def write_outputs(df: pd.DataFrame, out_dir: Path, condition: str, params: BeamParams, phase2_run: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{condition}_beam_waveforms.csv"
    summary_path = out_dir / f"{condition}_summary.json"
    df.to_csv(csv_path, index=False)
    summary = {
        "condition": condition,
        "rows": int(df.shape[0]),
        "t_start_ms": float(df["t_ms"].min()),
        "t_stop_ms": float(df["t_ms"].max()),
        "effective_dt_ms": float(np.nanmedian(np.diff(df["t_ms"].to_numpy(dtype=float)))) if df.shape[0] > 1 else None,
        "vertical_peak_abs": float(np.nanmax(np.abs(df["vertical"].to_numpy(dtype=float)))),
        "horizontal_peak_abs": float(np.nanmax(np.abs(df["horizontal"].to_numpy(dtype=float)))),
        "vector_peak": float(np.nanmax(df["vector"].to_numpy(dtype=float))),
        "response_events": int(df.attrs.get("successes", 0)),
        "jump_decision": df.attrs.get("jump_decision"),
        "gate_source": df.attrs.get("gate_source"),
        "phase2_motor_spike_count": int(df.attrs.get("phase2_motor_spike_count", 0)),
        "phase2_run": str(phase2_run) if phase2_run else None,
        "params": params.__dict__,
        "notes": [
            "Proxy waveform: tune against digitized paper figures before treating as recovered data.",
            "For fast jump traces, units are beam displacement proxies unless calibrated to force externally.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Elliott and Sparrow 2012 flexible-beam proxy waveforms.")
    parser.add_argument(
        "--condition",
        default="wildtype_jump",
        help="Condition name, e.g. wildtype_jump, phase2_gated_jump, phase2_gated_shakB2, standing_still, walking, flight_downdraft, parkin25.",
    )
    parser.add_argument("--phase2-run", default=None, help="Optional Phase 2 run directory containing spike_times.csv.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV and summary JSON.")
    parser.add_argument("--duration-ms", type=float, default=80.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--stim-time-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = BeamParams(
        duration_ms=float(args.duration_ms),
        dt_ms=float(args.dt_ms),
        stim_time_ms=float(args.stim_time_ms),
        seed=int(args.seed),
    )
    phase2_run = Path(args.phase2_run).expanduser().resolve() if args.phase2_run else None
    df = generate_condition(args.condition, params, phase2_run=phase2_run)
    write_outputs(df, Path(args.out_dir).expanduser().resolve(), args.condition, params, phase2_run)


if __name__ == "__main__":
    main()
