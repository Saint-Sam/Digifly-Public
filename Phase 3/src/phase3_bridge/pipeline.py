from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_phase2_spike_times(result_dir: Path | str) -> pd.DataFrame:
    """Load spike_times.csv from a Phase 2 run folder."""
    result_dir = Path(result_dir).expanduser().resolve()
    spike_csv = result_dir / "spike_times.csv"
    if not spike_csv.exists():
        raise FileNotFoundError(f"Missing spike_times.csv: {spike_csv}")

    sp = pd.read_csv(spike_csv)
    needed = {"neuron_id", "spike_time_ms"}
    missing = needed.difference(sp.columns)
    if missing:
        raise ValueError(f"spike_times.csv missing required columns: {sorted(missing)}")

    sp = sp.copy()
    sp["neuron_id"] = pd.to_numeric(sp["neuron_id"], errors="coerce").astype("Int64")
    sp["spike_time_ms"] = pd.to_numeric(sp["spike_time_ms"], errors="coerce")
    sp = sp.dropna(subset=["neuron_id", "spike_time_ms"])
    sp["neuron_id"] = sp["neuron_id"].astype(int)
    return sp.sort_values(["neuron_id", "spike_time_ms"]).reset_index(drop=True)


def load_phase2_motor_rates(result_dir: Path | str) -> pd.DataFrame:
    """Load motor_rates.csv from a paper-rate Phase 2 run folder."""
    result_dir = Path(result_dir).expanduser().resolve()
    rates_csv = result_dir / "motor_rates.csv"
    if not rates_csv.exists():
        raise FileNotFoundError(f"Missing motor_rates.csv: {rates_csv}")

    df = pd.read_csv(rates_csv)
    if "t_ms" not in df.columns:
        raise ValueError(f"motor_rates.csv missing required column: t_ms ({rates_csv})")

    df = df.copy()
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce")
    df = df.dropna(subset=["t_ms"]).sort_values("t_ms").reset_index(drop=True)

    valid_cols = ["t_ms"]
    for col in df.columns:
        if col == "t_ms":
            continue
        if re.fullmatch(r"\d+_rate_hz", str(col).strip()):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            valid_cols.append(col)
    if len(valid_cols) <= 1:
        raise ValueError(f"motor_rates.csv contains no <gid>_rate_hz columns: {rates_csv}")
    return df[valid_cols].reset_index(drop=True)


def _motor_rate_id_columns(rate_df: pd.DataFrame) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for col in rate_df.columns:
        if col == "t_ms":
            continue
        m = re.fullmatch(r"(\d+)_rate_hz", str(col).strip())
        if m:
            out[int(m.group(1))] = str(col)
    return out


def filter_motor_rates_to_neuron_ids(rate_df: pd.DataFrame, neuron_ids: Iterable[int]) -> pd.DataFrame:
    """Keep only the selected motor-rate columns, preserving t_ms."""
    id_cols = _motor_rate_id_columns(rate_df)
    keep = ["t_ms"]
    for nid in neuron_ids:
        col = id_cols.get(int(nid))
        if col is not None:
            keep.append(col)
    return rate_df[keep].copy()


def load_phase2_timebase_ms(
    result_dir: Path | str,
    uniform_dt_ms: float = 0.025,
) -> np.ndarray:
    """Use records.csv time span and return a stable, uniform timebase."""
    result_dir = Path(result_dir).expanduser().resolve()
    rec_csv = result_dir / "records.csv"
    if not rec_csv.exists():
        return np.arange(0.0, 50.0 + 0.025, 0.025, dtype=float)

    df = pd.read_csv(rec_csv, usecols=["t_ms"])
    t_raw = pd.to_numeric(df["t_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if t_raw.size < 2:
        return np.arange(0.0, 50.0 + 0.025, 0.025, dtype=float)

    # records.csv from variable-step runs can contain repeated/non-monotonic rows
    # with extremely tiny adaptive steps. We normalize to a uniform grid so
    # downstream profile transforms (settle/phase/loop) stay computationally stable.
    t = np.unique(np.sort(t_raw))
    if t.size < 2:
        return np.arange(0.0, 50.0 + 0.025, 0.025, dtype=float)
    if uniform_dt_ms is not None and float(uniform_dt_ms) > 0:
        dt = float(uniform_dt_ms)
        return np.arange(float(t[0]), float(t[-1]) + 0.5 * dt, dt, dtype=float)
    return t


def load_mapping_csv(mapping_csv: Path | str) -> pd.DataFrame:
    """Load MN->actuator mapping and normalize optional numeric fields."""
    mapping_csv = Path(mapping_csv).expanduser().resolve()
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Missing mapping CSV: {mapping_csv}")

    mp = pd.read_csv(mapping_csv)
    needed = {"mn_id", "actuator_name"}
    missing = needed.difference(mp.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {sorted(missing)}")

    mp = mp.copy()
    mp["mn_id"] = pd.to_numeric(mp["mn_id"], errors="coerce").astype("Int64")
    mp["actuator_name"] = mp["actuator_name"].astype(str).str.strip()
    mp = mp[mp["actuator_name"].ne("")].dropna(subset=["mn_id"]).copy()
    mp["mn_id"] = mp["mn_id"].astype(int)

    for col, default in (("gain", 1.0), ("sign", 1.0), ("bias", 0.0), ("weight", 1.0)):
        if col in mp.columns:
            mp[col] = pd.to_numeric(mp[col], errors="coerce").fillna(default)
        else:
            mp[col] = default
    return mp.reset_index(drop=True)


def summarize_mapping_coverage(spikes: pd.DataFrame, mapping: pd.DataFrame) -> Dict[str, int]:
    spike_ids = set(spikes["neuron_id"].astype(int).unique())
    map_ids = set(mapping["mn_id"].astype(int).unique())
    active_ids = spike_ids.intersection(map_ids)
    return {
        "spiking_neuron_count": len(spike_ids),
        "mapped_neuron_count": len(map_ids),
        "spiking_and_mapped_count": len(active_ids),
        "unmapped_spiking_count": len(spike_ids - map_ids),
        "mapped_not_spiking_count": len(map_ids - spike_ids),
    }


def summarize_rate_mapping_coverage(
    rate_df: pd.DataFrame,
    mapping: pd.DataFrame,
    active_threshold_hz: float = 0.01,
) -> Dict[str, int]:
    rate_cols = _motor_rate_id_columns(rate_df)
    rate_ids = set(rate_cols)
    map_ids = set(mapping["mn_id"].astype(int).unique())
    active_ids = set()
    threshold = float(active_threshold_hz)
    for nid, col in rate_cols.items():
        vals = pd.to_numeric(rate_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if vals.size and float(np.nanmax(vals)) >= threshold:
            active_ids.add(int(nid))
    return {
        "rate_neuron_count": len(rate_ids),
        "mapped_neuron_count": len(map_ids),
        "rate_and_mapped_count": len(rate_ids.intersection(map_ids)),
        "active_rate_neuron_count": len(active_ids),
        "active_and_mapped_count": len(active_ids.intersection(map_ids)),
        "mapped_not_in_rates_count": len(map_ids - rate_ids),
    }


def _spikes_to_activation(
    t_ms: np.ndarray,
    spikes_ms: Iterable[float],
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
) -> np.ndarray:
    tau_rise_ms = max(1e-6, float(tau_rise_ms))
    tau_decay_ms = max(1e-6, float(tau_decay_ms))
    if tau_rise_ms > tau_decay_ms:
        tau_rise_ms, tau_decay_ms = tau_decay_ms, tau_rise_ms

    t = t_ms.astype(float, copy=False)
    y = np.zeros_like(t, dtype=float)

    # Difference-of-exponentials PSP kernel normalized to peak=1.
    t_peak = (tau_rise_ms * tau_decay_ms / (tau_decay_ms - tau_rise_ms)) * np.log(
        tau_decay_ms / tau_rise_ms
    )
    peak = np.exp(-t_peak / tau_decay_ms) - np.exp(-t_peak / tau_rise_ms)
    if peak <= 0:
        peak = 1.0

    for ts in spikes_ms:
        dt = t - float(ts)
        mask = dt >= 0.0
        if not np.any(mask):
            continue
        k = np.exp(-dt[mask] / tau_decay_ms) - np.exp(-dt[mask] / tau_rise_ms)
        y[mask] += (k / peak)
    return y


def build_actuator_controls_from_spikes(
    spikes: pd.DataFrame,
    mapping: pd.DataFrame,
    t_ms: np.ndarray,
    tau_rise_ms: float = 1.0,
    tau_decay_ms: float = 6.0,
    scale: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build actuator controls on the provided timebase from spike times + mapping."""
    if t_ms.ndim != 1 or t_ms.size < 2:
        raise ValueError("t_ms must be a 1D array with at least 2 samples.")

    spikes_by_id = (
        spikes.groupby("neuron_id")["spike_time_ms"].apply(lambda s: s.to_numpy(dtype=float)).to_dict()
    )

    controls = pd.DataFrame({"t_ms": t_ms})
    used_rows = 0
    used_spike_neurons = set()
    dropped_rows = 0

    for actuator, rows in mapping.groupby("actuator_name", sort=True):
        trace = np.zeros_like(t_ms, dtype=float)
        bias = 0.0
        for _, row in rows.iterrows():
            nid = int(row["mn_id"])
            spikes_ms = spikes_by_id.get(nid)
            if spikes_ms is None or spikes_ms.size == 0:
                dropped_rows += 1
                continue
            used_rows += 1
            used_spike_neurons.add(nid)
            base = _spikes_to_activation(
                t_ms=t_ms,
                spikes_ms=spikes_ms,
                tau_rise_ms=tau_rise_ms,
                tau_decay_ms=tau_decay_ms,
            )
            gain = float(row["gain"])
            sign = float(row["sign"])
            weight = float(row["weight"])
            trace += (sign * gain * weight * float(scale) * base)
            bias += float(row["bias"])
        controls[actuator] = trace + bias

    stats = {
        "mapping_rows_total": int(mapping.shape[0]),
        "mapping_rows_used": int(used_rows),
        "mapping_rows_dropped_no_spike": int(dropped_rows),
        "spiking_ids_used": int(len(used_spike_neurons)),
        "actuator_count": int(max(0, controls.shape[1] - 1)),
    }
    return controls, stats


def build_actuator_controls_from_rates(
    rate_df: pd.DataFrame,
    mapping: pd.DataFrame,
    scale: float = 1.0,
    rate_norm_hz: float = 200.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build actuator controls directly from motor firing-rate traces."""
    if "t_ms" not in rate_df.columns:
        raise ValueError("rate_df must include a t_ms column.")

    t_ms = pd.to_numeric(rate_df["t_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if t_ms.ndim != 1 or t_ms.size < 2:
        raise ValueError("rate_df must contain at least 2 time samples.")

    rate_cols = _motor_rate_id_columns(rate_df)
    controls = pd.DataFrame({"t_ms": t_ms})
    used_rows = 0
    used_rate_neurons = set()
    dropped_rows = 0
    denom = float(rate_norm_hz)
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0

    for actuator, rows in mapping.groupby("actuator_name", sort=True):
        trace = np.zeros_like(t_ms, dtype=float)
        bias = 0.0
        for _, row in rows.iterrows():
            nid = int(row["mn_id"])
            col = rate_cols.get(nid)
            if col is None:
                dropped_rows += 1
                continue
            used_rows += 1
            used_rate_neurons.add(nid)
            rate = pd.to_numeric(rate_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            gain = float(row["gain"])
            sign = float(row["sign"])
            weight = float(row["weight"])
            trace += (sign * gain * weight * float(scale) * (rate / denom))
            bias += float(row["bias"])
        controls[actuator] = trace + bias

    stats = {
        "mapping_rows_total": int(mapping.shape[0]),
        "mapping_rows_used": int(used_rows),
        "mapping_rows_dropped_no_rate": int(dropped_rows),
        "rate_ids_used": int(len(used_rate_neurons)),
        "actuator_count": int(max(0, controls.shape[1] - 1)),
        "rate_norm_hz": float(denom),
    }
    return controls, stats


def save_controls_csv(controls: pd.DataFrame, out_csv: Path | str) -> Path:
    out_csv = Path(out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    controls.to_csv(out_csv, index=False)
    return out_csv


def plot_actuator_controls(
    controls: pd.DataFrame,
    out_path: Path | str,
    top_n: int = 12,
) -> Path:
    out_pdf = Path(out_path).expanduser().resolve().with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if "t_ms" not in controls.columns:
        raise ValueError("controls DataFrame must include t_ms column.")
    actuator_cols = [c for c in controls.columns if c != "t_ms"]
    if not actuator_cols:
        raise ValueError("controls DataFrame has no actuator columns.")

    top_n = max(1, int(top_n))
    ranked = sorted(
        actuator_cols,
        key=lambda c: float(np.nanmax(np.abs(controls[c].to_numpy(dtype=float)))),
        reverse=True,
    )[:top_n]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    t = controls["t_ms"].to_numpy(dtype=float)
    cmap = plt.get_cmap("tab20", max(20, len(ranked)))
    for idx, col in enumerate(ranked):
        ax.plot(
            t,
            controls[col].to_numpy(dtype=float),
            lw=1.7,
            color=cmap(idx),
            label=col,
        )
    ax.set_title(f"Phase 3 actuator controls (top {len(ranked)} by peak abs amplitude)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Control")
    ax.grid(True, color="#d1d5db", alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#111827")
        spine.set_linewidth(0.9)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#d1d5db")
    leg.get_frame().set_alpha(0.95)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
