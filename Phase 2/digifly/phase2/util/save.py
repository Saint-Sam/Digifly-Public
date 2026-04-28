from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import numpy as np
import pandas as pd

def _vec_to_numpy(v, expected_len: Optional[int] = None) -> np.ndarray:
    """
    Robust conversion for NEURON vectors.

    Some environments expose multiple conversion paths with different behavior
    (e.g., one may return a collapsed length-1 array). Try several and prefer
    a candidate that matches expected_len when provided.
    """
    candidates = []

    if hasattr(v, "to_python"):
        try:
            candidates.append(np.array(v.to_python(), dtype=float))
        except Exception:
            pass

    if hasattr(v, "as_numpy"):
        try:
            candidates.append(np.array(v.as_numpy(), dtype=float))
        except Exception:
            pass

    try:
        candidates.append(np.array(v, dtype=float))
    except Exception:
        pass

    try:
        candidates.append(np.array(list(v), dtype=float))
    except Exception:
        pass

    candidates = [c for c in candidates if isinstance(c, np.ndarray)]
    if not candidates:
        return np.array([], dtype=float)

    if expected_len is not None:
        for c in candidates:
            if int(c.size) == int(expected_len):
                return c

    # Fall back to the longest non-empty candidate.
    candidates.sort(key=lambda c: int(c.size), reverse=True)
    return candidates[0]

def save_config(cfg: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "config.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)


def _record_payload(net) -> Dict[str, Any]:
    rec = getattr(net, "records", {}) or {}
    tvecs = rec.get("t", [])
    t = _vec_to_numpy(tvecs[0]).tolist() if tvecs else []
    n_t = len(t)

    columns: Dict[str, List[float]] = {}
    for k, vecs in rec.items():
        if k == "t":
            continue
        for i, v in enumerate(vecs):
            col = f"{k}" if len(vecs) == 1 else f"{k}__{i}"
            arr = _vec_to_numpy(v, expected_len=n_t if n_t > 0 else None)
            columns[col] = arr.tolist()

    return {
        "rank": int(getattr(net, "parallel_rank", 0)),
        "t_ms": t,
        "columns": columns,
    }


def _merge_record_payloads(payloads: Iterable[Dict[str, Any] | None]) -> pd.DataFrame:
    payload_list = [payload for payload in payloads if isinstance(payload, dict)]
    if not payload_list:
        return pd.DataFrame()

    t = np.array([], dtype=float)
    for payload in payload_list:
        cand = np.array(payload.get("t_ms", []), dtype=float)
        if cand.size:
            t = cand
            break

    data: Dict[str, Any] = {"t_ms": t}
    n_t = int(t.size)
    for payload in payload_list:
        cols = payload.get("columns") or {}
        if not isinstance(cols, dict):
            continue
        for col, values in cols.items():
            arr = np.array(values, dtype=float)
            if n_t > 1 and arr.size not in (0, n_t):
                print(
                    f"[warn] distributed record length mismatch for {col}: "
                    f"{arr.size} vs t={n_t} (will be NaN-padded in CSV)"
                )
            data[str(col)] = arr

    return pd.DataFrame(data)


def _spike_payload(spike_map: Dict[int, Any], *, rank: int = 0) -> Dict[str, Any]:
    out: Dict[int, List[float]] = {}
    for nid, v in spike_map.items():
        out[int(nid)] = [float(x) for x in _vec_to_numpy(v)]
    return {"rank": int(rank), "spikes": out}


def _merge_spike_payloads(payloads: Iterable[Dict[str, Any] | None]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        spikes = payload.get("spikes") or {}
        if not isinstance(spikes, dict):
            continue
        for nid, values in spikes.items():
            for t in values:
                rows.append({"neuron_id": int(nid), "spike_time_ms": float(t)})
    rows.sort(key=lambda row: (int(row["neuron_id"]), float(row["spike_time_ms"])))
    return pd.DataFrame(rows, columns=["neuron_id", "spike_time_ms"])

def save_records(net, out_dir: Path) -> None:
    """
    Save recorded vectors in a columnar CSV:
      - time is taken from records['t'][0] if present
      - each recorded trace key becomes a column
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(net, "is_distributed", False)):
        pc = getattr(net, "_pc", None)
        gathered = pc.py_gather(_record_payload(net), int(getattr(net, "_root_rank", 0)))
        if not bool(getattr(net, "is_root_rank", False)):
            return
        df = _merge_record_payloads(gathered or [])
        if df.empty:
            return
        df.to_csv(out_dir / "records.csv", index=False)
        return

    rec = getattr(net, "records", {}) or {}
    tvecs = rec.get("t", [])
    if not tvecs:
        return

    t = _vec_to_numpy(tvecs[0])
    data = {"t_ms": t}
    n_t = int(t.size)

    for k, vecs in rec.items():
        if k == "t":
            continue
        # if multiple vectors recorded under same key, save each with suffix
        for i, v in enumerate(vecs):
            col = f"{k}" if len(vecs) == 1 else f"{k}__{i}"
            arr = _vec_to_numpy(v, expected_len=n_t if n_t > 0 else None)
            if n_t > 1 and arr.size not in (0, n_t):
                print(
                    f"[warn] record length mismatch for {col}: "
                    f"{arr.size} vs t={n_t} (will be NaN-padded in CSV)"
                )
            data[col] = arr

    df = pd.DataFrame(data)
    df.to_csv(out_dir / "records.csv", index=False)

def save_spikes(spike_map: Dict[int, Any], out_dir: Path, *, net=None) -> None:
    """
    spike_map: {nid: h.Vector spike_times_ms}

    Writes two files for compatibility:
      - spike_times.csv  (Phase 2 Master-compatible columns)
      - spikes.csv       (legacy columns)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if net is not None and bool(getattr(net, "is_distributed", False)):
        pc = getattr(net, "_pc", None)
        gathered = pc.py_gather(
            _spike_payload(spike_map, rank=int(getattr(net, "parallel_rank", 0))),
            int(getattr(net, "_root_rank", 0)),
        )
        if not bool(getattr(net, "is_root_rank", False)):
            return
        df_master = _merge_spike_payloads(gathered or [])
        df_master.to_csv(out_dir / "spike_times.csv", index=False)
        df_legacy = df_master.rename(columns={"neuron_id": "nid", "spike_time_ms": "t_ms"})
        df_legacy.to_csv(out_dir / "spikes.csv", index=False)
        return

    rows = []
    for nid, v in spike_map.items():
        arr = _vec_to_numpy(v)
        for t in arr:
            rows.append({"neuron_id": int(nid), "spike_time_ms": float(t)})

    # Master-compatible schema (always include headers)
    df_master = pd.DataFrame(rows, columns=["neuron_id", "spike_time_ms"])
    df_master.to_csv(out_dir / "spike_times.csv", index=False)

    # Legacy schema (always include headers)
    df_legacy = df_master.rename(columns={"neuron_id": "nid", "spike_time_ms": "t_ms"})
    df_legacy.to_csv(out_dir / "spikes.csv", index=False)
