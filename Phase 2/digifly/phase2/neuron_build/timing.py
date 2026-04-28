from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

import numpy as np
from neuron import h

from .config import SYN_PRESETS, NT_TO_CLASS, DEFAULT_GLOBAL_TIMING

def syn_class(pred_nt: Any, gt: Dict[str, Any]) -> str:
    if isinstance(pred_nt, str):
        return NT_TO_CLASS.get(pred_nt.strip().lower(), gt["default_class"])
    return gt["default_class"]

def edge_delay_from_row(row: Dict[str, Any], gt: Dict[str, Any]) -> float:
    if gt["delay_col"] in row and np.isfinite(row[gt["delay_col"]]):
        return float(row[gt["delay_col"]])
    pl = float(row.get(gt["pathlen_col"], np.nan))
    if not np.isfinite(pl):
        if all(k in row for k in ("pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z")):
            dx = float(row["pre_x"]) - float(row["post_x"])
            dy = float(row["pre_y"]) - float(row["post_y"])
            dz = float(row["pre_z"]) - float(row["post_z"])
            pl = float((dx * dx + dy * dy + dz * dz) ** 0.5)
        else:
            pl = 0.0
    return float(gt["base_release_delay_ms"] + pl / max(gt["vel_um_per_ms"], 1e-9))

def timing_from_row(row, cfg, gt=None):
    """
    Return (w_uS, delay_ms, erev_mV, tau1_ms, tau2_ms).

    This function is Phase-2-compatible with multiple historical column names:
      - Erev: syn_e_rev_mV OR erev_mV
      - Tau: tau1_ms/tau2_ms (required unless defaults are provided in cfg)
      - Weight/Delay: can come from row or cfg defaults
    """
    if gt is None:
        gt = cfg.get("global_timing", {}) or {}

    # Candidate column names (support both your current schema and older schema).
    #
    # IMPORTANT:
    #   By default we do NOT treat legacy "weight" as uS, because in older
    #   Phase-2 exports that column often encoded synapse counts (or other
    #   non-uS weights). This preserves Copy2-like behavior where missing
    #   weight_uS falls back to CONFIG['default_weight_uS'].
    #   If needed, opt in via:
    #     cfg["allow_legacy_weight_column"] = True
    #   or cfg["global_timing"]["allow_legacy_weight_column"] = True
    allow_legacy_weight = bool(
        cfg.get("allow_legacy_weight_column", False)
        or (cfg.get("global_timing") or {}).get("allow_legacy_weight_column", False)
    )
    weight_candidates = [gt.get("weight_col"), "weight_uS", "w_uS"]
    if allow_legacy_weight:
        weight_candidates.append("weight")
    delay_candidates  = [gt.get("delay_col"),  "delay_ms", "delay", "d_ms"]
    erev_candidates   = [gt.get("erev_col"),   "syn_e_rev_mV", "erev_mV", "e_rev_mV", "syn_erev_mV"]
    tau1_candidates   = [gt.get("tau1_col"),   "tau1_ms", "tau1"]
    tau2_candidates   = [gt.get("tau2_col"),   "tau2_ms", "tau2"]

    # row is typically a pandas Series
    idx = getattr(row, "index", None)

    def _first_present(cands):
        for c in cands:
            if not c:
                continue
            if idx is not None and c in idx:
                val = row[c]
                # treat NaN as missing
                try:
                    if val != val:  # NaN check
                        continue
                except Exception:
                    pass
                return c, val
            # some codepaths use dict-like rows
            try:
                if isinstance(row, dict) and c in row:
                    val = row[c]
                    # treat NaN as missing for dict rows too
                    try:
                        if val != val:  # NaN check
                            continue
                    except Exception:
                        pass
                    return c, val
            except Exception:
                pass
        return None, None

    # Weight (uS)
    _, w_val = _first_present(weight_candidates)
    if w_val is None:
        w_default = cfg.get("default_weight_uS", None)
        if w_default is None:
            raise ValueError(
                "Edges row is missing weight. Provide column 'weight_uS' "
                "or set CONFIG['default_weight_uS']."
            )
        w = float(w_default)
    else:
        w = float(w_val)

    # Optional global scaling (parity with Master-Copy2 behavior).
    # Controlled via cfg["global_timing"]["global_weight_scale"].
    try:
        w_scale = float((cfg.get("global_timing") or {}).get("global_weight_scale", 1.0))
    except Exception:
        w_scale = 1.0
    w *= w_scale

    # Delay (ms)
    _, d_val = _first_present(delay_candidates)
    if d_val is None:
        d_default = cfg.get("default_delay_ms", None)
        if d_default is None:
            # If you truly want delay to be optional, choose a hard default here.
            d = 0.0
        else:
            d = float(d_default)
    else:
        d = float(d_val)

    # Erev (mV)
    erev_col, e_val = _first_present(erev_candidates)
    if e_val is None:
        e_default = cfg.get("syn_e_rev_mV", None)
        if e_default is None:
            raise ValueError(
                "Edges row is missing Erev column. Accepted names include "
                "'syn_e_rev_mV' and 'erev_mV'. "
                f"Tried: {[c for c in erev_candidates if c]}. "
                f"Available columns: {list(idx) if idx is not None else 'unknown'}"
            )
        e = float(e_default)
    else:
        e = float(e_val)

    # Tau1 (ms)
    _, t1_val = _first_present(tau1_candidates)
    if t1_val is None:
        t1_default = cfg.get("syn_tau1_ms", None)
        if t1_default is None:
            raise ValueError(
                "Edges row is missing tau1. Provide column 'tau1_ms' "
                "or set CONFIG['syn_tau1_ms']."
            )
        t1 = float(t1_default)
    else:
        t1 = float(t1_val)

    # Tau2 (ms)
    _, t2_val = _first_present(tau2_candidates)
    if t2_val is None:
        t2_default = cfg.get("syn_tau2_ms", None)
        if t2_default is None:
            raise ValueError(
                "Edges row is missing tau2. Provide column 'tau2_ms' "
                "or set CONFIG['syn_tau2_ms']."
            )
        t2 = float(t2_default)
    else:
        t2 = float(t2_val)

    return float(w), float(d), float(e), float(t1), float(t2)


def xyz_at_site(site) -> Optional[Tuple[float, float, float]]:
    try:
        sec, x = site
        n3d = int(h.n3d(sec=sec))
        if n3d < 2:
            return float(h.x3d(0, sec=sec)), float(h.y3d(0, sec=sec)), float(h.z3d(0, sec=sec))
        xs = np.array([float(h.x3d(i, sec=sec)) for i in range(n3d)])
        ys = np.array([float(h.y3d(i, sec=sec)) for i in range(n3d)])
        zs = np.array([float(h.z3d(i, sec=sec)) for i in range(n3d)])
        seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
        arc = np.r_[0.0, np.cumsum(seg)]
        total = arc[-1] if arc[-1] > 0 else 1.0
        target = float(x) * total
        i = int(np.clip(np.searchsorted(arc, target, side="right") - 1, 0, len(arc) - 2))
        t = (target - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
        X = xs[i] + t * (xs[i + 1] - xs[i])
        Y = ys[i] + t * (ys[i + 1] - ys[i])
        Z = zs[i] + t * (zs[i + 1] - zs[i])
        return float(X), float(Y), float(Z)
    except Exception:
        return None


def geom_delay_ms_from_xyz(
    pre_xyz: Tuple[float, float, float] | None,
    post_site,
    gt: Dict[str, Any] | None = None,
) -> Optional[float]:
    gt = dict(DEFAULT_GLOBAL_TIMING) if gt is None else dict(gt)
    try:
        if pre_xyz is None:
            return None
        post_xyz = xyz_at_site(post_site)
        if post_xyz is None:
            return None
        px, py, pz = (float(pre_xyz[0]), float(pre_xyz[1]), float(pre_xyz[2]))
        qx, qy, qz = post_xyz
        dist_um = float(np.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2))
        return float(gt["base_release_delay_ms"] + dist_um / max(gt["vel_um_per_ms"], 1e-9))
    except Exception:
        return None


def geom_delay_ms(pre_site, post_site, gt: Dict[str, Any] | None = None) -> Optional[float]:
    pre_xyz = xyz_at_site(pre_site)
    return geom_delay_ms_from_xyz(pre_xyz, post_site, gt=gt)
