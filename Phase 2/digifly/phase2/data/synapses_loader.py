from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from digifly.phase2.data.paths import _syn_csv_path



# ===================================================================
# Synapse catalog helpers (optional per-neuron synmap)
_SYN_CACHE = {}
_SYN_CURSOR = {}

def _detect_and_scale_synmap(df: pd.DataFrame, nid: int, verbose=True):
    """Auto-detect nm vs µm in synmap and scale to µm."""
    for c in ("x","y","z"):
        if c not in df.columns: df[c] = np.nan
    xyz = df[["x","y","z"]].to_numpy(float)
    med_abs = np.nanmedian(np.abs(xyz))
    need_scale = med_abs > 200.0
    if need_scale:
        df[["x","y","z"]] *= 0.001
        if verbose:
            print(f"[synmap] {nid}: scaled ÷1000 from nm  | pre=({(df.type.str.lower()=='pre').sum()}, 3) post=({(df.type.str.lower()=='post').sum()}, 3)")
    else:
        if verbose:
            print(f"[synmap] {nid}: already in µm         | pre=({(df.type.str.lower()=='pre').sum()}, 3) post=({(df.type.str.lower()=='post').sum()}, 3)")
    return df


def _load_syn_catalog(swc_root: str, nid: int, verbose=True):
    """Cache + return {'pre': Nx3, 'post': Mx3} synapse coordinates (µm)."""
    nid = int(nid)
    if nid in _SYN_CACHE:
        return _SYN_CACHE[nid]
    path = _syn_csv_path(swc_root, nid)
    if not path or not Path(path).exists():
        _SYN_CACHE[nid] = {"pre": np.empty((0,3)), "post": np.empty((0,3))}
        return _SYN_CACHE[nid]
    df = pd.read_csv(path)
    for c in ("x","y","z","type"):
        if c not in df.columns: df[c] = np.nan
    df = _detect_and_scale_synmap(df, nid, verbose=verbose)
    pre  = df.loc[df["type"].astype(str).str.lower().eq("pre"),  ["x","y","z"]].to_numpy(float)
    post = df.loc[df["type"].astype(str).str.lower().eq("post"), ["x","y","z"]].to_numpy(float)
    _SYN_CACHE[nid] = {"pre": pre, "post": post}
    _SYN_CURSOR[nid] = 0
    return _SYN_CACHE[nid]



def load_pre_synapses_for_presyn_id(swc_root: str | Path, pre_id: int):
    p = _syn_csv_path(swc_root, int(pre_id))
    if p is None:
        return pd.DataFrame(), None, "missing_file"

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        return pd.DataFrame(), Path(p), f"read_csv error: {type(e).__name__}: {e}"

    # Hard requirement for edge building
    missing = [c for c in ("post_id",) if c not in df.columns]
    if missing:
        return df, Path(p), f"missing_columns: {missing} (has: {list(df.columns)})"

    return df, Path(p), None

