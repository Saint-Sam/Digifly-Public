from __future__ import annotations

import os
import csv
from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from neuron import h

from .swc_cell import SWCCell, find_swc_with_fallback

def _morph_swc_root(cfg: Dict[str, Any]) -> str:
    v = cfg.get("morph_swc_dir")
    return str(v if v else cfg["swc_dir"])


def _resolve_swc_for_gid(cfg: Dict[str, Any], gid: int) -> str:
    return find_swc_with_fallback(cfg.get("morph_swc_dir"), cfg["swc_dir"], int(gid))


def _xyz_of_section_site(sec, xloc):
    n3d = int(h.n3d(sec=sec))
    if n3d < 2:
        return float(h.x3d(0, sec=sec)), float(h.y3d(0, sec=sec)), float(h.z3d(0, sec=sec))
    xs = np.array([float(h.x3d(i, sec=sec)) for i in range(n3d)])
    ys = np.array([float(h.y3d(i, sec=sec)) for i in range(n3d)])
    zs = np.array([float(h.z3d(i, sec=sec)) for i in range(n3d)])
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    arc = np.r_[0.0, np.cumsum(seg)]
    total = arc[-1] if arc[-1] > 0 else 1.0
    target = float(xloc) * total
    i = int(np.clip(np.searchsorted(arc, target, side="right") - 1, 0, len(arc) - 2))
    t = (target - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
    X = xs[i] + t * (xs[i + 1] - xs[i])
    Y = ys[i] + t * (ys[i + 1] - ys[i])
    Z = zs[i] + t * (zs[i + 1] - zs[i])
    return float(X), float(Y), float(Z)

def _ais_override_path_for_gid(cfg: Dict[str, Any], gid: int) -> Path:
    swc = _resolve_swc_for_gid(cfg, int(gid))
    return Path(swc).parent / str(cfg.get("ais_override_filename", "ais_overrides.csv"))

def _ais_override_read(cfg: Dict[str, Any], gid: int) -> Optional[Dict[str, Any]]:
    p = _ais_override_path_for_gid(cfg, gid)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "gid" not in df.columns:
            return None
        df = df.dropna(subset=["gid"])
        df = df.astype({"gid": "int64"}, errors="ignore")
        df = df[df["gid"] == int(gid)]
        if df.empty:
            return None
        row = df.iloc[-1].to_dict()
        out = {"gid": int(row["gid"])}
        out["node_id"] = int(row["node_id"]) if "node_id" in row and pd.notna(row["node_id"]) else None
        out["xloc"] = float(row["xloc"]) if "xloc" in row and pd.notna(row["xloc"]) else None
        out["swc_path"] = str(row["swc_path"]) if "swc_path" in row and pd.notna(row["swc_path"]) else None
        out["source"] = str(row.get("source", "override"))
        return out
    except Exception:
        return None

def _ais_override_write(cfg: Dict[str, Any], gid: int, node_id: int, xloc: float, cell: SWCCell, src: str = "manual"):
    p = _ais_override_path_for_gid(cfg, gid)
    p.parent.mkdir(parents=True, exist_ok=True)

    sec = cell._sec_for_node.get(int(node_id))
    if sec is None:
        raise ValueError(f"gid={gid}: node_id={node_id} not found in SWC")

    X, Y, Z = _xyz_of_section_site(sec, float(xloc))
    swc_path = getattr(cell, "swc_path", None)
    swc_mtime = None
    try:
        swc_mtime = os.path.getmtime(swc_path) if swc_path else None
    except Exception:
        pass

    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["gid", "node_id", "xloc", "x", "y", "z", "source", "swc_path", "swc_mtime", "created_utc"])
        w.writerow([int(gid), int(node_id), float(xloc), float(X), float(Y), float(Z), str(src),
                    ("" if swc_path is None else str(swc_path)),
                    ("" if swc_mtime is None else float(swc_mtime)),
                    pd.Timestamp.utcnow().isoformat()])

    return {"gid": int(gid), "node_id": int(node_id), "xloc": float(xloc)}

def _ais_cache_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("ais_cache_csv", Path(_morph_swc_root(cfg)) / "_ais_cache.csv"))

def _ais_cache_read(cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    p = _ais_cache_path(cfg)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if "gid" not in df.columns:
            return {}
        df = df.dropna(subset=["gid"]).astype({"gid": "int64"}, errors="ignore").drop_duplicates("gid", keep="last")
        out: Dict[int, Dict[str, Any]] = {}
        for _, r in df.iterrows():
            out[int(r["gid"])] = {
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "swc_path": (str(r["swc_path"]) if "swc_path" in r else None),
                "swc_mtime": (float(r["swc_mtime"]) if "swc_mtime" in r and pd.notna(r["swc_mtime"]) else None),
            }
        return out
    except Exception:
        out = {}
        with p.open("r", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    gid = int(r.get("gid"))
                    out[gid] = {
                        "x": float(r["x"]),
                        "y": float(r["y"]),
                        "z": float(r["z"]),
                        "swc_path": r.get("swc_path"),
                        "swc_mtime": (float(r["swc_mtime"]) if r.get("swc_mtime") not in (None, "") else None),
                    }
                except Exception:
                    pass
        return out

def _ais_cache_append(cfg: Dict[str, Any], gid: int, x: float, y: float, z: float, src: str, swc_path: Optional[str], swc_mtime: Optional[float], dist_um: Optional[float] = None):
    p = _ais_cache_path(cfg)
    write_header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["gid", "x", "y", "z", "source", "dist_to_soma_um", "swc_path", "swc_mtime"])
        w.writerow([int(gid), float(x), float(y), float(z), str(src),
                    ("" if dist_um is None else float(dist_um)),
                    ("" if swc_path is None else str(swc_path)),
                    ("" if swc_mtime is None else float(swc_mtime))])

def _axon_sections(cell: SWCCell):
    inv = {sec: n for n, sec in getattr(cell, "_sec_for_node", {}).items()}
    secs = []
    for sec in cell._cache_sec_pts.keys():
        nid = inv.get(sec)
        if nid is not None and int(cell._nodes.get(nid, {}).get("type", 0)) == 2:
            secs.append(sec)
    return secs

def _nearest_site_in_sections(cell: SWCCell, x, y, z, sections):
    best = (None, 0.5, 1e18)
    tgt = np.array([float(x), float(y), float(z)], dtype=float)
    for sec in sections:
        pts = cell._cache_sec_pts.get(sec)
        if pts is None or pts.shape[0] < 2:
            continue
        P = pts[:, :3]
        seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seglen)]
        for i in range(len(P) - 1):
            p, q = P[i], P[i + 1]
            v = q - p
            v2 = float(v @ v) or 1e-12
            t = float(np.clip(((tgt - p) @ v) / v2, 0.0, 1.0))
            proj = p + t * v
            d2 = float(np.sum((proj - tgt) ** 2))
            if d2 < best[2]:
                xloc = float((cum[i] + t * (cum[i + 1] - cum[i])) / (cum[-1] or 1.0))
                best = (sec, xloc, d2)
    return (best[0], best[1]) if best[0] is not None else None

def _map_xyz_to_site_axon_pref(cfg: Dict[str, Any], cell: SWCCell, x, y, z):
    if cfg.get("ais_strict_axon_map", True):
        ax_secs = _axon_sections(cell)
        if ax_secs:
            got = _nearest_site_in_sections(cell, x, y, z, ax_secs)
            if got is not None:
                return got
    return cell.nearest_site(float(x), float(y), float(z))

def _compute_spatial_ais_xyz(cell: SWCCell):
    nodes = getattr(cell, "_nodes", {})
    if not nodes:
        return None
    soma_id = getattr(cell, "_soma_id", None)
    if soma_id is None or soma_id not in nodes:
        return None

    direct = [ch for ch in cell._children.get(soma_id, []) if int(nodes[ch].get("type", 0)) == 2]
    if direct:
        sx, sy, sz = nodes[soma_id]["x"], nodes[soma_id]["y"], nodes[soma_id]["z"]
        best = min(direct, key=lambda n: ((nodes[n]["x"] - sx) ** 2 + (nodes[n]["y"] - sy) ** 2 + (nodes[n]["z"] - sz) ** 2))
        r = nodes[best]
        return float(r["x"]), float(r["y"]), float(r["z"]), None

    q = deque(cell._children.get(soma_id, []))
    seen = {soma_id}
    first_axons = []
    while q and not first_axons:
        level = list(q)
        q.clear()
        for n in level:
            if n in seen:
                continue
            seen.add(n)
            if int(nodes[n].get("type", 0)) == 2:
                first_axons.append(n)
            q.extend(cell._children.get(n, []))
    if first_axons:
        sx, sy, sz = nodes[soma_id]["x"], nodes[soma_id]["y"], nodes[soma_id]["z"]
        best = min(first_axons, key=lambda n: ((nodes[n]["x"] - sx) ** 2 + (nodes[n]["y"] - sy) ** 2 + (nodes[n]["z"] - sz) ** 2))
        r = nodes[best]
        return float(r["x"]), float(r["y"]), float(r["z"]), None

    axon_nodes = [n for n, r in nodes.items() if int(r.get("type", 0)) == 2]
    if axon_nodes:
        sx, sy, sz = nodes[soma_id]["x"], nodes[soma_id]["y"], nodes[soma_id]["z"]
        best = min(axon_nodes, key=lambda n: ((nodes[n]["x"] - sx) ** 2 + (nodes[n]["y"] - sy) ** 2 + (nodes[n]["z"] - sz) ** 2))
        r = nodes[best]
        return float(r["x"]), float(r["y"]), float(r["z"]), None
    return None

def _cached_spatial_ais_site(self: SWCCell, force: bool = False):
    if not force and getattr(self, "_ais_site", None) is not None:
        return self._ais_site

    cfg = self.cfg
    gid = int(getattr(self, "gid", -1))
    swc_path = getattr(self, "swc_path", None)
    try:
        current_mtime = os.path.getmtime(swc_path) if swc_path else None
    except Exception:
        current_mtime = None

    def _finish_with_xyz(x, y, z, src_label):
        got = _map_xyz_to_site_axon_pref(cfg, self, x, y, z)
        if not got or got[0] is None:
            self._ais_site = self.soma_site()
            return self._ais_site
        sec, xloc = got
        xloc = max(float(xloc), float(cfg.get("ais_min_xloc", 0.05)))
        self._ais_site = (sec, xloc)
        return self._ais_site

    # override
    ov = _ais_override_read(cfg, gid)
    if ov is not None and ov.get("node_id") is not None:
        node_id = int(ov["node_id"])
        xloc = float(ov["xloc"] if ov.get("xloc") is not None else cfg.get("ais_min_xloc", 0.05))
        sec = getattr(self, "_sec_for_node", {}).get(node_id)
        if sec is not None:
            xloc = max(float(xloc), float(cfg.get("ais_min_xloc", 0.05)))
            self._ais_site = (sec, xloc)
            return self._ais_site

    cache = {} if force else _ais_cache_read(cfg)
    entry = None if force else cache.get(gid)
    if entry is not None:
        if (entry.get("swc_path") == swc_path and
            entry.get("swc_mtime") is not None and current_mtime is not None and
            abs(entry.get("swc_mtime") - current_mtime) < 1e-6):
            return _finish_with_xyz(entry["x"], entry["y"], entry["z"], "cache")

    got = _compute_spatial_ais_xyz(self)
    if got is not None:
        x, y, z, dist = got
        _ais_cache_append(cfg, gid, x, y, z, src="computed", dist_um=dist, swc_path=swc_path, swc_mtime=current_mtime)
        return _finish_with_xyz(x, y, z, "computed")

    self._ais_site = self.soma_site()
    return self._ais_site

def attach_ais_methods():
    """Monkey-patch SWCCell.axon_ais_site -> cached AIS site selection."""
    SWCCell.axon_ais_site = _cached_spatial_ais_site
