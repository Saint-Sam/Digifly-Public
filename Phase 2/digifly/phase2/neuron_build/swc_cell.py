from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from neuron import h

# Load only what we need (NO nrngui!)
for hoc in ("stdrun.hoc", "import3d.hoc"):
    try:
        h.load_file(hoc)
    except Exception:
        pass

# ---------------- Synapse catalog helpers ----------------

_SYN_CACHE: Dict[int, Dict[str, np.ndarray]] = {}
_SYN_CURSOR: Dict[int, int] = {}

def detect_and_scale_synmap(df: pd.DataFrame, nid: int, verbose: bool = True) -> pd.DataFrame:
    """Auto-detect nm vs µm in synmap and scale to µm."""
    for c in ("x", "y", "z"):
        if c not in df.columns:
            df[c] = np.nan
    xyz = df[["x", "y", "z"]].to_numpy(float)
    med_abs = np.nanmedian(np.abs(xyz))
    need_scale = bool(med_abs > 200.0)
    if need_scale:
        df[["x", "y", "z"]] *= 0.001
        if verbose and "type" in df.columns:
            t = df["type"].astype(str).str.lower()
            print(f"[synmap] {nid}: scaled ÷1000 from nm | pre={(t=='pre').sum()} post={(t=='post').sum()}")
    else:
        if verbose and "type" in df.columns:
            t = df["type"].astype(str).str.lower()
            print(f"[synmap] {nid}: already in µm       | pre={(t=='pre').sum()} post={(t=='post').sum()}")
    return df

def syn_csv_path(swc_root: str | Path, nid: int) -> Optional[Path]:
    """Find <nid>_synapses_new.csv with flexible layouts."""
    p = Path(swc_root)
    nid = int(nid)
    for pat in (
        f"**/{nid}/{nid}_synapses_new.csv",
        f"**/{nid}_synapses_new.csv",
        f"**/*{nid}*synapses_new*.csv",
    ):
        hits = list(p.glob(pat))
        if hits:
            hits.sort(key=lambda q: len(str(q)))
            return hits[0]
    return None

def load_syn_catalog(swc_root: str | Path, nid: int, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Cache + return {'pre': Nx3, 'post': Mx3} synapse coordinates (µm)."""
    nid = int(nid)
    if nid in _SYN_CACHE:
        return _SYN_CACHE[nid]
    path = syn_csv_path(swc_root, nid)
    if not path or not path.exists():
        _SYN_CACHE[nid] = {"pre": np.empty((0, 3)), "post": np.empty((0, 3))}
        _SYN_CURSOR[nid] = 0
        return _SYN_CACHE[nid]
    df = pd.read_csv(path)
    for c in ("x", "y", "z", "type"):
        if c not in df.columns:
            df[c] = np.nan
    df = detect_and_scale_synmap(df, nid, verbose=verbose)
    pre = df.loc[df["type"].astype(str).str.lower().eq("pre"), ["x", "y", "z"]].to_numpy(float)
    post = df.loc[df["type"].astype(str).str.lower().eq("post"), ["x", "y", "z"]].to_numpy(float)
    _SYN_CACHE[nid] = {"pre": pre, "post": post}
    _SYN_CURSOR[nid] = 0
    return _SYN_CACHE[nid]

def pick_post_site(cell: "SWCCell", row: Dict[str, Any], swc_root: str | Path) -> tuple[Any, float]:
    """Choose NEURON site for a postsyn location from row or from synmap; fallback soma."""
    # Support both schema variants used across Phase-2 exports:
    #   - post_x/post_y/post_z
    #   - x_post/y_post/z_post
    xyz = None
    xyz_sets = (("post_x", "post_y", "post_z"), ("x_post", "y_post", "z_post"))
    for xk, yk, zk in xyz_sets:
        if all(k in row and pd.notna(row[k]) for k in (xk, yk, zk)):
            try:
                x = float(row[xk])
                y = float(row[yk])
                z = float(row[zk])
                # Match synmap behavior: if values look nanometer-scale, convert to µm.
                if max(abs(x), abs(y), abs(z)) > 200.0:
                    x *= 0.001
                    y *= 0.001
                    z *= 0.001
                xyz = (x, y, z)
                break
            except Exception:
                pass

    if xyz is not None:
        return cell.nearest_site(*xyz)

    post_id = None
    try:
        if "post_id" in row and pd.notna(row["post_id"]):
            post_id = int(row["post_id"])
    except Exception:
        post_id = None
    if post_id is not None:
        cat = load_syn_catalog(swc_root, post_id, verbose=False)
        xyzs = cat.get("post", np.empty((0, 3)))
        if len(xyzs) > 0:
            j = None
            try:
                if "post_syn_index" in row and pd.notna(row["post_syn_index"]):
                    jj = int(float(row["post_syn_index"]))
                    if 0 <= jj < len(xyzs):
                        j = jj
            except Exception:
                j = None
            if j is None:
                j = _SYN_CURSOR.get(post_id, 0) % len(xyzs)
            _SYN_CURSOR[post_id] = j + 1
            x, y, z = map(float, xyzs[j])
            return cell.nearest_site(x, y, z)
    return cell.soma_site()

def pick_pre_site(cell: "SWCCell", row: Dict[str, Any], swc_root: str | Path) -> tuple[Any, float]:
    """Choose NEURON site for a presyn location from row or from synmap; fallback soma."""
    xyz = None
    xyz_sets = (("pre_x", "pre_y", "pre_z"), ("x_pre", "y_pre", "z_pre"))
    for xk, yk, zk in xyz_sets:
        if all(k in row and pd.notna(row[k]) for k in (xk, yk, zk)):
            try:
                x = float(row[xk])
                y = float(row[yk])
                z = float(row[zk])
                if max(abs(x), abs(y), abs(z)) > 200.0:
                    x *= 0.001
                    y *= 0.001
                    z *= 0.001
                xyz = (x, y, z)
                break
            except Exception:
                pass

    if xyz is not None:
        return cell.nearest_site(*xyz)

    pre_id = None
    try:
        if "pre_id" in row and pd.notna(row["pre_id"]):
            pre_id = int(row["pre_id"])
    except Exception:
        pre_id = None
    if pre_id is not None:
        cat = load_syn_catalog(swc_root, pre_id, verbose=False)
        xyzs = cat.get("pre", np.empty((0, 3)))
        if len(xyzs) > 0:
            j = None
            try:
                if "pre_syn_index" in row and pd.notna(row["pre_syn_index"]):
                    jj = int(float(row["pre_syn_index"]))
                    if 0 <= jj < len(xyzs):
                        j = jj
            except Exception:
                j = None
            if j is None:
                j = _SYN_CURSOR.get(pre_id, 0) % len(xyzs)
            _SYN_CURSOR[pre_id] = j + 1
            x, y, z = map(float, xyzs[j])
            return cell.nearest_site(x, y, z)
    return cell.soma_site()

# ---------------- SWC resolver ----------------

_SWC_ID_PATHS_CACHE: Dict[str, Dict[int, list[Path]]] = {}
_SWC_FILE_MTIME_CACHE: Dict[str, float] = {}


def _extract_id_hint_from_swc_path(p: Path) -> Optional[int]:
    parent = p.parent.name
    if parent.isdigit():
        try:
            return int(parent)
        except Exception:
            pass

    stem = p.stem
    m = re.match(r"^(\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def _score_swc_candidate_for_nid(p: Path, nid: int) -> tuple[int, int]:
    name = p.name
    stem = p.stem
    parent_is_nid = p.parent.name == str(nid)
    score = 99

    if name == f"{nid}_axodendro_with_synapses.swc":
        score = 0 if parent_is_nid else 1
    elif (str(nid) in stem) and ("with_synapses" in stem):
        score = 2
    elif name == f"{nid}_healed_final.swc":
        score = 3 if parent_is_nid else 4
    elif name == f"{nid}_healed.swc":
        score = 5 if parent_is_nid else 6
    elif name == f"{nid}.swc":
        score = 7
    elif str(nid) in stem:
        score = 8
    return score, len(str(p))


def _build_swc_id_paths_index(swc_root: Path) -> Dict[int, list[Path]]:
    idx: Dict[int, list[Path]] = defaultdict(list)
    for p in swc_root.rglob("*.swc"):
        hid = _extract_id_hint_from_swc_path(p)
        if hid is None:
            continue
        idx[int(hid)].append(p)
    return idx


def _get_swc_id_paths_index(swc_root: Path) -> Dict[int, list[Path]]:
    key = str(swc_root.resolve())
    try:
        root_mtime = float(swc_root.stat().st_mtime)
    except Exception:
        root_mtime = -1.0
    cached_mtime = _SWC_FILE_MTIME_CACHE.get(key)
    cached = _SWC_ID_PATHS_CACHE.get(key)
    if cached is not None and cached_mtime == root_mtime:
        return cached
    idx = _build_swc_id_paths_index(swc_root)
    _SWC_ID_PATHS_CACHE[key] = idx
    _SWC_FILE_MTIME_CACHE[key] = root_mtime
    return idx


def _find_swc_via_index(swc_root: Path, nid: int) -> Optional[str]:
    idx = _get_swc_id_paths_index(swc_root)
    cands = idx.get(int(nid), [])
    if not cands:
        return None
    best = min(cands, key=lambda p: _score_swc_candidate_for_nid(p, int(nid)))
    return str(best)

def find_swc(swc_root: str | Path, nid: int) -> str:
    """Pick the best SWC for nid (synapses→healed→generic)."""
    nid = int(nid)
    p = Path(swc_root)
    if p.is_file() and p.suffix.lower() == ".swc":
        if str(nid) in p.name:
            return str(p)
        p = p.parent

    fast = _find_swc_via_index(p, nid)
    if fast is not None:
        return fast

    patterns = [
        f"**/{nid}/{nid}_axodendro_with_synapses.swc",
        f"**/{nid}_axodendro_with_synapses.swc",
        f"**/*{nid}*with_synapses*.swc",
        f"**/{nid}/{nid}_healed_final.swc",
        f"**/{nid}_healed_final.swc",
        f"**/{nid}/{nid}_healed.swc",
        f"**/{nid}_healed.swc",
        f"**/{nid}.swc",
        f"**/*{nid}*.swc",
    ]
    for pat in patterns:
        hits = list(p.glob(pat))
        if hits:
            hits.sort(key=lambda q: len(str(q)))
            return str(hits[0])
    raise FileNotFoundError(f"No SWC for id={nid} under {p}")


def find_swc_with_fallback(primary_root: str | Path | None, fallback_root: str | Path, nid: int) -> str:
    """Resolve an SWC from the primary root and fall back to the base export root if needed."""
    if primary_root not in (None, "", False):
        try:
            return find_swc(primary_root, nid)
        except FileNotFoundError:
            pass
    return find_swc(fallback_root, nid)

# ---------------- SWC parsing + auto-repair ----------------

def _load_swc_rows(path: str | Path) -> list[list[float]]:
    """Parse SWC into rows [[id,type,x,y,z,r,parent], ...]."""
    rows: list[list[float]] = []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\s+", s)
            if len(parts) < 7:
                continue
            n = int(float(parts[0]))
            typ = int(float(parts[1]))
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            r = float(parts[5])
            parent = int(float(parts[6]))
            rows.append([n, typ, x, y, z, r, parent])
    if not rows:
        raise RuntimeError(f"Empty/invalid SWC: {path}")
    rows.sort(key=lambda t: t[0])
    return rows

def _auto_repair(rows: list[list[float]]):
    """
    Auto-fix:
      - nm→µm scaling if medians look nanometer-scale
      - choose soma hint: widest type==1; else widest root; else widest node
    """
    nodes = {
        int(n): dict(type=int(t), x=float(x), y=float(y), z=float(z), r=float(r), parent=int(parent))
        for n, t, x, y, z, r, parent in rows
    }

    segs, diams = [], []
    for n, rec in nodes.items():
        diams.append(2.0 * rec["r"])
        p = rec["parent"]
        if p in nodes:
            xp, yp, zp = nodes[p]["x"], nodes[p]["y"], nodes[p]["z"]
            segs.append(((rec["x"] - xp) ** 2 + (rec["y"] - yp) ** 2 + (rec["z"] - zp) ** 2) ** 0.5)

    med_seg = float(np.median(segs)) if segs else 0.0
    med_diam = float(np.median(diams)) if diams else 0.0
    scale_pos = 1000.0 if (med_seg > 100.0 or med_diam > 100.0) else 1.0

    if scale_pos != 1.0:
        for r in nodes.values():
            r["x"] /= scale_pos
            r["y"] /= scale_pos
            r["z"] /= scale_pos
            r["r"] /= scale_pos

    soma_ids = [n for n, r in nodes.items() if int(r.get("type", 0)) == 1]
    if soma_ids:
        soma_id_hint = max(soma_ids, key=lambda n: nodes[n]["r"])
    else:
        roots = [n for n, r in nodes.items() if r["parent"] == -1 or r["parent"] not in nodes]
        soma_id_hint = max(roots, key=lambda n: nodes[n]["r"]) if roots else max(nodes, key=lambda n: nodes[n]["r"])

    fixed = [
        [int(n), int(nodes[n]["type"]), float(nodes[n]["x"]), float(nodes[n]["y"]), float(nodes[n]["z"]), float(nodes[n]["r"]), int(nodes[n]["parent"])]
        for n in sorted(nodes.keys())
    ]
    return fixed, scale_pos, int(soma_id_hint)

def _rebuild_pt3d_cache(cell: "SWCCell"):
    """Populate cell._cache_sec_pts: dict[sec] -> array[[x,y,z,arc], ...]."""
    cache = {}
    for sec in cell._secs:
        n3d = int(h.n3d(sec=sec))
        if n3d < 1:
            continue
        xs = [float(h.x3d(i, sec=sec)) for i in range(n3d)]
        ys = [float(h.y3d(i, sec=sec)) for i in range(n3d)]
        zs = [float(h.z3d(i, sec=sec)) for i in range(n3d)]
        if n3d >= 2:
            seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
        else:
            arc = [0.0]
        cache[sec] = np.column_stack([xs, ys, zs, arc])
    cell._cache_sec_pts = cache

class SWCCell:
    """
    Robust SWC→NEURON loader (no Import3d).
    One Section per node; geometry encoded with pt3d lines (parent→child).
    Soma: prefer widest type==1; else widest root; else widest node.
    """
    def __init__(self, gid: int, swc_path: str, cfg: Dict[str, Any]):
        self.gid = int(gid)
        self.swc_path = str(swc_path)
        self.cfg = dict(cfg)

        self.sections: Dict[str, Any] = {}
        self._secs: list[Any] = []
        self.soma_sec = None

        self._cache_sec_pts: Dict[Any, np.ndarray] = {}
        self._nodes: Dict[int, Dict[str, Any]] = {}
        self._children: Dict[int, list[int]] = defaultdict(list)
        self._sec_for_node: Dict[int, Any] = {}

        self._soma_id: Optional[int] = None

        # AIS cache fields (set by ais.attach_ais_methods)
        self._ais_site = None
        self._ais_verbose_once = False

        self._build()

    def _build(self):
        rows = _load_swc_rows(self.swc_path)
        rows, _, _ = _auto_repair(rows)
        nodes = {n: dict(type=t, x=x, y=y, z=z, r=r, parent=p) for n, t, x, y, z, r, p in rows}
        self._nodes = nodes

        for n, rec in nodes.items():
            p = rec["parent"]
            if p in nodes:
                self._children[p].append(n)

        sec_of: Dict[int, Any] = {}
        for n in nodes:
            sec = h.Section(name=f"cell{self.gid}_n{n}")
            sec.Ra = float(self.cfg.get("Ra", 100.0))
            sec.cm = float(self.cfg.get("cm", 1.0))
            sec_of[n] = sec
            self._secs.append(sec)
            self.sections[sec.name()] = sec
        self._sec_for_node = sec_of

        for n, rec in nodes.items():
            sec = sec_of[n]
            h.pt3dclear(sec=sec)
            p = rec["parent"]
            if p in nodes:
                xp, yp, zp, rp = nodes[p]["x"], nodes[p]["y"], nodes[p]["z"], nodes[p]["r"]
                x, y, z, r = rec["x"], rec["y"], rec["z"], rec["r"]
                h.pt3dadd(xp, yp, zp, max(2.0 * rp, 1e-3), sec=sec)
                h.pt3dadd(x, y, z, max(2.0 * r, 1e-3), sec=sec)
                sec.connect(sec_of[p](1.0))
            else:
                x, y, z, r = rec["x"], rec["y"], rec["z"], rec["r"]
                h.pt3dadd(x, y, z, max(2.0 * r, 1e-3), sec=sec)
                h.pt3dadd(x + 1e-3, y, z, max(2.0 * r, 1e-3), sec=sec)

        try:
            h.define_shape()
        except Exception:
            pass

        _rebuild_pt3d_cache(self)

        soma_ids = [n for n, r in nodes.items() if int(r.get("type", 0)) == 1]
        if soma_ids:
            soma_id = max(soma_ids, key=lambda n: nodes[n]["r"])
        else:
            roots = [n for n, r in nodes.items() if r["parent"] not in nodes or r["parent"] == -1]
            soma_id = max(roots, key=lambda n: nodes[n]["r"]) if roots else max(nodes, key=lambda n: nodes[n]["r"])
        self._soma_id = int(soma_id)
        self.soma_sec = sec_of.get(self._soma_id, self._secs[0])

    def soma_site(self):
        return self.soma_sec, 0.5

    def nearest_site(self, x, y, z):
        best = (None, 0.5, 1e18)
        tgt = np.array([float(x), float(y), float(z)], dtype=float)
        for sec, pts in self._cache_sec_pts.items():
            if pts.shape[0] < 2:
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
        return best[0], best[1]
