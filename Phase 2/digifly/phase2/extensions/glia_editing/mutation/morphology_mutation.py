from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SWC_COLUMNS = ["id", "type", "x", "y", "z", "r", "parent"]


@dataclass
class MutationOperation:
    timestamp_utc: str
    neuron_id: int
    op: str
    params: Dict[str, Any]
    changed_node_ids: List[int]


@dataclass
class MutationConnection:
    pre_neuron_id: int
    pre_node_id: int
    post_neuron_id: int
    post_node_id: int
    chemical_synapses: int = 1
    gap_junctions: int = 0
    gap_mode: str = "none"  # none|non_rectifying|rectifying
    gap_direction: Optional[str] = None  # a_to_b|b_to_a|bidirectional
    note: str = ""


@dataclass
class MutationBiophysPolicy:
    neuron_id: int
    passive_node_ids: List[int]
    active_node_ids: List[int]
    note: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def find_swc_path(swc_root: str | Path, neuron_id: int) -> Path:
    swc_root = Path(swc_root).expanduser().resolve()
    nid = int(neuron_id)

    try:
        from digifly.phase2.neuron_build.swc_cell import find_swc  # type: ignore

        return Path(find_swc(swc_root, nid)).expanduser().resolve()
    except Exception:
        pass

    patterns = [
        f"**/{nid}/{nid}_axodendro_with_synapses.swc",
        f"**/{nid}_axodendro_with_synapses.swc",
        f"**/*{nid}*with_synapses*.swc",
        f"**/{nid}_healed_final.swc",
        f"**/{nid}_healed.swc",
        f"**/{nid}.swc",
        f"**/*{nid}*.swc",
    ]
    for pat in patterns:
        hits = sorted(swc_root.glob(pat), key=lambda p: (len(str(p)), str(p)))
        if hits:
            return hits[0].resolve()

    raise FileNotFoundError(f"No SWC found for neuron_id={nid} under {swc_root}")



def load_swc_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    [
                        int(float(parts[0])),
                        int(float(parts[1])),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        int(float(parts[6])),
                    ]
                )
            except Exception:
                continue

    if not rows:
        raise RuntimeError(f"Empty/invalid SWC: {path}")

    df = pd.DataFrame(rows, columns=SWC_COLUMNS)
    df = df.sort_values("id").reset_index(drop=True)
    return _normalize_swc_df(df)



def write_swc_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    header_lines: Optional[Sequence[str]] = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ndf = _normalize_swc_df(df)
    with path.open("w", encoding="utf-8") as f:
        if header_lines:
            for line in header_lines:
                f.write(f"# {line}\n")
        for _, r in ndf.iterrows():
            f.write(
                f"{int(r['id'])} {int(r['type'])} "
                f"{float(r['x']):.6f} {float(r['y']):.6f} {float(r['z']):.6f} "
                f"{float(r['r']):.6f} {int(r['parent'])}\n"
            )
    return path



def _normalize_swc_df(df: pd.DataFrame) -> pd.DataFrame:
    ndf = df.copy()
    missing = [c for c in SWC_COLUMNS if c not in ndf.columns]
    if missing:
        raise ValueError(f"SWC table missing required columns: {missing}")
    ndf = ndf[SWC_COLUMNS].copy()
    ndf["id"] = pd.to_numeric(ndf["id"], errors="coerce").astype("Int64")
    ndf["type"] = pd.to_numeric(ndf["type"], errors="coerce").fillna(3).astype("Int64")
    for c in ["x", "y", "z", "r"]:
        ndf[c] = pd.to_numeric(ndf[c], errors="coerce").astype(float)
    ndf["parent"] = pd.to_numeric(ndf["parent"], errors="coerce").fillna(-1).astype("Int64")

    if ndf["id"].isna().any():
        raise ValueError("SWC table has non-numeric/NaN node ids")
    if ndf[["x", "y", "z", "r"]].isna().any().any():
        raise ValueError("SWC table has NaNs in geometry/radius columns")

    ndf["id"] = ndf["id"].astype(int)
    ndf["type"] = ndf["type"].astype(int)
    ndf["parent"] = ndf["parent"].astype(int)
    ndf = ndf.sort_values("id").reset_index(drop=True)
    return ndf



def _id_set(df: pd.DataFrame) -> set[int]:
    return set(int(x) for x in df["id"].tolist())



def _children_map(df: pd.DataFrame) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {int(n): [] for n in df["id"].tolist()}
    for _, r in df.iterrows():
        nid = int(r["id"])
        parent = int(r["parent"])
        if parent in out:
            out[parent].append(nid)
    return out



def subtree_node_ids(df: pd.DataFrame, start_node_id: int) -> List[int]:
    start = int(start_node_id)
    kids = _children_map(df)
    if start not in kids:
        return []
    out: List[int] = []
    stack = [start]
    seen: set[int] = set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        stack.extend(kids.get(nid, []))
    return out



def connected_components(df: pd.DataFrame) -> List[List[int]]:
    ndf = _normalize_swc_df(df)
    ids = _id_set(ndf)
    adj: Dict[int, List[int]] = {nid: [] for nid in ids}
    for _, r in ndf.iterrows():
        nid = int(r["id"])
        parent = int(r["parent"])
        if parent in adj:
            adj[nid].append(parent)
            adj[parent].append(nid)

    out: List[List[int]] = []
    seen: set[int] = set()
    for nid in sorted(ids):
        if nid in seen:
            continue
        comp: List[int] = []
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj.get(cur, []))
        out.append(sorted(comp))
    return out



def _cycle_nodes(df: pd.DataFrame) -> List[int]:
    ndf = _normalize_swc_df(df)
    parent_of = {int(r["id"]): int(r["parent"]) for _, r in ndf.iterrows()}
    ids = set(parent_of.keys())

    state: Dict[int, int] = {nid: 0 for nid in ids}  # 0=unseen,1=visiting,2=done
    cycle_hits: set[int] = set()

    for nid in ids:
        if state[nid] != 0:
            continue
        cur = nid
        stack: List[int] = []
        index_in_stack: Dict[int, int] = {}

        while True:
            if cur not in ids:
                for v in stack:
                    state[v] = 2
                break
            if state[cur] == 2:
                for v in stack:
                    state[v] = 2
                break
            if cur in index_in_stack:
                loop_start = index_in_stack[cur]
                cycle_hits.update(stack[loop_start:])
                for v in stack:
                    state[v] = 2
                break
            if state[cur] == 1:
                for v in stack:
                    state[v] = 2
                break

            state[cur] = 1
            index_in_stack[cur] = len(stack)
            stack.append(cur)
            nxt = parent_of.get(cur, -1)
            if nxt == -1:
                for v in stack:
                    state[v] = 2
                break
            cur = int(nxt)

    return sorted(int(x) for x in cycle_hits)



def validate_swc_table(df: pd.DataFrame, *, require_single_component: bool = True) -> Dict[str, Any]:
    ndf = _normalize_swc_df(df)

    node_ids = [int(x) for x in ndf["id"].tolist()]
    id_set = set(node_ids)
    duplicates = sorted(int(x) for x in ndf[ndf["id"].duplicated()]["id"].tolist())

    roots = sorted(int(r["id"]) for _, r in ndf.iterrows() if int(r["parent"]) == -1)
    missing_parent_rows = ndf[(ndf["parent"] != -1) & (~ndf["parent"].isin(list(id_set)))]
    missing_parent_nodes = sorted(int(x) for x in missing_parent_rows["id"].tolist())
    negative_radius_nodes = sorted(int(r["id"]) for _, r in ndf.iterrows() if float(r["r"]) <= 0.0)

    cycles = _cycle_nodes(ndf)
    comps = connected_components(ndf)

    is_valid = True
    if duplicates:
        is_valid = False
    if missing_parent_nodes:
        is_valid = False
    if cycles:
        is_valid = False
    if negative_radius_nodes:
        is_valid = False
    if require_single_component and len(comps) != 1:
        is_valid = False

    return {
        "valid": bool(is_valid),
        "n_nodes": int(len(ndf)),
        "n_roots": int(len(roots)),
        "root_node_ids": roots,
        "n_components": int(len(comps)),
        "component_sizes": [int(len(c)) for c in comps],
        "duplicate_node_ids": duplicates,
        "missing_parent_node_ids": missing_parent_nodes,
        "negative_or_zero_radius_node_ids": negative_radius_nodes,
        "cycle_node_ids": cycles,
    }



def _next_node_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(np.max(pd.to_numeric(df["id"], errors="coerce").to_numpy(dtype=float))) + 1



def _require_nodes_exist(df: pd.DataFrame, node_ids: Iterable[int]) -> None:
    ids = _id_set(df)
    missing = sorted(int(n) for n in set(int(x) for x in node_ids) if int(n) not in ids)
    if missing:
        raise KeyError(f"Node ids not found: {missing}")



def scale_radii(
    df: pd.DataFrame,
    node_ids: Sequence[int],
    *,
    factor: float,
    include_subtree: bool = False,
    min_radius_um: float = 0.01,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    seeds = [int(x) for x in node_ids]
    _require_nodes_exist(ndf, seeds)

    target: set[int] = set()
    if include_subtree:
        for nid in seeds:
            target.update(subtree_node_ids(ndf, nid))
    else:
        target.update(seeds)

    mask = ndf["id"].isin(sorted(target))
    ndf.loc[mask, "r"] = np.maximum(float(min_radius_um), ndf.loc[mask, "r"] * float(factor))
    return ndf, sorted(target)



def translate_nodes(
    df: pd.DataFrame,
    node_ids: Sequence[int],
    *,
    dx_um: float,
    dy_um: float,
    dz_um: float,
    include_subtree: bool = False,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    seeds = [int(x) for x in node_ids]
    _require_nodes_exist(ndf, seeds)

    target: set[int] = set()
    if include_subtree:
        for nid in seeds:
            target.update(subtree_node_ids(ndf, nid))
    else:
        target.update(seeds)

    mask = ndf["id"].isin(sorted(target))
    ndf.loc[mask, "x"] = ndf.loc[mask, "x"] + float(dx_um)
    ndf.loc[mask, "y"] = ndf.loc[mask, "y"] + float(dy_um)
    ndf.loc[mask, "z"] = ndf.loc[mask, "z"] + float(dz_um)
    return ndf, sorted(target)



def split_edges(
    df: pd.DataFrame,
    child_node_ids: Sequence[int],
    *,
    frac: float = 0.5,
    radius_scale: float = 1.0,
    node_type: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    frac = float(np.clip(float(frac), 1e-6, 1.0 - 1e-6))
    kids = [int(x) for x in child_node_ids]
    _require_nodes_exist(ndf, kids)

    id_to_row = {int(r["id"]): r for _, r in ndf.iterrows()}
    new_rows: List[Dict[str, Any]] = []
    inserted: List[int] = []
    next_id = _next_node_id(ndf)

    for child in kids:
        crow = id_to_row[int(child)]
        parent = int(crow["parent"])
        if parent == -1 or parent not in id_to_row:
            continue

        prow = id_to_row[parent]
        new_id = int(next_id)
        next_id += 1

        px, py, pz = float(prow["x"]), float(prow["y"]), float(prow["z"])
        cx, cy, cz = float(crow["x"]), float(crow["y"]), float(crow["z"])
        nx = px + frac * (cx - px)
        ny = py + frac * (cy - py)
        nz = pz + frac * (cz - pz)

        rr = max(0.01, float(crow["r"]) * float(radius_scale))
        tt = int(node_type) if node_type is not None else int(crow["type"])

        new_rows.append(
            {
                "id": new_id,
                "type": tt,
                "x": nx,
                "y": ny,
                "z": nz,
                "r": rr,
                "parent": parent,
            }
        )
        inserted.append(new_id)
        ndf.loc[ndf["id"] == child, "parent"] = new_id

    if new_rows:
        ndf = pd.concat([ndf, pd.DataFrame(new_rows)], ignore_index=True)
    return _normalize_swc_df(ndf), sorted(inserted)



def _node_xyz(df: pd.DataFrame, node_id: int) -> np.ndarray:
    row = df.loc[df["id"] == int(node_id)]
    if row.empty:
        raise KeyError(f"node id not found: {node_id}")
    r = row.iloc[0]
    return np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)



def _infer_tangent(df: pd.DataFrame, parent_id: int) -> np.ndarray:
    ndf = _normalize_swc_df(df)
    id_to_parent = {int(r["id"]): int(r["parent"]) for _, r in ndf.iterrows()}
    pid = int(parent_id)

    pxyz = _node_xyz(ndf, pid)
    parent_of_parent = int(id_to_parent.get(pid, -1))
    if parent_of_parent != -1 and parent_of_parent in _id_set(ndf):
        v = pxyz - _node_xyz(ndf, parent_of_parent)
    else:
        kids = ndf.loc[ndf["parent"] == pid, "id"].astype(int).tolist()
        if kids:
            v = _node_xyz(ndf, int(kids[0])) - pxyz
        else:
            v = np.array([1.0, 0.0, 0.0], dtype=float)

    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / norm



def grow_branch_to_point(
    df: pd.DataFrame,
    *,
    parent_node_id: int,
    target_xyz_um: Sequence[float],
    segments: int = 4,
    node_type: Optional[int] = None,
    radius_scale: float = 0.85,
    absolute_radius_um: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    pid = int(parent_node_id)
    _require_nodes_exist(ndf, [pid])

    segments = max(1, int(segments))
    target = np.asarray(target_xyz_um, dtype=float).reshape(3)

    prow = ndf.loc[ndf["id"] == pid].iloc[0]
    start = np.array([float(prow["x"]), float(prow["y"]), float(prow["z"])], dtype=float)
    ptype = int(prow["type"])
    pr = float(prow["r"])

    step = (target - start) / float(segments)
    next_id = _next_node_id(ndf)
    new_rows: List[Dict[str, Any]] = []
    new_ids: List[int] = []

    prev = pid
    for i in range(segments):
        nid = int(next_id)
        next_id += 1
        alpha = float(i + 1) / float(segments)
        xyz = start + step * float(i + 1)

        if absolute_radius_um is not None:
            rr = float(max(0.01, float(absolute_radius_um)))
        else:
            rr = float(max(0.01, pr * float(radius_scale) * (1.0 - 0.35 * alpha)))

        new_rows.append(
            {
                "id": nid,
                "type": int(node_type) if node_type is not None else ptype,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "r": rr,
                "parent": int(prev),
            }
        )
        prev = nid
        new_ids.append(nid)

    out = pd.concat([ndf, pd.DataFrame(new_rows)], ignore_index=True)
    return _normalize_swc_df(out), new_ids



def grow_branch_along_tangent(
    df: pd.DataFrame,
    *,
    parent_node_id: int,
    length_um: float,
    segments: int = 4,
    node_type: Optional[int] = None,
    radius_scale: float = 0.85,
    absolute_radius_um: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    pid = int(parent_node_id)
    _require_nodes_exist(ndf, [pid])

    start = _node_xyz(ndf, pid)
    direction = _infer_tangent(ndf, pid)
    target = start + direction * float(length_um)
    return grow_branch_to_point(
        ndf,
        parent_node_id=pid,
        target_xyz_um=target,
        segments=int(segments),
        node_type=node_type,
        radius_scale=float(radius_scale),
        absolute_radius_um=absolute_radius_um,
    )



def detach_nodes(df: pd.DataFrame, node_ids: Sequence[int]) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    ids = [int(x) for x in node_ids]
    _require_nodes_exist(ndf, ids)
    ndf.loc[ndf["id"].isin(ids), "parent"] = -1
    return ndf, sorted(ids)



def reparent_nodes(
    df: pd.DataFrame,
    node_parent_pairs: Sequence[Tuple[int, int]],
    *,
    allow_cycles: bool = False,
) -> Tuple[pd.DataFrame, List[int]]:
    ndf = _normalize_swc_df(df)
    if not node_parent_pairs:
        return ndf, []

    ids = _id_set(ndf)
    changed: List[int] = []

    for node_id, new_parent in node_parent_pairs:
        nid = int(node_id)
        pid = int(new_parent)
        if nid not in ids:
            raise KeyError(f"node id not found: {nid}")
        if pid not in ids:
            raise KeyError(f"parent id not found: {pid}")
        if nid == pid:
            raise ValueError("node cannot be parent of itself")

        if not allow_cycles:
            descendants = set(subtree_node_ids(ndf, nid))
            if pid in descendants:
                raise ValueError(
                    f"Reparent would create a cycle: node={nid}, new_parent={pid} is in subtree(node)."
                )

        ndf.loc[ndf["id"] == nid, "parent"] = pid
        changed.append(nid)

    return _normalize_swc_df(ndf), sorted(set(changed))



class MorphologyMutationProject:
    def __init__(self, swc_root: str | Path, neuron_ids: Sequence[int]):
        self.swc_root = Path(swc_root).expanduser().resolve()
        self.neuron_ids = [int(x) for x in neuron_ids]
        self.base_paths: Dict[int, Path] = {}
        self.tables: Dict[int, pd.DataFrame] = {}
        self.operations: List[MutationOperation] = []
        self.connections: List[MutationConnection] = []
        self.biophys_policies: Dict[int, Dict[str, set[int]]] = {
            int(nid): {"passive": set(), "active": set()} for nid in self.neuron_ids
        }

        seen: set[int] = set()
        for nid in self.neuron_ids:
            if nid in seen:
                continue
            seen.add(nid)
            swc_path = find_swc_path(self.swc_root, nid)
            self.base_paths[nid] = swc_path
            self.tables[nid] = load_swc_table(swc_path)

        self.neuron_ids = sorted(self.tables.keys())

    @classmethod
    def from_neuron_ids(cls, swc_root: str | Path, neuron_ids: Sequence[int]) -> "MorphologyMutationProject":
        return cls(swc_root, neuron_ids)

    def table(self, neuron_id: int) -> pd.DataFrame:
        nid = int(neuron_id)
        if nid not in self.tables:
            raise KeyError(f"Neuron id not loaded: {nid}")
        return self.tables[nid]

    def _record_op(self, neuron_id: int, op: str, params: Dict[str, Any], changed: Sequence[int]) -> None:
        self.operations.append(
            MutationOperation(
                timestamp_utc=_utc_now(),
                neuron_id=int(neuron_id),
                op=str(op),
                params=dict(params),
                changed_node_ids=sorted(int(x) for x in set(changed)),
            )
        )

    def apply_scale_radii(
        self,
        neuron_id: int,
        node_ids: Sequence[int],
        *,
        factor: float,
        include_subtree: bool = False,
        min_radius_um: float = 0.01,
    ) -> List[int]:
        nid = int(neuron_id)
        out, changed = scale_radii(
            self.table(nid),
            node_ids,
            factor=float(factor),
            include_subtree=bool(include_subtree),
            min_radius_um=float(min_radius_um),
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "scale_radii",
            {
                "factor": float(factor),
                "include_subtree": bool(include_subtree),
                "min_radius_um": float(min_radius_um),
                "node_ids": [int(x) for x in node_ids],
            },
            changed,
        )
        return changed

    def apply_translate(
        self,
        neuron_id: int,
        node_ids: Sequence[int],
        *,
        dx_um: float,
        dy_um: float,
        dz_um: float,
        include_subtree: bool = False,
    ) -> List[int]:
        nid = int(neuron_id)
        out, changed = translate_nodes(
            self.table(nid),
            node_ids,
            dx_um=float(dx_um),
            dy_um=float(dy_um),
            dz_um=float(dz_um),
            include_subtree=bool(include_subtree),
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "translate_nodes",
            {
                "dx_um": float(dx_um),
                "dy_um": float(dy_um),
                "dz_um": float(dz_um),
                "include_subtree": bool(include_subtree),
                "node_ids": [int(x) for x in node_ids],
            },
            changed,
        )
        return changed

    def apply_split_edges(
        self,
        neuron_id: int,
        child_node_ids: Sequence[int],
        *,
        frac: float = 0.5,
        radius_scale: float = 1.0,
        node_type: Optional[int] = None,
    ) -> List[int]:
        nid = int(neuron_id)
        out, inserted = split_edges(
            self.table(nid),
            child_node_ids,
            frac=float(frac),
            radius_scale=float(radius_scale),
            node_type=node_type,
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "split_edges",
            {
                "frac": float(frac),
                "radius_scale": float(radius_scale),
                "node_type": int(node_type) if node_type is not None else None,
                "child_node_ids": [int(x) for x in child_node_ids],
            },
            inserted,
        )
        return inserted

    def apply_grow_branch_along_tangent(
        self,
        neuron_id: int,
        *,
        parent_node_id: int,
        length_um: float,
        segments: int = 4,
        node_type: Optional[int] = None,
        radius_scale: float = 0.85,
        absolute_radius_um: Optional[float] = None,
    ) -> List[int]:
        nid = int(neuron_id)
        out, new_ids = grow_branch_along_tangent(
            self.table(nid),
            parent_node_id=int(parent_node_id),
            length_um=float(length_um),
            segments=int(segments),
            node_type=node_type,
            radius_scale=float(radius_scale),
            absolute_radius_um=absolute_radius_um,
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "grow_branch_along_tangent",
            {
                "parent_node_id": int(parent_node_id),
                "length_um": float(length_um),
                "segments": int(segments),
                "node_type": int(node_type) if node_type is not None else None,
                "radius_scale": float(radius_scale),
                "absolute_radius_um": float(absolute_radius_um)
                if absolute_radius_um is not None
                else None,
            },
            new_ids,
        )
        return new_ids

    def apply_grow_branch_to_point(
        self,
        neuron_id: int,
        *,
        parent_node_id: int,
        target_xyz_um: Sequence[float],
        segments: int = 4,
        node_type: Optional[int] = None,
        radius_scale: float = 0.85,
        absolute_radius_um: Optional[float] = None,
    ) -> List[int]:
        nid = int(neuron_id)
        out, new_ids = grow_branch_to_point(
            self.table(nid),
            parent_node_id=int(parent_node_id),
            target_xyz_um=target_xyz_um,
            segments=int(segments),
            node_type=node_type,
            radius_scale=float(radius_scale),
            absolute_radius_um=absolute_radius_um,
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "grow_branch_to_point",
            {
                "parent_node_id": int(parent_node_id),
                "target_xyz_um": [float(x) for x in target_xyz_um],
                "segments": int(segments),
                "node_type": int(node_type) if node_type is not None else None,
                "radius_scale": float(radius_scale),
                "absolute_radius_um": float(absolute_radius_um)
                if absolute_radius_um is not None
                else None,
            },
            new_ids,
        )
        return new_ids

    def apply_detach(self, neuron_id: int, node_ids: Sequence[int]) -> List[int]:
        nid = int(neuron_id)
        out, changed = detach_nodes(self.table(nid), node_ids)
        self.tables[nid] = out
        self._record_op(
            nid,
            "detach_nodes",
            {"node_ids": [int(x) for x in node_ids]},
            changed,
        )
        return changed

    def apply_reparent(
        self,
        neuron_id: int,
        node_parent_pairs: Sequence[Tuple[int, int]],
        *,
        allow_cycles: bool = False,
    ) -> List[int]:
        nid = int(neuron_id)
        out, changed = reparent_nodes(
            self.table(nid), node_parent_pairs, allow_cycles=bool(allow_cycles)
        )
        self.tables[nid] = out
        self._record_op(
            nid,
            "reparent_nodes",
            {
                "node_parent_pairs": [[int(a), int(b)] for a, b in node_parent_pairs],
                "allow_cycles": bool(allow_cycles),
            },
            changed,
        )
        return changed

    def add_connection(
        self,
        *,
        pre_neuron_id: int,
        pre_node_id: int,
        post_neuron_id: int,
        post_node_id: int,
        chemical_synapses: int = 1,
        gap_junctions: int = 0,
        gap_mode: str = "none",
        gap_direction: Optional[str] = None,
        note: str = "",
    ) -> MutationConnection:
        pre_n = int(pre_neuron_id)
        post_n = int(post_neuron_id)
        if pre_n not in self.tables:
            raise KeyError(f"pre_neuron_id not loaded: {pre_n}")
        if post_n not in self.tables:
            raise KeyError(f"post_neuron_id not loaded: {post_n}")

        _require_nodes_exist(self.table(pre_n), [int(pre_node_id)])
        _require_nodes_exist(self.table(post_n), [int(post_node_id)])

        mode = str(gap_mode).strip().lower()
        if mode not in {"none", "non_rectifying", "rectifying"}:
            raise ValueError("gap_mode must be one of: none, non_rectifying, rectifying")

        conn = MutationConnection(
            pre_neuron_id=pre_n,
            pre_node_id=int(pre_node_id),
            post_neuron_id=post_n,
            post_node_id=int(post_node_id),
            chemical_synapses=max(0, int(chemical_synapses)),
            gap_junctions=max(0, int(gap_junctions)),
            gap_mode=mode,
            gap_direction=(str(gap_direction) if gap_direction is not None else None),
            note=str(note),
        )
        self.connections.append(conn)
        return conn

    def _policy_bucket(self, neuron_id: int) -> Dict[str, set[int]]:
        nid = int(neuron_id)
        if nid not in self.tables:
            raise KeyError(f"Neuron id not loaded: {nid}")
        if nid not in self.biophys_policies:
            self.biophys_policies[nid] = {"passive": set(), "active": set()}
        return self.biophys_policies[nid]

    def _expand_node_targets(
        self,
        neuron_id: int,
        node_ids: Sequence[int],
        *,
        include_subtree: bool = False,
    ) -> List[int]:
        nid = int(neuron_id)
        seeds = sorted(set(int(x) for x in node_ids))
        _require_nodes_exist(self.table(nid), seeds)
        if not include_subtree:
            return seeds
        out: set[int] = set()
        df = self.table(nid)
        for node_id in seeds:
            out.update(int(x) for x in subtree_node_ids(df, int(node_id)))
        return sorted(out)

    def set_nodes_passive(
        self,
        neuron_id: int,
        node_ids: Sequence[int],
        *,
        include_subtree: bool = False,
        replace: bool = False,
        note: str = "",
    ) -> List[int]:
        nid = int(neuron_id)
        target = self._expand_node_targets(nid, node_ids, include_subtree=bool(include_subtree))
        bucket = self._policy_bucket(nid)

        if replace:
            bucket["passive"] = set(int(x) for x in target)
            bucket["active"] = set(int(x) for x in bucket.get("active", set()) if int(x) not in bucket["passive"])
        else:
            bucket["passive"].update(int(x) for x in target)
            bucket["active"].difference_update(int(x) for x in target)

        self._record_op(
            nid,
            "set_nodes_passive",
            {
                "node_ids": [int(x) for x in node_ids],
                "resolved_node_ids": [int(x) for x in target],
                "include_subtree": bool(include_subtree),
                "replace": bool(replace),
                "note": str(note),
            },
            target,
        )
        return target

    def set_nodes_active(
        self,
        neuron_id: int,
        node_ids: Sequence[int],
        *,
        include_subtree: bool = False,
        replace: bool = False,
        note: str = "",
    ) -> List[int]:
        nid = int(neuron_id)
        target = self._expand_node_targets(nid, node_ids, include_subtree=bool(include_subtree))
        bucket = self._policy_bucket(nid)

        if replace:
            bucket["active"] = set(int(x) for x in target)
            bucket["passive"] = set(int(x) for x in bucket.get("passive", set()) if int(x) not in bucket["active"])
        else:
            bucket["active"].update(int(x) for x in target)
            bucket["passive"].difference_update(int(x) for x in target)

        self._record_op(
            nid,
            "set_nodes_active",
            {
                "node_ids": [int(x) for x in node_ids],
                "resolved_node_ids": [int(x) for x in target],
                "include_subtree": bool(include_subtree),
                "replace": bool(replace),
                "note": str(note),
            },
            target,
        )
        return target

    def set_nodes_biophys_by_type(
        self,
        neuron_id: int,
        *,
        passive_types: Optional[Sequence[int]] = None,
        active_types: Optional[Sequence[int]] = None,
        replace: bool = True,
        note: str = "",
    ) -> Dict[str, List[int]]:
        nid = int(neuron_id)
        df = self.table(nid)

        pset = set(int(x) for x in (passive_types or []))
        aset = set(int(x) for x in (active_types or []))

        passive_ids: List[int] = []
        active_ids: List[int] = []

        if pset:
            passive_ids = sorted(int(x) for x in df.loc[df["type"].isin(sorted(pset)), "id"].astype(int).tolist())
        if aset:
            active_ids = sorted(int(x) for x in df.loc[df["type"].isin(sorted(aset)), "id"].astype(int).tolist())

        bucket = self._policy_bucket(nid)
        if replace:
            bucket["passive"].clear()
            bucket["active"].clear()

        bucket["passive"].update(int(x) for x in passive_ids)
        bucket["active"].update(int(x) for x in active_ids)
        # Active policy wins on overlaps.
        bucket["passive"].difference_update(bucket["active"])

        changed = sorted(set(passive_ids).union(active_ids))
        self._record_op(
            nid,
            "set_nodes_biophys_by_type",
            {
                "passive_types": sorted(int(x) for x in pset),
                "active_types": sorted(int(x) for x in aset),
                "replace": bool(replace),
                "note": str(note),
                "resolved_passive_node_ids": sorted(int(x) for x in bucket["passive"]),
                "resolved_active_node_ids": sorted(int(x) for x in bucket["active"]),
            },
            changed,
        )

        return {
            "passive_node_ids": sorted(int(x) for x in bucket["passive"]),
            "active_node_ids": sorted(int(x) for x in bucket["active"]),
        }

    def clear_nodes_biophys_policy(
        self,
        neuron_id: int,
        node_ids: Optional[Sequence[int]] = None,
        *,
        include_subtree: bool = False,
        clear_all: bool = False,
        note: str = "",
    ) -> List[int]:
        nid = int(neuron_id)
        bucket = self._policy_bucket(nid)

        if clear_all:
            changed = sorted(set(int(x) for x in bucket["passive"]).union(int(x) for x in bucket["active"]))
            bucket["passive"].clear()
            bucket["active"].clear()
        else:
            if node_ids is None:
                raise ValueError("node_ids is required when clear_all=False")
            target = self._expand_node_targets(nid, node_ids, include_subtree=bool(include_subtree))
            changed = sorted(int(x) for x in target)
            bucket["passive"].difference_update(int(x) for x in target)
            bucket["active"].difference_update(int(x) for x in target)

        self._record_op(
            nid,
            "clear_nodes_biophys_policy",
            {
                "node_ids": [int(x) for x in (node_ids or [])],
                "include_subtree": bool(include_subtree),
                "clear_all": bool(clear_all),
                "note": str(note),
            },
            changed,
        )
        return changed

    def biophys_policy(self, neuron_id: int) -> MutationBiophysPolicy:
        nid = int(neuron_id)
        bucket = self._policy_bucket(nid)
        return MutationBiophysPolicy(
            neuron_id=int(nid),
            passive_node_ids=sorted(int(x) for x in bucket["passive"]),
            active_node_ids=sorted(int(x) for x in bucket["active"]),
        )

    def _biophys_policies_manifest(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for nid in sorted(self.tables.keys()):
            pol = self.biophys_policy(int(nid))
            out[str(int(nid))] = {
                "passive_node_ids": [int(x) for x in pol.passive_node_ids],
                "active_node_ids": [int(x) for x in pol.active_node_ids],
            }
        return out

    def validate_all(self, *, require_single_component: bool = True) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for nid in self.neuron_ids:
            out[int(nid)] = validate_swc_table(
                self.table(nid), require_single_component=bool(require_single_component)
            )
        return out

    def save_mutated_swcs(
        self,
        *,
        output_dir: str | Path,
        tag: str,
    ) -> Dict[int, Path]:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = str(tag).strip().replace(" ", "_")

        out: Dict[int, Path] = {}
        for nid in self.neuron_ids:
            path = out_dir / f"{int(nid)}_{safe_tag}.swc"
            header = [
                f"generated_utc={_utc_now()}",
                f"base_swc={self.base_paths[int(nid)]}",
                f"neuron_id={int(nid)}",
                f"tag={safe_tag}",
            ]
            write_swc_table(self.table(nid), path, header_lines=header)
            out[int(nid)] = path
        return out

    def save_phase2_overlay(
        self,
        *,
        overlay_dir: str | Path,
        mutated_paths: Dict[int, Path],
        symlink: bool = True,
    ) -> Path:
        ov = Path(overlay_dir).expanduser().resolve()
        ov.mkdir(parents=True, exist_ok=True)

        for nid in self.neuron_ids:
            src = Path(mutated_paths[int(nid)]).expanduser().resolve()
            dst = ov / f"{int(nid)}_axodendro_with_synapses.swc"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if symlink:
                try:
                    dst.symlink_to(src)
                except Exception:
                    dst.write_bytes(src.read_bytes())
            else:
                dst.write_bytes(src.read_bytes())
        return ov

    def export_bundle(
        self,
        *,
        output_root: str | Path,
        tag: str,
        require_single_component: bool = True,
        write_phase2_overlay: bool = True,
    ) -> Dict[str, Any]:
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_tag = str(tag).strip().replace(" ", "_")
        run_dir = root / f"morphology_mutation_{safe_tag}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        mutated_dir = run_dir / "mutated_swc"
        mutated_paths = self.save_mutated_swcs(output_dir=mutated_dir, tag=safe_tag)

        overlay_dir = None
        if write_phase2_overlay:
            overlay_dir = self.save_phase2_overlay(
                overlay_dir=run_dir / "phase2_morph_overlay",
                mutated_paths=mutated_paths,
                symlink=True,
            )

        validation = self.validate_all(require_single_component=bool(require_single_component))

        manifest = {
            "generated_utc": _utc_now(),
            "swc_root": str(self.swc_root),
            "tag": safe_tag,
            "neuron_ids": [int(x) for x in self.neuron_ids],
            "base_swcs": {str(k): str(v) for k, v in self.base_paths.items()},
            "mutated_swcs": {str(k): str(v) for k, v in mutated_paths.items()},
            "phase2_overlay_dir": str(overlay_dir) if overlay_dir is not None else None,
            "operations": [asdict(op) for op in self.operations],
            "connections": [asdict(c) for c in self.connections],
            "biophys_policies": self._biophys_policies_manifest(),
            "validation": {str(k): v for k, v in validation.items()},
        }

        manifest_path = run_dir / "morphology_mutation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        conn_json = run_dir / "mutation_connections.json"
        conn_json.write_text(
            json.dumps([asdict(c) for c in self.connections], indent=2),
            encoding="utf-8",
        )

        conn_csv = run_dir / "mutation_connections.csv"
        pd.DataFrame([asdict(c) for c in self.connections]).to_csv(conn_csv, index=False)

        biophys_json = run_dir / "mutation_biophys_policies.json"
        biophys_json.write_text(
            json.dumps(self._biophys_policies_manifest(), indent=2),
            encoding="utf-8",
        )

        val_json = run_dir / "mutation_validation.json"
        val_json.write_text(json.dumps(validation, indent=2), encoding="utf-8")

        return {
            "run_dir": str(run_dir),
            "manifest_json": str(manifest_path),
            "connections_json": str(conn_json),
            "connections_csv": str(conn_csv),
            "biophys_policies_json": str(biophys_json),
            "validation_json": str(val_json),
            "mutated_swcs": {int(k): str(v) for k, v in mutated_paths.items()},
            "phase2_overlay_dir": str(overlay_dir) if overlay_dir is not None else None,
        }
