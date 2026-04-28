"""digifly.tools.reduce_swc_dataset

Non-destructive SWC reduction for large morphology datasets.

Design goals:
  - Never mutate originals.
  - Keep topology (roots/branches/terminals) and soma nodes.
  - Optionally protect nodes nearest synapse coordinates.
  - Reduce linear chains using geometric/diameter thresholds.
  - Preserve relative folder structure in output root.

Example:
  python -m digifly.tools.reduce_swc_dataset \
      --input-root "/path/export_swc" \
      --output-root "/path/export_swc_reduced/v1" \
      --workers 8 \
      --protect-synapses
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReducerParams:
    max_path_um: float
    max_turn_deg: float
    max_diam_rel: float
    protect_synapses: bool
    max_syn_points: int
    write_map: bool
    overwrite: bool
    dry_run: bool


def _is_subpath(path: Path, maybe_parent: Path) -> bool:
    try:
        path.resolve().relative_to(maybe_parent.resolve())
        return True
    except Exception:
        return False


def _extract_first_int(text: str) -> Optional[int]:
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_neuron_id_from_swc_path(swc_path: Path) -> Optional[int]:
    # Prefer explicit leading id in common naming patterns.
    stem = swc_path.stem
    m = re.match(r"^(\d+)", stem)
    if m:
        return int(m.group(1))

    # Fall back to any integer in stem.
    any_id = _extract_first_int(stem)
    if any_id is not None:
        return any_id

    # Last attempt: parent directory name.
    return _extract_first_int(swc_path.parent.name)


def _iter_swc_files(input_root: Path, output_root: Path, match: Optional[str] = None) -> List[Path]:
    out: List[Path] = []
    for p in input_root.rglob("*.swc"):
        if _is_subpath(p, output_root):
            continue
        if match and match not in str(p):
            continue
        out.append(p)
    out.sort()
    return out


def _build_syn_csv_index(input_root: Path) -> Dict[int, str]:
    idx: Dict[int, str] = {}
    for p in input_root.rglob("*_synapses_new.csv"):
        nid = _extract_neuron_id_from_swc_path(p)
        if nid is None:
            continue
        s = str(p.resolve())
        prev = idx.get(nid)
        if prev is None or len(s) < len(prev):
            idx[nid] = s
    return idx


def _parse_swc(path: Path) -> List[List[float]]:
    rows: Dict[int, List[float]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\s+", s)
            if len(parts) < 7:
                continue
            try:
                n = int(float(parts[0]))
                typ = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                r = float(parts[5])
                parent = int(float(parts[6]))
            except Exception:
                continue
            rows[n] = [n, typ, x, y, z, r, parent]
    if not rows:
        raise RuntimeError(f"Empty/invalid SWC: {path}")
    return [rows[k] for k in sorted(rows.keys())]


def _write_swc(path: Path, rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for n, typ, x, y, z, r, parent in rows:
            f.write(
                f"{int(n)} {int(typ)} "
                f"{float(x):.6f} {float(y):.6f} {float(z):.6f} "
                f"{max(float(r), 1e-6):.6f} {int(parent)}\n"
            )


def _detect_and_scale_synmap(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("x", "y", "z"):
        if c not in df.columns:
            df[c] = np.nan
    xyz = df[["x", "y", "z"]].to_numpy(float)
    med_abs = np.nanmedian(np.abs(xyz))
    if np.isfinite(med_abs) and med_abs > 200.0:
        df[["x", "y", "z"]] *= 0.001
    return df


def _load_syn_points(syn_csv_path: Optional[str], max_syn_points: int) -> np.ndarray:
    if not syn_csv_path:
        return np.empty((0, 3), dtype=float)
    p = Path(syn_csv_path)
    if not p.exists():
        return np.empty((0, 3), dtype=float)
    try:
        df = pd.read_csv(p)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if "type" not in df.columns:
        df["type"] = ""
    for c in ("x", "y", "z"):
        if c not in df.columns:
            df[c] = np.nan
    df = _detect_and_scale_synmap(df)
    xyz = df.loc[df["type"].astype(str).str.lower().isin({"pre", "post"}), ["x", "y", "z"]].to_numpy(float)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    ok = np.isfinite(xyz).all(axis=1)
    xyz = xyz[ok]
    if xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    if max_syn_points > 0 and len(xyz) > max_syn_points:
        keep = np.linspace(0, len(xyz) - 1, max_syn_points, dtype=int)
        xyz = xyz[keep]
    return xyz


def _turn_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = b - a
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    cos_th = float(np.dot(u, v) / (nu * nv))
    cos_th = max(-1.0, min(1.0, cos_th))
    return float(math.degrees(math.acos(cos_th)))


def _nearest_node_ids(node_ids: np.ndarray, node_xyz: np.ndarray, points_xyz: np.ndarray) -> List[int]:
    out: List[int] = []
    if len(points_xyz) == 0 or len(node_xyz) == 0:
        return out
    chunk = 128
    for i in range(0, len(points_xyz), chunk):
        pts = points_xyz[i : i + chunk]  # (m, 3)
        # (m, n) pairwise squared distances, chunked to keep memory bounded.
        d2 = np.sum((pts[:, None, :] - node_xyz[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        out.extend(int(node_ids[int(j)]) for j in idx)
    return out


def _build_graph(rows: Sequence[Sequence[float]]):
    nodes: Dict[int, Dict[str, float]] = {}
    children: Dict[int, List[int]] = {}
    for n, typ, x, y, z, r, parent in rows:
        nid = int(n)
        nodes[nid] = {
            "type": int(typ),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "r": max(float(r), 1e-6),
            "parent": int(parent),
        }
        children[nid] = []
    for nid, rec in nodes.items():
        p = int(rec["parent"])
        if p in nodes:
            children[p].append(nid)
    for k in children:
        children[k].sort()
    roots = [nid for nid, rec in nodes.items() if int(rec["parent"]) not in nodes or int(rec["parent"]) == -1]
    roots.sort()
    if not roots:
        roots = [min(nodes.keys())]
    return nodes, children, roots


def _select_chain_keeps(
    parent_old: int,
    chain: Sequence[int],
    nodes: Dict[int, Dict[str, float]],
    protected: set[int],
    max_path_um: float,
    max_turn_deg: float,
    max_diam_rel: float,
) -> List[int]:
    keeps: List[int] = []
    p0 = np.array([nodes[parent_old]["x"], nodes[parent_old]["y"], nodes[parent_old]["z"]], dtype=float)
    prev_xyz = p0
    dist_since_keep = 0.0
    last_keep_r = float(nodes[parent_old]["r"])

    for i, oid in enumerate(chain):
        xyz = np.array([nodes[oid]["x"], nodes[oid]["y"], nodes[oid]["z"]], dtype=float)
        dist_since_keep += float(np.linalg.norm(xyz - prev_xyz))
        prev_xyz = xyz

        keep = False
        if oid in protected:
            keep = True
        if max_path_um > 0.0 and dist_since_keep >= max_path_um:
            keep = True

        r = float(nodes[oid]["r"])
        rel = abs(r - last_keep_r) / max(last_keep_r, 1e-6)
        if max_diam_rel > 0.0 and rel >= max_diam_rel:
            keep = True

        if max_turn_deg > 0.0 and i < len(chain) - 1:
            a_id = parent_old if i == 0 else chain[i - 1]
            c_id = chain[i + 1]
            a = np.array([nodes[a_id]["x"], nodes[a_id]["y"], nodes[a_id]["z"]], dtype=float)
            b = xyz
            c = np.array([nodes[c_id]["x"], nodes[c_id]["y"], nodes[c_id]["z"]], dtype=float)
            if _turn_angle_deg(a, b, c) >= max_turn_deg:
                keep = True

        if i == len(chain) - 1:
            keep = True

        if keep:
            keeps.append(oid)
            dist_since_keep = 0.0
            last_keep_r = r
    return keeps


def _reduce_rows(
    rows: Sequence[Sequence[float]],
    params: ReducerParams,
    syn_points_xyz: np.ndarray,
):
    nodes, children, roots = _build_graph(rows)
    node_ids = np.array(sorted(nodes.keys()), dtype=int)
    node_xyz = np.array([[nodes[n]["x"], nodes[n]["y"], nodes[n]["z"]] for n in node_ids], dtype=float)

    protected: set[int] = set()
    protected.update(roots)
    protected.update(n for n in nodes if len(children[n]) != 1)  # branch points + leaves
    protected.update(n for n, rec in nodes.items() if int(rec.get("type", 0)) == 1)  # soma class

    if params.max_diam_rel > 0.0:
        for n, rec in nodes.items():
            p = int(rec["parent"])
            if p not in nodes:
                continue
            pr = float(nodes[p]["r"])
            nr = float(rec["r"])
            rel = abs(nr - pr) / max(pr, 1e-6)
            if rel >= params.max_diam_rel:
                protected.add(n)
                protected.add(p)

    syn_protected = 0
    if params.protect_synapses and len(syn_points_xyz) > 0:
        nearest = _nearest_node_ids(node_ids, node_xyz, syn_points_xyz)
        syn_nodes = set(nearest)
        syn_protected = len(syn_nodes)
        protected.update(syn_nodes)

    new_rows: List[List[float]] = []
    kept_old_to_new: Dict[int, int] = {}
    old_to_rep_new: Dict[int, int] = {}

    def add_node(old_id: int, parent_new: int) -> int:
        rec = nodes[old_id]
        new_id = len(new_rows) + 1
        new_rows.append(
            [
                int(new_id),
                int(rec["type"]),
                float(rec["x"]),
                float(rec["y"]),
                float(rec["z"]),
                float(rec["r"]),
                int(parent_new),
            ]
        )
        kept_old_to_new[old_id] = new_id
        return new_id

    for root in roots:
        root_new = add_node(int(root), -1)
        old_to_rep_new[int(root)] = int(root_new)

        stack: List[Tuple[int, int]] = [(int(root), int(root_new))]
        while stack:
            parent_old, parent_new = stack.pop()
            for child in sorted(children[parent_old], reverse=True):
                chain = [int(child)]
                cur = int(child)
                while cur not in protected and len(children[cur]) == 1:
                    cur = int(children[cur][0])
                    chain.append(cur)

                keep_old = _select_chain_keeps(
                    parent_old=parent_old,
                    chain=chain,
                    nodes=nodes,
                    protected=protected,
                    max_path_um=params.max_path_um,
                    max_turn_deg=params.max_turn_deg,
                    max_diam_rel=params.max_diam_rel,
                )
                if not keep_old:
                    keep_old = [chain[-1]]

                pos_of = {oid: i for i, oid in enumerate(chain)}
                keep_pos = [pos_of[oid] for oid in keep_old]
                keep_new: List[int] = []
                prev_new = parent_new
                for oid in keep_old:
                    prev_new = add_node(int(oid), int(prev_new))
                    keep_new.append(int(prev_new))

                # Map each original node in this chain to a representative reduced node.
                for i, oid in enumerate(chain):
                    rep_j = 0
                    for j, p in enumerate(keep_pos):
                        if p >= i:
                            rep_j = j
                            break
                        rep_j = j
                    old_to_rep_new[int(oid)] = int(keep_new[rep_j])

                end_old = int(chain[-1])
                stack.append((end_old, int(prev_new)))

    map_rows: List[Dict[str, int]] = []
    for old_id in sorted(nodes.keys()):
        map_rows.append(
            {
                "old_id": int(old_id),
                "mapped_new_id": int(old_to_rep_new.get(old_id, -1)),
                "kept": int(1 if old_id in kept_old_to_new else 0),
                "kept_new_id": int(kept_old_to_new.get(old_id, -1)),
                "old_parent": int(nodes[old_id]["parent"]),
            }
        )

    stats = {
        "old_nodes": int(len(nodes)),
        "new_nodes": int(len(new_rows)),
        "ratio": (float(len(new_rows)) / float(len(nodes))) if len(nodes) > 0 else 1.0,
        "protected_nodes": int(len(protected)),
        "syn_protected_nodes": int(syn_protected),
    }
    return new_rows, map_rows, stats


def _reduce_one(
    swc_path_str: str,
    input_root_str: str,
    output_root_str: str,
    params_dict: Dict[str, object],
    syn_idx: Dict[int, str],
) -> Dict[str, object]:
    t0 = time.perf_counter()
    swc_path = Path(swc_path_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)
    params = ReducerParams(**params_dict)

    rel = swc_path.resolve().relative_to(input_root.resolve())
    out_swc = output_root / rel
    map_out = out_swc.with_suffix(out_swc.suffix + ".map.csv")
    nid = _extract_neuron_id_from_swc_path(swc_path)

    if out_swc.exists() and not params.overwrite:
        return {
            "rel_path": str(rel),
            "swc_in": str(swc_path),
            "swc_out": str(out_swc),
            "neuron_id": int(nid) if nid is not None else None,
            "status": "skipped_exists",
            "elapsed_s": time.perf_counter() - t0,
        }

    try:
        rows = _parse_swc(swc_path)
        syn_csv = syn_idx.get(int(nid)) if (params.protect_synapses and nid is not None) else None
        syn_pts = _load_syn_points(syn_csv, max_syn_points=int(params.max_syn_points))
        new_rows, map_rows, stats = _reduce_rows(rows, params, syn_pts)

        if not params.dry_run:
            _write_swc(out_swc, new_rows)
            if params.write_map:
                map_out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(map_rows).to_csv(map_out, index=False)

        return {
            "rel_path": str(rel),
            "swc_in": str(swc_path),
            "swc_out": str(out_swc),
            "map_out": (str(map_out) if params.write_map else None),
            "neuron_id": int(nid) if nid is not None else None,
            "status": "ok",
            "elapsed_s": time.perf_counter() - t0,
            **stats,
        }
    except Exception as e:
        return {
            "rel_path": str(rel),
            "swc_in": str(swc_path),
            "swc_out": str(out_swc),
            "neuron_id": int(nid) if nid is not None else None,
            "status": "error",
            "error": str(e),
            "elapsed_s": time.perf_counter() - t0,
        }


def _format_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-destructive SWC reducer for Digifly datasets.")
    p.add_argument("--input-root", required=True, help="Original SWC dataset root (export_swc).")
    p.add_argument("--output-root", required=True, help="Destination root for reduced SWCs.")
    p.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 1, 1), help="Parallel worker count.")
    p.add_argument("--max-path-um", type=float, default=8.0, help="Max path distance between retained nodes.")
    p.add_argument("--max-turn-deg", type=float, default=35.0, help="Keep points at/above this turn angle.")
    p.add_argument("--max-diam-rel", type=float, default=0.35, help="Keep nodes where relative radius jump exceeds this.")
    p.add_argument("--max-syn-points", type=int, default=2000, help="Cap synapse points per neuron for protection.")
    g_syn = p.add_mutually_exclusive_group()
    g_syn.add_argument("--protect-synapses", dest="protect_synapses", action="store_true", help="Protect nearest nodes to synapse coordinates (default).")
    g_syn.add_argument("--no-protect-synapses", dest="protect_synapses", action="store_false", help="Disable synapse protection.")
    p.set_defaults(protect_synapses=True)
    p.add_argument("--write-map", action="store_true", help="Write per-neuron old->new node mapping CSVs.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing reduced SWCs.")
    p.add_argument("--dry-run", action="store_true", help="Analyze only; do not write output files.")
    p.add_argument("--summary-name", default="_swc_reduction_summary.csv", help="Summary CSV file name in output root.")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of SWC files to process.")
    p.add_argument("--match", default=None, help="Optional substring filter on SWC path.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input-root is not a directory: {input_root}")
    if _is_subpath(output_root, input_root):
        print(f"[warn] output-root is inside input-root: {output_root}")
        print("[warn] reducer will skip writing/reading any SWCs under output-root.")

    protect_synapses = bool(args.protect_synapses)
    params = ReducerParams(
        max_path_um=float(args.max_path_um),
        max_turn_deg=float(args.max_turn_deg),
        max_diam_rel=float(args.max_diam_rel),
        protect_synapses=protect_synapses,
        max_syn_points=int(args.max_syn_points),
        write_map=bool(args.write_map),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    swc_files = _iter_swc_files(input_root, output_root, match=args.match)
    if int(args.limit) > 0:
        swc_files = swc_files[: int(args.limit)]
    if not swc_files:
        print("[error] no .swc files found")
        return 2
    print(f"[scan] swc files: {len(swc_files)}")

    syn_idx: Dict[int, str] = {}
    if params.protect_synapses:
        syn_idx = _build_syn_csv_index(input_root)
        print(f"[scan] synapse catalogs indexed: {len(syn_idx)}")

    if not params.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    params_dict = {
        "max_path_um": params.max_path_um,
        "max_turn_deg": params.max_turn_deg,
        "max_diam_rel": params.max_diam_rel,
        "protect_synapses": params.protect_synapses,
        "max_syn_points": params.max_syn_points,
        "write_map": params.write_map,
        "overwrite": params.overwrite,
        "dry_run": params.dry_run,
    }

    t0 = time.perf_counter()
    results: List[Dict[str, object]] = []
    workers = max(int(args.workers), 1)
    if workers == 1:
        for i, swc in enumerate(swc_files, 1):
            res = _reduce_one(str(swc), str(input_root), str(output_root), params_dict, syn_idx)
            results.append(res)
            if i % 250 == 0 or i == len(swc_files):
                print(f"[progress] {i}/{len(swc_files)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _reduce_one,
                    str(swc),
                    str(input_root),
                    str(output_root),
                    params_dict,
                    syn_idx,
                )
                for swc in swc_files
            ]
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 250 == 0 or done == len(swc_files):
                    print(f"[progress] {done}/{len(swc_files)}")

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(results)
    if "status" not in df.columns:
        df["status"] = "error"
    ok_df = df[df["status"] == "ok"]
    err_df = df[df["status"] == "error"]
    skip_df = df[df["status"] == "skipped_exists"]

    summary_path = output_root / str(args.summary_name)
    if not params.dry_run:
        df.to_csv(summary_path, index=False)
        manifest = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "params": params_dict,
            "counts": {
                "total": int(len(df)),
                "ok": int(len(ok_df)),
                "error": int(len(err_df)),
                "skipped_exists": int(len(skip_df)),
            },
            "elapsed_s": float(elapsed),
            "created_utc": pd.Timestamp.utcnow().isoformat(),
        }
        with (output_root / "_swc_reduction_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"[done] elapsed_s={elapsed:.2f}")
    print(f"[done] total={len(df)} ok={len(ok_df)} error={len(err_df)} skipped={len(skip_df)}")
    if len(ok_df) > 0 and {"old_nodes", "new_nodes"}.issubset(ok_df.columns):
        old_sum = int(pd.to_numeric(ok_df["old_nodes"], errors="coerce").fillna(0).sum())
        new_sum = int(pd.to_numeric(ok_df["new_nodes"], errors="coerce").fillna(0).sum())
        ratio = (float(new_sum) / float(old_sum)) if old_sum > 0 else 1.0
        print(f"[done] nodes old={old_sum} new={new_sum} ratio={ratio:.3f} ({_format_pct(1.0 - ratio)} reduction)")
    if not params.dry_run:
        print(f"[done] summary={summary_path}")
    if len(err_df) > 0:
        print("[error] first failures:")
        for _, r in err_df.head(10).iterrows():
            print(f"  - {r.get('rel_path', '?')}: {r.get('error', 'unknown')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
