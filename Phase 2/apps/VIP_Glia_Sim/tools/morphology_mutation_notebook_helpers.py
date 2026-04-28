from __future__ import annotations

from pathlib import Path
import shutil
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .morphology_mutation import load_swc_table


def load_mutation_manifest(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Mutation manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Mutation manifest is not a JSON object: {p}")
    return data


def load_mutation_connections(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load connection specs from either:
      - manifest JSON (embedded `connections` or sidecar file), or
      - direct mutation_connections.json path.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Connection/manifest path not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(x) for x in data]
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported JSON type at {p}: {type(data)}")

    if isinstance(data.get("connections"), list):
        return [dict(x) for x in data.get("connections", [])]

    sidecar = p.parent / "mutation_connections.json"
    if sidecar.exists():
        side = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(side, list):
            return [dict(x) for x in side]

    return []


def load_mutation_biophys_policies(path: str | Path) -> Dict[int, Dict[str, List[int]]]:
    """
    Load passive/active node policy maps from either:
      - manifest JSON field `biophys_policies`, or
      - sidecar `mutation_biophys_policies.json` in same run dir.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Policy/manifest path not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    raw: Any = None
    if isinstance(data, dict):
        raw = data.get("biophys_policies")

    if raw is None and p.is_file():
        sidecar = p.parent / "mutation_biophys_policies.json"
        if sidecar.exists():
            raw = json.loads(sidecar.read_text(encoding="utf-8"))

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"biophys_policies has unsupported type: {type(raw)}")

    out: Dict[int, Dict[str, List[int]]] = {}
    for k, v in raw.items():
        try:
            nid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        passive_ids = sorted({int(x) for x in (v.get("passive_node_ids") or [])})
        active_ids = sorted({int(x) for x in (v.get("active_node_ids") or [])})
        passive_ids = [int(x) for x in passive_ids if int(x) not in set(active_ids)]
        out[int(nid)] = {
            "passive_node_ids": passive_ids,
            "active_node_ids": active_ids,
        }

    return out


def load_mutation_ais_policies(path: str | Path) -> Dict[int, Dict[str, Any]]:
    """
    Load AIS policy maps from either:
      - manifest JSON field `ais_policies`, or
      - sidecar `mutation_ais_policies.json` in same run dir.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"AIS policy/manifest path not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    raw: Any = None
    if isinstance(data, dict):
        raw = data.get("ais_policies")

    if raw is None and p.is_file():
        sidecar = p.parent / "mutation_ais_policies.json"
        if sidecar.exists():
            raw = json.loads(sidecar.read_text(encoding="utf-8"))

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"ais_policies has unsupported type: {type(raw)}")

    out: Dict[int, Dict[str, Any]] = {}
    for k, v in raw.items():
        try:
            nid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        primary_node_id = v.get("primary_node_id")
        if primary_node_id is not None:
            try:
                primary_node_id = int(primary_node_id)
            except Exception:
                primary_node_id = None
        extra_node_ids = sorted({int(x) for x in (v.get("extra_node_ids") or [])})
        out[int(nid)] = {
            "primary_node_id": primary_node_id,
            "primary_xloc": float(v.get("primary_xloc", 0.5)),
            "extra_node_ids": extra_node_ids,
        }
    return out


def mutation_overlay_dir(manifest: str | Path) -> Path:
    m = load_mutation_manifest(manifest)
    ov = m.get("phase2_overlay_dir")
    if not ov:
        raise ValueError(
            "Manifest has no phase2_overlay_dir. "
            "Re-save bundle with write_phase2_overlay=True."
        )
    p = Path(ov).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Overlay dir from manifest does not exist: {p}")
    return p


def mutation_neuron_ids(manifest: str | Path) -> List[int]:
    m = load_mutation_manifest(manifest)
    vals = m.get("neuron_ids") or []
    return sorted({int(x) for x in vals})


def _read_edges_table(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if sfx in {".feather", ".ftr"}:
        return pd.read_feather(path)
    return pd.read_csv(path)


def _resolve_mutated_swc_paths_from_manifest(manifest_data: Dict[str, Any]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    raw = manifest_data.get("mutated_swcs") or {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            nid = int(k)
        except Exception:
            continue
        p = Path(str(v)).expanduser().resolve()
        if p.exists():
            out[nid] = p
    return out


def _dedupe_ints(vals: Iterable[Any]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in vals:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _extract_requested_neuron_ids(overrides: Optional[Dict[str, Any]]) -> List[int]:
    cfg = dict(overrides or {})
    ids: List[int] = []

    sel = dict(cfg.get("selection") or {})
    if sel.get("neuron_id") is not None:
        ids.append(int(sel.get("neuron_id")))
    ids.extend(sel.get("neuron_ids") or [])

    if cfg.get("neuron_id") is not None:
        ids.append(int(cfg.get("neuron_id")))
    ids.extend(cfg.get("neuron_ids") or [])
    ids.extend(cfg.get("seeds") or [])

    stim = cfg.get("stim") or {}
    if isinstance(stim, dict):
        for key in ["neuron_ids", "seed_ids", "target_ids"]:
            ids.extend(stim.get(key) or [])
        iclamp = stim.get("iclamp") or {}
        if isinstance(iclamp, dict):
            for key in ["neuron_ids", "target_ids"]:
                ids.extend(iclamp.get(key) or [])

    return _dedupe_ints(ids)


def _find_base_swc_path(swc_root: str | Path, nid: int) -> Path:
    root = Path(swc_root).expanduser().resolve()
    nid = int(nid)
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
        hits = sorted(root.glob(pat), key=lambda q: len(str(q)))
        if hits:
            return hits[0].resolve()
    raise FileNotFoundError(f"No SWC for id={nid} under {root}")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def materialize_simulation_swc_overlay(
    manifest: str | Path,
    *,
    required_neuron_ids: Optional[Iterable[int]] = None,
) -> Path:
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_data = load_mutation_manifest(manifest_path)
    mutated_paths = _resolve_mutated_swc_paths_from_manifest(manifest_data)
    base_overlay = mutation_overlay_dir(manifest_path)

    required_ids = _dedupe_ints(required_neuron_ids or [])
    if not required_ids:
        return base_overlay

    mutated_ids = set(int(x) for x in mutated_paths.keys())
    if set(required_ids).issubset(mutated_ids):
        return base_overlay

    swc_root = manifest_data.get("swc_root")
    if not swc_root:
        raise ValueError("Mutation manifest has no swc_root; cannot materialize merged overlay.")

    merged_dir = manifest_path.parent / "phase2_morph_overlay_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    expected = [merged_dir / f"{int(nid)}_axodendro_with_synapses.swc" for nid in required_ids]
    src_ais = base_overlay / "ais_overrides.csv"
    merged_ais = merged_dir / "ais_overrides.csv"
    if all(p.exists() or p.is_symlink() for p in expected) and ((not src_ais.exists()) or merged_ais.exists() or merged_ais.is_symlink()):
        return merged_dir

    for nid in required_ids:
        nid = int(nid)
        src = mutated_paths.get(nid)
        if src is None:
            src = _find_base_swc_path(swc_root, nid)
        dst = merged_dir / f"{nid}_axodendro_with_synapses.swc"
        _link_or_copy(Path(src).expanduser().resolve(), dst)

    if src_ais.exists():
        _link_or_copy(src_ais.resolve(), merged_ais)

    return merged_dir


def _node_xyz_from_swc_table(df: pd.DataFrame, node_id: int) -> Tuple[float, float, float]:
    row = df.loc[df["id"] == int(node_id)]
    if row.empty:
        raise KeyError(f"node_id {int(node_id)} not found in SWC table")
    r = row.iloc[0]
    return float(r["x"]), float(r["y"]), float(r["z"])


def build_forced_chem_edges_from_mutation_connections(
    manifest: str | Path,
    *,
    base_edges_path: Optional[str | Path],
    output_csv: Optional[str | Path] = None,
    default_weight_uS: float = 6e-6,
    default_delay_ms: Optional[float] = None,
    default_tau1_ms: Optional[float] = None,
    default_tau2_ms: Optional[float] = None,
    default_syn_e_rev_mV: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert mutation connection specs into a simulation-ready forced chem edges overlay.

    - Uses exact node coordinates from mutated SWCs.
    - Emits one row per `chemical_synapses` count per connection.
    - Matches the broad schema used by existing `_forced_chem_only_edges` CSVs.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    m = load_mutation_manifest(manifest_path)
    conns = load_mutation_connections(manifest_path)
    mutated_swcs = _resolve_mutated_swc_paths_from_manifest(m)
    if not mutated_swcs:
        raise RuntimeError(
            "Manifest does not expose readable mutated_swcs paths. "
            "Re-export mutation bundle and retry."
        )

    swc_tables: Dict[int, pd.DataFrame] = {}
    for nid, swc_path in mutated_swcs.items():
        swc_tables[int(nid)] = load_swc_table(swc_path)

    add_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for c in conns:
        pre_id = int(c.get("pre_neuron_id"))
        post_id = int(c.get("post_neuron_id"))
        pre_node = int(c.get("pre_node_id"))
        post_node = int(c.get("post_node_id"))
        n_chem = max(0, int(c.get("chemical_synapses", 0)))
        if n_chem <= 0:
            continue

        if pre_id not in swc_tables:
            raise KeyError(f"pre_neuron_id {pre_id} not present in mutated SWCs from manifest")
        if post_id not in swc_tables:
            raise KeyError(f"post_neuron_id {post_id} not present in mutated SWCs from manifest")

        pre_x, pre_y, pre_z = _node_xyz_from_swc_table(swc_tables[pre_id], pre_node)
        post_x, post_y, post_z = _node_xyz_from_swc_table(swc_tables[post_id], post_node)

        for j in range(n_chem):
            syn_idx = int(j)
            add_rows.append(
                {
                    "pre_id": int(pre_id),
                    "post_id": int(post_id),
                    "weight_uS": float(default_weight_uS),
                    "delay_ms": (float(default_delay_ms) if default_delay_ms is not None else np.nan),
                    "tau1_ms": (float(default_tau1_ms) if default_tau1_ms is not None else np.nan),
                    "tau2_ms": (float(default_tau2_ms) if default_tau2_ms is not None else np.nan),
                    "syn_e_rev_mV": (
                        float(default_syn_e_rev_mV) if default_syn_e_rev_mV is not None else np.nan
                    ),
                    "pre_x": float(pre_x),
                    "pre_y": float(pre_y),
                    "pre_z": float(pre_z),
                    "post_x": float(post_x),
                    "post_y": float(post_y),
                    "post_z": float(post_z),
                    "syn_index": float(syn_idx),
                    "pre_syn_index": float(syn_idx),
                    "post_syn_index": float(syn_idx),
                    "pre_match_um": 0.0,
                    "post_match_um": 0.0,
                }
            )

        summary.append(
            {
                "pre_id": int(pre_id),
                "post_id": int(post_id),
                "pre_node_id": int(pre_node),
                "post_node_id": int(post_node),
                "added_rows": int(n_chem),
                "gap_junctions": int(c.get("gap_junctions", 0) or 0),
                "gap_mode": str(c.get("gap_mode", "none")),
                "gap_direction": c.get("gap_direction"),
            }
        )

    if not add_rows:
        raise RuntimeError("No chemical rows to add from mutation connections (chemical_synapses <= 0).")

    add_df = pd.DataFrame(add_rows)

    base_df = pd.DataFrame()
    if base_edges_path:
        base_p = Path(base_edges_path).expanduser().resolve()
        if not base_p.exists():
            raise FileNotFoundError(f"base_edges_path not found: {base_p}")
        base_df = _read_edges_table(base_p)

    merged = pd.concat([base_df, add_df], ignore_index=True, sort=False)
    if "pre_id" not in merged.columns or "post_id" not in merged.columns:
        raise RuntimeError("Merged edges table missing pre_id/post_id columns.")

    merged["pre_id"] = pd.to_numeric(merged["pre_id"], errors="coerce")
    merged["post_id"] = pd.to_numeric(merged["post_id"], errors="coerce")
    merged = merged.dropna(subset=["pre_id", "post_id"]).copy()
    merged["pre_id"] = merged["pre_id"].astype(int)
    merged["post_id"] = merged["post_id"].astype(int)

    dedup_cols = [
        c
        for c in [
            "pre_id",
            "post_id",
            "pre_x",
            "pre_y",
            "pre_z",
            "post_x",
            "post_y",
            "post_z",
            "pre_syn_index",
            "post_syn_index",
        ]
        if c in merged.columns
    ]
    if dedup_cols:
        merged = merged.drop_duplicates(subset=dedup_cols, keep="first")

    if output_csv is None:
        output_csv = manifest_path.parent / "mutation_forced_chem_edges.csv"
    out_p = Path(output_csv).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_p, index=False)

    return {
        "overlay_csv": str(out_p),
        "rows_total": int(len(merged)),
        "rows_added_from_mutation": int(len(add_df)),
        "added_connection_summary": summary,
    }


def build_sim_overrides_from_mutation_manifest(
    manifest: str | Path,
    *,
    base_overrides: Optional[Dict[str, Any]] = None,
    force_neuron_ids: Optional[Iterable[int]] = None,
    include_mutation_connections: bool = False,
    base_edges_path: Optional[str | Path] = None,
    output_edges_csv: Optional[str | Path] = None,
    default_weight_uS: float = 6e-6,
    default_delay_ms: Optional[float] = None,
    default_tau1_ms: Optional[float] = None,
    default_tau2_ms: Optional[float] = None,
    default_syn_e_rev_mV: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Helper for glia_simulation/run_simulation notebook cells.

    Returns a copy of `base_overrides` with:
      - morph_swc_dir set to a simulation-ready overlay directory
      - mutated SWCs supplied from the mutation bundle
      - non-mutated required SWCs supplied from the manifest swc_root
      - selection.neuron_ids / neuron_ids preserved from the scenario when available

    Optional:
      - include_mutation_connections=True builds a forced chem edges overlay from mutation
        connection specs and injects:
            edges_path, edges_csv, edge_cache.enabled=False
    """
    manifest_path = Path(manifest).expanduser().resolve()
    m = load_mutation_manifest(manifest_path)
    out = dict(base_overrides or {})

    existing_ids = _extract_requested_neuron_ids(base_overrides)
    forced_ids = _dedupe_ints(force_neuron_ids or [])
    manifest_ids = _dedupe_ints(m.get("neuron_ids") or [])
    ids = forced_ids or existing_ids or manifest_ids

    overlay = materialize_simulation_swc_overlay(
        manifest_path,
        required_neuron_ids=ids,
    )
    out["morph_swc_dir"] = str(overlay)
    out["_mutation_overlay_info"] = {
        "manifest": str(manifest_path),
        "overlay_dir": str(overlay),
        "required_neuron_ids": list(ids),
        "mutated_neuron_ids": list(manifest_ids),
    }

    if ids:
        out["neuron_ids"] = list(ids)
        sel = dict(out.get("selection") or {})
        if sel.get("mode") is None:
            sel["mode"] = "custom"
        if len(ids) == 1 and sel.get("mode") == "single":
            sel["neuron_id"] = int(ids[0])
        else:
            sel["mode"] = "custom"
            sel["neuron_ids"] = list(ids)
        out["selection"] = sel

    if bool(include_mutation_connections):
        conn_rows = [c for c in (m.get("connections") or []) if isinstance(c, dict)]
        has_chem_rows = any(int(c.get("chemical_synapses", 0) or 0) > 0 for c in conn_rows)
        if has_chem_rows:
            info = build_forced_chem_edges_from_mutation_connections(
                manifest_path,
                base_edges_path=base_edges_path,
                output_csv=output_edges_csv,
                default_weight_uS=float(default_weight_uS),
                default_delay_ms=default_delay_ms,
                default_tau1_ms=default_tau1_ms,
                default_tau2_ms=default_tau2_ms,
                default_syn_e_rev_mV=default_syn_e_rev_mV,
            )
            out["edges_path"] = str(info["overlay_csv"])
            out["edges_csv"] = str(info["overlay_csv"])
            edge_cache_cfg = dict(out.get("edge_cache") or {})
            edge_cache_cfg["enabled"] = False
            out["edge_cache"] = edge_cache_cfg
            out["_mutation_connection_overlay_info"] = info
        else:
            out["_mutation_connection_overlay_info"] = {
                "overlay_csv": None,
                "rows_total": 0,
                "rows_added_from_mutation": 0,
                "added_connection_summary": [],
                "skipped_reason": "no_chemical_mutation_connections",
            }

    return out
