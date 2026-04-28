from __future__ import annotations

from pathlib import Path
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
      - morph_swc_dir set to mutation phase2 overlay directory
      - selection.neuron_ids / neuron_ids updated from manifest unless force_neuron_ids is given

    Optional:
      - include_mutation_connections=True builds a forced chem edges overlay from mutation
        connection specs and injects:
            edges_path, edges_csv, edge_cache.enabled=False
    """
    m = load_mutation_manifest(manifest)
    out = dict(base_overrides or {})

    overlay = mutation_overlay_dir(manifest)
    out["morph_swc_dir"] = str(overlay)

    if force_neuron_ids is not None:
        ids = [int(x) for x in force_neuron_ids]
    else:
        ids = [int(x) for x in (m.get("neuron_ids") or [])]

    if ids:
        out["neuron_ids"] = list(ids)
        sel = dict(out.get("selection") or {})
        sel["mode"] = "custom"
        sel["neuron_ids"] = list(ids)
        out["selection"] = sel

    if bool(include_mutation_connections):
        info = build_forced_chem_edges_from_mutation_connections(
            manifest,
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

    return out
