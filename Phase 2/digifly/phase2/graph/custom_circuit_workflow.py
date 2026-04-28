from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def dedupe_preserve_order(seq: Iterable[Any]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def apply_recording_policy(
    record_dict: Mapping[str, Any] | None,
    loaded_ids: Sequence[int],
    stim_ids: Sequence[int],
    *,
    record_soma_for_all: bool = False,
) -> Dict[str, Any]:
    """Notebook-friendly recording policy normalizer used by run/glia notebooks."""
    rd = dict(record_dict) if isinstance(record_dict, Mapping) else {}
    loaded_ids = [int(x) for x in loaded_ids]
    stim_ids = [int(x) for x in stim_ids]

    if "spikes" in rd:
        rd["spikes"] = list(loaded_ids)
    if "spike_ids" in rd:
        rd["spike_ids"] = list(loaded_ids)
    if "spike" in rd and isinstance(rd["spike"], Mapping):
        spike_cfg = dict(rd["spike"])
        for k in ("ids", "neuron_ids", "cells"):
            if k in spike_cfg:
                spike_cfg[k] = list(loaded_ids)
        rd["spike"] = spike_cfg
    if ("spikes" not in rd) and ("spike_ids" not in rd) and ("spike" not in rd):
        rd["spikes"] = list(loaded_ids)

    target_v_ids = list(loaded_ids) if record_soma_for_all else list(stim_ids)
    if "soma_v" in rd:
        rd["soma_v"] = target_v_ids
    if "voltages" in rd and isinstance(rd["voltages"], Mapping):
        vcfg = dict(rd["voltages"])
        for k in ("soma", "soma_ids"):
            if k in vcfg:
                vcfg[k] = target_v_ids
        rd["voltages"] = vcfg
    if "v" in rd and isinstance(rd["v"], Mapping):
        vcfg = dict(rd["v"])
        for k in ("soma", "soma_ids"):
            if k in vcfg:
                vcfg[k] = target_v_ids
        rd["v"] = vcfg
    return rd


def read_edges_table(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    sfx = p.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(p)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if sfx in {".feather", ".ftr"}:
        return pd.read_feather(p)
    return pd.read_csv(p)


def build_chem_only_pairs(
    chem_map: Mapping[Any, Sequence[Any]] | None,
    *,
    direction: str = "source_to_postsyn",
) -> Dict[str, Any]:
    """Normalize notebook chem-only mapping into explicit directed edge pairs."""
    direction_norm = str(direction or "source_to_postsyn").strip().lower()
    mapping = dict(chem_map or {}) if isinstance(chem_map, Mapping) else {}
    pairs: List[Tuple[int, int]] = []
    added_ids: List[int] = []
    source_ids: List[int] = []
    target_ids: List[int] = []
    seen_added = set()
    seen_sources = set()
    seen_targets = set()

    if direction_norm == "source_to_postsyn":
        for src, dsts in mapping.items():
            s = int(src)
            if s not in seen_sources:
                seen_sources.add(s)
                source_ids.append(s)
            for dst in (dsts or []):
                d = int(dst)
                pairs.append((s, d))
                if d not in seen_targets:
                    seen_targets.add(d)
                    target_ids.append(d)
                if d not in seen_added:
                    seen_added.add(d)
                    added_ids.append(d)
    else:
        # Legacy interpretation: target -> presyn inputs
        for tgt, pres in mapping.items():
            t = int(tgt)
            if t not in seen_targets:
                seen_targets.add(t)
                target_ids.append(t)
            for pre in (pres or []):
                p = int(pre)
                pairs.append((p, t))
                if p not in seen_sources:
                    seen_sources.add(p)
                    source_ids.append(p)
                if p not in seen_added:
                    seen_added.add(p)
                    added_ids.append(p)

    return {
        "direction": direction_norm,
        "mapping": mapping,
        "pairs": [(int(a), int(b)) for a, b in pairs],
        "added_ids": [int(x) for x in added_ids],
        "source_ids": [int(x) for x in source_ids],
        "target_ids": [int(x) for x in target_ids],
    }


def extend_neuron_ids_with_chem_only(
    neuron_ids: Sequence[int],
    *,
    chem_map: Mapping[Any, Sequence[Any]] | None,
    direction: str = "source_to_postsyn",
) -> Dict[str, Any]:
    spec = build_chem_only_pairs(chem_map, direction=direction)
    merged = dedupe_preserve_order(list(neuron_ids) + spec["source_ids"] + spec["target_ids"])
    out = dict(spec)
    out["merged_neuron_ids"] = merged
    return out


def edge_pair_counts_from_df(df_edges: pd.DataFrame, pairs: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    if df_edges is None or df_edges.empty or not pairs:
        return {tuple(map(int, p)): 0 for p in pairs}
    if not {"pre_id", "post_id"}.issubset(set(df_edges.columns)):
        raise ValueError("edges table missing pre_id/post_id columns")
    pp = df_edges[["pre_id", "post_id"]].copy()
    pp["pre_id"] = pd.to_numeric(pp["pre_id"], errors="coerce").fillna(-1).astype(int)
    pp["post_id"] = pd.to_numeric(pp["post_id"], errors="coerce").fillna(-1).astype(int)
    counts: Dict[Tuple[int, int], int] = {}
    for a, b in pairs:
        ai = int(a)
        bi = int(b)
        counts[(ai, bi)] = int(((pp["pre_id"] == ai) & (pp["post_id"] == bi)).sum())
    return counts


def edge_pair_counts_from_file(path: str | Path, pairs: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    return edge_pair_counts_from_df(read_edges_table(path), pairs)


def rawsyn_pair_counts_for_pairs(
    swc_root: str | Path,
    pairs: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    """Count directed pair rows from rawsyn source exports (ground truth for current dataset export)."""
    root = Path(swc_root).expanduser().resolve()
    by_source: Dict[int, List[int]] = {}
    for a, b in pairs:
        by_source.setdefault(int(a), []).append(int(b))

    out: Dict[Tuple[int, int], int] = { (int(a), int(b)): 0 for a, b in pairs }
    for src, posts in by_source.items():
        hits = sorted(root.rglob(f"edges_ego_{int(src)}__rawsyn.csv"))
        if not hits:
            continue
        p = hits[0]
        try:
            df = pd.read_csv(p, usecols=["pre_id", "post_id"])
        except Exception:
            df = pd.read_csv(p)
            if not {"pre_id", "post_id"}.issubset(df.columns):
                continue
            df = df[["pre_id", "post_id"]]
        df["pre_id"] = pd.to_numeric(df["pre_id"], errors="coerce").fillna(-1).astype(int)
        df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce").fillna(-1).astype(int)
        sub = df[df["pre_id"] == int(src)]
        for post in posts:
            out[(int(src), int(post))] = int((sub["post_id"] == int(post)).sum())
    return out


def _resolve_cache_subset_edges_if_enabled(cfg: Dict[str, Any]) -> Tuple[Optional[Path], Optional[List[int]], Optional[str]]:
    ec = dict(cfg.get("edge_cache") or {})
    if not bool(ec.get("enabled", False)):
        return None, None, None
    try:
        from digifly.phase2.graph.edge_cache import resolve_custom_edges_from_cache
    except Exception as e:
        return None, None, f"import edge_cache failed: {e}"
    try:
        sel_ids = [int(x) for x in ((cfg.get("selection") or {}).get("neuron_ids") or [])]
        seeds = [int(x) for x in (cfg.get("seeds") or [])]
        out_path, resolved_ids = resolve_custom_edges_from_cache(cfg, loaded_ids=sel_ids, seed_ids=seeds)
        return Path(out_path).expanduser().resolve(), [int(x) for x in (resolved_ids or [])], None
    except Exception as e:
        return None, None, str(e)


def _syn_catalog_xyzs(swc_root: str | Path, nid: int, kind: str) -> np.ndarray:
    try:
        from digifly.phase2.neuron_build.swc_cell import load_syn_catalog
    except Exception:
        return np.empty((0, 3), dtype=float)
    try:
        cat = load_syn_catalog(swc_root, int(nid), verbose=False)
        arr = np.asarray(cat.get(str(kind), np.empty((0, 3))), dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
            return np.empty((0, 3), dtype=float)
        return arr[:, :3].copy()
    except Exception:
        return np.empty((0, 3), dtype=float)


def apply_chem_only_workflow_to_cfg(
    cfg: Mapping[str, Any],
    *,
    scenario_label: str = "run",
    chem_only_map: Mapping[Any, Sequence[Any]] | None = None,
    chem_only_direction: str = "source_to_postsyn",
    record_soma_for_all: bool = True,
    force_overlay_for_missing_pairs: bool = True,
    force_min_rows_per_pair: int = 128,
    force_weight_scale: float = 1.0,
    force_delay_ms: float | None = None,
    force_erev_mV: float | None = None,
    force_tau1_ms: float | None = None,
    force_tau2_ms: float | None = None,
    force_wire_soma: bool = False,
    base_edges_path: str | Path | None = None,
    swc_root: str | Path | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply notebook chem-only custom-circuit workflow to a built config.

    - Restores chem-only ids into selection + recording.
    - Optionally resolves cache subset edges early and uses it as base.
    - Synthesizes directed source->postsyn rows when requested pairs are absent/sparse.
    - Forces the run to use the generated overlay when synthetic rows are added.
    """
    cfg_out: Dict[str, Any] = dict(cfg or {})
    info: Dict[str, Any] = {
        "applied": False,
        "direction": str(chem_only_direction or "source_to_postsyn"),
        "pairs": [],
        "added_ids": [],
        "used_cache_subset_base": False,
        "cache_subset_path": None,
        "cache_subset_error": None,
        "base_edge_path": None,
        "overlay_path": None,
        "pair_counts_base": {},
        "pair_counts_rawsyn": {},
        "synth_added_rows": [],
    }

    sel = dict(cfg_out.get("selection") or {})
    if str(sel.get("mode", "")).strip().lower() != "custom":
        return cfg_out, info

    chem_spec = build_chem_only_pairs(chem_only_map, direction=chem_only_direction)
    chem_pairs = [(int(a), int(b)) for a, b in (chem_spec.get("pairs") or [])]
    info["pairs"] = list(chem_pairs)
    info["added_ids"] = [int(x) for x in (chem_spec.get("added_ids") or [])]
    if not chem_pairs:
        return cfg_out, info

    cur_ids = [int(x) for x in (sel.get("neuron_ids") or [])]
    merged_ids = dedupe_preserve_order(cur_ids + chem_spec["source_ids"] + chem_spec["target_ids"])
    if merged_ids != cur_ids:
        sel["neuron_ids"] = list(merged_ids)
        cfg_out["selection"] = sel
    else:
        merged_ids = cur_ids

    seeds = [int(x) for x in (cfg_out.get("seeds") or [])]
    cfg_out["record"] = apply_recording_policy(cfg_out.get("record") or {}, merged_ids, seeds, record_soma_for_all=bool(record_soma_for_all))
    info["applied"] = True

    swc_root_eff = str((cfg_out.get("morph_swc_dir") or cfg_out.get("swc_dir") or swc_root or "")).strip() or None
    if swc_root_eff:
        try:
            info["pair_counts_rawsyn"] = {f"{a}->{b}": int(n) for (a, b), n in rawsyn_pair_counts_for_pairs(swc_root_eff, chem_pairs).items()}
        except Exception as e:
            info["pair_counts_rawsyn_error"] = str(e)

    cache_subset_path: Optional[Path] = None
    resolved_ids: Optional[List[int]] = None
    cache_err: Optional[str] = None
    cache_subset_path, resolved_ids, cache_err = _resolve_cache_subset_edges_if_enabled(cfg_out)
    if cache_subset_path is not None:
        info["used_cache_subset_base"] = True
        info["cache_subset_path"] = str(cache_subset_path)
        if resolved_ids:
            merged2 = dedupe_preserve_order(merged_ids + resolved_ids)
            if merged2 != merged_ids:
                sel = dict(cfg_out.get("selection") or {})
                sel["neuron_ids"] = list(merged2)
                cfg_out["selection"] = sel
                merged_ids = merged2
                cfg_out["record"] = apply_recording_policy(cfg_out.get("record") or {}, merged_ids, seeds, record_soma_for_all=bool(record_soma_for_all))
    elif cache_err:
        info["cache_subset_error"] = str(cache_err)

    base_candidates: List[Path] = []
    for cand in (cache_subset_path, cfg_out.get("edges_path"), base_edges_path):
        if not cand:
            continue
        try:
            p = Path(str(cand)).expanduser().resolve()
        except Exception:
            continue
        if p.exists() and p not in base_candidates:
            base_candidates.append(p)
    if not base_candidates:
        raise RuntimeError(
            "No usable base edges file found for chem-only workflow. Provide cfg['edges_path'] or base_edges_path."
        )

    base_edge_path = base_candidates[0]
    info["base_edge_path"] = str(base_edge_path)
    df_edges = read_edges_table(base_edge_path)
    if not {"pre_id", "post_id"}.issubset(set(df_edges.columns)):
        raise RuntimeError(f"Base edges file missing pre_id/post_id: {base_edge_path}")
    for c in ("pre_id", "post_id"):
        df_edges[c] = pd.to_numeric(df_edges[c], errors="coerce")
    df_edges = df_edges.dropna(subset=["pre_id", "post_id"]).copy()
    df_edges["pre_id"] = df_edges["pre_id"].astype(int)
    df_edges["post_id"] = df_edges["post_id"].astype(int)

    pair_counts_base = edge_pair_counts_from_df(df_edges, chem_pairs)
    info["pair_counts_base"] = {f"{a}->{b}": int(n) for (a, b), n in pair_counts_base.items()}

    if not bool(force_overlay_for_missing_pairs):
        return cfg_out, info

    min_rows = int(force_min_rows_per_pair)
    loaded_set = set(int(x) for x in ((cfg_out.get("selection") or {}).get("neuron_ids") or []))

    w_default = float(cfg_out.get("default_weight_uS", 6e-6) or 6e-6) * float(force_weight_scale)
    d_default = float(force_delay_ms if force_delay_ms is not None else (cfg_out.get("default_delay_ms", 1.0) or 1.0))
    e_default = float(force_erev_mV if force_erev_mV is not None else (cfg_out.get("syn_e_rev_mV", 0.0) or 0.0))
    t1_default = float(force_tau1_ms if force_tau1_ms is not None else (cfg_out.get("syn_tau1_ms", 0.5) or 0.5))
    t2_default = float(force_tau2_ms if force_tau2_ms is not None else (cfg_out.get("syn_tau2_ms", 3.0) or 3.0))

    swc_root_for_catalog = swc_root_eff
    if not swc_root_for_catalog:
        raise RuntimeError("chem-only forced overlay requires swc_root/morph_swc_dir to locate syn catalogs")

    _catalog_cache: Dict[Tuple[int, str], np.ndarray] = {}
    def _xyzs(nid: int, kind: str) -> np.ndarray:
        key = (int(nid), str(kind))
        if key not in _catalog_cache:
            _catalog_cache[key] = _syn_catalog_xyzs(swc_root_for_catalog, int(nid), kind)
        return _catalog_cache[key]

    extras: List[pd.DataFrame] = []
    synth_added_rows: List[Dict[str, Any]] = []
    for src, dst in chem_pairs:
        src_i = int(src); dst_i = int(dst)
        if loaded_set and ((src_i not in loaded_set) or (dst_i not in loaded_set)):
            continue
        have = int(pair_counts_base.get((src_i, dst_i), 0))
        need = max(0, int(min_rows) - have)
        if need <= 0:
            continue

        pre_xyz = _xyzs(src_i, "pre")
        post_xyz = _xyzs(dst_i, "post")
        n_pre = int(len(pre_xyz)); n_post = int(len(post_xyz))
        rows: List[Dict[str, Any]] = []
        for j in range(need):
            row: Dict[str, Any] = {
                "pre_id": src_i,
                "post_id": dst_i,
                "pre_match_um": 0.0,
                "post_match_um": 0.0,
                "weight_uS": float(w_default),
                "delay_ms": float(d_default),
                "syn_e_rev_mV": float(e_default),
                "tau1_ms": float(t1_default),
                "tau2_ms": float(t2_default),
            }
            if n_pre > 0:
                kpre = int((have + j) % n_pre)
                row.update({
                    "pre_syn_index": int(kpre),
                    "pre_x": float(pre_xyz[kpre, 0]),
                    "pre_y": float(pre_xyz[kpre, 1]),
                    "pre_z": float(pre_xyz[kpre, 2]),
                })
            else:
                row["pre_syn_index"] = np.nan
            if n_post > 0:
                kpost = int((have + j) % n_post)
                row.update({
                    "post_syn_index": int(kpost),
                    "post_x": float(post_xyz[kpost, 0]),
                    "post_y": float(post_xyz[kpost, 1]),
                    "post_z": float(post_xyz[kpost, 2]),
                })
            else:
                row["post_syn_index"] = np.nan
            rows.append(row)
        if rows:
            extras.append(pd.DataFrame(rows))
            synth_added_rows.append({
                "pre_id": src_i,
                "post_id": dst_i,
                "had_rows": have,
                "added_rows": len(rows),
                "target_min_rows": int(min_rows),
                "pre_catalog_n": n_pre,
                "post_catalog_n": n_post,
            })

    info["synth_added_rows"] = synth_added_rows
    if not extras:
        if verbose:
            print("[chem-only] shared workflow: no synthetic overlay needed; requested pairs already present.")
        return cfg_out, info

    df_aug = pd.concat([df_edges] + extras, ignore_index=True, sort=False)
    for c in ("pre_id", "post_id"):
        df_aug[c] = pd.to_numeric(df_aug[c], errors="coerce")
    df_aug = df_aug.dropna(subset=["pre_id", "post_id"]).copy()
    df_aug["pre_id"] = df_aug["pre_id"].astype(int)
    df_aug["post_id"] = df_aug["post_id"].astype(int)

    runs_root = Path(str(cfg_out.get("runs_root") or Path.cwd())).expanduser().resolve()
    overlay_dir = (runs_root / "_forced_chem_only_edges").resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg_out.get("run_id") or "run")
    safe_label = "".join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in str(scenario_label))
    overlay_path = overlay_dir / f"{run_id}_{safe_label}_forced_chem_edges.csv"
    df_aug.to_csv(overlay_path, index=False)

    ecfg = dict(cfg_out.get("edge_cache") or {})
    ecfg["enabled"] = False
    cfg_out["edge_cache"] = ecfg
    cfg_out["edges_path"] = str(overlay_path)
    if "edges_csv" in cfg_out:
        cfg_out["edges_csv"] = str(overlay_path)
    if bool(force_wire_soma):
        cfg_out["wire_force_soma"] = True

    info["overlay_path"] = str(overlay_path)
    if verbose:
        print(f"[chem-only] shared workflow: forced overlay -> {overlay_path}")
        if info.get("used_cache_subset_base"):
            print(f"[chem-only] shared workflow: base from cache subset -> {info.get('cache_subset_path')}")
        elif info.get("base_edge_path"):
            print(f"[chem-only] shared workflow: base from edges_path -> {info.get('base_edge_path')}")
        for row in synth_added_rows:
            print(
                "[chem-only] shared workflow synth",
                f"{row['pre_id']}->{row['post_id']}",
                f"had={row['had_rows']}",
                f"added={row['added_rows']}",
                f"target_min={row['target_min_rows']}",
                f"pre_catalog_n={row['pre_catalog_n']}",
                f"post_catalog_n={row['post_catalog_n']}",
            )
    return cfg_out, info
