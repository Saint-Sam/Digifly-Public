from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import re
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from digifly.phase2.config.loader import build_config
from digifly.phase2.hemi.runner import run_hemilineage
from digifly.phase2.graph.edge_cache import resolve_custom_edges_from_cache
from digifly.phase2.neuron_build.network import Network
from digifly.phase2.neuron_build.builders import build_network_driven_subset
from digifly.phase2.neuron_build.gaps import apply_gap_config
from digifly.phase2.neuron_build.biophys import make_passive, make_active
from digifly.phase2.util.save import save_config, save_records, save_spikes


def _read_edges_table_for_sanity(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def _maybe_tqdm(iterable, total=None, desc="", use_tqdm=True):
    if use_tqdm:
        try:
            from tqdm import tqdm
            return tqdm(iterable, total=total, desc=desc, leave=False)
        except Exception:
            pass
    return iterable


@contextmanager
def _phase_timer(label: str, *, collector: Optional[List[Dict[str, Any]]] = None):
    t0 = time.perf_counter()
    print(f"[phase] {label}: start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if collector is not None:
            collector.append({"label": str(label), "wall_s": float(dt)})
        print(f"[phase] {label}: done wall_s={dt:.3f}")


_SWC_ID_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _extract_id_hint_from_swc_path(p: Path) -> Optional[int]:
    """
    Fast ID hint extraction from SWC path/name.
    Priority:
      1) numeric parent directory (common layout: .../<nid>/<nid>_*.swc)
      2) leading integer in stem
      3) first integer in stem
    """
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


def _build_swc_id_index(swc_root: Path, *, use_tqdm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ids: set[int] = set()
    files_scanned = 0
    it = _maybe_tqdm(
        swc_root.rglob("*.swc"),
        desc="Indexing SWCs",
        use_tqdm=use_tqdm,
    )
    for p in it:
        files_scanned += 1
        hid = _extract_id_hint_from_swc_path(p)
        if hid is not None:
            ids.add(int(hid))
    return {
        "ids": ids,
        "files_scanned": int(files_scanned),
        "elapsed_s": float(time.perf_counter() - t0),
    }


def _get_swc_id_index(swc_root: Path, *, use_tqdm: bool = True) -> Dict[str, Any]:
    key = str(swc_root.resolve())
    cached = _SWC_ID_INDEX_CACHE.get(key)
    if cached is not None:
        print(
            "[swc] using cached id-index "
            f"ids={len(cached['ids']):,} files={int(cached['files_scanned']):,}"
        )
        return cached
    built = _build_swc_id_index(swc_root, use_tqdm=use_tqdm)
    _SWC_ID_INDEX_CACHE[key] = built
    print(
        "[swc] built id-index "
        f"ids={len(built['ids']):,} files={int(built['files_scanned']):,} wall_s={float(built['elapsed_s']):.3f}"
    )
    return built


def _find_edges_for_label(edges_root: Path, label: str) -> Optional[Path]:
    """
    Find an existing hemilineage edges file for a label.
    Accepts .csv or .parquet. Prefers '*_from_synapses*' if present.
    """
    lab = str(label).strip()
    if not lab:
        return None

    candidates: List[Path] = []
    for ext in (".csv", ".parquet"):
        candidates += list(edges_root.glob(f"*{lab}*{ext}"))

    if not candidates:
        return None

    # Prefer from_synapses outputs (Phase 2 behavior)
    candidates.sort(key=lambda p: (("from_synapses" not in p.name), len(p.name)))
    return candidates[0]


def _filter_ids_with_existing_swc(
    cfg: Dict[str, Any],
    ids: List[int],
    *,
    use_tqdm: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Keep only neuron ids with resolvable SWC files under morph_swc_dir or swc_dir.
    Returns (kept_ids, dropped_ids_missing_swc).
    """
    if not ids:
        return [], []
    primary_root = cfg.get("morph_swc_dir")
    base_root = Path(cfg["swc_dir"]).expanduser().resolve()
    roots: List[Path] = []
    if primary_root not in (None, "", False):
        roots.append(Path(primary_root).expanduser().resolve())
    roots.append(base_root)

    id_set = set()
    for idx_root in roots:
        idx = _get_swc_id_index(idx_root, use_tqdm=use_tqdm)
        id_set.update(idx["ids"])

    kept: List[int] = []
    dropped: List[int] = []
    for nid in _maybe_tqdm(
        ids,
        total=len(ids),
        desc="Checking SWC availability",
        use_tqdm=use_tqdm,
    ):
        n = int(nid)
        if n in id_set:
            kept.append(n)
        else:
            dropped.append(n)
    print(
        "[swc] availability summary "
        f"requested={len(ids):,} kept={len(kept):,} dropped={len(dropped):,}"
    )
    return kept, dropped


def _filter_edges_to_loaded_nodes(
    edges_path: Path,
    loaded_ids: List[int],
    *,
    runs_root: Path,
    run_id: str,
) -> Path:
    """
    Write a filtered edge overlay containing only edges among loaded_ids.
    If no filtering is needed, returns the original edges_path.
    """
    try:
        df = _read_edges_table_for_sanity(edges_path)
    except Exception as e:
        print(f"[edges] warning: could not read edges for filtering ({e}); using {edges_path}")
        return edges_path

    if not {"pre_id", "post_id"}.issubset(df.columns):
        print(f"[edges] warning: edges missing pre_id/post_id; using {edges_path}")
        return edges_path

    keep = set(int(x) for x in loaded_ids)
    pre = pd.to_numeric(df["pre_id"], errors="coerce")
    post = pd.to_numeric(df["post_id"], errors="coerce")
    mask = pre.isin(list(keep)) & post.isin(list(keep))

    if bool(mask.all()):
        return edges_path

    df2 = df.loc[mask].copy()
    overlay_dir = (runs_root / "_edge_cache_overlays").resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = (overlay_dir / f"{run_id}_filtered_loaded_edges.csv").resolve()
    df2.to_csv(out_path, index=False)
    print(
        f"[edges] filtered to loaded SWC-backed nodes: kept {len(df2):,}/{len(df):,} rows -> {out_path}"
    )
    return out_path


def _resolve_seeds(cfg: Dict[str, Any], neuron_ids: List[int]) -> List[int]:
    if cfg.get("seeds"):
        return [int(x) for x in cfg["seeds"]]
    # default: all neurons in population are seeds (your confirmed behavior)
    return [int(x) for x in neuron_ids]


def _resolve_clamp_site(cfg: Dict[str, Any]) -> str:
    """
    Clamp site policy:
      - honors loader-mapped cfg['iclamp_location'] from stim.iclamp.location
      - falls back to 'ais' to match Master-Copy2 behavior
    """
    site = str(cfg.get("iclamp_location", "ais")).strip().lower()
    if site not in {"soma", "ais"}:
        print(f"[warn] Unknown iclamp location '{site}', falling back to 'ais'")
        return "ais"
    return site


def _resolve_neg_pulse(cfg: Dict[str, Any], fallback_site: str) -> Optional[Dict[str, Any]]:
    neg = cfg.get("neg_pulse") or {}
    if not isinstance(neg, dict):
        return None
    if not bool(neg.get("enabled", False)):
        return None

    try:
        amp = float(neg.get("amp_nA", 0.0))
        delay = float(neg.get("delay_ms", 0.0))
        dur = float(neg.get("dur_ms", 0.0))
    except Exception as e:
        print(f"[warn] Invalid neg_pulse spec ({e}); skipping second pulse.")
        return None

    if dur <= 0.0:
        print("[warn] neg_pulse.dur_ms <= 0; skipping second pulse.")
        return None

    site = str(neg.get("location", fallback_site)).strip().lower()
    if site not in {"soma", "ais"}:
        print(f"[warn] Unknown neg_pulse location '{site}', falling back to '{fallback_site}'")
        site = str(fallback_site)

    return {
        "enabled": True,
        "amp": amp,
        "delay": delay,
        "dur": dur,
        "site": site,
    }


def _clamp_site_for_cell(cell, site_name: str):
    return cell.axon_ais_site() if str(site_name).lower() == "ais" else cell.soma_site()


def _resolve_pulse_train(cfg: Dict[str, Any], fallback_site: str) -> Optional[Dict[str, Any]]:
    tr = cfg.get("pulse_train") or {}
    if not isinstance(tr, dict):
        return None
    if not bool(tr.get("enabled", False)):
        return None

    freq_raw = tr.get("freq_hz", tr.get("frequency_hz"))
    if freq_raw is None:
        print("[warn] pulse_train enabled but freq_hz is missing; skipping pulse_train.")
        return None

    try:
        freq_hz = float(freq_raw)
        amp = float(tr.get("amp_nA", cfg.get("iclamp_amp_nA", 0.0)))
        delay_ms = float(tr.get("delay_ms", cfg.get("iclamp_delay_ms", 0.0)))
        dur_ms = float(tr.get("dur_ms", cfg.get("iclamp_dur_ms", 0.0)))
        tstop_ms = float(cfg.get("tstop_ms", 0.0))
    except Exception as e:
        print(f"[warn] Invalid pulse_train values ({e}); skipping pulse_train.")
        return None

    if freq_hz <= 0.0:
        print("[warn] pulse_train.freq_hz must be > 0; skipping pulse_train.")
        return None
    if dur_ms <= 0.0:
        print("[warn] pulse_train.dur_ms must be > 0; skipping pulse_train.")
        return None

    stop_ms_raw = tr.get("stop_ms", None)
    if stop_ms_raw is None:
        stop_ms = tstop_ms
    else:
        try:
            stop_ms = float(stop_ms_raw)
        except Exception:
            print("[warn] pulse_train.stop_ms is invalid; using tstop_ms.")
            stop_ms = tstop_ms
    stop_ms = min(stop_ms, tstop_ms)

    delay_ms = max(0.0, delay_ms)
    if stop_ms <= delay_ms:
        print("[warn] pulse_train.stop_ms <= delay_ms; skipping pulse_train.")
        return None

    site = str(tr.get("location", fallback_site)).strip().lower()
    if site not in {"soma", "ais"}:
        print(f"[warn] Unknown pulse_train location '{site}', falling back to '{fallback_site}'")
        site = str(fallback_site)

    period_ms = 1000.0 / freq_hz
    max_pulses_raw = tr.get("max_pulses", None)
    max_pulses: Optional[int] = None
    if max_pulses_raw is not None:
        try:
            max_pulses = max(0, int(max_pulses_raw))
        except Exception:
            max_pulses = None

    hard_cap = 200000
    if max_pulses is None:
        max_pulses = hard_cap
    else:
        max_pulses = min(max_pulses, hard_cap)

    delays: List[float] = []
    eps = 1e-12
    for i in range(int(max_pulses)):
        t = delay_ms + i * period_ms
        if t > stop_ms + eps:
            break
        if t + dur_ms > tstop_ms + eps:
            break
        delays.append(float(t))

    if not delays:
        print("[warn] pulse_train produced zero pulses within tstop_ms; skipping pulse_train.")
        return None

    return {
        "enabled": True,
        "freq_hz": float(freq_hz),
        "amp": float(amp),
        "dur": float(dur_ms),
        "site": str(site),
        "delays": delays,
        "include_base_iclamp": bool(tr.get("include_base_iclamp", False)),
    }


def _load_master_ids_for_hemilineage(master_csv: Path, hemilineage_label: str) -> List[int]:
    df = pd.read_csv(master_csv, low_memory=False)
    col = "hemilineage"
    if col not in df.columns:
        raise ValueError(f"Master CSV is missing '{col}' column: {master_csv}")
    ids = (
        df.loc[df[col].astype(str) == str(hemilineage_label), "bodyId"]
        .dropna()
        .astype(int)
        .tolist()
    )
    return ids


def _record_setup(net: Network, cfg: Dict[str, Any], seeds: List[int], neuron_ids: List[int]) -> Dict[int, Any]:
    net.record_time()

    rec = cfg.get("record", {}) or {}
    soma_spec = rec.get("soma_v", "seeds")
    spike_spec = rec.get("spikes", "seeds")
    thresh = float(rec.get("spike_thresh_mV", 0.0))

    def _resolve(spec):
        if spec in (None, "none", False):
            return []
        if spec == "seeds":
            return list(seeds)
        if spec == "all":
            return list(neuron_ids)
        if isinstance(spec, (list, tuple, set)):
            return [int(x) for x in spec]
        return []

    soma_ids = _resolve(soma_spec)
    spike_ids = _resolve(spike_spec)

    for nid in soma_ids:
        net.record_soma(int(nid))

    spike_map: Dict[int, Any] = {}
    for nid in spike_ids:
        spikes = net.count_spikes(int(nid), thresh=thresh)
        if spikes is not None:
            spike_map[int(nid)] = spikes

    return spike_map


def _dump_cell_biophys(net: Network, neuron_ids: List[int], out_dir: Path) -> None:
    rows = []
    for nid in neuron_ids:
        if bool(getattr(net, "is_distributed", False)) and not bool(net.is_local_gid(int(nid))):
            continue
        try:
            cell = net.ensure_cell(int(nid))
            soma_sec = cell.soma_sec
            soma_seg = soma_sec(0.5)
            soma_has_hh = hasattr(soma_seg, "gnabar_hh")
            soma_gnabar = float(getattr(soma_seg, "gnabar_hh", float("nan")))
            soma_gkbar = float(getattr(soma_seg, "gkbar_hh", float("nan")))
            soma_gl = float(getattr(soma_seg, "gl_hh", float("nan")))
            soma_e_pas = float(getattr(soma_seg, "e_pas", float("nan")))

            ais_sec, ais_x = cell.axon_ais_site()
            ais_seg = ais_sec(float(ais_x))
            ais_has_hh = hasattr(ais_seg, "gnabar_hh")
            ais_gnabar = float(getattr(ais_seg, "gnabar_hh", float("nan")))
            ais_gkbar = float(getattr(ais_seg, "gkbar_hh", float("nan")))
            ais_gl = float(getattr(ais_seg, "gl_hh", float("nan")))
            ais_e_pas = float(getattr(ais_seg, "e_pas", float("nan")))

            rows.append(
                {
                    "neuron_id": int(nid),
                    "soma_sec": str(soma_sec.name()),
                    "soma_has_hh": bool(soma_has_hh),
                    "soma_gnabar_hh": soma_gnabar,
                    "soma_gkbar_hh": soma_gkbar,
                    "soma_gl_hh": soma_gl,
                    "soma_e_pas": soma_e_pas,
                    "ais_sec": str(ais_sec.name()),
                    "ais_x": float(ais_x),
                    "ais_has_hh": bool(ais_has_hh),
                    "ais_gnabar_hh": ais_gnabar,
                    "ais_gkbar_hh": ais_gkbar,
                    "ais_gl_hh": ais_gl,
                    "ais_e_pas": ais_e_pas,
                }
            )
        except Exception as e:
            rows.append({"neuron_id": int(nid), "error": str(e)})

    if bool(getattr(net, "is_distributed", False)):
        gathered = net._pc.py_gather(rows, int(getattr(net, "_root_rank", 0)))
        if not bool(getattr(net, "is_root_rank", False)):
            return
        rows = [row for chunk in (gathered or []) if isinstance(chunk, list) for row in chunk]

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "cell_biophys.csv", index=False)


def run_walking_simulation(
    user_config: Dict[str, Any],
    *,
    strict: bool = True,
    pre_run_hook=None,
    return_net: bool = False,
    ) -> Union[Path, Tuple[Path, Network]]:

    """
    Unified runner that matches Phase 2 Master-Copy1 behavior and extends it to:
      - single neuron
      - custom network (explicit ids; edges file OR edge_cache)
      - hemilineage (build/reuse edges)
    Writes outputs to:
      <swc_dir>/hemi_runs/<run_id>/
    """
    cfg = build_config(user_config, strict=strict)
    net: Optional[Network] = None
    keep_net = False
    run_t0 = time.perf_counter()
    phase_timings: List[Dict[str, Any]] = []
    sim_wall_s = float("nan")
    backend = "unknown"
    integrator = "unknown"
    out_dir: Optional[Path] = None

    try:
        swc_root = Path(cfg["swc_dir"]).expanduser().resolve()
        edges_root = Path(cfg["edges_root"]).expanduser().resolve()
        runs_root = Path(cfg["runs_root"]).expanduser().resolve()
        master_csv = Path(cfg["master_csv"]).expanduser().resolve()

        sel = cfg.get("selection", {}) or {}
        mode = sel.get("mode")

        # -------------------------------------------------------------------------
        # Resolve neuron_ids and edges file
        # -------------------------------------------------------------------------
        neuron_ids: List[int]
        edges_path: Optional[Path] = None
        with _phase_timer("resolve selection + edges", collector=phase_timings):
            if mode == "single":
                nid = sel.get("neuron_id")
                if nid is None:
                    raise ValueError("selection.neuron_id is required for mode='single'")
                neuron_ids = [int(nid)]

            elif mode == "custom":
                ids = sel.get("neuron_ids")
                if not ids:
                    raise ValueError("selection.neuron_ids is required for mode='custom'")
                neuron_ids = [int(x) for x in ids]

                use_edge_cache = bool((cfg.get("edge_cache") or {}).get("enabled", False))
                if use_edge_cache:
                    # If provided, keep manual edges_path as a safety fallback in case the cache
                    # query returns a disconnected/irrelevant subset for this custom run.
                    fallback_edges_path: Optional[Path] = None
                    fallback_raw = cfg.get("edges_path")
                    if fallback_raw:
                        p = Path(str(fallback_raw)).expanduser().resolve()
                        if p.exists():
                            fallback_edges_path = p

                    with _phase_timer("edge-cache resolve", collector=phase_timings):
                        edges_path, resolved_ids = resolve_custom_edges_from_cache(
                            cfg,
                            loaded_ids=neuron_ids,
                            seed_ids=cfg.get("seeds"),
                        )
                    neuron_ids = [int(x) for x in resolved_ids]
                    sel["neuron_ids"] = list(neuron_ids)
                    cfg["selection"] = sel
                    cfg["edges_path"] = str(edges_path)

                    # Cache sanity gate:
                    # If the cached subset has no edges among loaded ids, or no seed->post edges,
                    # fall back to the explicit manual edges file when available.
                    if fallback_edges_path is not None and edges_path != fallback_edges_path:
                        try:
                            df_ec = _read_edges_table_for_sanity(edges_path)
                            if {"pre_id", "post_id"}.issubset(df_ec.columns):
                                pre = pd.to_numeric(df_ec["pre_id"], errors="coerce")
                                post = pd.to_numeric(df_ec["post_id"], errors="coerce")
                                loaded_set = set(int(x) for x in neuron_ids)
                                seeds_set = set(int(x) for x in (cfg.get("seeds") or []))
                                posts_set = loaded_set - seeds_set

                                has_within_loaded = bool(
                                    (pre.isin(list(loaded_set)) & post.isin(list(loaded_set))).any()
                                )
                                has_seed_to_posts = True
                                if seeds_set and posts_set:
                                    has_seed_to_posts = bool(
                                        (pre.isin(list(seeds_set)) & post.isin(list(posts_set))).any()
                                    )

                                if (not has_within_loaded) or (not has_seed_to_posts):
                                    edges_path = fallback_edges_path
                                    cfg["edges_path"] = str(edges_path)
                                    print(
                                        "[edge-cache] warning: cache subset failed connectivity sanity "
                                        f"(within_loaded={has_within_loaded}, seed_to_posts={has_seed_to_posts}); "
                                        f"falling back to explicit edges_path: {edges_path}"
                                    )
                        except Exception as e:
                            print(f"[edge-cache] warning: sanity check failed ({e}); keeping cache subset.")
                else:
                    ep = cfg.get("edges_path")
                    if not ep:
                        raise ValueError("CONFIG['edges_path'] is required for mode='custom' when edge_cache.enabled=False")
                    edges_path = Path(ep).expanduser().resolve()
                    if not edges_path.exists():
                        raise FileNotFoundError(f"edges_path not found: {edges_path}")

            elif mode == "hemilineage":
                label = sel.get("label")
                if not label:
                    raise ValueError("selection.label is required for mode='hemilineage'")

                neuron_ids = _load_master_ids_for_hemilineage(master_csv, str(label))
                if not neuron_ids:
                    raise ValueError(f"No neurons found for hemilineage '{label}' in master CSV")

                # Prefer explicit edges_path, then reuse, then build
                ep = cfg.get("edges_path")
                if ep:
                    edges_path = Path(ep).expanduser().resolve()
                    if not edges_path.exists():
                        raise FileNotFoundError(f"edges_path not found: {edges_path}")
                else:
                    edges_path = _find_edges_for_label(edges_root, str(label))
                    if edges_path is None:
                        seeds = _resolve_seeds(cfg, neuron_ids)
                        edges_path, _ = run_hemilineage(
                            hemilineage_label=str(label),
                            seeds=seeds,
                            swc_root=str(swc_root),
                            default_weight_uS=float(cfg.get("default_weight_uS", 6e-6)),
                            master_csv=str(master_csv),
                            edges_root=str(edges_root),
                            results_root=str(runs_root),
                            smoke_test=False,
                            pres_limit=None,
                        )

            else:
                raise ValueError("selection.mode must be one of: 'single'|'custom'|'hemilineage'")

        # Guard: edge-cache expansions can include nodes lacking SWC in current dataset.
        # Drop those nodes up-front so network build does not crash on find_swc().
        with _phase_timer("swc availability check", collector=phase_timings):
            neuron_ids, dropped_missing_swc = _filter_ids_with_existing_swc(
                cfg,
                neuron_ids,
                use_tqdm=bool(cfg.get("use_tqdm", True)),
            )
        if dropped_missing_swc:
            explicit_seeds = [int(x) for x in (cfg.get("seeds") or [])]
            dropped_seed_ids = sorted(set(dropped_missing_swc).intersection(explicit_seeds))
            if dropped_seed_ids:
                active_roots = [cfg.get("morph_swc_dir"), cfg.get("swc_dir")]
                active_roots = [str(x) for x in active_roots if x not in (None, "", False)]
                raise FileNotFoundError(
                    "Requested seed ids are missing SWCs in the active morphology roots "
                    f"({active_roots}): {dropped_seed_ids}"
                )
            preview = dropped_missing_swc[:20]
            suffix = " ..." if len(dropped_missing_swc) > 20 else ""
            active_roots = [cfg.get("morph_swc_dir"), cfg.get("swc_dir")]
            active_roots = [str(x) for x in active_roots if x not in (None, "", False)]
            print(
                "[swc] warning: dropping ids with no SWC in current morphology roots "
                f"{active_roots} ({len(dropped_missing_swc)}): {preview}{suffix}"
            )

            if mode == "custom":
                sel["neuron_ids"] = list(neuron_ids)
                cfg["selection"] = sel

            if edges_path is not None:
                edges_path = _filter_edges_to_loaded_nodes(
                    edges_path,
                    neuron_ids,
                    runs_root=runs_root,
                    run_id=str(cfg.get("run_id", "run")),
                )
                cfg["edges_path"] = str(edges_path)

        if not neuron_ids:
            raise RuntimeError(
                "No loadable neurons remain after SWC availability filtering. "
                "Verify SWC dataset coverage for the selected ids."
            )

        seeds = _resolve_seeds(cfg, neuron_ids)
        clamp_site = _resolve_clamp_site(cfg)
        neg_pulse = _resolve_neg_pulse(cfg, clamp_site)
        pulse_train = _resolve_pulse_train(cfg, clamp_site)

        # -------------------------------------------------------------------------
        # Build + wire
        # -------------------------------------------------------------------------
        spike_map: Dict[int, Any] = {}
        drivers: Dict[int, Dict[str, Any]] = {}

        if mode == "single":
            # Build a tiny network directly (no edges file required)
            net = Network(cfg)
            with _phase_timer("single: load cells", collector=phase_timings):
                for nid in _maybe_tqdm(
                    neuron_ids,
                    total=len(neuron_ids),
                    desc="Loading cells",
                    use_tqdm=bool(cfg.get("use_tqdm", True)),
                ):
                    net.ensure_cell(int(nid))

            # baseline passive, then activate seed for stimulation
            with _phase_timer("single: passive baseline", collector=phase_timings):
                cells_all = list(net.cells.values())
                for cell in _maybe_tqdm(
                    cells_all,
                    total=len(cells_all),
                    desc="Applying passive",
                    use_tqdm=bool(cfg.get("use_tqdm", True)),
                ):
                    make_passive(cell, cfg)

            with _phase_timer("single: activate seeds", collector=phase_timings):
                for nid in _maybe_tqdm(
                    seeds,
                    total=len(seeds),
                    desc="Activating seeds",
                    use_tqdm=bool(cfg.get("use_tqdm", True)),
                ):
                    make_active(net.cells[int(nid)], cfg, cfg["pre_soma_hh"], cfg["pre_branch_hh"])

            with _phase_timer("single: apply gaps", collector=phase_timings):
                apply_gap_config(net, cfg)

            # record
            with _phase_timer("single: record setup", collector=phase_timings):
                spike_map = _record_setup(net, cfg, seeds, neuron_ids)

            # stimulate all seeds (IClamp options come from drivers specs)
            with _phase_timer("single: place IClamps", collector=phase_timings):
                for nid in _maybe_tqdm(
                    seeds,
                    total=len(seeds),
                    desc="Placing IClamps",
                    use_tqdm=bool(cfg.get("use_tqdm", True)),
                ):
                    cell = net.cells[int(nid)]
                    site = _clamp_site_for_cell(cell, clamp_site)
                    if pulse_train is None or bool(pulse_train.get("include_base_iclamp", False)):
                        net.add_iclamp_site(
                            int(nid),
                            site,
                            amp_nA=float(cfg["iclamp_amp_nA"]),
                            delay_ms=float(cfg["iclamp_delay_ms"]),
                            dur_ms=float(cfg["iclamp_dur_ms"]),
                            kind="base",
                        )
                    if pulse_train is not None:
                        tr_site = _clamp_site_for_cell(cell, str(pulse_train["site"]))
                        for dly in pulse_train.get("delays", []):
                            net.add_iclamp_site(
                                int(nid),
                                tr_site,
                                amp_nA=float(pulse_train["amp"]),
                                delay_ms=float(dly),
                                dur_ms=float(pulse_train["dur"]),
                                kind="pulse_train",
                            )
                    if neg_pulse is not None:
                        neg_site = _clamp_site_for_cell(cell, str(neg_pulse["site"]))
                        net.add_iclamp_site(
                            int(nid),
                            neg_site,
                            amp_nA=float(neg_pulse["amp"]),
                            delay_ms=float(neg_pulse["delay"]),
                            dur_ms=float(neg_pulse["dur"]),
                            kind="neg_pulse",
                        )

        else:
            assert edges_path is not None
            cfg["edges_csv"] = str(edges_path)

            drivers = {
                int(nid): {
                    "amp": float(cfg["iclamp_amp_nA"]),
                    "delay": float(cfg["iclamp_delay_ms"]),
                    "dur": float(cfg["iclamp_dur_ms"]),
                    "site": clamp_site,
                    "neg_pulse": (dict(neg_pulse) if neg_pulse is not None else None),
                    "pulse_train": (dict(pulse_train) if pulse_train is not None else None),
                }
                for nid in seeds
            }

            active_posts_mode = str(cfg.get("active_posts_mode", "all_selected")).strip().lower()
            if active_posts_mode in {"drivers_only", "seeds_only", "drivers"}:
                active_post_ids = sorted(int(nid) for nid in seeds)
            elif active_posts_mode in {"none", "passive_only"}:
                active_post_ids = []
            else:
                active_post_ids = sorted(int(nid) for nid in neuron_ids)

            with _phase_timer("build driven subset", collector=phase_timings):
                net, _df_sub = build_network_driven_subset(
                    cfg,
                    drivers=drivers,
                    active_posts=active_post_ids,
                    loaded_nodes=neuron_ids,
                )

            with _phase_timer("apply gaps", collector=phase_timings):
                apply_gap_config(net, cfg)

            # record
            with _phase_timer("record setup", collector=phase_timings):
                spike_map = _record_setup(net, cfg, seeds, neuron_ids)

        # -------------------------------------------------------------------------
        # Run + save
        # -------------------------------------------------------------------------
        run_id = str(cfg["run_id"])
        out_dir = (runs_root / run_id).resolve()
        with _phase_timer("prepare output + save config", collector=phase_timings):
            if not bool(getattr(net, "is_distributed", False)) or bool(getattr(net, "is_root_rank", False)):
                out_dir.mkdir(parents=True, exist_ok=True)
                save_config(cfg, out_dir)
            _dump_cell_biophys(net, neuron_ids, out_dir)
            if bool(getattr(net, "is_distributed", False)):
                net._pc.barrier()
        print("[debug] drivers passed to network:", list(drivers.keys()))
        print("[debug] clamp site =", clamp_site)

        from neuron import h

        print("[debug] tstop_ms =", float(cfg["tstop_ms"]))
        print("[debug] dt_ms    =", float(cfg["dt_ms"]))

        print("[debug] cfg iclamp amp/delay/dur:",
              float(cfg.get("iclamp_amp_nA", -999)),
              float(cfg.get("iclamp_delay_ms", -999)),
              float(cfg.get("iclamp_dur_ms", -999)))
        if pulse_train is not None:
            print(
                "[debug] cfg pulse_train freq/amp/dur/site/pulses:",
                float(pulse_train.get("freq_hz", -999)),
                float(pulse_train.get("amp", -999)),
                float(pulse_train.get("dur", -999)),
                str(pulse_train.get("site", "")),
                int(len(pulse_train.get("delays", []))),
            )
        if neg_pulse is not None:
            print(
                "[debug] cfg neg_pulse amp/delay/dur/site:",
                float(neg_pulse.get("amp", -999)),
                float(neg_pulse.get("delay", -999)),
                float(neg_pulse.get("dur", -999)),
                str(neg_pulse.get("site", "")),
            )

        print("[debug] net.iclamps =", len(getattr(net, "iclamps", [])))
        if getattr(net, "iclamps", []):
            s0 = net.iclamps[0]
            print("[debug] first IClamp amp/delay/dur:",
                  float(s0.amp), float(s0.delay), float(s0.dur))


        # -------------------------------------------------------------------------
        # Optional debug hook: attach extra recordings before the sim runs
        # -------------------------------------------------------------------------
        if pre_run_hook is not None:
            with _phase_timer("pre-run hook", collector=phase_timings):
                pre_run_hook(net, cfg, out_dir)

        print("[phase] simulate: start")
        t_start = time.perf_counter()
        net.run(tstop_ms=float(cfg["tstop_ms"]), dt_ms=float(cfg["dt_ms"]), show_progress=bool(cfg.get("progress", True)))
        sim_wall_s = time.perf_counter() - t_start
        phase_timings.append({"label": "simulate", "wall_s": float(sim_wall_s)})

        cv_cfg = cfg.get("cvode", {}) or {}
        cv_enabled = bool(cv_cfg.get("enabled", False)) if isinstance(cv_cfg, dict) else False
        backend = "coreneuron" if bool(getattr(net, "_coreneuron_on", False)) else "neuron"
        if cv_enabled:
            integrator = "cvode"
        else:
            integrator = f"fixed-step(dt_ms={float(cfg['dt_ms'])})"
        print(f"[timing] sim_wall_s={sim_wall_s:.3f} backend={backend} integrator={integrator}")
        print(f"[phase] simulate: done wall_s={sim_wall_s:.3f}")

        with _phase_timer("save outputs", collector=phase_timings):
            io_workers_raw = cfg.get("io_workers", None)
            try:
                io_workers = max(1, int(io_workers_raw)) if io_workers_raw not in (None, "", False) else 1
            except Exception:
                io_workers = 1

            if bool(getattr(net, "is_distributed", False)):
                save_records(net, out_dir)
                save_spikes(spike_map, out_dir, net=net)
                net._pc.barrier()
            elif io_workers > 1:
                with ThreadPoolExecutor(max_workers=min(int(io_workers), 2)) as ex:
                    fut_records = ex.submit(save_records, net, out_dir)
                    fut_spikes = ex.submit(save_spikes, spike_map, out_dir)
                    fut_records.result()
                    fut_spikes.result()
            else:
                save_records(net, out_dir)
                save_spikes(spike_map, out_dir)

        total_wall_s = time.perf_counter() - run_t0
        save_phase_wall_s = next(
            (float(row["wall_s"]) for row in phase_timings if str(row.get("label")) == "save outputs"),
            float("nan"),
        )
        timing_summary = {
            "run_id": run_id,
            "mode": str(mode),
            "selection_count": int(len(neuron_ids)),
            "seed_count": int(len(seeds)),
            "build_wall_s": float(t_start - run_t0),
            "pre_sim_wall_s": float(t_start - run_t0),
            "sim_wall_s": float(sim_wall_s),
            "post_sim_save_wall_s": float(save_phase_wall_s),
            "total_wall_s": float(total_wall_s),
            "backend": str(backend),
            "integrator": str(integrator),
            "phase_rows": phase_timings,
        }
        if not bool(getattr(net, "is_distributed", False)) or bool(getattr(net, "is_root_rank", False)):
            (out_dir / "_phase_timings.json").write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")

        if return_net:
            keep_net = True
            return out_dir, net
        return out_dir
    finally:
        if net is not None and not keep_net:
            try:
                net.close()
            except Exception as e:
                print(f"[warn] network cleanup failed: {e}")
