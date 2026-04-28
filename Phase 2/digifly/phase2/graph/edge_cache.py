from __future__ import annotations

import json
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from digifly.phase2.data.synapses_loader import _detect_and_scale_synmap


def _dedupe_ints(values: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        x = int(v)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _normalize_optional_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        f = float(v)
        if not np.isfinite(f):
            return None
        return int(f)
    except Exception:
        return None


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".feather", ".ftr"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported edge source format: {path}")


def _col_num(df: pd.DataFrame, names: Sequence[str], default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _normalize_edges_df(df: pd.DataFrame, *, default_weight_uS: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    out["pre_id"] = _col_num(df, ("pre_id", "preId"))
    out["post_id"] = _col_num(df, ("post_id", "postId"))
    out = out.dropna(subset=["pre_id", "post_id"]).copy()
    out["pre_id"] = out["pre_id"].astype("int64")
    out["post_id"] = out["post_id"].astype("int64")

    # Coordinates and syn indices (optional).
    out["post_x"] = _col_num(df.loc[out.index], ("post_x", "x_post", "x"))
    out["post_y"] = _col_num(df.loc[out.index], ("post_y", "y_post", "y"))
    out["post_z"] = _col_num(df.loc[out.index], ("post_z", "z_post", "z"))
    out["pre_x"] = _col_num(df.loc[out.index], ("pre_x", "x_pre"))
    out["pre_y"] = _col_num(df.loc[out.index], ("pre_y", "y_pre"))
    out["pre_z"] = _col_num(df.loc[out.index], ("pre_z", "z_pre"))

    syn_idx = _col_num(df.loc[out.index], ("syn_index", "post_syn_index", "pre_syn_index"))
    if syn_idx.isna().all():
        syn_idx = pd.Series(np.arange(len(out), dtype=np.int64), index=out.index)
    out["syn_index"] = syn_idx
    out["pre_syn_index"] = _col_num(df.loc[out.index], ("pre_syn_index",), default=np.nan).fillna(out["syn_index"])
    out["post_syn_index"] = _col_num(df.loc[out.index], ("post_syn_index",), default=np.nan).fillna(out["syn_index"])

    # Timing / mechanism columns expected by the runner.
    out["weight_uS"] = _col_num(df.loc[out.index], ("weight_uS", "w_uS"), default=np.nan).fillna(float(default_weight_uS))
    out["delay_ms"] = _col_num(df.loc[out.index], ("delay_ms", "delay"), default=np.nan)
    out["tau1_ms"] = _col_num(df.loc[out.index], ("tau1_ms", "tau1"), default=np.nan)
    out["tau2_ms"] = _col_num(df.loc[out.index], ("tau2_ms", "tau2"), default=np.nan)
    out["syn_e_rev_mV"] = _col_num(df.loc[out.index], ("syn_e_rev_mV", "e_rev_mV", "erev_mV"), default=np.nan)

    # Keep rows alive through epsilon filtering in builders.
    out["pre_match_um"] = _col_num(df.loc[out.index], ("pre_match_um",), default=0.0).fillna(0.0)
    out["post_match_um"] = _col_num(df.loc[out.index], ("post_match_um",), default=0.0).fillna(0.0)

    return out.reset_index(drop=True)


def _infer_pre_id_from_syn_file(path: Path) -> int | None:
    m = re.search(r"(\d+)_synapses_new\.csv$", path.name)
    if m:
        return int(m.group(1))
    # Common layout: .../<nid>/<nid>_synapses_new.csv
    if path.parent.name.isdigit():
        return int(path.parent.name)
    return None


def _rows_from_synapse_csv(path: Path, *, default_weight_uS: float) -> pd.DataFrame:
    pre_id = _infer_pre_id_from_syn_file(path)
    if pre_id is None:
        raise ValueError(f"Cannot infer pre_id from synapse file name: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "post_id" not in df.columns:
        raise ValueError(f"Missing post_id column: {path}")

    if "type" not in df.columns:
        df["type"] = "pre"
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower() == "pre"].copy()

    df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce")
    df = df[df["post_id"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    for c in ("x", "y", "z"):
        if c not in df.columns:
            df[c] = np.nan

    df = _detect_and_scale_synmap(df, int(pre_id), verbose=False)
    out = pd.DataFrame(index=df.index.copy())
    out["pre_id"] = int(pre_id)
    out["post_id"] = df["post_id"].astype("int64")
    out["weight_uS"] = float(default_weight_uS)
    out["post_x"] = pd.to_numeric(df["x"], errors="coerce")
    out["post_y"] = pd.to_numeric(df["y"], errors="coerce")
    out["post_z"] = pd.to_numeric(df["z"], errors="coerce")
    out["pre_x"] = _col_num(df, ("pre_x", "x_pre"), default=np.nan)
    out["pre_y"] = _col_num(df, ("pre_y", "y_pre"), default=np.nan)
    out["pre_z"] = _col_num(df, ("pre_z", "z_pre"), default=np.nan)
    out["syn_index"] = np.arange(len(df), dtype=np.int64)
    out["pre_syn_index"] = out["syn_index"]
    out["post_syn_index"] = out["syn_index"]
    out["delay_ms"] = np.nan
    out["tau1_ms"] = np.nan
    out["tau2_ms"] = np.nan
    out["syn_e_rev_mV"] = np.nan
    out["pre_match_um"] = 0.0
    out["post_match_um"] = 0.0
    return out.reset_index(drop=True)


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


def _sqlite_init(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            pre_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            weight_uS REAL,
            delay_ms REAL,
            tau1_ms REAL,
            tau2_ms REAL,
            syn_e_rev_mV REAL,
            pre_x REAL,
            pre_y REAL,
            pre_z REAL,
            post_x REAL,
            post_y REAL,
            post_z REAL,
            syn_index INTEGER,
            pre_syn_index INTEGER,
            post_syn_index INTEGER,
            pre_match_um REAL,
            post_match_um REAL
        )
        """
    )
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")


def _sqlite_reindex(con: sqlite3.Connection) -> None:
    con.execute("CREATE INDEX IF NOT EXISTS idx_edges_pre_id ON edges(pre_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_edges_post_id ON edges(post_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_edges_pre_post ON edges(pre_id, post_id)")


def _sqlite_set_meta(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)",
        (str(key), json.dumps(value)),
    )


def _sqlite_get_row_count(con: sqlite3.Connection) -> int:
    cur = con.execute("SELECT COUNT(*) FROM edges")
    return int(cur.fetchone()[0])


def _discover_default_edge_sources(edges_root: Path) -> List[Path]:
    pats = ("*_from_synapses.csv", "*_from_synapses.parquet", "*_from_synapses.feather")
    files: List[Path] = []
    for pat in pats:
        files.extend(sorted(edges_root.glob(pat)))
    return files


def _build_cache_from_edge_files(
    con: sqlite3.Connection,
    source_paths: Sequence[Path],
    *,
    default_weight_uS: float,
) -> Dict[str, Any]:
    total_rows = 0
    used_sources: List[str] = []
    start = time.perf_counter()
    for src in source_paths:
        if not src.exists():
            raise FileNotFoundError(f"edge_cache source not found: {src}")
        raw = _read_table(src)
        norm = _normalize_edges_df(raw, default_weight_uS=default_weight_uS)
        if norm.empty:
            continue
        norm.to_sql("edges", con, if_exists="append", index=False, chunksize=25_000)
        total_rows += int(len(norm))
        used_sources.append(str(src))
        print(f"[edge-cache] loaded {len(norm):,} rows from {src.name}")
    elapsed = time.perf_counter() - start
    return {"rows_inserted": total_rows, "sources": used_sources, "elapsed_s": elapsed}


def _build_cache_from_synapses(
    con: sqlite3.Connection,
    swc_root: Path,
    *,
    default_weight_uS: float,
    workers: int = 1,
    chunk_size: int = 64,
) -> Dict[str, Any]:
    files = sorted(swc_root.rglob("*_synapses_new.csv"))
    total_rows = 0
    ok_files = 0
    bad_files = 0
    start = time.perf_counter()
    workers_use = max(1, int(workers))
    chunk_size_use = max(1, int(chunk_size))
    if workers_use <= 1 or len(files) <= 1:
        for i, p in enumerate(files, start=1):
            try:
                rows = _rows_from_synapse_csv(p, default_weight_uS=default_weight_uS)
                if not rows.empty:
                    rows.to_sql("edges", con, if_exists="append", index=False, chunksize=25_000)
                    total_rows += int(len(rows))
                ok_files += 1
            except Exception as e:
                bad_files += 1
                if bad_files <= 10:
                    print(f"[edge-cache] skip {p}: {type(e).__name__}: {e}")
            if i % 500 == 0:
                print(f"[edge-cache] scanned {i:,}/{len(files):,} synapse files  rows={total_rows:,}")
    else:
        chunks: List[List[str]] = [
            [str(p) for p in files[i : i + chunk_size_use]]
            for i in range(0, len(files), chunk_size_use)
        ]
        chunk_total = len(chunks)
        print(
            f"[edge-cache] parallel synapse-cache build with workers={workers_use} "
            f"chunk_size={chunk_size_use} files={len(files):,} chunks={chunk_total:,}"
        )
        with ProcessPoolExecutor(max_workers=min(workers_use, chunk_total)) as executor:
            futures = {
                executor.submit(
                    _rows_from_synapse_chunk,
                    chunk,
                    default_weight_uS=default_weight_uS,
                ): idx
                for idx, chunk in enumerate(chunks, start=1)
            }
            finished_chunks = 0
            for fut in as_completed(futures):
                finished_chunks += 1
                payload = fut.result()
                rows = payload.pop("rows_df")
                if not rows.empty:
                    rows.to_sql("edges", con, if_exists="append", index=False, chunksize=25_000)
                    total_rows += int(len(rows))
                ok_files += int(payload.get("ok_files", 0))
                bad_files += int(payload.get("bad_files", 0))
                for msg in payload.get("bad_messages", []):
                    if bad_files <= 10:
                        print(f"[edge-cache] skip {msg}")
                if finished_chunks % 10 == 0 or finished_chunks == chunk_total:
                    scanned = min(finished_chunks * chunk_size_use, len(files))
                    print(
                        f"[edge-cache] scanned {scanned:,}/{len(files):,} synapse files  "
                        f"rows={total_rows:,} chunks={finished_chunks:,}/{chunk_total:,}"
                    )
    elapsed = time.perf_counter() - start
    return {
        "rows_inserted": total_rows,
        "synapse_files_total": len(files),
        "synapse_files_ok": ok_files,
        "synapse_files_failed": bad_files,
        "workers": workers_use,
        "chunk_size": chunk_size_use,
        "elapsed_s": elapsed,
    }


def _rows_from_synapse_chunk(
    files: Sequence[str],
    *,
    default_weight_uS: float,
) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    ok_files = 0
    bad_files = 0
    bad_messages: List[str] = []
    for raw_path in files:
        path = Path(raw_path)
        try:
            rows = _rows_from_synapse_csv(path, default_weight_uS=default_weight_uS)
            if not rows.empty:
                frames.append(rows)
            ok_files += 1
        except Exception as exc:
            bad_files += 1
            if len(bad_messages) < 3:
                bad_messages.append(f"{path}: {type(exc).__name__}: {exc}")
    rows_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "rows_df": rows_df,
        "ok_files": ok_files,
        "bad_files": bad_files,
        "bad_messages": bad_messages,
    }


def ensure_edge_cache(cfg: Dict[str, Any]) -> Path:
    swc_root = Path(cfg["swc_dir"]).expanduser().resolve()
    edges_root = Path(cfg["edges_root"]).expanduser().resolve()
    ecfg = dict(cfg.get("edge_cache") or {})

    db_path = Path(ecfg.get("db_path") or (edges_root / "master_edges_cache.sqlite")).expanduser().resolve()
    build_if_missing = bool(ecfg.get("build_if_missing", True))
    force_rebuild = bool(ecfg.get("force_rebuild", False))
    build_mode = str(ecfg.get("build_mode", "from_edges_files")).strip().lower()
    build_workers = max(1, int(ecfg.get("workers", 1)))
    build_chunk_size = max(1, int(ecfg.get("chunk_size", 64)))
    default_weight_uS = float(cfg.get("default_weight_uS", 6e-6))

    if force_rebuild and db_path.exists():
        db_path.unlink()

    if db_path.exists():
        return db_path
    if not build_if_missing:
        raise FileNotFoundError(f"edge_cache db not found and build_if_missing=False: {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _sqlite_connect(db_path)
    try:
        _sqlite_init(con)
        if build_mode == "from_edges_files":
            srcs = ecfg.get("source_paths") or []
            if srcs:
                source_paths = [Path(s).expanduser().resolve() for s in srcs]
            else:
                source_paths = _discover_default_edge_sources(edges_root)
            if not source_paths:
                raise FileNotFoundError(
                    "edge_cache build_mode='from_edges_files' found no default sources under "
                    f"{edges_root}. Set edge_cache.source_paths explicitly or switch to build_mode='from_synapses_csv'."
                )
            stats = _build_cache_from_edge_files(
                con,
                source_paths,
                default_weight_uS=default_weight_uS,
            )
        elif build_mode == "from_synapses_csv":
            stats = _build_cache_from_synapses(
                con,
                swc_root,
                default_weight_uS=default_weight_uS,
                workers=build_workers,
                chunk_size=build_chunk_size,
            )
        else:
            raise ValueError(f"Unsupported edge_cache.build_mode: {build_mode}")

        _sqlite_reindex(con)
        total_rows = _sqlite_get_row_count(con)
        meta = {
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_mode": build_mode,
            "swc_dir": str(swc_root),
            "edges_root": str(edges_root),
            "row_count": total_rows,
            "stats": stats,
        }
        _sqlite_set_meta(con, "build_info", meta)
        con.commit()
        print(f"[edge-cache] built {total_rows:,} rows -> {db_path}")
    finally:
        con.close()
    return db_path


def _sqlite_temp_ids(con: sqlite3.Connection, table_name: str, ids: Sequence[int]) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TEMP TABLE {table_name}(id INTEGER PRIMARY KEY)")
    con.executemany(f"INSERT INTO {table_name}(id) VALUES (?)", [(int(x),) for x in ids])


def _query_neighbors_one_hop(con: sqlite3.Connection, seed_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    if not seed_ids:
        return [], []
    _sqlite_temp_ids(con, "_seed_ids", seed_ids)
    out_rows = con.execute(
        "SELECT DISTINCT e.post_id FROM edges e JOIN _seed_ids s ON e.pre_id = s.id"
    ).fetchall()
    in_rows = con.execute(
        "SELECT DISTINCT e.pre_id FROM edges e JOIN _seed_ids s ON e.post_id = s.id"
    ).fetchall()
    outs = [int(r[0]) for r in out_rows if r[0] is not None]
    ins = [int(r[0]) for r in in_rows if r[0] is not None]
    return ins, outs


def _query_edges_for_nodes(con: sqlite3.Connection, node_ids: Sequence[int], *, max_rows: int | None = None) -> pd.DataFrame:
    _sqlite_temp_ids(con, "_node_ids", node_ids)
    q_count = (
        "SELECT COUNT(*) FROM edges e "
        "JOIN _node_ids p ON e.pre_id = p.id "
        "JOIN _node_ids q ON e.post_id = q.id"
    )
    n_rows = int(con.execute(q_count).fetchone()[0])
    if max_rows is not None and n_rows > int(max_rows):
        raise RuntimeError(
            f"edge_cache query returned {n_rows:,} rows, exceeding max_rows={int(max_rows):,}. "
            "Raise edge_cache.query.max_rows or reduce node set."
        )
    q = (
        "SELECT e.pre_id, e.post_id, e.weight_uS, e.delay_ms, e.tau1_ms, e.tau2_ms, e.syn_e_rev_mV, "
        "e.pre_x, e.pre_y, e.pre_z, e.post_x, e.post_y, e.post_z, "
        "e.syn_index, e.pre_syn_index, e.post_syn_index, e.pre_match_um, e.post_match_um "
        "FROM edges e "
        "JOIN _node_ids p ON e.pre_id = p.id "
        "JOIN _node_ids q ON e.post_id = q.id"
    )
    return pd.read_sql_query(q, con)


def _resolve_nodes_for_query(
    *,
    mode: str,
    loaded_ids: Sequence[int],
    seed_ids: Sequence[int],
    include_loaded: bool,
    ins: Sequence[int],
    outs: Sequence[int],
) -> List[int]:
    mode = mode.strip().lower()
    loaded = _dedupe_ints([int(x) for x in loaded_ids])
    seeds = _dedupe_ints([int(x) for x in seed_ids])
    if mode == "loaded_subgraph":
        return loaded
    if mode == "seed_io_1hop":
        nodes: List[int] = []
        seen = set()
        # Deterministic ordering: seeds, loaded ids (optional), then discovered neighbors.
        for x in seeds:
            if x not in seen:
                nodes.append(x)
                seen.add(x)
        if include_loaded:
            for x in loaded:
                if x not in seen:
                    nodes.append(x)
                    seen.add(x)
        for x in _dedupe_ints([int(v) for v in outs]):
            if x not in seen:
                nodes.append(x)
                seen.add(x)
        for x in _dedupe_ints([int(v) for v in ins]):
            if x not in seen:
                nodes.append(x)
                seen.add(x)
        return nodes
    raise ValueError(f"Unsupported edge_cache.query.mode: {mode}")


def resolve_custom_edges_from_cache(
    cfg: Dict[str, Any],
    *,
    loaded_ids: Sequence[int],
    seed_ids: Sequence[int] | None,
) -> Tuple[Path, List[int]]:
    db_path = ensure_edge_cache(cfg)
    ecfg = dict(cfg.get("edge_cache") or {})
    qcfg = dict(ecfg.get("query") or {})
    mode = str(qcfg.get("mode", "loaded_subgraph")).strip().lower()
    include_loaded = bool(qcfg.get("include_loaded", True))
    max_nodes = _normalize_optional_int(qcfg.get("max_nodes"))
    max_rows = _normalize_optional_int(qcfg.get("max_rows"))

    loaded = _dedupe_ints([int(x) for x in loaded_ids])
    seeds = _dedupe_ints([int(x) for x in (seed_ids if seed_ids else loaded)])
    if not loaded:
        raise ValueError("edge_cache custom mode requires non-empty selection.neuron_ids")

    con = _sqlite_connect(db_path)
    try:
        ins: List[int] = []
        outs: List[int] = []
        if mode == "seed_io_1hop":
            ins, outs = _query_neighbors_one_hop(con, seeds)
        nodes = _resolve_nodes_for_query(
            mode=mode,
            loaded_ids=loaded,
            seed_ids=seeds,
            include_loaded=include_loaded,
            ins=ins,
            outs=outs,
        )
        if max_nodes is not None and len(nodes) > int(max_nodes):
            raise RuntimeError(
                f"edge_cache node set size {len(nodes):,} exceeds max_nodes={int(max_nodes):,}. "
                "Raise edge_cache.query.max_nodes or tighten selection."
            )
        df = _query_edges_for_nodes(con, nodes, max_rows=max_rows)
    finally:
        con.close()

    overlay_dir = Path(ecfg.get("overlay_dir") or (Path(cfg["runs_root"]).expanduser().resolve() / "_edge_cache_overlays")).resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.get("run_id", "run"))
    out_path = overlay_dir / f"{run_id}_{mode}_edges.csv"
    df.to_csv(out_path, index=False)

    print(
        f"[edge-cache] mode={mode} loaded={len(loaded)} seeds={len(seeds)} "
        f"resolved_nodes={len(nodes)} edges={len(df):,}"
    )
    print(f"[edge-cache] subset edges csv -> {out_path}")
    return out_path, nodes
