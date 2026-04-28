from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from digifly.phase2.data.paths import _syn_csv_path
from digifly.phase2.data.synapses_loader import _detect_and_scale_synmap
from digifly.phase2.config.defaults import (
    HEMI_SWC_ROOT,
    HEMI_MASTER_CSV,
    HEMI_EDGES_ROOT,
    HEMI_RUN_ROOT,
    CONFIG,
)

try:
    tqdm  # noqa
except NameError:  # pragma: no cover
    from tqdm import tqdm  # type: ignore


# ============================================================================
# Edge construction from synapses_new.csv
# ============================================================================

def _hemi_fetch_edges_from_synapse_csvs(
    seeds: list[int],
    hemi_ids: list[int],
    label: str,
    swc_root: Path,
    default_weight_uS: float,
    *,
    one_row_per_synapse: bool = True,
    smoke_test: bool = False,
    pres_limit: int | None = None,
    workers: int = 16,
    smoke_seeds_only: bool = False,
) -> pd.DataFrame:
    """
    Build an edge table using local <id>_synapses_new.csv files.
    """

    swc_root = Path(swc_root)
    seeds = [int(s) for s in seeds]
    hemi_ids = [int(h) for h in hemi_ids]

    if smoke_test and smoke_seeds_only:
        pres_full = sorted(set(seeds))
        pres = pres_full if pres_limit is None else pres_full[:pres_limit]
    else:
        pres_full = sorted(set(seeds) | set(hemi_ids))
        pres = pres_full
        if smoke_test:
            pres = pres_full[: pres_limit or min(20, len(pres_full))]

    default_w = float(default_weight_uS)

    rows = []
    missing_files = []
    load_errors = []
    no_pre_rows = []

    start = time.perf_counter()

    def _process_pre(pre_id: int):
        local_rows = []
        local_missing = []
        local_errors = []
        local_no_pre = []

        path = _syn_csv_path(swc_root, pre_id)
        if not path or not Path(path).exists():
            local_missing.append(pre_id)
            return local_rows, local_missing, local_errors, local_no_pre

        try:
            df = pd.read_csv(path)
        except Exception as e:
            local_errors.append((pre_id, f"read_csv error: {e}"))
            return local_rows, local_missing, local_errors, local_no_pre

        if "post_id" not in df.columns:
            local_errors.append((pre_id, "missing post_id column"))
            return local_rows, local_missing, local_errors, local_no_pre

        if "type" in df.columns:
            df = df[df["type"].astype(str).str.lower() == "pre"].copy()

        df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce")
        df = df[df["post_id"].notna()].copy()

        if df.empty:
            local_no_pre.append(pre_id)
            return local_rows, local_missing, local_errors, local_no_pre

        for c in ("x", "y", "z"):
            if c not in df.columns:
                df[c] = np.nan

        df = _detect_and_scale_synmap(df, pre_id, verbose=False)

        if one_row_per_synapse:
            for i, (_, r) in enumerate(df.iterrows()):
                local_rows.append(
                    dict(
                        pre_id=pre_id,
                        post_id=int(r["post_id"]),
                        weight_uS=default_w,
                        post_x=float(r["x"]),
                        post_y=float(r["y"]),
                        post_z=float(r["z"]),
                        syn_index=i,
                    )
                )
        else:
            for post_id, sub in df.groupby("post_id"):
                local_rows.append(
                    dict(
                        pre_id=pre_id,
                        post_id=int(post_id),
                        weight_uS=len(sub) * default_w,
                        post_x=float(sub["x"].mean()),
                        post_y=float(sub["y"].mean()),
                        post_z=float(sub["z"].mean()),
                        nsyn=len(sub),
                    )
                )

        return local_rows, local_missing, local_errors, local_no_pre

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = {ex.submit(_process_pre, pid): pid for pid in pres}
        for fut in tqdm(as_completed(futures), total=len(futures), leave=False):
            r, m, e, n = fut.result()
            rows.extend(r)
            missing_files.extend(m)
            load_errors.extend(e)
            no_pre_rows.extend(n)

    df_edges = pd.DataFrame(rows)
    elapsed = time.perf_counter() - start

    print(f"[edges] built {len(df_edges)} rows in {elapsed:0.2f}s")

    return df_edges


# ============================================================================
# Hemilineage edge build wrapper
# ============================================================================

def build_hemi_edges_from_synapse_csvs(
    hemilineage_label: str,
    seeds: list[int],
    master_csv: str | None = None,
    force_rebuild: bool = False,
    smoke_test: bool = False,
    pres_limit: int | None = None,
):
    """
    Build and persist edges for a hemilineage using synapses_new.csv files.
    """

    lab = hemilineage_label.strip()
    if not lab:
        raise ValueError("hemilineage_label must be non-empty")

    mc = Path(master_csv) if master_csv is not None else HEMI_MASTER_CSV

    edges_path = HEMI_EDGES_ROOT / (
        f"hemi_{lab}_edges_from_synapses_smoke.csv"
        if smoke_test
        else f"hemi_{lab}_edges_from_synapses.csv"
    )

    if edges_path.exists() and not force_rebuild:
        return edges_path, pd.read_csv(edges_path)

    df_master = pd.read_csv(mc)
    hemi_ids = (
        df_master.loc[
            df_master["hemilineage"].astype(str).str.lower() == lab.lower(),
            "bodyId",
        ]
        .astype(int)
        .tolist()
    )

    default_weight_uS = float(CONFIG.get("default_weight_uS", 6e-6))

    df_edges = _hemi_fetch_edges_from_synapse_csvs(
        seeds=seeds,
        hemi_ids=hemi_ids,
        label=lab,
        swc_root=HEMI_SWC_ROOT,
        default_weight_uS=default_weight_uS,
        smoke_test=smoke_test,
        pres_limit=pres_limit,
        smoke_seeds_only=True,
    )

    edges_path.parent.mkdir(parents=True, exist_ok=True)
    df_edges.to_csv(edges_path, index=False)

    hemi_run_dir = HEMI_RUN_ROOT / f"hemi_{lab}"
    hemi_run_dir.mkdir(parents=True, exist_ok=True)
    mirror = hemi_run_dir / edges_path.name
    if not mirror.exists():
        mirror.write_bytes(edges_path.read_bytes())

    return edges_path, df_edges
