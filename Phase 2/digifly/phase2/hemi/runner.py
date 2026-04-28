from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

from digifly.phase2.graph.edges_from_synapses import (
    _hemi_fetch_edges_from_synapse_csvs,
)

# -----------------------------------------------------------------------------
# Timing + logging (EXACT behavior preserved)
# -----------------------------------------------------------------------------
_HEMI_TIMING = []


@contextmanager
def _hemi_stage(label: str):
    start = time.perf_counter()
    print(f"[hemi] {label} …")
    yield
    elapsed = time.perf_counter() - start
    _HEMI_TIMING.append((label, elapsed))
    print(f"[hemi] {label} done in {elapsed:0.3f}s")


def _hemi_print_timing_summary():
    if not _HEMI_TIMING:
        return
    print("\n[hemi] ===== Hemilineage timing summary =====")
    for label, t in _HEMI_TIMING:
        print(f"[hemi] {label:<40} {t:8.3f}s")
    print("[hemi] ======================================")
    _HEMI_TIMING.clear()


# -----------------------------------------------------------------------------
# Hemilineage membership (EXACT logic from Phase 2)
# -----------------------------------------------------------------------------
def _get_ids_for_hemilineage(hemilineage_label: str, master_csv: Path) -> list[int]:
    """
    Return all neuron IDs belonging to a hemilineage.
    """
    df = pd.read_csv(master_csv)

    # Normalize column names defensively
    cols = {c.lower(): c for c in df.columns}

    if "hemilineage" not in cols or "bodyid" not in cols:
        raise ValueError(
            "Master CSV must contain 'hemilineage' and 'bodyId' columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    hemi_col = cols["hemilineage"]
    id_col = cols["bodyid"]

    mask = (
        df[hemi_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == str(hemilineage_label).strip().lower()
    )

    ids = sorted(
        pd.to_numeric(df.loc[mask, id_col], errors="coerce")
        .dropna()
        .astype(int)
    )
    return ids


# -----------------------------------------------------------------------------
# PUBLIC ENTRYPOINT: hemilineage execution
# -----------------------------------------------------------------------------
def run_hemilineage(
    hemilineage_label: str,
    seeds: list[int],
    *,
    swc_root,
    default_weight_uS: float,
    master_csv=None,
    edges_root=None,
    results_root=None,
    smoke_test: bool = False,
    pres_limit: int | None = None,
):
    """
    Run a hemilineage build (graph construction only).

    This matches Phase 2 behavior:
      - resolve hemilineage membership
      - build edges from synapses_new.csv
      - save + mirror edges
      - print timing summary
    """

    lab = str(hemilineage_label).strip()
    if not lab:
        raise ValueError("hemilineage_label must be non-empty")

    # FIX: this whole block MUST be inside the function
    swc_root = Path(swc_root).expanduser().resolve()
    if master_csv is None:
        master_csv = swc_root.parent / "all_neurons_neuroncriteria_template.csv"
    master_csv = Path(master_csv).expanduser().resolve()

    if edges_root is None:
        edges_root = swc_root / "edges"
    edges_root = Path(edges_root).expanduser().resolve()

    if results_root is None:
        results_root = swc_root / "hemi_runs"
    results_root = Path(results_root).expanduser().resolve()

    edges_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Load hemilineage members
    # -------------------------------------------------------------------------
    with _hemi_stage("load master table (hemilineage members)"):
        hemi_ids = _get_ids_for_hemilineage(lab, master_csv)

    print(f"[hemi] hemilineage='{lab}' → {len(hemi_ids)} neurons")
    print(f"[hemi] seeds={seeds}")

    # -------------------------------------------------------------------------
    # 2) Build edges
    # -------------------------------------------------------------------------
    with _hemi_stage("build edges from synapses_new.csv"):
        df_edges = _hemi_fetch_edges_from_synapse_csvs(
            seeds=seeds,
            hemi_ids=hemi_ids,
            label=lab,
            swc_root=swc_root,
            default_weight_uS=default_weight_uS,
            one_row_per_synapse=True,
            smoke_test=smoke_test,
            pres_limit=pres_limit,
            smoke_seeds_only=True,
        )

    # -------------------------------------------------------------------------
    # 3) Save + mirror
    # -------------------------------------------------------------------------
    edges_path = edges_root / (
        f"hemi_{lab}_edges_from_synapses_smoke.csv"
        if smoke_test
        else f"hemi_{lab}_edges_from_synapses.csv"
    )

    with _hemi_stage("save hemilineage edges CSV"):
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        df_edges.to_csv(edges_path, index=False)
        print(f"[hemi] Saved hemilineage edges → {edges_path}")

        hemi_run_dir = results_root / f"hemi_{lab}"
        hemi_run_dir.mkdir(parents=True, exist_ok=True)
        mirror = hemi_run_dir / edges_path.name
        if not mirror.exists():
            mirror.write_bytes(edges_path.read_bytes())
            print(f"[hemi] edges CSV also copied → {mirror}")

    _hemi_print_timing_summary()

    return edges_path, df_edges
