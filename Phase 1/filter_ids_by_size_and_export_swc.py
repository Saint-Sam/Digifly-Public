#!/usr/bin/env python3
"""
Filter neuron body IDs from an Excel list by size, then export SWC skeletons.

This script is designed to match the existing neuPrint -> SWC workflow already
used in this project:
- read body IDs from Excel
- compute a size metric (native neuPrint n.size or bbox metrics from skeleton XYZ)
- drop IDs below a chosen threshold
- export healed/upsampled SWCs for the remaining IDs
- save remaining and dropped neuron tables
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from neuprint import Client
from neuprint.skeleton import heal_skeleton, upsample_skeleton

try:
    from digifly_phase1.token_store import get_neuprint_token, phase1_root
except Exception:  # pragma: no cover - allows standalone script copying
    get_neuprint_token = None
    phase1_root = None


SIZE_METRICS = (
    "n_size",
    "bbox_x_um",
    "bbox_y_um",
    "bbox_z_um",
    "bbox_diag_um",
    "bbox_volume_um3",
)


def _phase1_path(path: Union[str, Path]) -> Path:
    """Resolve relative output paths under the Phase 1 folder."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if phase1_root is not None:
        return (phase1_root() / candidate).resolve()
    return (Path(__file__).resolve().parent / candidate).resolve()


def option26_prompt_and_run(
    client: Client,
    default_output_root: Union[str, Path] = "export_swc",
) -> Dict[str, Path]:
    """
    Notebook-friendly interactive runner (Option26 style).

    Example in notebook:
        from filter_ids_by_size_and_export_swc import option26_prompt_and_run
        option26_prompt_and_run(client=client)
    """
    if client is None:
        raise ValueError("client is required")

    print("\n[Option26-style] Excel IDs -> size filter -> SWC export")
    excel_raw = input("Excel path with body IDs: ").strip()
    if not excel_raw:
        raise ValueError("Excel path is required.")
    excel_path = Path(excel_raw).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    sheet_raw = input("Sheet (name or index) [0]: ").strip() or "0"
    sheet_name = _coerce_sheet_name(sheet_raw)

    id_col_raw = input("ID column name [auto-detect]: ").strip()
    id_column = id_col_raw if id_col_raw else None

    print("\nChoose size metric:")
    for i, m in enumerate(SIZE_METRICS, start=1):
        print(f"  {i}. {m}")
    metric_raw = input("Metric number [1]: ").strip() or "1"
    try:
        metric_idx = int(metric_raw)
        size_metric = SIZE_METRICS[metric_idx - 1]
    except Exception:  # noqa: BLE001
        raise ValueError(f"Invalid metric choice: {metric_raw}")

    min_size = float(input("Minimum size threshold (keep >= this): ").strip())
    upsample_nm = float(
        input("Upsample max segment length (nm) [2000]: ").strip() or "2000"
    )
    metadata_batch_size = int(
        input("Metadata query batch size [500]: ").strip() or "500"
    )

    default_label = f"filtered_{excel_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_label = input(f"Output subfolder name [{default_label}]: ").strip() or default_label
    output_dir = _phase1_path(default_output_root) / out_label

    skip_existing_raw = input("Skip existing SWCs if already present? [Y/n]: ").strip().lower()
    skip_existing = skip_existing_raw not in {"n", "no"}

    print("\n[run] starting...")
    result = run_filter_and_export(
        client=client,
        excel_path=excel_path,
        output_dir=output_dir,
        id_column=id_column,
        sheet_name=sheet_name,
        size_metric=size_metric,
        min_size=min_size,
        upsample_nm=upsample_nm,
        metadata_batch_size=metadata_batch_size,
        skip_existing=skip_existing,
    )
    print("[run] done.")
    return result


def _coerce_sheet_name(sheet: str) -> Union[int, str]:
    try:
        return int(sheet)
    except (TypeError, ValueError):
        return sheet


def _detect_id_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested id column '{requested}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    preferred = (
        "bodyId",
        "body_id",
        "bodyid",
        "neuron_id",
        "neuronId",
        "id",
        "ID",
    )
    for name in preferred:
        if name in df.columns:
            return name

    if len(df.columns) == 1:
        return str(df.columns[0])

    raise ValueError(
        "Could not infer ID column. Use --id-column.\n"
        f"Available columns: {list(df.columns)}"
    )


def _read_ids_from_excel(
    excel_path: Path,
    id_column: Optional[str],
    sheet_name: Union[int, str],
) -> Tuple[List[int], pd.DataFrame, str]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.empty:
        raise ValueError(f"Excel sheet is empty: {excel_path} (sheet={sheet_name})")

    id_col = _detect_id_column(df, id_column)
    ids = pd.to_numeric(df[id_col], errors="coerce").dropna().astype(np.int64).tolist()
    ids = list(dict.fromkeys(ids))  # keep order, drop duplicates

    if not ids:
        raise ValueError(
            f"No valid numeric IDs found in column '{id_col}' from {excel_path}"
        )
    return ids, df, id_col


def _chunked(values: Iterable[int], size: int) -> Iterable[List[int]]:
    values = list(values)
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _fetch_metadata(client: Client, body_ids: List[int], batch_size: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    total = len(body_ids)
    for i, chunk in enumerate(_chunked(body_ids, max(1, int(batch_size))), start=1):
        ids_csv = ",".join(str(int(b)) for b in chunk)
        query = f"""
MATCH (n:Neuron)
WHERE n.bodyId IN [{ids_csv}]
RETURN
  n.bodyId   AS bodyId,
  n.instance AS instance,
  n.type     AS type,
  n.status   AS status,
  n.pre      AS n_pre,
  n.post     AS n_post,
  n.size     AS n_size
ORDER BY bodyId
"""
        df = client.fetch_custom(query)
        if df is None:
            df = pd.DataFrame(columns=["bodyId", "instance", "type", "status", "n_pre", "n_post", "n_size"])
        frames.append(df)
        print(f"[meta] batch {i}: fetched {len(df)} rows ({min(i * batch_size, total)}/{total} IDs)")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(
            columns=["bodyId", "instance", "type", "status", "n_pre", "n_post", "n_size"]
        )

    out["bodyId"] = pd.to_numeric(out["bodyId"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["bodyId"]).copy()
    out["bodyId"] = out["bodyId"].astype(np.int64)
    out = out.drop_duplicates("bodyId")
    return out


def _fetch_healed_skeleton(client: Client, body_id: int, upsample_nm: float) -> pd.DataFrame:
    skel = client.fetch_skeleton(int(body_id), heal=False, format="pandas")
    if skel is None or len(skel) == 0:
        raise RuntimeError("empty skeleton")
    skel = heal_skeleton(skel, max_distance=np.inf, root_parent=-1)
    skel = upsample_skeleton(skel, max_segment_length=float(upsample_nm))
    if skel is None or len(skel) == 0:
        raise RuntimeError("empty skeleton after heal/upsample")
    return skel


def _bbox_metrics_um(skel: pd.DataFrame) -> Dict[str, float]:
    if not {"x", "y", "z"}.issubset(skel.columns):
        raise RuntimeError("skeleton missing x/y/z columns")
    xyz_nm = skel[["x", "y", "z"]].astype(float).to_numpy()
    if xyz_nm.size == 0:
        raise RuntimeError("skeleton has no xyz points")

    lo = np.min(xyz_nm, axis=0)
    hi = np.max(xyz_nm, axis=0)
    span_um = (hi - lo) / 1000.0

    return {
        "bbox_x_um": float(span_um[0]),
        "bbox_y_um": float(span_um[1]),
        "bbox_z_um": float(span_um[2]),
        "bbox_diag_um": float(np.linalg.norm(span_um)),
        "bbox_volume_um3": float(np.prod(span_um)),
    }


def _to_ordered_swc_table(skel: pd.DataFrame) -> pd.DataFrame:
    required = {"rowId", "link", "x", "y", "z"}
    missing = required.difference(skel.columns)
    if missing:
        raise RuntimeError(f"skeleton missing required columns: {sorted(missing)}")

    work = skel.copy()
    if "radius" not in work.columns:
        work["radius"] = 0.01

    def parent_id(value: object) -> int:
        if pd.isna(value):
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    row_ids = [int(rid) for rid in work["rowId"].tolist()]
    row_id_set = set(row_ids)
    parent_by_id: Dict[int, int] = {}
    children: Dict[int, List[int]] = defaultdict(list)
    for _, row in work.iterrows():
        rid = int(row["rowId"])
        parent = parent_id(row["link"])
        parent_by_id[rid] = parent
        if parent != rid:
            children[parent].append(rid)

    for key in children:
        children[key].sort()

    roots = [
        rid
        for rid in row_ids
        if parent_by_id.get(rid, -1) < 0
        or parent_by_id.get(rid) not in row_id_set
        or parent_by_id.get(rid) == rid
    ]

    ordered: List[int] = []
    seen: set[int] = set()

    def visit(start_id: int) -> None:
        stack = [int(start_id)]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
            for child_id in reversed(children.get(node_id, [])):
                if child_id not in seen:
                    stack.append(child_id)

    for root in sorted(set(roots)):
        visit(root)

    # If graph has disconnected pieces or cycles, append missing nodes deterministically.
    for rid in row_ids:
        if rid not in seen:
            visit(rid)

    work = work.set_index("rowId").loc[ordered].reset_index()

    # Convert nm -> um for SWC output.
    work[["x", "y", "z", "radius"]] = work[["x", "y", "z", "radius"]].astype(float) / 1000.0
    bad_radius = ~np.isfinite(work["radius"]) | (work["radius"] <= 0)
    work.loc[bad_radius, "radius"] = 0.01

    work["swc_type"] = 3
    work["new_id"] = np.arange(1, len(work) + 1, dtype=int)
    id_map = dict(zip(work["rowId"].astype(int), work["new_id"].astype(int)))

    def map_parent(row: pd.Series) -> int:
        p = parent_id(row["link"])
        if p < 0 or p == int(row["rowId"]):
            return -1
        mapped_parent = id_map.get(p)
        if mapped_parent is None or int(mapped_parent) >= int(row["new_id"]):
            return -1
        return int(mapped_parent)

    work["new_parent"] = work.apply(map_parent, axis=1).astype(int)
    return work


def _write_swc(skel: pd.DataFrame, body_id: int, out_path: Path) -> None:
    table = _to_ordered_swc_table(skel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# bodyId {int(body_id)}\n")
        for _, row in table.iterrows():
            f.write(
                f"{int(row['new_id'])} {int(row['swc_type'])} "
                f"{row['x']:.3f} {row['y']:.3f} {row['z']:.3f} "
                f"{row['radius']:.3f} {int(row['new_parent'])}\n"
            )


def run_filter_and_export(
    *,
    client: Client,
    excel_path: Path,
    output_dir: Path,
    id_column: Optional[str],
    sheet_name: Union[int, str],
    size_metric: str,
    min_size: float,
    upsample_nm: float,
    metadata_batch_size: int,
    skip_existing: bool,
) -> Dict[str, Path]:
    body_ids, _, id_col = _read_ids_from_excel(excel_path, id_column, sheet_name)
    print(f"[input] {len(body_ids)} unique IDs from {excel_path} (column '{id_col}', sheet={sheet_name})")

    meta = _fetch_metadata(client, body_ids, batch_size=metadata_batch_size)
    base = pd.DataFrame({"bodyId": body_ids}).merge(meta, on="bodyId", how="left")

    skeleton_cache: Dict[int, pd.DataFrame] = {}
    size_values: List[float] = []
    size_errors: List[str] = []

    if size_metric == "n_size":
        size_values = pd.to_numeric(base["n_size"], errors="coerce").astype(float).tolist()
        size_errors = [""] * len(base)
    else:
        print(f"[size] computing {size_metric} from skeleton XYZ for {len(base)} IDs...")
        for idx, bid in enumerate(base["bodyId"].tolist(), start=1):
            try:
                skel = _fetch_healed_skeleton(client, int(bid), upsample_nm=upsample_nm)
                skeleton_cache[int(bid)] = skel
                metrics = _bbox_metrics_um(skel)
                size_values.append(float(metrics[size_metric]))
                size_errors.append("")
            except Exception as exc:  # noqa: BLE001
                size_values.append(np.nan)
                size_errors.append(str(exc))
            if idx % 25 == 0 or idx == len(base):
                print(f"[size] {idx}/{len(base)} done")

    base["size_metric"] = size_metric
    base["size_value"] = size_values
    base["size_fetch_error"] = size_errors
    base["keep_by_size"] = base["size_value"] >= float(min_size)
    base.loc[base["size_value"].isna(), "keep_by_size"] = False

    kept = base[base["keep_by_size"]].copy()
    dropped = base[~base["keep_by_size"]].copy()
    print(
        f"[filter] min_size={min_size} on '{size_metric}' -> "
        f"kept={len(kept)}, dropped={len(dropped)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    swc_dir = output_dir / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    export_status: List[str] = []
    export_error: List[str] = []
    export_path: List[str] = []

    for idx, bid in enumerate(kept["bodyId"].astype(int).tolist(), start=1):
        out_swc = swc_dir / f"{bid}_healed.swc"
        try:
            if skip_existing and out_swc.exists():
                export_status.append("already_exists")
                export_error.append("")
                export_path.append(str(out_swc))
                continue

            skel = skeleton_cache.get(int(bid))
            if skel is None:
                skel = _fetch_healed_skeleton(client, int(bid), upsample_nm=upsample_nm)

            _write_swc(skel, bid, out_swc)
            export_status.append("exported")
            export_error.append("")
            export_path.append(str(out_swc))
        except Exception as exc:  # noqa: BLE001
            export_status.append("failed")
            export_error.append(str(exc))
            export_path.append(str(out_swc))
        if idx % 25 == 0 or idx == len(kept):
            print(f"[export] {idx}/{len(kept)} done")

    if len(kept) > 0:
        kept["export_status"] = export_status
        kept["export_error"] = export_error
        kept["swc_path"] = export_path
    else:
        kept["export_status"] = pd.Series(dtype="object")
        kept["export_error"] = pd.Series(dtype="object")
        kept["swc_path"] = pd.Series(dtype="object")

    remaining = kept[kept["export_status"] != "failed"].copy()
    failed_exports = kept[kept["export_status"] == "failed"].copy()

    all_csv = output_dir / "all_ids_with_size_and_status.csv"
    kept_csv = output_dir / "remaining_neurons.csv"
    dropped_csv = output_dir / "dropped_neurons.csv"
    failed_csv = output_dir / "failed_exports.csv"
    ids_txt = output_dir / "remaining_bodyIds.txt"

    all_xlsx = output_dir / "all_ids_with_size_and_status.xlsx"
    kept_xlsx = output_dir / "remaining_neurons.xlsx"
    dropped_xlsx = output_dir / "dropped_neurons.xlsx"

    base.sort_values(["keep_by_size", "size_value"], ascending=[False, False]).to_csv(all_csv, index=False)
    remaining.sort_values("size_value", ascending=False).to_csv(kept_csv, index=False)
    dropped.sort_values("size_value", ascending=True).to_csv(dropped_csv, index=False)
    failed_exports.to_csv(failed_csv, index=False)

    with ids_txt.open("w", encoding="utf-8") as f:
        for bid in remaining["bodyId"].astype(int).tolist():
            f.write(f"{bid}\n")

    with pd.ExcelWriter(all_xlsx, engine="openpyxl") as w:
        base.sort_values(["keep_by_size", "size_value"], ascending=[False, False]).to_excel(
            w, sheet_name="all", index=False
        )
        remaining.sort_values("size_value", ascending=False).to_excel(w, sheet_name="remaining", index=False)
        dropped.sort_values("size_value", ascending=True).to_excel(w, sheet_name="dropped", index=False)
        failed_exports.to_excel(w, sheet_name="failed_exports", index=False)

    remaining.sort_values("size_value", ascending=False).to_excel(kept_xlsx, index=False)
    dropped.sort_values("size_value", ascending=True).to_excel(dropped_xlsx, index=False)

    print(f"[save] remaining list  -> {kept_csv}")
    print(f"[save] dropped list    -> {dropped_csv}")
    print(f"[save] bodyId txt      -> {ids_txt}")
    print(f"[save] SWC folder      -> {swc_dir}")
    print(
        f"[done] remaining={len(remaining)}, failed_exports={len(failed_exports)}, "
        f"swc_files={(remaining['export_status'] != 'failed').sum()}"
    )

    return {
        "output_dir": output_dir,
        "swc_dir": swc_dir,
        "remaining_csv": kept_csv,
        "dropped_csv": dropped_csv,
        "failed_csv": failed_csv,
        "remaining_ids_txt": ids_txt,
    }


def _build_client(server: str, dataset: str, token: Optional[str]) -> Client:
    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token and get_neuprint_token is not None:
        token = get_neuprint_token(required=False)
    if not token:
        raise RuntimeError(
            "No neuPrint token found. Pass --token, set NEUPRINT_TOKEN, "
            "or save it to Neuprint Token.txt."
        )
    return Client(server, dataset=dataset, token=token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter body IDs from Excel by size and export SWC skeletons."
    )
    parser.add_argument("--excel", required=True, help="Path to Excel file with neuron IDs.")
    parser.add_argument("--id-column", default=None, help="Column name containing body IDs.")
    parser.add_argument("--sheet", default="0", help="Sheet index/name (default: 0).")
    parser.add_argument(
        "--size-metric",
        choices=SIZE_METRICS,
        default="n_size",
        help="Size metric used for filtering.",
    )
    parser.add_argument(
        "--min-size",
        required=True,
        type=float,
        help="Minimum size threshold; IDs below this are dropped.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: export_swc/filtered_<excel_stem>_<timestamp>",
    )
    parser.add_argument(
        "--upsample-nm",
        type=float,
        default=2000.0,
        help="Max segment length for upsample_skeleton during SWC export.",
    )
    parser.add_argument(
        "--metadata-batch-size",
        type=int,
        default=500,
        help="Batch size for metadata fetch_custom calls.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-export SWCs even if file already exists.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"),
        help="neuPrint server URL.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("NEUPRINT_DATASET", "manc:v1.2.1"),
        help="neuPrint dataset name.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_TOKEN"),
        help="neuPrint token (or set NEUPRINT_TOKEN).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excel_path = Path(args.excel).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = _phase1_path("export_swc") / f"filtered_{excel_path.stem}_{stamp}"
    output_dir = _phase1_path(args.output_dir) if args.output_dir else default_out

    client = _build_client(args.server, args.dataset, args.token)
    print(f"[client] server={args.server} dataset={args.dataset}")

    run_filter_and_export(
        client=client,
        excel_path=excel_path,
        output_dir=output_dir,
        id_column=args.id_column,
        sheet_name=_coerce_sheet_name(args.sheet),
        size_metric=args.size_metric,
        min_size=float(args.min_size),
        upsample_nm=float(args.upsample_nm),
        metadata_batch_size=int(args.metadata_batch_size),
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
