"""Choice 8: build connectivity matrices from synapse parquet files."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TITLE = "Build connectivity matrix from synapse parquet"

PHASE1_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PHASE1_ROOT / "outputs" / "connectivity_matrices"

NEUROTRANSMITTERS = [
    "acetylcholine",
    "gaba",
    "glutamate",
    "dopamine",
    "serotonin",
    "octopamine",
    "histamine",
    "tyramine",
]

DEFAULT_MEAN_COLUMNS = [
    "confidence",
    "syn_top_p",
    "acetylcholine",
    "gaba",
    "glutamate",
    "dopamine",
    "serotonin",
    "octopamine",
    "histamine",
    "tyramine",
    "size",
]

MAX_DEFAULT_MATRIX_CELLS = 5_000_000
MAX_MATRIX_EDGE_ROWS = 5_000_000
DEFAULT_DISPLAY_MATRIX_CELLS = 2_500
DEFAULT_PLOT_MATRIX_CELLS = 20_000
NEUPRINT_SERVER = "https://neuprint.janelia.org"
KNOWN_NEUPRINT_DATASETS_BY_SOURCE = {
    "male-cns": "male-cns:v0.9",
    "malecns": "male-cns:v0.9",
}


@dataclass(frozen=True)
class ParquetInfo:
    path: Path
    relpath: str
    rows: int | None
    columns: tuple[str, ...]
    column_types: dict[str, str]
    size_bytes: int
    row_groups: int | None = None
    metadata_error: str | None = None


@dataclass(frozen=True)
class ColumnFilter:
    column: str
    op: str
    value: str | float | None = None
    values: tuple[str, ...] = ()


@dataclass
class SourceResult:
    source: str
    edge_csv: str | None
    matrix_csv: str | None
    edges: int
    pre_nodes: int
    post_nodes: int
    skipped_filters: list[str]
    skipped_summaries: list[str]
    error: str | None = None
    matrix_plot_png: str | None = None


@dataclass
class NeuronFilter:
    role: str = ""
    mode: str = "none"
    query: str = ""
    field: str = ""
    match_mode: str = ""
    ids_by_source: dict[str, set[str]] = dataclass_field(default_factory=dict)
    errors_by_source: dict[str, str] = dataclass_field(default_factory=dict)

    def is_active(self) -> bool:
        return self.mode != "none"

    def ids_for(self, info: ParquetInfo) -> set[str] | None:
        if not self.is_active():
            return None
        if "*" in self.ids_by_source:
            return self.ids_by_source["*"]
        return self.ids_by_source.get(info.relpath, set())

    def summary(self, sources: list[ParquetInfo]) -> dict:
        counts = {}
        if self.is_active():
            for info in sources:
                ids = self.ids_for(info)
                counts[info.relpath] = len(ids or set())
        return {
            "role": self.role,
            "mode": self.mode,
            "query": self.query,
            "field": self.field,
            "match_mode": self.match_mode,
            "counts_by_source": counts,
            "errors_by_source": dict(self.errors_by_source),
        }

    def copy_for(self, role: str) -> "NeuronFilter":
        return NeuronFilter(
            role=role,
            mode=self.mode,
            query=self.query,
            field=self.field,
            match_mode=self.match_mode,
            ids_by_source={key: set(value) for key, value in self.ids_by_source.items()},
            errors_by_source=dict(self.errors_by_source),
        )


def run(client=None):  # noqa: ARG001 - menu runners share this signature
    print("\n[Choice 8] Build connectivity matrix from synapse parquet")
    print(
        "This creates a sparse pre->post edge table and, when the selected neuron "
        "set is small enough, a wide matrix CSV."
    )

    sources = discover_parquet_files()
    if not sources:
        print("[Choice 8] No parquet files found under this Phase 1 folder.")
        return None

    selected = prompt_sources(sources)
    if not selected:
        print("[Choice 8] No files selected.")
        return None

    pre_col = prompt_column("Presynaptic neuron column", selected, default="pre")
    post_col = prompt_column("Postsynaptic neuron column", selected, default="post")
    if not pre_col or not post_col:
        print("[Choice 8] Both pre and post columns are required.")
        return None

    pre_filter = prompt_neuron_filter("Restrict PRE/source neurons", selected)
    post_filter = prompt_neuron_filter("Restrict POST/target neurons", selected)
    if pre_filter.is_active() and not post_filter.is_active() and yes_no(
        "Use the same PRE/source neuron filter for POST/target too?", default=False
    ):
        post_filter = pre_filter.copy_for("post")
    if post_filter.is_active() and not pre_filter.is_active() and yes_no(
        "Use the same POST/target neuron filter for PRE/source too?", default=False
    ):
        pre_filter = post_filter.copy_for("pre")

    filters = prompt_parameter_filters(selected, reserved={pre_col, post_col})
    min_synapses = prompt_int("Minimum synapse count per connection", default=1, minimum=1)
    mean_columns = prompt_mean_columns(selected, reserved={pre_col, post_col})
    nt_breakdown = yes_no("Include syn_top_nt neurotransmitter count columns?", default=True)
    write_matrix = yes_no("Write wide matrix CSV when size is manageable?", default=True)
    max_matrix_cells = prompt_int(
        "Maximum wide-matrix cells to write",
        default=MAX_DEFAULT_MATRIX_CELLS,
        minimum=1,
    )
    display_matrix = False
    max_display_cells = DEFAULT_DISPLAY_MATRIX_CELLS
    plot_matrix = False
    max_plot_cells = DEFAULT_PLOT_MATRIX_CELLS
    if write_matrix:
        display_matrix = yes_no("Print the wide matrix table after it is written?", default=False)
        if display_matrix:
            max_display_cells = prompt_int(
                "Maximum matrix cells to display",
                default=DEFAULT_DISPLAY_MATRIX_CELLS,
                minimum=1,
            )
        plot_matrix = yes_no("Display a heatmap plot after the matrix is written?", default=True)
        if plot_matrix:
            max_plot_cells = prompt_int(
                "Maximum matrix cells to plot",
                default=DEFAULT_PLOT_MATRIX_CELLS,
                minimum=1,
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_utc": timestamp,
        "selected_files": [info.relpath for info in selected],
        "pre_column": pre_col,
        "post_column": post_col,
        "pre_filter": pre_filter.summary(selected),
        "post_filter": post_filter.summary(selected),
        "filters": [asdict(spec) for spec in filters],
        "min_synapses": min_synapses,
        "mean_columns": mean_columns,
        "nt_breakdown": nt_breakdown,
        "write_matrix": write_matrix,
        "max_matrix_cells": max_matrix_cells,
        "display_matrix": display_matrix,
        "max_display_cells": max_display_cells,
        "plot_matrix": plot_matrix,
        "max_plot_cells": max_plot_cells,
    }
    (out_dir / "settings.json").write_text(json.dumps(config, indent=2) + "\n")

    engine = choose_engine()
    if engine is None:
        print("[Choice 8] Could not find a usable parquet engine.")
        print("Install DuckDB in the notebook kernel with: %pip install duckdb")
        return None

    print(f"\n[Choice 8] Output folder: {out_dir}")
    print(f"[Choice 8] Engine: {engine}")

    results: list[SourceResult] = []
    for info in selected:
        print(f"\n[Choice 8] Processing {info.relpath}")
        pre_ids = pre_filter.ids_for(info)
        post_ids = post_filter.ids_for(info)
        if pre_filter.is_active():
            print(f"[Choice 8] PRE/source filter IDs for this file: {len(pre_ids or set()):,}")
        if post_filter.is_active():
            print(f"[Choice 8] POST/target filter IDs for this file: {len(post_ids or set()):,}")
        try:
            if engine == "duckdb":
                result = run_duckdb_source(
                    info=info,
                    out_dir=out_dir,
                    pre_col=pre_col,
                    post_col=post_col,
                    pre_ids=pre_ids,
                    post_ids=post_ids,
                    filters=filters,
                    min_synapses=min_synapses,
                    mean_columns=mean_columns,
                    nt_breakdown=nt_breakdown,
                    write_matrix=write_matrix,
                    max_matrix_cells=max_matrix_cells,
                )
            else:
                result = run_pyarrow_source(
                    info=info,
                    out_dir=out_dir,
                    pre_col=pre_col,
                    post_col=post_col,
                    pre_ids=pre_ids,
                    post_ids=post_ids,
                    filters=filters,
                    min_synapses=min_synapses,
                    mean_columns=mean_columns,
                    nt_breakdown=nt_breakdown,
                    write_matrix=write_matrix,
                    max_matrix_cells=max_matrix_cells,
                )
        except Exception as exc:  # keep batch/all-three mode moving
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[Choice 8] ERROR {info.relpath}: {msg}")
            if "Repetition level histogram size mismatch" in msg:
                print(
                    "[Choice 8] This usually means the installed PyArrow is too old "
                    "for the parquet writer metadata. Install DuckDB or upgrade PyArrow."
                )
            result = SourceResult(
                source=info.relpath,
                edge_csv=None,
                matrix_csv=None,
                edges=0,
                pre_nodes=0,
                post_nodes=0,
                skipped_filters=[],
                skipped_summaries=[],
                error=msg,
            )
        results.append(result)

    combined = combine_edge_csvs(results, out_dir)

    print("\n[Choice 8] Done.")
    if combined:
        print(f"[Choice 8] Combined sparse edge table: {combined}")
    for result in results:
        if result.error:
            print(f"  - {result.source}: ERROR {result.error}")
            continue
        print(
            f"  - {result.source}: {result.edges:,} edges, "
            f"{result.pre_nodes:,} pre nodes, {result.post_nodes:,} post nodes"
        )
        if result.edge_csv:
            print(f"    edge CSV: {result.edge_csv}")
        if result.matrix_csv:
            print(f"    matrix CSV: {result.matrix_csv}")
            if plot_matrix:
                result.matrix_plot_png = plot_matrix_csv(
                    result.matrix_csv,
                    max_cells=max_plot_cells,
                    title=f"{result.source} connectivity",
                )
                if result.matrix_plot_png:
                    print(f"    matrix heatmap PNG: {result.matrix_plot_png}")
            if display_matrix:
                display_matrix_csv(result.matrix_csv, max_cells=max_display_cells)
        if result.skipped_filters:
            print(f"    skipped filters: {', '.join(result.skipped_filters)}")
        if result.skipped_summaries:
            print(f"    skipped summaries: {', '.join(result.skipped_summaries)}")
    summary = {
        "output_folder": str(out_dir),
        "combined_edges_csv": str(combined) if combined else None,
        "results": [asdict(result) for result in results],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def discover_parquet_files() -> list[ParquetInfo]:
    ignored_parts = {".ipynb_checkpoints", "outputs", "__pycache__"}
    paths = [
        path
        for path in PHASE1_ROOT.rglob("*.parquet")
        if not ignored_parts.intersection(path.relative_to(PHASE1_ROOT).parts)
    ]
    infos = [read_parquet_info(path) for path in sorted(paths)]
    return sorted(infos, key=lambda info: (preferred_source_rank(info.relpath), info.relpath))


def preferred_source_rank(relpath: str) -> int:
    lowered = relpath.lower()
    if lowered.startswith("male-cns") or "malecns" in lowered:
        return 0
    if lowered.startswith("fafb"):
        return 1
    if lowered.startswith("banc"):
        return 2
    return 3


def read_parquet_info(path: Path) -> ParquetInfo:
    try:
        relpath = str(path.relative_to(PHASE1_ROOT))
    except ValueError:
        relpath = str(path)
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        columns = tuple(field.name for field in schema)
        column_types = {field.name: str(field.type) for field in schema}
        return ParquetInfo(
            path=path,
            relpath=relpath,
            rows=pf.metadata.num_rows,
            columns=columns,
            column_types=column_types,
            size_bytes=path.stat().st_size,
            row_groups=pf.metadata.num_row_groups,
        )
    except Exception as exc:
        return ParquetInfo(
            path=path,
            relpath=relpath,
            rows=None,
            columns=(),
            column_types={},
            size_bytes=path.stat().st_size,
            metadata_error=f"{type(exc).__name__}: {exc}",
        )


def prompt_sources(sources: list[ParquetInfo]) -> list[ParquetInfo]:
    print("\nAvailable parquet files:")
    for idx, info in enumerate(sources, start=1):
        rows = f"{info.rows:,} rows" if info.rows is not None else "rows unknown"
        cols = f"{len(info.columns)} cols" if info.columns else "schema unavailable"
        print(f"  {idx}. {info.relpath} ({rows}, {cols}, {format_bytes(info.size_bytes)})")
        if info.metadata_error:
            print(f"     metadata warning: {info.metadata_error}")
    print("  A. All listed parquet files")

    raw = input("Select parquet file(s) by number, comma list, A, or absolute path: ").strip()
    if not raw:
        return []
    if raw.lower() == "a":
        return sources

    direct_path = Path(raw).expanduser()
    if not direct_path.is_absolute():
        direct_path = PHASE1_ROOT / raw
    if direct_path.exists() and direct_path.is_file():
        return [read_parquet_info(direct_path.resolve())]

    selected: list[ParquetInfo] = []
    by_rel = {info.relpath: info for info in sources}
    for token in split_tokens(raw):
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(sources):
                selected.append(sources[idx - 1])
            else:
                print(f"[Choice 8] Ignoring invalid source number: {token}")
            continue
        path = Path(token).expanduser()
        if not path.is_absolute():
            if token in by_rel:
                selected.append(by_rel[token])
                continue
            path = PHASE1_ROOT / token
        if path.exists():
            selected.append(read_parquet_info(path.resolve()))
        else:
            print(f"[Choice 8] Ignoring missing parquet path: {token}")

    deduped: list[ParquetInfo] = []
    seen: set[Path] = set()
    for info in selected:
        if info.path not in seen:
            deduped.append(info)
            seen.add(info.path)
    return deduped


def prompt_mode() -> str:
    print("\nWhat would you like to do?")
    print("  1. Preview parquet head")
    print("  2. Build connectivity matrix")
    raw = input("Mode [2]: ").strip()
    if raw == "1" or raw.lower() in {"head", "preview", "h"}:
        return "preview"
    return "matrix"


def preview_parquet_heads(sources: list[ParquetInfo]):
    row_count = prompt_int("Rows to display", default=10, minimum=1)
    selected_columns = prompt_preview_columns(sources)
    engine = choose_engine()
    if engine is None:
        print("[Choice 8] Could not find a usable parquet engine.")
        print("Install DuckDB in the notebook kernel with: %pip install duckdb")
        return None

    print(f"\n[Choice 8] Preview engine: {engine}")
    previews = []
    for info in sources:
        print(f"\n===== {info.relpath} | head({row_count}) =====")
        try:
            if engine == "duckdb":
                df = read_head_duckdb(info, row_count=row_count, columns=selected_columns)
            else:
                df = read_head_pyarrow(info, row_count=row_count, columns=selected_columns)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[Choice 8] ERROR previewing {info.relpath}: {msg}")
            if "Repetition level histogram size mismatch" in msg:
                print(
                    "[Choice 8] This usually means the installed PyArrow is too old "
                    "for the parquet writer metadata. Install DuckDB or upgrade PyArrow."
                )
            previews.append({"source": info.relpath, "error": msg})
            continue
        if df.empty:
            print("(no rows)")
        else:
            print(df.to_string(index=False, max_cols=None))
        previews.append(
            {
                "source": info.relpath,
                "rows_displayed": int(len(df)),
                "columns_displayed": list(df.columns),
            }
        )
    return previews


def prompt_preview_columns(sources: list[ParquetInfo]) -> list[str]:
    columns = union_columns(sources)
    print("\nPreview columns:")
    print("  blank = all columns")
    print_column_choices(sources, columns=columns)
    raw = input("Columns to display (numbers or names, comma separated): ").strip()
    if not raw:
        return []
    return resolve_columns(raw, columns)


def read_head_duckdb(info: ParquetInfo, *, row_count: int, columns: list[str]):
    import duckdb

    con = duckdb.connect()
    available = set(info.columns)
    selected = [col for col in columns if col in available]
    select_sql = ", ".join(qident(col) for col in selected) if selected else "*"
    query = f"SELECT {select_sql} FROM read_parquet(?) LIMIT {int(row_count)}"
    try:
        return con.execute(query, [str(info.path)]).fetchdf()
    finally:
        con.close()


def read_head_pyarrow(info: ParquetInfo, *, row_count: int, columns: list[str]):
    import pyarrow.parquet as pq

    selected = [col for col in columns if col in set(info.columns)] or None
    pf = pq.ParquetFile(info.path)
    table = pf.read_row_group(0, columns=selected).slice(0, row_count)
    return table.to_pandas()


def prompt_column(label: str, sources: list[ParquetInfo], *, default: str) -> str | None:
    all_columns = union_columns(sources)
    common = common_columns(sources)
    if default in common:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        if raw in all_columns:
            return raw
        print(
            f"[Choice 8] '{raw}' is not a parquet column. Keeping '{default}'. "
            "Enter neuron types such as EPG/PEG at the neuron-filter prompts next."
        )
        return default

    print(f"\n{label}:")
    print_column_choices(sources, columns=all_columns)
    raw = input(f"Column name [{default}]: ").strip() or default
    if raw not in all_columns:
        print(f"[Choice 8] Column '{raw}' is not present in the selected parquet schemas.")
        return None
    return raw


def prompt_neuron_filter(label: str, sources: list[ParquetInfo]) -> NeuronFilter:
    print(f"\n{label}")
    print("  blank = no restriction")
    print("  examples: 720575940596125868,720575940627502824")
    print("            EPG")
    print("            type:EPG")
    print("            instance:.*EPG.*")
    raw = input("IDs, CSV/TXT path, or neuPrint type/instance query: ").strip()
    if not raw:
        return NeuronFilter(role=label, mode="none")

    path = resolve_input_path(raw)
    if path and path.exists() and path.is_file():
        values = read_values_or_tokens(raw)
        print(f"[Choice 8] Loaded {len(values):,} body IDs from file for {label.lower()}.")
        return NeuronFilter(
            role=label,
            mode="bodyId",
            query=str(path),
            field="bodyId",
            ids_by_source={"*": values},
        )

    tokens = split_tokens(raw)
    if tokens and all(token.lstrip("-").isdigit() for token in tokens):
        values = set(tokens)
        print(f"[Choice 8] Using {len(values):,} explicit body IDs for {label.lower()}.")
        return NeuronFilter(
            role=label,
            mode="bodyId",
            query=raw,
            field="bodyId",
            ids_by_source={"*": values},
        )

    field_name = "type"
    query = raw
    if ":" in raw:
        prefix, value = raw.split(":", 1)
        prefix = prefix.strip().lower()
        if prefix in {"type", "instance"}:
            field_name = prefix
            query = value.strip()
    if not query:
        print(f"[Choice 8] Empty neuron query for {label.lower()}; no restriction applied.")
        return NeuronFilter(role=label, mode="none")

    match_mode = "regex" if looks_regexy(query) else "contains"
    resolved = resolve_neuprint_filter_ids(
        query=query,
        field_name=field_name,
        match_mode=match_mode,
        sources=sources,
    )
    return NeuronFilter(
        role=label,
        mode="neuprint",
        query=query,
        field=field_name,
        match_mode=match_mode,
        ids_by_source=resolved[0],
        errors_by_source=resolved[1],
    )


def resolve_neuprint_filter_ids(
    *,
    query: str,
    field_name: str,
    match_mode: str,
    sources: list[ParquetInfo],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    ids_by_source: dict[str, set[str]] = {}
    errors_by_source: dict[str, str] = {}
    for info in sources:
        dataset = infer_neuprint_dataset(info)
        if dataset is None:
            raw_dataset = input(
                f"neuPrint dataset for {info.relpath} to resolve {field_name}='{query}' "
                "(blank = skip this file): "
            ).strip()
            dataset = raw_dataset or None
        if dataset is None:
            ids_by_source[info.relpath] = set()
            errors_by_source[info.relpath] = "No neuPrint dataset selected for type lookup."
            print(f"[Choice 8] Skipping neuPrint lookup for {info.relpath}.")
            continue

        try:
            ids = fetch_neuprint_body_ids(
                dataset=dataset,
                field_name=field_name,
                query=query,
                match_mode=match_mode,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            ids_by_source[info.relpath] = set()
            errors_by_source[info.relpath] = msg
            print(f"[Choice 8] ERROR resolving {field_name}='{query}' in {dataset}: {msg}")
            continue

        ids_by_source[info.relpath] = ids
        print(
            f"[Choice 8] {info.relpath}: resolved {field_name}='{query}' "
            f"in {dataset} to {len(ids):,} body IDs."
        )
    return ids_by_source, errors_by_source


def fetch_neuprint_body_ids(
    *,
    dataset: str,
    field_name: str,
    query: str,
    match_mode: str,
) -> set[str]:
    import pandas as pd
    import neuprint as neu

    from .token_store import get_neuprint_token

    token = get_neuprint_token(required=True)
    client = neu.Client(NEUPRINT_SERVER, dataset=dataset, token=token)
    pattern = query if match_mode == "regex" else contains_regex(query)
    crit_kwargs = {
        field_name: pattern,
        "regex": True,
        "status": "Traced",
    }
    crit = neu.NeuronCriteria(**crit_kwargs)
    result = neu.fetch_neurons(crit, omit_rois=True, client=client)
    neurons_df = result[0] if isinstance(result, tuple) else result
    if neurons_df is None or neurons_df.empty or "bodyId" not in neurons_df.columns:
        return set()

    preview_cols = [col for col in ["bodyId", "instance", "type"] if col in neurons_df.columns]
    if preview_cols:
        print(neurons_df[preview_cols].head(10).to_string(index=False))

    ids = pd.to_numeric(neurons_df["bodyId"], errors="coerce").dropna().astype("int64")
    return {str(value) for value in ids.tolist()}


def infer_neuprint_dataset(info: ParquetInfo) -> str | None:
    lowered = info.relpath.lower().replace("_", "-")
    for token, dataset in KNOWN_NEUPRINT_DATASETS_BY_SOURCE.items():
        if token in lowered:
            return dataset
    return None


def contains_regex(text: str) -> str:
    return f".*{re.escape(str(text).strip())}.*"


def looks_regexy(text: str) -> bool:
    return any(ch in str(text) for ch in r".^$*+?{}[]\|()")


def resolve_input_path(raw: str) -> Path | None:
    if not raw or any(sep in raw.strip() for sep in [",", ";", "\n", "\t"]):
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PHASE1_ROOT / raw
    return path


def prompt_id_filter(label: str) -> set[str]:
    print(f"\n{label}")
    raw = input("IDs as comma/space list or CSV/TXT path (blank = no restriction): ").strip()
    if not raw:
        return set()
    values = read_values_or_tokens(raw)
    print(f"[Choice 8] Loaded {len(values):,} IDs for {label.lower()}.")
    return values


def prompt_parameter_filters(
    sources: list[ParquetInfo], *, reserved: set[str]
) -> list[ColumnFilter]:
    available = [col for col in union_columns(sources) if col not in reserved]
    if not available:
        return []

    print("\nFilterable columns:")
    print_column_choices(sources, columns=available)
    raw = input(
        "Columns to filter by (numbers or names, comma separated; blank = none): "
    ).strip()
    if not raw:
        return []

    chosen = resolve_columns(raw, available)
    filters: list[ColumnFilter] = []
    for col in chosen:
        typ = type_summary(sources, col)
        print(f"\nFilter for {col} ({typ})")
        if is_numeric_type(typ):
            min_raw = input("  minimum value, blank = none: ").strip()
            max_raw = input("  maximum value, blank = none: ").strip()
            allowed_raw = input("  exact allowed values, comma list, blank = none: ").strip()
            if min_raw:
                filters.append(ColumnFilter(column=col, op="min", value=float(min_raw)))
            if max_raw:
                filters.append(ColumnFilter(column=col, op="max", value=float(max_raw)))
            if allowed_raw:
                filters.append(
                    ColumnFilter(column=col, op="in", values=tuple(sorted(split_tokens(allowed_raw))))
                )
        else:
            allowed_raw = input("  allowed values, comma list, blank = none: ").strip()
            contains_raw = input("  text contains, blank = none: ").strip()
            if allowed_raw:
                filters.append(
                    ColumnFilter(column=col, op="in", values=tuple(sorted(split_tokens(allowed_raw))))
                )
            if contains_raw:
                filters.append(ColumnFilter(column=col, op="contains", value=contains_raw))
            if yes_no("  require non-empty/non-null values?", default=False):
                filters.append(ColumnFilter(column=col, op="not_null"))
    return filters


def prompt_mean_columns(sources: list[ParquetInfo], *, reserved: set[str]) -> list[str]:
    numeric = [
        col
        for col in union_columns(sources)
        if col not in reserved and is_numeric_type(type_summary(sources, col))
    ]
    if not numeric:
        return []
    defaults = [col for col in DEFAULT_MEAN_COLUMNS if col in numeric]
    default_text = ",".join(defaults)
    print("\nNumeric columns that can be averaged per connection:")
    print_numbered(numeric)
    raw = input(
        "Columns to average into the edge table "
        f"[default: {default_text or 'none'}; blank keeps default; '-' = none]: "
    ).strip()
    if raw == "-":
        return []
    if not raw:
        return defaults
    return resolve_columns(raw, numeric)


def run_duckdb_source(
    *,
    info: ParquetInfo,
    out_dir: Path,
    pre_col: str,
    post_col: str,
    pre_ids: set[str] | None,
    post_ids: set[str] | None,
    filters: list[ColumnFilter],
    min_synapses: int,
    mean_columns: list[str],
    nt_breakdown: bool,
    write_matrix: bool,
    max_matrix_cells: int,
) -> SourceResult:
    import duckdb
    import pandas as pd

    con = duckdb.connect()
    source_slug = safe_slug(info.relpath)
    edge_csv = out_dir / f"{source_slug}_edges.csv"
    matrix_csv = out_dir / f"{source_slug}_matrix.csv"

    available = set(info.columns)
    if pre_col not in available or post_col not in available:
        raise ValueError(f"Missing required columns {pre_col!r}/{post_col!r}")

    skipped_filters: list[str] = []
    skipped_summaries: list[str] = []
    where = [
        f"{qident(pre_col)} IS NOT NULL",
        f"{qident(post_col)} IS NOT NULL",
    ]

    if pre_ids is not None:
        con.register("__pre_ids", pd.DataFrame({"id": sorted(pre_ids)}))
        where.append(f"CAST({qident(pre_col)} AS VARCHAR) IN (SELECT id FROM __pre_ids)")
    if post_ids is not None:
        con.register("__post_ids", pd.DataFrame({"id": sorted(post_ids)}))
        where.append(f"CAST({qident(post_col)} AS VARCHAR) IN (SELECT id FROM __post_ids)")

    for idx, spec in enumerate(filters):
        clause = duckdb_filter_clause(con, idx, spec, available)
        if clause is None:
            skipped_filters.append(spec.column)
        else:
            where.append(clause)

    select_cols = [
        f"{sql_string(info.relpath)} AS source",
        f"{sql_string(str(info.path))} AS parquet_file",
        f"CAST({qident(pre_col)} AS VARCHAR) AS pre",
        f"CAST({qident(post_col)} AS VARCHAR) AS post",
        "COUNT(*) AS synapse_count",
    ]
    for col in mean_columns:
        if col not in available:
            skipped_summaries.append(col)
            continue
        select_cols.append(f"AVG(TRY_CAST({qident(col)} AS DOUBLE)) AS {qident('mean_' + col)}")

    if nt_breakdown and "syn_top_nt" in available:
        for nt in NEUROTRANSMITTERS:
            alias = f"nt_{nt}_count"
            select_cols.append(
                "SUM(CASE WHEN LOWER(CAST(syn_top_nt AS VARCHAR)) = "
                f"{sql_string(nt)} THEN 1 ELSE 0 END) AS {qident(alias)}"
            )
    elif nt_breakdown:
        skipped_summaries.append("syn_top_nt")

    where_sql = " AND ".join(f"({clause})" for clause in where)
    parquet_path = sql_string(str(info.path))
    query = f"""
        SELECT
            {", ".join(select_cols)}
        FROM read_parquet({parquet_path})
        WHERE {where_sql}
        GROUP BY pre, post
        HAVING COUNT(*) >= {int(min_synapses)}
        ORDER BY pre, post
    """

    con.execute("CREATE TEMP TABLE edge_counts AS " + query)
    con.execute(f"COPY edge_counts TO {sql_string(str(edge_csv))} (HEADER, DELIMITER ',')")
    edges, pre_nodes, post_nodes = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pre), COUNT(DISTINCT post) FROM edge_counts"
    ).fetchone()

    written_matrix: str | None = None
    cells = int(pre_nodes or 0) * int(post_nodes or 0)
    if write_matrix and edges and cells <= max_matrix_cells and edges <= MAX_MATRIX_EDGE_ROWS:
        df = con.execute("SELECT pre, post, synapse_count FROM edge_counts").fetchdf()
        write_wide_matrix(
            df,
            matrix_csv,
            pre_ids=pre_ids,
            post_ids=post_ids,
            max_matrix_cells=max_matrix_cells,
        )
        written_matrix = str(matrix_csv)
    elif write_matrix and edges:
        print(
            f"[Choice 8] Skipping matrix for {info.relpath}: "
            f"{pre_nodes:,} x {post_nodes:,} = {cells:,} cells "
            f"or {edges:,} sparse edges exceeds the safety limit."
        )

    con.close()
    return SourceResult(
        source=info.relpath,
        edge_csv=str(edge_csv),
        matrix_csv=written_matrix,
        edges=int(edges or 0),
        pre_nodes=int(pre_nodes or 0),
        post_nodes=int(post_nodes or 0),
        skipped_filters=sorted(set(skipped_filters)),
        skipped_summaries=sorted(set(skipped_summaries)),
    )


def duckdb_filter_clause(con, idx: int, spec: ColumnFilter, available: set[str]) -> str | None:
    import pandas as pd

    if spec.column not in available:
        return None
    col = qident(spec.column)
    if spec.op == "min":
        return f"TRY_CAST({col} AS DOUBLE) >= {float(spec.value)}"
    if spec.op == "max":
        return f"TRY_CAST({col} AS DOUBLE) <= {float(spec.value)}"
    if spec.op == "contains":
        pattern = f"%{escape_like(str(spec.value or ''))}%"
        return f"CAST({col} AS VARCHAR) ILIKE {sql_string(pattern)}"
    if spec.op == "not_null":
        return f"{col} IS NOT NULL AND NULLIF(TRIM(CAST({col} AS VARCHAR)), '') IS NOT NULL"
    if spec.op == "in":
        table_name = f"__filter_{idx}"
        con.register(table_name, pd.DataFrame({"value": list(spec.values)}))
        return f"CAST({col} AS VARCHAR) IN (SELECT value FROM {table_name})"
    raise ValueError(f"Unsupported filter op: {spec.op}")


def run_pyarrow_source(
    *,
    info: ParquetInfo,
    out_dir: Path,
    pre_col: str,
    post_col: str,
    pre_ids: set[str] | None,
    post_ids: set[str] | None,
    filters: list[ColumnFilter],
    min_synapses: int,
    mean_columns: list[str],
    nt_breakdown: bool,
    write_matrix: bool,
    max_matrix_cells: int,
) -> SourceResult:
    import pandas as pd
    import pyarrow.parquet as pq

    source_slug = safe_slug(info.relpath)
    edge_csv = out_dir / f"{source_slug}_edges.csv"
    matrix_csv = out_dir / f"{source_slug}_matrix.csv"
    available = set(info.columns)
    if pre_col not in available or post_col not in available:
        raise ValueError(f"Missing required columns {pre_col!r}/{post_col!r}")

    skipped_filters = sorted({spec.column for spec in filters if spec.column not in available})
    active_filters = [spec for spec in filters if spec.column in available]
    skipped_summaries = [col for col in mean_columns if col not in available]
    active_mean_columns = [col for col in mean_columns if col in available]
    use_nt = nt_breakdown and "syn_top_nt" in available
    if nt_breakdown and not use_nt:
        skipped_summaries.append("syn_top_nt")

    needed = {pre_col, post_col, *(spec.column for spec in active_filters), *active_mean_columns}
    if use_nt:
        needed.add("syn_top_nt")

    pf = pq.ParquetFile(info.path)
    chunks: list[pd.DataFrame] = []
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=sorted(needed))
        df = table.to_pandas()
        df = apply_pandas_filters(
            df,
            pre_col=pre_col,
            post_col=post_col,
            pre_ids=pre_ids,
            post_ids=post_ids,
            filters=active_filters,
        )
        if df.empty:
            continue
        df["pre"] = df[pre_col].astype(str)
        df["post"] = df[post_col].astype(str)
        grouped = df.groupby(["pre", "post"], dropna=True).size().rename("synapse_count").reset_index()
        for col in active_mean_columns:
            sums = (
                df.groupby(["pre", "post"], dropna=True)[col]
                .sum()
                .rename(f"sum_{col}")
                .reset_index()
            )
            grouped = grouped.merge(sums, on=["pre", "post"], how="left")
        if use_nt:
            nt_counts = (
                df.assign(_nt=df["syn_top_nt"].astype(str).str.lower())
                .groupby(["pre", "post", "_nt"], dropna=True)
                .size()
                .unstack("_nt", fill_value=0)
                .reset_index()
            )
            for nt in NEUROTRANSMITTERS:
                if nt in nt_counts.columns:
                    grouped = grouped.merge(
                        nt_counts[["pre", "post", nt]].rename(columns={nt: f"nt_{nt}_count"}),
                        on=["pre", "post"],
                        how="left",
                    )
        chunks.append(grouped)

    if chunks:
        edges_df = pd.concat(chunks, ignore_index=True)
        sum_cols = [col for col in edges_df.columns if col.startswith("sum_") or col.startswith("nt_")]
        agg = {"synapse_count": "sum", **{col: "sum" for col in sum_cols}}
        edges_df = edges_df.groupby(["pre", "post"], as_index=False).agg(agg)
        for col in active_mean_columns:
            sum_col = f"sum_{col}"
            if sum_col in edges_df.columns:
                edges_df[f"mean_{col}"] = edges_df[sum_col] / edges_df["synapse_count"]
                edges_df = edges_df.drop(columns=[sum_col])
        edges_df = edges_df[edges_df["synapse_count"] >= min_synapses]
        edges_df.insert(0, "parquet_file", str(info.path))
        edges_df.insert(0, "source", info.relpath)
    else:
        edges_df = pd.DataFrame(columns=["source", "parquet_file", "pre", "post", "synapse_count"])

    edges_df.to_csv(edge_csv, index=False)
    pre_nodes = int(edges_df["pre"].nunique()) if not edges_df.empty else 0
    post_nodes = int(edges_df["post"].nunique()) if not edges_df.empty else 0

    written_matrix = None
    cells = pre_nodes * post_nodes
    if write_matrix and not edges_df.empty and cells <= max_matrix_cells:
        write_wide_matrix(
            edges_df[["pre", "post", "synapse_count"]],
            matrix_csv,
            pre_ids=pre_ids,
            post_ids=post_ids,
            max_matrix_cells=max_matrix_cells,
        )
        written_matrix = str(matrix_csv)
    elif write_matrix and not edges_df.empty:
        print(
            f"[Choice 8] Skipping matrix for {info.relpath}: "
            f"{pre_nodes:,} x {post_nodes:,} = {cells:,} cells exceeds the safety limit."
        )

    return SourceResult(
        source=info.relpath,
        edge_csv=str(edge_csv),
        matrix_csv=written_matrix,
        edges=len(edges_df),
        pre_nodes=pre_nodes,
        post_nodes=post_nodes,
        skipped_filters=skipped_filters,
        skipped_summaries=sorted(set(skipped_summaries)),
    )


def apply_pandas_filters(
    df,
    *,
    pre_col: str,
    post_col: str,
    pre_ids: set[str] | None,
    post_ids: set[str] | None,
    filters: list[ColumnFilter],
):
    mask = df[pre_col].notna() & df[post_col].notna()
    if pre_ids is not None:
        mask &= df[pre_col].astype(str).isin(pre_ids)
    if post_ids is not None:
        mask &= df[post_col].astype(str).isin(post_ids)
    for spec in filters:
        series = df[spec.column]
        if spec.op == "min":
            mask &= series.astype(float) >= float(spec.value)
        elif spec.op == "max":
            mask &= series.astype(float) <= float(spec.value)
        elif spec.op == "contains":
            mask &= series.astype(str).str.contains(str(spec.value), case=False, na=False)
        elif spec.op == "not_null":
            mask &= series.notna() & (series.astype(str).str.strip() != "")
        elif spec.op == "in":
            mask &= series.astype(str).isin(spec.values)
        else:
            raise ValueError(f"Unsupported filter op: {spec.op}")
    return df.loc[mask].copy()


def write_wide_matrix(
    edges_df,
    path: Path,
    *,
    pre_ids: set[str] | None,
    post_ids: set[str] | None,
    max_matrix_cells: int,
) -> None:
    rows = sorted(pre_ids) if pre_ids is not None else sorted(edges_df["pre"].dropna().astype(str).unique())
    cols = sorted(post_ids) if post_ids is not None else sorted(edges_df["post"].dropna().astype(str).unique())
    cells = len(rows) * len(cols)
    if cells > max_matrix_cells:
        raise ValueError(f"Matrix would contain {cells:,} cells, above limit {max_matrix_cells:,}")
    matrix = edges_df.pivot_table(
        index="pre",
        columns="post",
        values="synapse_count",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.reindex(index=rows, columns=cols, fill_value=0)
    matrix.to_csv(path, index_label="pre")


def display_matrix_csv(path: str | Path, *, max_cells: int) -> None:
    import pandas as pd

    path = Path(path)
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        print(f"    matrix display skipped: could not read matrix CSV ({exc})")
        return

    n_cols = max(len(header.columns) - 1, 0)
    try:
        n_rows = sum(1 for _ in path.open(newline="")) - 1
    except Exception as exc:
        print(f"    matrix display skipped: could not count matrix rows ({exc})")
        return
    n_rows = max(n_rows, 0)
    cells = n_rows * n_cols
    if cells > max_cells:
        print(
            f"    matrix display skipped: {n_rows:,} x {n_cols:,} = {cells:,} cells "
            f"exceeds display limit {max_cells:,}."
        )
        return

    matrix = pd.read_csv(path, index_col=0)
    print(f"\n===== Matrix Preview: {path.name} ({n_rows:,} x {n_cols:,}) =====")
    if matrix.empty:
        print("(empty matrix)")
    else:
        print(matrix.to_string(max_rows=None, max_cols=None))


def plot_matrix_csv(path: str | Path, *, max_cells: int, title: str) -> str | None:
    import numpy as np
    import pandas as pd

    path = Path(path)
    try:
        matrix = pd.read_csv(path, index_col=0)
    except Exception as exc:
        print(f"    matrix heatmap skipped: could not read matrix CSV ({exc})")
        return None

    n_rows, n_cols = matrix.shape
    cells = n_rows * n_cols
    if cells == 0:
        print("    matrix heatmap skipped: empty matrix.")
        return None
    if cells > max_cells:
        print(
            f"    matrix heatmap skipped: {n_rows:,} x {n_cols:,} = {cells:,} cells "
            f"exceeds plot limit {max_cells:,}."
        )
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"    matrix heatmap skipped: matplotlib is unavailable ({exc})")
        return None

    values = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
    plot_values = np.log1p(values.to_numpy(dtype=float))
    max_count = float(values.to_numpy(dtype=float).max()) if cells else 0.0

    width = min(max(7.0, 2.0 + 0.35 * n_cols), 26.0)
    height = min(max(5.0, 2.0 + 0.28 * n_rows), 26.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(plot_values, aspect="auto", interpolation="nearest", cmap="viridis")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("log1p(synapse count)")

    ax.set_title(title)
    ax.set_xlabel("post neuron")
    ax.set_ylabel("pre neuron")
    apply_axis_labels(ax, values.index.astype(str).tolist(), axis="y")
    apply_axis_labels(ax, values.columns.astype(str).tolist(), axis="x")

    if cells <= 400:
        threshold = np.log1p(max_count) * 0.55 if max_count > 0 else 0
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                count = int(values.iat[row_idx, col_idx])
                if count == 0:
                    continue
                color = "white" if plot_values[row_idx, col_idx] > threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    str(count),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    fig.tight_layout()
    png_path = path.with_name(path.stem + "_heatmap.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    if "agg" not in str(plt.get_backend()).lower():
        plt.show()
    plt.close(fig)
    return str(png_path)


def apply_axis_labels(ax, labels: list[str], *, axis: str) -> None:
    limit = 60
    count = len(labels)
    if count <= limit:
        ticks = list(range(count))
    else:
        step = max(1, math.ceil(count / limit))
        ticks = list(range(0, count, step))
    shown = [labels[idx] for idx in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(shown, rotation=90 if count > 8 else 45, ha="center", fontsize=8)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(shown, fontsize=8)


def combine_edge_csvs(results: list[SourceResult], out_dir: Path) -> Path | None:
    edge_paths = [Path(result.edge_csv) for result in results if result.edge_csv]
    edge_paths = [path for path in edge_paths if path.exists()]
    if not edge_paths:
        return None
    combined = out_dir / "combined_edges.csv"
    fieldnames: list[str] = []
    for path in edge_paths:
        with path.open(newline="") as in_f:
            reader = csv.DictReader(in_f)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
    with combined.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for path in edge_paths:
            with path.open(newline="") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    for field in fieldnames:
                        row.setdefault(field, "")
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
    return combined


def choose_engine() -> str | None:
    try:
        import duckdb  # noqa: F401

        return "duckdb"
    except Exception:
        pass

    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401

        print(
            "[Choice 8] DuckDB is not installed; falling back to PyArrow. "
            "DuckDB is recommended for these large files."
        )
        return "pyarrow"
    except Exception:
        return None


def union_columns(sources: Iterable[ParquetInfo]) -> list[str]:
    seen: dict[str, None] = {}
    for info in sources:
        for col in info.columns:
            seen.setdefault(col, None)
    return list(seen)


def common_columns(sources: Iterable[ParquetInfo]) -> set[str]:
    sources = list(sources)
    if not sources:
        return set()
    common = set(sources[0].columns)
    for info in sources[1:]:
        common &= set(info.columns)
    return common


def type_summary(sources: Iterable[ParquetInfo], column: str) -> str:
    types = sorted({info.column_types[column] for info in sources if column in info.column_types})
    return " | ".join(types) if types else "unknown"


def print_column_choices(sources: list[ParquetInfo], *, columns: list[str]) -> None:
    common = common_columns(sources)
    for idx, col in enumerate(columns, start=1):
        marker = "common" if col in common else "partial"
        print(f"  {idx:2d}. {col} ({type_summary(sources, col)}; {marker})")


def print_numbered(values: list[str]) -> None:
    for idx, value in enumerate(values, start=1):
        print(f"  {idx:2d}. {value}")


def resolve_columns(raw: str, columns: list[str]) -> list[str]:
    selected: list[str] = []
    for token in split_tokens(raw):
        value: str | None = None
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(columns):
                value = columns[idx - 1]
        elif token in columns:
            value = token
        if value is None:
            print(f"[Choice 8] Ignoring unknown column selection: {token}")
            continue
        if value not in selected:
            selected.append(value)
    return selected


def read_values_or_tokens(raw: str) -> set[str]:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PHASE1_ROOT / raw
    if path.exists() and path.is_file():
        values: set[str] = set()
        with path.open(newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            if "," in sample:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        token = str(row[0]).strip()
                        if token and token.lower() not in {"id", "bodyid", "body_id", "pre", "post"}:
                            values.add(token)
            else:
                for line in f:
                    values.update(split_tokens(line))
        return values
    return set(split_tokens(raw))


def split_tokens(raw: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\s,;]+", str(raw).strip()) if token.strip()]


def yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


def prompt_int(prompt: str, *, default: int, minimum: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[Choice 8] Invalid integer '{raw}', using {default}.")
        return default
    if value < minimum:
        print(f"[Choice 8] Value below {minimum}, using {minimum}.")
        return minimum
    return value


def is_numeric_type(type_text: str) -> bool:
    lowered = type_text.lower()
    return any(
        token in lowered
        for token in ["int", "float", "double", "decimal", "uint", "half_float"]
    )


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return slug.strip("._") or "parquet"


def format_bytes(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    order = min(int(math.log(size, 1024)), len(units) - 1)
    return f"{size / (1024**order):.2f} {units[order]}"
