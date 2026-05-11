"""Choice 9: preview top rows from a parquet file."""

from __future__ import annotations

from .choice_8_connectivity_matrix import discover_parquet_files, preview_parquet_heads


TITLE = "Preview parquet head (top rows)"


def run(client=None):  # noqa: ARG001 - menu runners share this signature
    print("\n[Choice 9] Preview parquet head")
    sources = discover_parquet_files()
    if not sources:
        print("[Choice 9] No parquet files found under this Phase 1 folder.")
        return None

    from .choice_8_connectivity_matrix import prompt_sources

    selected = prompt_sources(sources)
    if not selected:
        print("[Choice 9] No files selected.")
        return None
    return preview_parquet_heads(selected)
