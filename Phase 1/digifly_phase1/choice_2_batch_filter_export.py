"""Choice 2: batch body ID list/size filter to SWC export."""

from __future__ import annotations

from .clients import get_default_client
from . import workflow_core


TITLE = "Batch bodyId list/size filter -> SWC export helper"


def run(client=None):
    client = client or get_default_client()
    workflow_core.set_active_client(client)

    try:
        from filter_ids_by_size_and_export_swc import option26_prompt_and_run
    except Exception as exc:
        raise RuntimeError(
            "Could not import filter_ids_by_size_and_export_swc.py. "
            "It should live next to Phase 1.ipynb."
        ) from exc

    return option26_prompt_and_run(
        client=client,
        default_output_root=workflow_core.phase1_output_path("export_swc"),
    )
