"""Choice 1: build healed SWC and synapse mapping outputs."""

from __future__ import annotations

from .clients import choose_neuprint_dataset, dataset_slug, get_default_client
from . import workflow_core


TITLE = "Build healed SWC + synapse mapping (Phase 2 core)"


def run(client=None):
    dataset = getattr(client, "dataset", None)
    if client is None:
        dataset = choose_neuprint_dataset()
        client = get_default_client(dataset=dataset)

    dataset_folder = dataset_slug(dataset or getattr(client, "dataset", "neuprint"))
    base_out = workflow_core.phase1_output_path(f"{dataset_folder}/export_swc")
    unlabeled_root = workflow_core.phase1_output_path(f"{dataset_folder}/Glia IDs")

    print(f"[Choice 1] Dataset: {dataset or getattr(client, 'dataset', '(unknown)')}")
    print(f"[Choice 1] SWC output root: {base_out}")
    print(f"[Choice 1] Unlabeled output root: {unlabeled_root}")

    workflow_core.set_active_client(client)
    return workflow_core.option_20_build_and_map(
        client,
        base_out=base_out,
        unlabeled_export_root=unlabeled_root,
    )
