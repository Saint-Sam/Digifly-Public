"""neuPrint client construction for Phase 1 workflows."""

from __future__ import annotations

import os

import neuprint as neu

from .token_store import get_neuprint_token


DEFAULT_NEUPRINT_SERVER = "https://neuprint.janelia.org"
DEFAULT_NEUPRINT_DATASET = "manc:v1.2.1"
KNOWN_NEUPRINT_DATASETS = [
    ("manc:v1.2.1", "MANC v1.2.1"),
    ("manc:v1.2.3", "MANC v1.2.3"),
    ("manc:v1.0", "MANC v1.0"),
    ("male-cns:v0.9", "Male CNS v0.9"),
]
DATASET_ALIASES = {
    "manc:v1.0.0": "manc:v1.0",
    "mancv1.0.0": "manc:v1.0",
    "manc-v1.0.0": "manc:v1.0",
}


def normalize_neuprint_dataset_name(dataset: str) -> str:
    """Normalize common user-entered dataset spellings into neuPrint names."""
    raw = str(dataset).strip()
    compact = raw.lower().replace(" ", "").replace("_", "-")
    compact = DATASET_ALIASES.get(compact, compact)
    if compact in DATASET_ALIASES:
        return DATASET_ALIASES[compact]
    if ":" in compact:
        return compact
    if compact.startswith("mancv"):
        return compact.replace("mancv", "manc:v", 1)
    if compact.startswith("male-cnsv"):
        return compact.replace("male-cnsv", "male-cns:v", 1)
    if compact.startswith("malecnsv"):
        return compact.replace("malecnsv", "male-cns:v", 1)
    return raw


def dataset_slug(dataset: str) -> str:
    """Return a filesystem-safe folder name for a neuPrint dataset."""
    text = normalize_neuprint_dataset_name(dataset)
    return (
        text.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def choose_neuprint_dataset(default: str = DEFAULT_NEUPRINT_DATASET) -> str:
    """Prompt for a known neuPrint dataset or a custom dataset string."""
    print("\nChoose neuPrint connectome/dataset:")
    for idx, (dataset, label) in enumerate(KNOWN_NEUPRINT_DATASETS, start=1):
        suffix = " [default]" if dataset == default else ""
        print(f"  {idx}. {label} ({dataset}){suffix}")
    print("  C. Custom dataset name")

    raw = input(f"Dataset [{default}]: ").strip()
    if not raw:
        return normalize_neuprint_dataset_name(default)
    if raw.lower() == "c":
        custom = input("Custom neuPrint dataset name, e.g. manc:v1.2.1: ").strip()
        if not custom:
            print(f"[dataset] Empty custom value; using {default}")
            return normalize_neuprint_dataset_name(default)
        return normalize_neuprint_dataset_name(custom)
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(KNOWN_NEUPRINT_DATASETS):
            return normalize_neuprint_dataset_name(KNOWN_NEUPRINT_DATASETS[idx - 1][0])
        print(f"[dataset] Invalid number {raw}; using {default}")
        return normalize_neuprint_dataset_name(default)
    return normalize_neuprint_dataset_name(raw)


def make_client(
    *,
    server: str | None = None,
    dataset: str | None = None,
    token: str | None = None,
    register: bool = True,
):
    server_use = server or os.environ.get("NEUPRINT_SERVER") or DEFAULT_NEUPRINT_SERVER
    dataset_use = normalize_neuprint_dataset_name(
        dataset or os.environ.get("NEUPRINT_DATASET") or DEFAULT_NEUPRINT_DATASET
    )
    token_use = token or get_neuprint_token(required=True)
    client = neu.Client(server_use, dataset=dataset_use, token=token_use)

    if register:
        from . import workflow_core

        workflow_core.set_active_client(client)

    return client


def get_default_client(
    *,
    server: str | None = None,
    dataset: str | None = None,
    token: str | None = None,
    register: bool = True,
):
    return make_client(
        server=server,
        dataset=dataset,
        token=token,
        register=register,
    )
