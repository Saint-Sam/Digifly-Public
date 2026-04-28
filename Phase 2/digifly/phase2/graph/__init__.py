from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_MALE_CNS_DATASET",
    "DEFAULT_NEUPRINT_DATASET",
    "EDGE_SET_COLUMNS",
    "build_neuprint_dataset_name",
    "coerce_int_ids",
    "default_edges_registry_root",
    "default_neuprint_dataset_version",
    "edge_request_signature",
    "ensure_edges_registry_layout",
    "ensure_named_edge_set",
    "ensure_phase1_exports_if_needed",
    "expand_requested_network",
    "find_registered_edge_set",
    "known_neuprint_dataset_versions",
    "normalize_edge_set_name",
    "normalize_neuprint_dataset",
    "normalize_neuprint_dataset_family",
    "refresh_master_edges_csv",
    "resolve_neuprint_dataset_choice",
]

_REQUESTED_EDGE_SET_EXPORTS = {
    name: (".requested_edge_sets", name)
    for name in __all__
}


def __getattr__(name: str):
    if name not in _REQUESTED_EDGE_SET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _REQUESTED_EDGE_SET_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
