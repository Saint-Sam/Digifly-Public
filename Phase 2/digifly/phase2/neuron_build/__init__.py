from __future__ import annotations

from importlib import import_module

from .config import DEFAULT_CONFIG, DEFAULT_GLOBAL_TIMING, SYN_PRESETS, NT_TO_CLASS, merge_cfg
from .ownership import OwnershipPlan, build_cell_ownership, ownership_from_cfg
from .wiring_plan import ConnectionPlan, NetworkBuildPlan, build_network_plan

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_GLOBAL_TIMING",
    "SYN_PRESETS",
    "NT_TO_CLASS",
    "merge_cfg",
    "OwnershipPlan",
    "build_cell_ownership",
    "ownership_from_cfg",
    "ConnectionPlan",
    "NetworkBuildPlan",
    "build_network_plan",
    "SWCCell",
    "find_swc",
    "syn_csv_path",
    "load_syn_catalog",
    "pick_post_site",
    "attach_ais_methods",
    "set_pas",
    "set_hh",
    "make_passive",
    "make_active",
    "apply_biophys",
    "timing_from_row",
    "geom_delay_ms",
    "Network",
    "get_parallel_context",
    "get_parallel_state",
    "configure_parallel_context",
    "apply_thread_partitions",
    "reset_parallel_context",
    "gap_pair_ohmic",
    "gap_pair_rectifying",
    "gap_pair_directed",
    "ensure_gap_mechanism_available",
    "apply_gap_config",
    "build_network_driven_subset",
    "expand_from_edges",
    "build_pair_only",
    "run_pair_demo",
    "visualize_ais_strict",
    "fix_and_visualize_soma_ais",
    "visualize_ais",
]

_LAZY_EXPORTS = {
    "SWCCell": (".swc_cell", "SWCCell"),
    "find_swc": (".swc_cell", "find_swc"),
    "syn_csv_path": (".swc_cell", "syn_csv_path"),
    "load_syn_catalog": (".swc_cell", "load_syn_catalog"),
    "pick_post_site": (".swc_cell", "pick_post_site"),
    "attach_ais_methods": (".ais", "attach_ais_methods"),
    "set_pas": (".biophys", "set_pas"),
    "set_hh": (".biophys", "set_hh"),
    "make_passive": (".biophys", "make_passive"),
    "make_active": (".biophys", "make_active"),
    "apply_biophys": (".biophys", "apply_biophys"),
    "timing_from_row": (".timing", "timing_from_row"),
    "geom_delay_ms": (".timing", "geom_delay_ms"),
    "Network": (".network", "Network"),
    "get_parallel_context": (".parallel", "get_parallel_context"),
    "get_parallel_state": (".parallel", "get_parallel_state"),
    "configure_parallel_context": (".parallel", "configure_parallel_context"),
    "apply_thread_partitions": (".parallel", "apply_thread_partitions"),
    "reset_parallel_context": (".parallel", "reset_parallel_context"),
    "gap_pair_ohmic": (".gaps", "gap_pair_ohmic"),
    "gap_pair_rectifying": (".gaps", "gap_pair_rectifying"),
    "gap_pair_directed": (".gaps", "gap_pair_directed"),
    "ensure_gap_mechanism_available": (".gaps", "ensure_gap_mechanism_available"),
    "apply_gap_config": (".gaps", "apply_gap_config"),
    "build_network_driven_subset": (".builders", "build_network_driven_subset"),
    "expand_from_edges": (".builders", "expand_from_edges"),
    "build_pair_only": (".builders", "build_pair_only"),
    "run_pair_demo": (".builders", "run_pair_demo"),
    "visualize_ais_strict": (".viz_ais", "visualize_ais_strict"),
    "fix_and_visualize_soma_ais": (".viz_ais", "fix_and_visualize_soma_ais"),
    "visualize_ais": (".viz_ais", "visualize_ais"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
