from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_COALESCE_PROFILES",
    "DEFAULT_REDUCTION_PROFILES",
    "DEFAULT_THRESHOLDS",
    "RUN_SIMULATION_GAP_PAIRS",
    "build_edges_for_hemilineage_ids",
    "build_gap_config_from_run_simulation_preset",
    "build_master_config",
    "compare_runs",
    "default_master_csv_for_swc_dir",
    "expand_core_network_to_immediate_motor_postsynaptic",
    "hemilineage_project_name",
    "infer_latency_pairs",
    "infer_track_ids",
    "load_phase_timing_summary",
    "normalize_hemilineage_label",
    "preview_hemilineage_benchmark_plan",
    "project_paths",
    "run_coalescing_sweep",
    "run_combined_optimization",
    "run_full_hemilineage_project",
    "run_hemilineage_benchmark",
    "run_reduction_sweep",
    "summarize_run",
]

_SIM_PROJECT_EXPORTS = {
    name: (".sim_project", name)
    for name in __all__
}


def __getattr__(name: str):
    if name not in _SIM_PROJECT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _SIM_PROJECT_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
