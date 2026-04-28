"""Phase 3 helpers for converting Phase 2 outputs into MuJoCo controls."""

from .pipeline import (
    build_actuator_controls_from_spikes,
    load_mapping_csv,
    load_phase2_spike_times,
    load_phase2_timebase_ms,
    plot_actuator_controls,
    save_controls_csv,
    summarize_mapping_coverage,
)
from .video_pipeline import (
    apply_profile_transforms,
    canonicalize_actuator_controls_for_model,
    loop_signals_simple,
    remap_to_ctrlrange_auto,
    render_controls_mujoco,
)
from .mapping_enrichment import run_mapping_enrichment
from .hemilineage import (
    build_spike_mapping_summary,
    filter_spikes_to_neuron_ids,
    list_hemilineage_folders,
    list_run_folders,
    load_added_motor_neuron_ids,
    resolve_hemilineage_dir,
    summarize_focus_neuron_overlap,
)
from .gait_audit import run_gait_expectation_audit
from .gait_compare import compare_gait_to_expected
from .expected_gait import build_expected_gait_controls, build_tripod_phase_channels, render_expected_gait_video
from .inverse_gait import derive_expected_mn_drive

__all__ = [
    "build_actuator_controls_from_spikes",
    "load_mapping_csv",
    "load_phase2_spike_times",
    "load_phase2_timebase_ms",
    "plot_actuator_controls",
    "save_controls_csv",
    "summarize_mapping_coverage",
    "apply_profile_transforms",
    "canonicalize_actuator_controls_for_model",
    "loop_signals_simple",
    "remap_to_ctrlrange_auto",
    "render_controls_mujoco",
    "run_mapping_enrichment",
    "build_spike_mapping_summary",
    "filter_spikes_to_neuron_ids",
    "list_hemilineage_folders",
    "list_run_folders",
    "load_added_motor_neuron_ids",
    "resolve_hemilineage_dir",
    "summarize_focus_neuron_overlap",
    "run_gait_expectation_audit",
    "compare_gait_to_expected",
    "build_tripod_phase_channels",
    "build_expected_gait_controls",
    "render_expected_gait_video",
    "derive_expected_mn_drive",
]
