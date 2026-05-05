from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PUBLIC_MANC_LABEL = "manc_v1.2.1"


@dataclass(frozen=True)
class ControlSpec:
    key: str
    label: str
    section: str
    control_type: str
    default: Any
    help_text: str
    choices: Sequence[str] = field(default_factory=tuple)
    cache_impact: str = "build_time"
    runner_scopes: Sequence[str] = field(default_factory=lambda: ("shared_runner", "hemilineage_project"))
    modes: Sequence[str] = field(default_factory=tuple)


CONTROL_SPECS: List[ControlSpec] = [
    ControlSpec(
        key="runner_kind",
        label="Execution surface",
        section="Project",
        control_type="choice",
        default="shared_runner",
        choices=("shared_runner", "hemilineage_project"),
        help_text="Use the shared runner for single/custom/hemilineage runs. Use hemilineage project for the fuller Phase 2 pipeline.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="run_id",
        label="Run ID",
        section="Project",
        control_type="text",
        default="phase2_workbench_run",
        help_text="Short name used in output folders and manifest files.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="run_notes",
        label="Run notes",
        section="Project",
        control_type="textarea",
        default="",
        help_text="Free-text notes saved alongside the workbench manifests.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="projects_root",
        label="Projects root",
        section="Project",
        control_type="text",
        default="",
        help_text="Used by the hemilineage project pipeline. A project folder is created here.",
        cache_impact="analysis_only",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="swc_dir",
        label="SWC root",
        section="Paths",
        control_type="text",
        default="",
        help_text="Phase 1 export_swc root. This is the main required input for both runner types.",
    ),
    ControlSpec(
        key="morph_swc_dir",
        label="Morphology override SWC root",
        section="Paths",
        control_type="text",
        default="",
        help_text="Optional reduced or healed SWC root used only for morphology loading.",
    ),
    ControlSpec(
        key="master_csv",
        label="Master metadata CSV",
        section="Paths",
        control_type="text",
        default="",
        help_text="Optional override for all_neurons_neuroncriteria_template.csv. Leave blank to use the Phase 2 default next to SWC_DIR.",
    ),
    ControlSpec(
        key="edges_root",
        label="Edges root override",
        section="Paths",
        control_type="text",
        default="",
        help_text="Shared runner only. Leave blank to use SWC_DIR/edges.",
        runner_scopes=("shared_runner",),
    ),
    ControlSpec(
        key="runs_root",
        label="Runs root override",
        section="Paths",
        control_type="text",
        default="",
        help_text="Shared runner only. Defaults to SWC_DIR/hemi_runs so Docker writes run outputs into the mounted repo.",
        runner_scopes=("shared_runner",),
    ),
    ControlSpec(
        key="template_config_path",
        label="Template config path",
        section="Paths",
        control_type="text",
        default="",
        help_text="Optional hemilineage project template config path.",
        runner_scopes=("hemilineage_project",),
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="edge_set_name",
        label="Edge-set name",
        section="Paths",
        control_type="text",
        default="",
        help_text="Named edge-set to reuse for hemilineage project builds.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="edges_registry_root",
        label="Edges registry root",
        section="Paths",
        control_type="text",
        default="",
        help_text="Optional registry root for named edge sets.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="neuprint_dataset",
        label="neuPrint dataset",
        section="Paths",
        control_type="text",
        default="",
        help_text="Optional dataset tag such as manc:v1.2.1.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="mode",
        label="Selection mode",
        section="Selection",
        control_type="choice",
        default="hemilineage",
        choices=("single", "custom", "hemilineage"),
        help_text="Selection mode for the shared runner.",
        runner_scopes=("shared_runner",),
        cache_impact="build_time",
    ),
    ControlSpec(
        key="hemi_label",
        label="Hemilineage label",
        section="Selection",
        control_type="text",
        default="09A",
        help_text="Used by shared hemilineage mode and hemilineage project runs.",
    ),
    ControlSpec(
        key="neuron_id",
        label="Single neuron ID",
        section="Selection",
        control_type="int",
        default=10000,
        help_text="Used only for shared single-neuron runs.",
        runner_scopes=("shared_runner",),
        modes=("single",),
    ),
    ControlSpec(
        key="neuron_ids_text",
        label="Custom neuron IDs",
        section="Selection",
        control_type="textarea",
        default="10000, 10002",
        help_text="Comma-separated IDs for shared custom runs.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="hemi_core_ids_text",
        label="Core hemilineage IDs",
        section="Selection",
        control_type="textarea",
        default="",
        help_text="Optional comma-separated core IDs for the hemilineage project. Leave blank to derive from the master CSV and hemilineage label.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="seeds_text",
        label="Seed neuron IDs",
        section="Selection",
        control_type="textarea",
        default="",
        help_text="Optional comma-separated seed IDs. Leave blank to use all selected IDs.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="edges_path",
        label="Explicit edges file",
        section="Selection",
        control_type="text",
        default="",
        help_text="Required for shared custom mode unless edge cache is enabled.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="tstop_ms",
        label="tstop (ms)",
        section="Runtime",
        control_type="float",
        default=4000.0,
        help_text="Total simulation time.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="dt_ms",
        label="dt (ms)",
        section="Runtime",
        control_type="float",
        default=0.1,
        help_text="Fixed-step dt. Treat as runtime-safe with caution when comparing across cached runs.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="threads",
        label="Threads",
        section="Runtime",
        control_type="int",
        default=0,
        help_text="Leave at 0 for default behavior. Shared and project runtimes will ignore it when blank/zero.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="build_backend",
        label="Build backend",
        section="Runtime",
        control_type="choice",
        default="single_host",
        choices=("single_host", "distributed_gid"),
        help_text="Shared runner only. Choose distributed_gid for the distributed ownership path.",
        runner_scopes=("shared_runner",),
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="ownership_strategy",
        label="Ownership strategy",
        section="Runtime",
        control_type="choice",
        default="round_robin",
        choices=("round_robin", "contiguous"),
        help_text="Shared runner only. Used when distributed ownership is enabled.",
        runner_scopes=("shared_runner",),
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="maxstep_ms",
        label="Parallel maxstep (ms)",
        section="Runtime",
        control_type="float",
        default=10.0,
        help_text="Shared runner only. Used for spike delivery validity in distributed runs.",
        runner_scopes=("shared_runner",),
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="enable_coreneuron",
        label="Enable CoreNEURON",
        section="Runtime",
        control_type="bool",
        default=False,
        help_text="Requests CoreNEURON. The run manifest should always record whether it was actually used.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="coreneuron_gpu",
        label="Enable CoreNEURON GPU",
        section="Runtime",
        control_type="bool",
        default=False,
        help_text="Requests GPU mode when CoreNEURON is active.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="coreneuron_nthread",
        label="CoreNEURON threads",
        section="Runtime",
        control_type="int",
        default=0,
        help_text="Optional CoreNEURON thread count.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="io_workers",
        label="Output IO workers",
        section="Runtime",
        control_type="int",
        default=0,
        help_text="Optional parallel save stage worker count.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="progress",
        label="Show progress",
        section="Runtime",
        control_type="bool",
        default=True,
        help_text="Enable progress reporting where supported.",
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="default_weight_uS",
        label="Default weight (uS)",
        section="Synapses",
        control_type="float",
        default=0.000003,
        help_text="Used when explicit edge weights are absent or during some hemilineage build paths.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="default_delay_ms",
        label="Default delay (ms)",
        section="Synapses",
        control_type="float",
        default=1.0,
        help_text="Default synaptic delay.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="syn_tau1_ms",
        label="syn tau1 (ms)",
        section="Synapses",
        control_type="float",
        default=0.5,
        help_text="Default synaptic rise constant.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="syn_tau2_ms",
        label="syn tau2 (ms)",
        section="Synapses",
        control_type="float",
        default=3.0,
        help_text="Default synaptic decay constant.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="syn_e_rev_mV",
        label="syn Erev (mV)",
        section="Synapses",
        control_type="float",
        default=0.0,
        help_text="Default synaptic reversal potential.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="post_active",
        label="Active postsynaptic cells",
        section="Biophysics",
        control_type="bool",
        default=True,
        help_text="Quick toggle for active versus passive postsynaptic targets.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="active_posts_mode",
        label="Active posts mode",
        section="Biophysics",
        control_type="choice",
        default="all_selected",
        choices=("all_selected", "drivers_only", "none"),
        help_text="Shared runner only. Controls which post cells are active.",
        runner_scopes=("shared_runner",),
        cache_impact="build_time",
    ),
    ControlSpec(
        key="iclamp_amp_nA",
        label="IClamp amp (nA)",
        section="Stimulation",
        control_type="float",
        default=2.5,
        help_text="Base IClamp amplitude.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="iclamp_delay_ms",
        label="IClamp delay (ms)",
        section="Stimulation",
        control_type="float",
        default=100.0,
        help_text="Base IClamp delay.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="iclamp_dur_ms",
        label="IClamp duration (ms)",
        section="Stimulation",
        control_type="float",
        default=200.0,
        help_text="Base IClamp duration.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="iclamp_location",
        label="IClamp location",
        section="Stimulation",
        control_type="choice",
        default="soma",
        choices=("soma", "ais"),
        help_text="Preferred clamp site.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="neg_pulse_enabled",
        label="Enable negative pulse",
        section="Stimulation",
        control_type="bool",
        default=False,
        help_text="Adds a negative pulse overlay to the base IClamp.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="neg_pulse_amp_nA",
        label="Negative pulse amp (nA)",
        section="Stimulation",
        control_type="float",
        default=-1.0,
        help_text="Negative pulse amplitude.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="neg_pulse_delay_ms",
        label="Negative pulse delay (ms)",
        section="Stimulation",
        control_type="float",
        default=150.0,
        help_text="Negative pulse delay.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="neg_pulse_dur_ms",
        label="Negative pulse duration (ms)",
        section="Stimulation",
        control_type="float",
        default=50.0,
        help_text="Negative pulse duration.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_enabled",
        label="Enable pulse train",
        section="Stimulation",
        control_type="bool",
        default=False,
        help_text="Adds a repeated pulse train.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_freq_hz",
        label="Pulse train frequency (Hz)",
        section="Stimulation",
        control_type="float",
        default=10.0,
        help_text="Pulse train frequency.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_amp_nA",
        label="Pulse train amp (nA)",
        section="Stimulation",
        control_type="float",
        default=2.5,
        help_text="Pulse train amplitude.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_delay_ms",
        label="Pulse train start delay (ms)",
        section="Stimulation",
        control_type="float",
        default=100.0,
        help_text="Pulse train start delay.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_dur_ms",
        label="Pulse train pulse width (ms)",
        section="Stimulation",
        control_type="float",
        default=20.0,
        help_text="Pulse train pulse width.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_stop_ms",
        label="Pulse train stop (ms)",
        section="Stimulation",
        control_type="float",
        default=0.0,
        help_text="0 means no explicit stop. Otherwise pulses stop at this time.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_max_pulses",
        label="Pulse train max pulses",
        section="Stimulation",
        control_type="int",
        default=0,
        help_text="0 means unlimited by count.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="pulse_train_include_base",
        label="Keep base IClamp with train",
        section="Stimulation",
        control_type="bool",
        default=False,
        help_text="When enabled, the base IClamp is kept alongside the pulse train.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="record_soma_v",
        label="Record soma V",
        section="Recording",
        control_type="choice",
        default="seeds",
        choices=("none", "seeds", "all"),
        help_text="Shared recording policy.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="record_spikes",
        label="Record spikes",
        section="Recording",
        control_type="choice",
        default="seeds",
        choices=("none", "seeds", "all"),
        help_text="Shared spike recording policy.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="spike_thresh_mV",
        label="Spike threshold (mV)",
        section="Recording",
        control_type="float",
        default=0.0,
        help_text="Spike threshold used by the shared runner.",
        cache_impact="runtime_safe",
    ),
    ControlSpec(
        key="gap_enabled",
        label="Enable gaps",
        section="Gap Junctions",
        control_type="bool",
        default=False,
        help_text="Quick gap toggle.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="gap_mechanisms_dir",
        label="Gap mechanisms dir",
        section="Gap Junctions",
        control_type="text",
        default="",
        help_text="Optional gap mechanism source or compiled mechanisms directory.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="gap_default_site",
        label="Gap default site",
        section="Gap Junctions",
        control_type="choice",
        default="ais",
        choices=("ais", "soma"),
        help_text="Default gap placement site.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="gap_default_g_uS",
        label="Gap default g (uS)",
        section="Gap Junctions",
        control_type="float",
        default=0.001,
        help_text="Default gap conductance.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="gap_pairs_json",
        label="Gap pairs JSON",
        section="Gap Junctions",
        control_type="textarea",
        default="[]",
        help_text="Optional per-pair gap list.",
        cache_impact="build_time",
    ),
    ControlSpec(
        key="edge_cache_enabled",
        label="Enable edge cache",
        section="Advanced Shared",
        control_type="bool",
        default=False,
        help_text="Shared custom mode only. Build/query the edge cache instead of requiring a manual edges file.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="edge_cache_build_if_missing",
        label="Build edge cache if missing",
        section="Advanced Shared",
        control_type="bool",
        default=True,
        help_text="Shared custom mode only.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="edge_cache_force_rebuild",
        label="Force edge cache rebuild",
        section="Advanced Shared",
        control_type="bool",
        default=False,
        help_text="Shared custom mode only.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="edge_cache_build_mode",
        label="Edge cache build mode",
        section="Advanced Shared",
        control_type="choice",
        default="from_edges_files",
        choices=("from_edges_files", "from_synapses_csv"),
        help_text="Shared custom mode only.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="edge_cache_query_mode",
        label="Edge cache query mode",
        section="Advanced Shared",
        control_type="choice",
        default="loaded_subgraph",
        choices=("loaded_subgraph", "seed_io_1hop"),
        help_text="Shared custom mode only.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="edge_cache_source_paths_text",
        label="Edge cache source paths",
        section="Advanced Shared",
        control_type="textarea",
        default="",
        help_text="Optional newline- or comma-separated edge files used to build the cache.",
        runner_scopes=("shared_runner",),
        modes=("custom",),
    ),
    ControlSpec(
        key="shared_overrides_json",
        label="Extra shared-runner overrides JSON",
        section="Advanced Shared",
        control_type="textarea",
        default="{}",
        help_text="Advanced top-level CONFIG overrides for the shared runner.",
        runner_scopes=("shared_runner",),
    ),
    ControlSpec(
        key="runtime_json",
        label="Runtime JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced runtime overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="timing_json",
        label="Timing JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced timing overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="biophysics_json",
        label="Biophysics JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced biophysics overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="stim_json",
        label="Stim JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced stimulation overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="record_json",
        label="Record JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced recording overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="gap_json",
        label="Gap JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Advanced gap overrides for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="thresholds_json",
        label="Thresholds JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Optional threshold overrides for benchmarking and summary analysis.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="reduction_profiles_json",
        label="Reduction profiles JSON",
        section="Advanced Project",
        control_type="textarea",
        default="[]",
        help_text="Optional reduction profile list for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="coalesce_profiles_json",
        label="Coalescing profiles JSON",
        section="Advanced Project",
        control_type="textarea",
        default="[]",
        help_text="Optional coalescing profile list for the hemilineage project runner.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="extra_overrides_json",
        label="Extra overrides JSON",
        section="Advanced Project",
        control_type="textarea",
        default="{}",
        help_text="Optional extra overrides merged into the hemilineage project base config.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="run_reduction_pipeline",
        label="Run reduction pipeline",
        section="Project Pipelines",
        control_type="bool",
        default=False,
        help_text="Hemilineage project only.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="run_coalescing_pipeline",
        label="Run coalescing pipeline",
        section="Project Pipelines",
        control_type="bool",
        default=False,
        help_text="Hemilineage project only.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="run_combined_pipeline",
        label="Run combined pipeline",
        section="Project Pipelines",
        control_type="bool",
        default=False,
        help_text="Hemilineage project only. Uses best reduction + coalescing candidates together.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="force_rebuild_edges",
        label="Force edge rebuild",
        section="Project Pipelines",
        control_type="bool",
        default=False,
        help_text="Hemilineage project only.",
        runner_scopes=("hemilineage_project",),
    ),
    ControlSpec(
        key="build_edges_workers",
        label="Edge-build workers",
        section="Project Pipelines",
        control_type="int",
        default=16,
        help_text="Hemilineage project only.",
        runner_scopes=("hemilineage_project",),
        cache_impact="analysis_only",
    ),
    ControlSpec(
        key="reduction_workers",
        label="Reduction workers",
        section="Project Pipelines",
        control_type="int",
        default=32,
        help_text="Hemilineage project only.",
        runner_scopes=("hemilineage_project",),
        cache_impact="analysis_only",
    ),
]


def default_state() -> Dict[str, Any]:
    state = {spec.key: spec.default for spec in CONTROL_SPECS}
    _apply_environment_path_defaults(state)
    return state


def _apply_environment_path_defaults(state: Dict[str, Any]) -> None:
    """Pre-fill path fields from Docker/local environment variables when available."""

    swc_dir = os.environ.get("DIGIFLY_SWC_DIR", "").strip()
    if swc_dir:
        swc_root = Path(swc_dir).expanduser().resolve()
        state["swc_dir"] = str(swc_root)
        state["runs_root"] = str((swc_root / "hemi_runs").resolve())
    else:
        swc_root = _infer_public_swc_dir()
        if swc_root is not None:
            state["swc_dir"] = str(swc_root)
            state["runs_root"] = str((swc_root / "hemi_runs").resolve())

    morph_swc_dir = os.environ.get("DIGIFLY_MORPH_SWC_DIR", "").strip()
    if morph_swc_dir:
        state["morph_swc_dir"] = morph_swc_dir

    gap_mech_dir = os.environ.get("DIGIFLY_GAP_MECH_DIR", "").strip()
    if gap_mech_dir:
        state["gap_mechanisms_dir"] = gap_mech_dir

    phase2_root = os.environ.get("DIGIFLY_PHASE2_ROOT", "").strip()
    workspace = os.environ.get("DIGIFLY_WORKSPACE", "").strip()
    if phase2_root:
        state["projects_root"] = str((Path(phase2_root).expanduser() / "outputs" / "workbench_projects").resolve())
    elif workspace:
        state["projects_root"] = str((Path(workspace).expanduser() / "Phase 2" / "outputs" / "workbench_projects").resolve())


def _infer_public_swc_dir() -> Path | None:
    """Infer the public Phase 1 SWC root from a repo checkout."""

    candidates: list[Path] = []
    for repo_root in _candidate_repo_roots():
        candidates.extend(
            [
                repo_root / "Phase 1" / PUBLIC_MANC_LABEL / "export_swc",
                repo_root / "Phase 1" / "export_swc",
                repo_root / "Phase 2" / "data" / "export_swc",
            ]
        )
        phase1_root = repo_root / "Phase 1"
        if phase1_root.exists():
            candidates.extend(sorted(phase1_root.glob("*/export_swc")))

    for candidate in _dedupe_paths(candidates):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _candidate_repo_roots() -> list[Path]:
    candidates: list[Path] = []

    workspace = os.environ.get("DIGIFLY_WORKSPACE", "").strip()
    if workspace:
        candidates.append(Path(workspace))

    phase2_root = os.environ.get("DIGIFLY_PHASE2_ROOT", "").strip()
    if phase2_root:
        candidates.append(Path(phase2_root).expanduser().parent)

    module_phase2_root = Path(__file__).resolve().parents[3]
    candidates.append(module_phase2_root.parent)

    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(cwd.parents)
    return _dedupe_paths(candidates)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def control_by_key() -> Dict[str, ControlSpec]:
    return {spec.key: spec for spec in CONTROL_SPECS}


def sections_for_state(state: Dict[str, Any]) -> List[str]:
    seen: List[str] = []
    runner_kind = str(state.get("runner_kind", "shared_runner"))
    mode = str(state.get("mode", "hemilineage"))
    for spec in CONTROL_SPECS:
        if _spec_visible(spec, runner_kind=runner_kind, mode=mode):
            if spec.section not in seen:
                seen.append(spec.section)
    return seen


def visible_specs(state: Dict[str, Any]) -> List[ControlSpec]:
    runner_kind = str(state.get("runner_kind", "shared_runner"))
    mode = str(state.get("mode", "hemilineage"))
    return [spec for spec in CONTROL_SPECS if _spec_visible(spec, runner_kind=runner_kind, mode=mode)]


def specs_in_section(section: str, state: Dict[str, Any]) -> List[ControlSpec]:
    return [spec for spec in visible_specs(state) if spec.section == section]


def _spec_visible(spec: ControlSpec, *, runner_kind: str, mode: str) -> bool:
    if spec.runner_scopes and runner_kind not in spec.runner_scopes:
        return False
    if spec.modes:
        if runner_kind != "shared_runner":
            return False
        return mode in spec.modes
    return True


def json_control_keys() -> Iterable[str]:
    for spec in CONTROL_SPECS:
        if spec.control_type == "textarea" and spec.key.endswith("_json"):
            yield spec.key
