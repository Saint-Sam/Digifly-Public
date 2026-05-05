from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .controls import default_state


@dataclass(frozen=True)
class PresetSpec:
    slug: str
    label: str
    description: str
    state_overrides: Mapping[str, Any]
    notes: Sequence[str] = field(default_factory=tuple)


PRESETS: List[PresetSpec] = [
    PresetSpec(
        slug="single-neuron-debug",
        label="Single Neuron Debug",
        description="Shared runner preset for a single-neuron sanity run.",
        state_overrides={
            "runner_kind": "shared_runner",
            "mode": "single",
            "run_id": "single_neuron_debug",
            "neuron_id": 10000,
            "tstop_ms": 1000.0,
            "iclamp_amp_nA": 0.0,
            "iclamp_delay_ms": 100.0,
            "iclamp_dur_ms": 0.0,
            "pulse_train_enabled": True,
            "pulse_train_freq_hz": 4.0,
            "pulse_train_amp_nA": 1.0,
            "pulse_train_delay_ms": 100.0,
            "pulse_train_dur_ms": 80.0,
            "pulse_train_stop_ms": 900.0,
            "pulse_train_max_pulses": 4,
            "pulse_train_include_base": False,
            "record_soma_v": "all",
            "record_spikes": "all",
        },
        notes=(
            "Best first preset for the public Docker runtime.",
            "Uses four repeated pulse-train events across the 1000 ms run for a clearer flow demo.",
            "Uses the bundled 10000 SWC and does not require the full metadata CSV.",
        ),
    ),
    PresetSpec(
        slug="hemilineage-network-quick",
        label="Hemilineage Network Quick",
        description="Shared runner preset for a single hemilineage build/run using run_walking_simulation().",
        state_overrides={
            "runner_kind": "shared_runner",
            "mode": "hemilineage",
            "run_id": "hemi_network_quick",
            "hemi_label": "09A",
            "record_soma_v": "seeds",
            "record_spikes": "seeds",
            "post_active": True,
            "active_posts_mode": "all_selected",
        },
        notes=(
            "Good first preset when a full Phase 1 metadata export is available.",
            "Uses the master CSV to resolve all neurons in the hemilineage.",
        ),
    ),
    PresetSpec(
        slug="custom-network-quick",
        label="Custom Network Quick",
        description="Shared runner preset for an explicit custom circuit plus an edges file or edge cache.",
        state_overrides={
            "runner_kind": "shared_runner",
            "mode": "custom",
            "run_id": "custom_network_quick",
            "neuron_ids_text": "10000, 10002",
            "record_soma_v": "seeds",
            "record_spikes": "seeds",
        },
        notes=("Set either edges_path or enable the edge cache before running.",),
    ),
    PresetSpec(
        slug="hemilineage-project-baseline",
        label="Hemilineage Project Baseline",
        description="Dedicated Phase 2 project pipeline preset using run_full_hemilineage_project() with only the baseline run enabled.",
        state_overrides={
            "runner_kind": "hemilineage_project",
            "run_id": "hemi_project_baseline",
            "hemi_label": "09A",
            "run_reduction_pipeline": False,
            "run_coalescing_pipeline": False,
            "run_combined_pipeline": False,
            "record_soma_v": "all",
            "record_spikes": "seeds",
        },
        notes=(
            "Best first preset for the fuller hemilineage project surface.",
            "If hemi_core_ids_text is blank, core IDs are derived from the master CSV using the hemilineage label.",
        ),
    ),
    PresetSpec(
        slug="hemilineage-project-full",
        label="Hemilineage Project Full Pipeline",
        description="Dedicated Phase 2 project pipeline preset with reduction and coalescing enabled.",
        state_overrides={
            "runner_kind": "hemilineage_project",
            "run_id": "hemi_project_full",
            "hemi_label": "09A",
            "run_reduction_pipeline": True,
            "run_coalescing_pipeline": True,
            "run_combined_pipeline": True,
            "force_rebuild_edges": False,
            "record_soma_v": "all",
            "record_spikes": "all",
        },
        notes=(
            "This is the notebook-first bridge toward the fuller Phase 2 pipeline.",
            "Use the advanced project JSON boxes for thresholds, reduction profiles, and coalescing profiles.",
        ),
    ),
]


def get_preset(slug: str) -> PresetSpec:
    for preset in PRESETS:
        if preset.slug == slug:
            return preset
    raise KeyError(f"Unknown preset: {slug}")


def apply_preset(slug: str, base_state: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    state = dict(default_state() if base_state is None else base_state)
    state.update(dict(get_preset(slug).state_overrides))
    return state


def preset_options() -> List[Tuple[str, str]]:
    return [(preset.label, preset.slug) for preset in PRESETS]


def iter_notes(slug: str) -> Iterable[str]:
    return get_preset(slug).notes
