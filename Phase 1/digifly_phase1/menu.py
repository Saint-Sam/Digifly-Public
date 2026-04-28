"""Data-driven Phase 1 menu."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill
from typing import Callable

from . import choice_1_build_exports
from . import choice_2_batch_filter_export
from . import choice_3_metadata_template
from . import choice_4_pathfinding
from . import choice_5_glia_volume
from . import choice_6_label_coverage
from . import choice_7_proximity_scan


@dataclass(frozen=True)
class Choice:
    code: str
    title: str
    description: str
    runner: Callable


CORE_CHOICES: list[Choice] = [
    Choice(
        "1",
        choice_1_build_exports.TITLE,
        "Core SWC export workflow. First asks which neuPrint connectome to use, "
        "then takes body IDs, cell-family prefixes, ALL, UNLABELED, or "
        "UNLABELED_STRICT; writes dataset-scoped SWCs, synapse CSVs, mapped "
        "SWCs, and metadata needed by later phases.",
        choice_1_build_exports.run,
    ),
    Choice(
        "2",
        choice_2_batch_filter_export.TITLE,
        "Reads body IDs from an Excel sheet, computes a size metric, keeps IDs "
        "above your threshold, then exports a filtered SWC bundle plus kept, "
        "dropped, failed, and bodyId-list reports.",
        choice_2_batch_filter_export.run,
    ),
    Choice(
        "3",
        choice_3_metadata_template.TITLE,
        "Exports a NeuronCriteria-style all-neuron metadata CSV. Use this as a "
        "Phase 2/3 planning table or as a master sheet for downstream filters.",
        choice_3_metadata_template.run,
    ),
    Choice(
        "4",
        choice_4_pathfinding.TITLE,
        "Runs male-CNS pathfinding between selected upstream and downstream "
        "neurons, annotates path rows with neurotransmitter predictions, prints "
        "intermediate summaries, and can save the combined path table.",
        choice_4_pathfinding.run,
    ),
]


UTILITY_CHOICES: list[Choice] = [
    Choice(
        "5",
        choice_5_glia_volume.TITLE,
        "Queries MANC glia, computes skeleton bounding-box size metrics, and "
        "exports glia volume tables as CSV and parquet for Phase 3 support.",
        choice_5_glia_volume.run,
    ),
    Choice(
        "6",
        choice_6_label_coverage.TITLE,
        "Counts labeled vs unlabeled neurons in the active dataset using type "
        "and instance fields, with an optional strict-unlabeled bodyId sample.",
        choice_6_label_coverage.run,
    ),
    Choice(
        "7",
        choice_7_proximity_scan.TITLE,
        "Loads a master metadata CSV from Choice 1, compares exported SWCs "
        "against reference skeletons, and saves neurons within the requested "
        "distance threshold.",
        choice_7_proximity_scan.run,
    ),
]

# Backward-compatible full registry for code that imports CHOICES directly.
CHOICES: list[Choice] = CORE_CHOICES + UTILITY_CHOICES


def register_choice(choice: Choice) -> None:
    if any(existing.code == choice.code for existing in CHOICES):
        raise ValueError(f"Choice code already registered: {choice.code}")
    CORE_CHOICES.append(choice)
    CHOICES.append(choice)


def _visible_choices(*, show_utilities: bool) -> list[Choice]:
    return CHOICES if show_utilities else CORE_CHOICES


def main_menu(client=None, *, show_utilities: bool = False):
    visible_choices = _visible_choices(show_utilities=show_utilities)
    exit_code = str(max(int(choice.code) for choice in visible_choices) + 1)

    while True:
        print("\n===== Phase 1 Menu (SWC + Phase2/3) =====")
        for choice in visible_choices:
            print(f"{choice.code}. {choice.title}")
            print(fill(choice.description, width=88, initial_indent="   ", subsequent_indent="   "))
        if not show_utilities:
            print("U. Show utility choices")
        print(f"{exit_code}. Exit")

        valid_codes = ",".join(c.code for c in visible_choices)
        utility_hint = ",U" if not show_utilities else ""
        selected = input(f"Choose ({valid_codes}{utility_hint},{exit_code}): ").strip()
        if not show_utilities and selected.lower() == "u":
            show_utilities = True
            visible_choices = _visible_choices(show_utilities=True)
            exit_code = str(max(int(choice.code) for choice in visible_choices) + 1)
            continue
        if selected == exit_code:
            print("Exiting.")
            break

        choice_map = {choice.code: choice for choice in visible_choices}
        choice = choice_map.get(selected)
        if choice is None:
            print("Invalid choice, try again.")
            continue

        choice.runner(client=client)
