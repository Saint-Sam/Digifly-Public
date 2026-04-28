"""Compatibility re-exports for the workbench preset registry."""

from digifly.phase2.workbench.presets import PRESETS, PresetSpec, apply_preset, get_preset

__all__ = ["PRESETS", "PresetSpec", "apply_preset", "get_preset"]
