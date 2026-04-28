from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .controls import CONTROL_SPECS
from .presets import apply_preset


def summarize_impacts(state: Mapping[str, Any], *, preset_slug: str | None = None) -> Dict[str, List[str]]:
    baseline = apply_preset(preset_slug) if preset_slug else {spec.key: spec.default for spec in CONTROL_SPECS}
    impacts: Dict[str, List[str]] = {
        "build_time": [],
        "runtime_safe": [],
        "analysis_only": [],
    }
    for spec in CONTROL_SPECS:
        current = _normalize(state.get(spec.key))
        base_value = _normalize(baseline.get(spec.key))
        if current != base_value:
            impacts.setdefault(spec.cache_impact, []).append(spec.label)
    return impacts


def format_impact_summary(impact_summary: Mapping[str, List[str]]) -> str:
    lines: List[str] = []
    for key in ("build_time", "runtime_safe", "analysis_only"):
        labels = impact_summary.get(key) or []
        if not labels:
            continue
        lines.append(f"{key}:")
        lines.extend(f"- {label}" for label in labels)
    return "\n".join(lines) if lines else "No preset deltas."


def _normalize(value: Any) -> Any:
    if value in ("", None):
        return None
    return value
