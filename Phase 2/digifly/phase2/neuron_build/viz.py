"""
Compatibility shim.

Older code referenced:
  digifly.phase2.neuron_build.viz

Your visualization functions currently live in:
  digifly.phase2.neuron_build.viz_ais

So we re-export the expected names here.
"""

from __future__ import annotations

from .viz_ais import (  # noqa: F401
    visualize_ais_strict,
    fix_and_visualize_soma_ais,
    visualize_ais,
)

__all__ = [
    "visualize_ais_strict",
    "fix_and_visualize_soma_ais",
    "visualize_ais",
]
