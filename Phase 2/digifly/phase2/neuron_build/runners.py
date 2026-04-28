"""
Compatibility shim.

Older notebook / refactor steps referenced:
  digifly.phase2.neuron_build.runners

Your implementation currently places most "runner" functions in:
  digifly.phase2.neuron_build.builders

So we re-export those names here to keep imports stable.
"""

from __future__ import annotations

# Re-export builder entrypoints under the expected module name
from .builders import (  # noqa: F401
    build_pair_only,
    run_pair_demo,
    build_network_driven_subset,
    expand_from_edges,
)

__all__ = [
    "build_pair_only",
    "run_pair_demo",
    "build_network_driven_subset",
    "expand_from_edges",
]
