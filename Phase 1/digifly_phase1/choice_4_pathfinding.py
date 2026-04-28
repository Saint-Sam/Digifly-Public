"""Choice 4: pathfinding utilities for downstream support."""

from __future__ import annotations

from . import workflow_core


TITLE = "Pathfinding utilities (Phase 3 support)"


def run(client=None):
    if client is not None:
        workflow_core.set_active_client(client)
    return workflow_core.run_pathfinding_option_26()
