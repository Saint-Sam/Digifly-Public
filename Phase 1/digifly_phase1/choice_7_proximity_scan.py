"""Choice 7: scan exported neurons for proximity to reference skeletons."""

from __future__ import annotations

from .clients import get_default_client
from . import workflow_core


TITLE = "Proximity scan to refs (10000/10002)"


def run(client=None):
    client = client or get_default_client()
    workflow_core.set_active_client(client)
    return workflow_core.find_neurons_near_reference_skeletons(client)
