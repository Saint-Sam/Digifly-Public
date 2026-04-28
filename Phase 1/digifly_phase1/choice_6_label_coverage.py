"""Choice 6: report dataset label coverage."""

from __future__ import annotations

from .clients import get_default_client
from . import workflow_core


TITLE = "Dataset label coverage (labeled vs unlabeled)"


def run(client=None):
    client = client or get_default_client()
    workflow_core.set_active_client(client)
    return workflow_core.report_dataset_label_coverage(client)
