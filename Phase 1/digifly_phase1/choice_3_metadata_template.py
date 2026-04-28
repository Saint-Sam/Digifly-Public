"""Choice 3: export all-neuron metadata template CSV."""

from __future__ import annotations

from .clients import get_default_client
from . import workflow_core


TITLE = "Export ALL neuron metadata template CSV (for Phase 2)"


def run(client=None):
    client = client or get_default_client()
    workflow_core.set_active_client(client)
    default_name = "outputs/all_neurons_neuroncriteria_template.csv"
    out = input(f"Output CSV filename [{default_name}]: ").strip() or default_name
    df_template = workflow_core.export_all_neuroncriteria_template(
        csv_out=out,
        criteria_kwargs=None,
    )
    print(df_template.head())
    return df_template
