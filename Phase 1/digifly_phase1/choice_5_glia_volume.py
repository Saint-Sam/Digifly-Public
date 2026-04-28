"""Choice 5: export glia volume table."""

from __future__ import annotations

from .clients import get_default_client
from . import workflow_core


TITLE = "Export manc:v1.2.1 glia volume table (Phase 3 support)"


def run(client=None):
    client = client or get_default_client()
    workflow_core.set_active_client(client)
    default_base = "outputs/manc_v1.2.1_glia_volume"
    out_base = input(f"Output base path (no extension) [{default_base}]: ").strip() or default_base
    df_glia = workflow_core.export_glia_volume_manc_v121(out_base=out_base)
    print(df_glia.head())
    return df_glia
