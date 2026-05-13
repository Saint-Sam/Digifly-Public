"""Public Phase 2 entry points."""

from digifly.phase2.cache import (
    read_cache_status,
    run_cached_simulation,
    shutdown_cached_session,
    start_cached_session,
    submit_cached_run,
)
from digifly.phase2.config import build_config, get_default_config
from digifly.phase2.runtime_env import (
    configure_phase2_environment,
    diagnose_phase2_environment,
    ensure_phase2_environment,
)


def run_walking_simulation(*args, **kwargs):
    from digifly.phase2.walking.runner import run_walking_simulation as _run_walking_simulation

    return _run_walking_simulation(*args, **kwargs)

__all__ = [
    "build_config",
    "configure_phase2_environment",
    "diagnose_phase2_environment",
    "ensure_phase2_environment",
    "get_default_config",
    "read_cache_status",
    "run_cached_simulation",
    "run_walking_simulation",
    "shutdown_cached_session",
    "start_cached_session",
    "submit_cached_run",
]
