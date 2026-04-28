"""
digifly/phase2/api.py

Phase 2 scaffold file.
Logic will be added during refactor.
"""
from digifly.phase2.walking.runner import run_walking_simulation

from digifly.phase2.config import build_config, get_default_config

__all__ = ["run_walking_simulation", "build_config", "get_default_config"]
