from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    workspace = Path(os.environ.get("DIGIFLY_WORKSPACE", "/workspace"))
    phase2 = workspace / "Phase 2"
    phase1 = workspace / "Phase 1"
    for path in (phase2, phase1, workspace):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from digifly.phase2.api import ensure_phase2_environment

    report = ensure_phase2_environment(
        profiles=("core", "notebook"),
        auto_install_python=False,
        check_gap_mechanisms=True,
        quiet=True,
    )
    if report.get("missing_python_packages"):
        raise RuntimeError(f"Missing Python packages in container: {report['missing_python_packages']}")
    if report.get("gap_mechanisms_available") is False:
        raise RuntimeError(f"Gap mechanisms are unavailable: {report.get('errors')}")

    mech_dir = Path(report.get("gap_mechanisms_dir") or os.environ.get("DIGIFLY_GAP_MECH_DIR", "/opt/digifly-mechanisms"))

    import digifly.phase2.api  # noqa: F401
    import digifly.phase2.neuron_build.network  # noqa: F401

    print("[ok] NEURON import works")
    print("[ok] Digifly Phase 2 imports work")
    print(f"[ok] mechanism dir = {mech_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
