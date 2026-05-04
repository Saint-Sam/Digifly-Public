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

    from neuron import h, load_mechanisms

    mech_dir = Path(os.environ.get("DIGIFLY_GAP_MECH_DIR", "/opt/digifly-mechanisms"))
    names = ("Gap", "RectGap", "HeteroRectGap")
    missing = [name for name in names if not hasattr(h, name)]
    if missing:
        try:
            load_mechanisms(str(mech_dir))
        except RuntimeError as exc:
            if "already exists" not in str(exc):
                raise
    missing = [name for name in names if not hasattr(h, name)]
    if missing:
        raise RuntimeError(f"Missing compiled NEURON mechanisms: {missing}")

    import digifly.phase2.api  # noqa: F401
    import digifly.phase2.neuron_build.network  # noqa: F401

    print("[ok] NEURON import works")
    print("[ok] Digifly Phase 2 imports work")
    print(f"[ok] mechanism dir = {mech_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
