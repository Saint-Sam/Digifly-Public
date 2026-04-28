"""Wrapper launcher for the shared Phase 2 SWC box selector app."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import sys


def main(argv: Optional[Sequence[str]] = None) -> int:
    phase2_root = Path(__file__).resolve().parents[3]
    if str(phase2_root) not in sys.path:
        sys.path.insert(0, str(phase2_root))

    from digifly.phase2.extensions.glia_editing.selectors.swc_box_selector_app import main as app_main

    args = list(argv if argv is not None else sys.argv[1:])
    if "--phase2-root" not in args:
        args.extend(["--phase2-root", str(phase2_root)])
    if "--output-spec-json" not in args:
        default_json = phase2_root / "outputs" / "glia_editing" / "selectors" / "selected_glia_spec.json"
        args.extend(["--output-spec-json", str(default_json)])
    return int(app_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
