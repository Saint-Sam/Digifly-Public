"""Wrapper launcher for the shared Phase 2 morphology mutation app."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import sys


def main(argv: Optional[Sequence[str]] = None) -> int:
    phase2_root = Path(__file__).resolve().parents[3]
    if str(phase2_root) not in sys.path:
        sys.path.insert(0, str(phase2_root))

    from digifly.phase2.extensions.glia_editing.mutation.morphology_mutation_app import main as app_main

    args = list(argv if argv is not None else sys.argv[1:])
    if "--phase2-root" not in args:
        args.extend(["--phase2-root", str(phase2_root)])
    if "--output-root" not in args:
        default_output = phase2_root / "outputs" / "glia_editing" / "mutation"
        args.extend(["--output-root", str(default_output)])
    return int(app_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
