from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class ProjectLayout:
    slug: str
    root: Path
    outputs: Path
    runs: Path
    workbench_runs: Path
    logs: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "root": str(self.root),
            "outputs": str(self.outputs),
            "runs": str(self.runs),
            "workbench_runs": str(self.workbench_runs),
            "logs": str(self.logs),
        }


def find_phase2_root(start: str | Path | None = None) -> Path:
    """Find the Phase 2 root from a project notebook or a repo checkout."""

    candidates: list[Path] = []
    env_root = os.environ.get("DIGIFLY_PHASE2_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    if start is not None:
        candidates.append(Path(start))
    try:
        candidates.append(Path.cwd())
    except FileNotFoundError:
        pass

    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.expanduser().resolve()
        except Exception:
            candidate = candidate.expanduser()
        for root in [candidate, *candidate.parents]:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            if (root / "digifly").exists() and (root / "Projects").exists():
                return root
            nested = root / "Phase 2"
            if (nested / "digifly").exists() and (nested / "Projects").exists():
                return nested.resolve()

    raise RuntimeError("Could not locate Phase 2. Set DIGIFLY_PHASE2_ROOT to the folder that contains digifly/.")


def project_layout(slug: str, *, phase2_root: str | Path | None = None) -> ProjectLayout:
    phase2 = Path(phase2_root).expanduser().resolve() if phase2_root else find_phase2_root()
    root = phase2 / "Projects" / str(slug)
    outputs = root / "outputs"
    return ProjectLayout(
        slug=str(slug),
        root=root,
        outputs=outputs,
        runs=outputs / "runs",
        workbench_runs=outputs / "workbench_runs",
        logs=outputs / "logs",
    )


def activate_project(slug: str, *, phase2_root: str | Path | None = None) -> ProjectLayout:
    """Create the project output folders and point notebook runs at them."""

    phase2 = Path(phase2_root).expanduser().resolve() if phase2_root else find_phase2_root()
    layout = project_layout(slug, phase2_root=phase2)
    for path in (layout.outputs, layout.runs, layout.workbench_runs, layout.logs):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["DIGIFLY_PHASE2_ROOT"] = str(phase2)
    os.environ["DIGIFLY_PROJECT_SLUG"] = layout.slug
    os.environ["DIGIFLY_PROJECT_ROOT"] = str(layout.root)
    os.environ["DIGIFLY_RUNS_ROOT"] = str(layout.runs)
    os.environ["DIGIFLY_PROJECTS_ROOT"] = str(layout.outputs / "workbench_projects")
    os.environ["DIGIFLY_WORKBENCH_RUNS_ROOT"] = str(layout.workbench_runs)

    for path in (phase2, phase2 / "Projects"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    return layout
