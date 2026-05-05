from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping

from .validation import clean_optional_text, parse_int_list


@dataclass(frozen=True)
class MutationLaunchPlan:
    command: list[str]
    app_root: Path
    app_path: Path
    swc_dir: Path
    flow_run_dir: Path
    output_root: Path
    log_path: Path
    neuron_ids: list[int]
    warning: str | None = None

    def command_text(self) -> str:
        return shlex.join(self.command)


def build_mutation_launch_plan(
    state: Mapping[str, Any],
    *,
    phase2_root: str | Path,
    flow_run_dir: str | Path,
    python_bin: str | Path | None = None,
) -> MutationLaunchPlan:
    """Build a morphology mutation app command for a completed Phase 2 run."""

    phase2_root = Path(phase2_root).expanduser().resolve()
    app_root = phase2_root / "apps" / "VIP_Glia_Sim"
    app_path = app_root / "tools" / "morphology_mutation_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Mutation app script not found: {app_path}")

    flow_run_dir = Path(flow_run_dir).expanduser().resolve()
    _validate_flow_run_dir(flow_run_dir)

    swc_dir = _resolve_swc_dir(state, phase2_root=phase2_root)
    neuron_ids = _recorded_neuron_ids(flow_run_dir) or _state_neuron_ids(state)
    if not neuron_ids:
        raise ValueError(f"Could not infer neuron IDs from records.csv or workbench state for {flow_run_dir}")

    output_root = app_root / "notebooks" / "debug" / "outputs"
    log_dir = output_root / "_launcher_logs"
    log_path = log_dir / f"morphology_mutation_from_workbench_{int(time.time())}.log"

    command = [
        str(_resolve_python_bin(python_bin)),
        str(app_path),
        "--swc-dir",
        str(swc_dir),
        "--phase2-root",
        str(phase2_root),
        "--neuron-ids",
        ",".join(str(int(x)) for x in neuron_ids),
        "--output-root",
        str(output_root),
        "--tag",
        f"workbench_{flow_run_dir.name}",
        "--render-mode",
        "neuroglancer",
        "--skeleton-line-width",
        "6.0",
        "--visual-style",
        "classic",
        "--neuroglancer-quality",
        "ultra",
        "--flow-run-dir",
        str(flow_run_dir),
        "--flow-fps",
        "30",
        "--flow-speed-um-per-ms",
        "25",
        "--flow-pulse-sigma-ms",
        "18",
        "--flow-max-ms",
        "0",
        "--flow-duration-sec",
        "20",
    ]
    if len(neuron_ids) == 1:
        command.extend(["--start-solo", "--start-neuron-id", str(int(neuron_ids[0]))])

    return MutationLaunchPlan(
        command=command,
        app_root=app_root,
        app_path=app_path,
        swc_dir=swc_dir,
        flow_run_dir=flow_run_dir,
        output_root=output_root,
        log_path=log_path,
        neuron_ids=neuron_ids,
        warning=_desktop_runtime_warning(),
    )


def launch_mutation_app(plan: MutationLaunchPlan) -> dict[str, Any]:
    """Launch the mutation app and return process/log details."""

    if plan.warning:
        return {
            "pid": None,
            "returncode": None,
            "log_path": None,
            "command": plan.command_text(),
            "blocked": True,
            "reason": plan.warning,
        }

    plan.output_root.mkdir(parents=True, exist_ok=True)
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with plan.log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            plan.command,
            cwd=str(plan.app_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    time.sleep(1.0)
    return {
        "pid": proc.pid,
        "returncode": proc.poll(),
        "log_path": str(plan.log_path),
        "command": plan.command_text(),
        "blocked": False,
        "reason": None,
    }


def log_tail(path: str | Path, *, max_chars: int = 4000) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")[-int(max_chars) :]
    except Exception as exc:
        return f"Could not read launcher log: {exc}"


def _validate_flow_run_dir(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Flow run directory not found: {run_dir}")
    missing = [name for name in ("config.json", "records.csv") if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Flow run directory is missing {', '.join(missing)}: {run_dir}")


def _resolve_swc_dir(state: Mapping[str, Any], *, phase2_root: Path) -> Path:
    swc_text = clean_optional_text(state.get("swc_dir"))
    if swc_text:
        swc_dir = Path(swc_text).expanduser().resolve()
    else:
        swc_dir = (phase2_root.parent / "Phase 1" / "manc_v1.2.1" / "export_swc").resolve()
    if not swc_dir.exists() or not swc_dir.is_dir():
        raise FileNotFoundError(f"SWC root not found for mutation app launch: {swc_dir}")
    return swc_dir


def _recorded_neuron_ids(run_dir: Path) -> list[int]:
    records_path = run_dir / "records.csv"
    header = records_path.open("r", encoding="utf-8").readline().strip().split(",")
    ids: list[int] = []
    for column in header:
        column = str(column).strip()
        if not column.endswith("_soma_v"):
            continue
        raw = column[: -len("_soma_v")]
        if raw.isdigit():
            ids.append(int(raw))
    return sorted(set(ids))


def _state_neuron_ids(state: Mapping[str, Any]) -> list[int]:
    mode = str(state.get("mode", "single")).strip()
    if mode == "single":
        value = state.get("neuron_id")
        return [int(value)] if str(value).strip() else []
    if mode == "custom":
        return parse_int_list(state.get("neuron_ids_text", ""), allow_empty=True)
    return parse_int_list(state.get("hemi_core_ids_text", ""), allow_empty=True)


def _resolve_python_bin(python_bin: str | Path | None) -> Path:
    candidates = [
        python_bin,
        os.environ.get("VIP_PYTHON_BIN", "").strip() or None,
        os.environ.get("PYTHON_BIN", "").strip() or None,
        "/opt/anaconda3/bin/python3.12",
        "/opt/anaconda3/bin/python3",
        "/opt/anaconda3/bin/python",
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def _desktop_runtime_warning() -> str | None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return (
            "This Python session does not expose DISPLAY/WAYLAND_DISPLAY. "
            "The PyVista desktop app may not open from a browser-only Docker session."
        )
    return None
