from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def prepare_workbench_bundle(phase2_root: str | Path, *, run_id: str, preset_slug: str, state: Mapping[str, Any], plan: Mapping[str, Any]) -> Path:
    bundle_dir = (Path(phase2_root).expanduser().resolve() / "workbench_runs" / str(run_id)).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "workbench_state.json", state)
    _write_json(bundle_dir / "execution_plan.json", plan)
    _write_json(
        bundle_dir / "run_manifest.json",
        {
            "created_at_utc": _utc_now_iso(),
            "run_id": str(run_id),
            "preset_slug": str(preset_slug),
            "runner_kind": str(plan.get("runner_kind")),
            "description": str(plan.get("description", "")),
        },
    )
    _ensure_placeholder_logs(bundle_dir)
    return bundle_dir


def record_execution_status(target_dir: str | Path, *, status: str, result: Mapping[str, Any] | None = None, failure: Mapping[str, Any] | None = None) -> None:
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "run_status.json",
        {
            "updated_at_utc": _utc_now_iso(),
            "status": str(status),
            "result": _json_ready(result or {}),
        },
    )
    if failure:
        _write_json(root / "failure_report.json", failure)
    _ensure_placeholder_logs(root)


def write_plan_copy(target_dir: str | Path, *, state: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "workbench_state.json", state)
    _write_json(root / "workbench_plan.json", plan)
    _ensure_placeholder_logs(root)


def build_failure_report(exc: Exception) -> Dict[str, Any]:
    return {
        "timestamp_utc": _utc_now_iso(),
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _ensure_placeholder_logs(root: Path) -> None:
    error_log = root / "error_log.md"
    if not error_log.exists():
        error_log.write_text(
            "# Error Log\n\n"
            "Record failures, warnings, and fixes for this run here.\n",
            encoding="utf-8",
        )
    fix_log = root / "fix_log.md"
    if not fix_log.exists():
        fix_log.write_text(
            "# Fix Log\n\n"
            "Use this file to note what changed between attempts.\n",
            encoding="utf-8",
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
