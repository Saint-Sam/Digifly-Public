from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from digifly.phase2.runtime_env import build_phase2_launch_env, ensure_phase2_environment, resolve_mpiexec


@dataclass(frozen=True)
class CachedSessionPaths:
    root: Path
    case_json: Path
    status_json: Path
    session_info_json: Path
    requests_dir: Path
    responses_dir: Path
    stdout_log: Path
    stderr_log: Path


def cached_session_paths(session_root: str | Path) -> CachedSessionPaths:
    root = Path(session_root).expanduser().resolve()
    return CachedSessionPaths(
        root=root,
        case_json=root / "case.json",
        status_json=root / "status.json",
        session_info_json=root / "session_info.json",
        requests_dir=root / "requests",
        responses_dir=root / "responses",
        stdout_log=root / "phase2_cache.stdout.txt",
        stderr_log=root / "phase2_cache.stderr.txt",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return out


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _safe_read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return None
        data = _read_json(resolved)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _phase2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_launch_env(
    base_env: Mapping[str, str] | None = None,
    *,
    include_mpi_library: bool = False,
) -> dict[str, str]:
    return build_phase2_launch_env(
        base_env or os.environ,
        phase2_dir=_phase2_root(),
        include_mpi_library=bool(include_mpi_library),
    )


def prepare_cached_config(cfg: Mapping[str, Any], *, nproc: int = 1) -> dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    if int(nproc) > 1:
        parallel_cfg = dict(out.get("parallel") or {})
        parallel_cfg["build_backend"] = "distributed_gid"
        out["parallel"] = parallel_cfg
    return out


def cache_fingerprint(cfg: Mapping[str, Any], *, nproc: int = 1) -> str:
    from .session import _build_cache_fingerprint

    return _build_cache_fingerprint(prepare_cached_config(cfg, nproc=nproc))


def _validate_child_python_env(
    *,
    python_exe: str,
    env: Mapping[str, str],
    cwd: str | Path,
) -> None:
    if str(env.get("DIGIFLY_SKIP_CACHE_IMPORT_PREFLIGHT", "")).strip() in {"1", "true", "TRUE"}:
        return
    probe = (
        "import importlib.util, os, sys; "
        "import numpy, pandas; "
        "print('numpy=' + str(numpy.__file__)); "
        "print('pandas=' + str(pandas.__file__))"
    )
    result = subprocess.run(
        [str(python_exe), "-c", probe],
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cached Phase 2 child Python cannot import numpy/pandas with the launch environment.\n"
            f"python={python_exe}\n"
            f"cwd={Path(cwd).expanduser().resolve()}\n"
            f"PYTHONPATH={env.get('PYTHONPATH', '')}\n"
            f"DYLD_LIBRARY_PATH={env.get('DYLD_LIBRARY_PATH', '')}\n"
            f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def read_cache_status(session_root: str | Path) -> dict[str, Any] | None:
    return _safe_read_json(cached_session_paths(session_root).status_json)


def _session_ready_for_config(session_root: str | Path, cfg: Mapping[str, Any], *, nproc: int = 1) -> bool:
    status = read_cache_status(session_root) or {}
    if str(status.get("state") or "") != "ready":
        return False
    return str(status.get("cache_fingerprint") or "") == cache_fingerprint(cfg, nproc=nproc)


def wait_for_cache_ready(
    session_root: str | Path,
    *,
    timeout_s: float = 900.0,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    paths = cached_session_paths(session_root)
    deadline = time.time() + float(timeout_s)
    while time.time() <= deadline:
        status = _safe_read_json(paths.status_json)
        if status:
            state = str(status.get("state") or "")
            if state == "ready":
                return status
            if state in {"error", "stopped"}:
                raise RuntimeError(f"Cached Phase 2 session is {state}: {status}")
        time.sleep(float(poll_s))
    raise TimeoutError(f"Timed out waiting for cached Phase 2 session to become ready: {paths.status_json}")


def start_cached_session(
    cfg: Mapping[str, Any],
    *,
    session_root: str | Path,
    nproc: int = 1,
    python_exe: str | Path | None = None,
    mpiexec: str | Path | None = None,
    force_restart: bool = False,
    wait_ready: bool = True,
    timeout_s: float = 900.0,
    auto_install_python: bool = False,
) -> dict[str, Any]:
    env_report = ensure_phase2_environment(
        profiles=("core",),
        auto_install_python=bool(auto_install_python),
        check_gap_mechanisms=False,
        quiet=True,
    )
    if env_report.get("missing_python_packages"):
        raise RuntimeError(
            "Cannot start cached Phase 2 session because required Python packages are missing: "
            f"{env_report['missing_python_packages']}. "
            "Pass auto_install_python=True from a notebook or install them into the active Python environment."
        )
    paths = cached_session_paths(session_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.requests_dir.mkdir(parents=True, exist_ok=True)
    paths.responses_dir.mkdir(parents=True, exist_ok=True)

    if force_restart:
        try:
            shutdown_cached_session(paths.root, timeout_s=20.0)
        except Exception:
            pass
    elif _session_ready_for_config(paths.root, cfg, nproc=nproc):
        status = read_cache_status(paths.root) or {}
        status["started"] = False
        return status

    for old_dir in (paths.requests_dir, paths.responses_dir):
        for old_path in old_dir.glob("*.json"):
            try:
                old_path.unlink()
            except Exception:
                pass

    cfg_use = prepare_cached_config(cfg, nproc=nproc)
    _write_json(paths.case_json, cfg_use)
    _write_json(
        paths.status_json,
        {
            "state": "starting",
            "backend": "distributed_gid" if int(nproc) > 1 else "single_host",
            "nproc": int(nproc),
            "cache_fingerprint": cache_fingerprint(cfg, nproc=nproc),
            "started_by": "phase2_cache_launcher",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    py = str(Path(python_exe).expanduser().resolve()) if python_exe else sys.executable
    cmd = [
        py,
        "-m",
        "digifly.phase2.cache.session",
        "--case-json",
        str(paths.case_json),
        "--session-root",
        str(paths.root),
    ]
    if int(nproc) > 1:
        mpi = str(Path(mpiexec).expanduser().resolve()) if mpiexec else (resolve_mpiexec() or "mpiexec")
        cmd = [mpi, "-n", str(int(nproc)), *cmd]

    env = build_launch_env(include_mpi_library=int(nproc) > 1)
    _validate_child_python_env(python_exe=py, env=env, cwd=_phase2_root())
    with paths.stdout_log.open("a", encoding="utf-8") as stdout, paths.stderr_log.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=str(_phase2_root()),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    launch_info = {
        "state": "starting",
        "started": True,
        "pid": int(process.pid),
        "cmd": cmd,
        "session_root": str(paths.root),
        "stdout_log": str(paths.stdout_log),
        "stderr_log": str(paths.stderr_log),
        "cache_fingerprint": cache_fingerprint(cfg, nproc=nproc),
    }
    _write_json(paths.root / "launcher.json", launch_info)
    if wait_ready:
        ready = wait_for_cache_ready(paths.root, timeout_s=timeout_s)
        ready["started"] = True
        ready["pid"] = int(process.pid)
        ready["stdout_log"] = str(paths.stdout_log)
        ready["stderr_log"] = str(paths.stderr_log)
        return ready
    return launch_info


def _next_command_id(paths: CachedSessionPaths) -> int:
    max_seen = 0
    for root in (paths.requests_dir, paths.responses_dir):
        if not root.exists():
            continue
        for path in root.glob("*.json"):
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            if digits:
                max_seen = max(max_seen, int(digits))
    return max_seen + 1


def submit_cached_run(
    session_root: str | Path,
    *,
    run_id: str,
    runtime_overrides: Mapping[str, Any] | None = None,
    stim_overrides: Mapping[str, Any] | None = None,
    record_overrides: Mapping[str, Any] | None = None,
    synapse_group_overrides: list[Mapping[str, Any]] | None = None,
    gap_group_overrides: list[Mapping[str, Any]] | None = None,
    cell_biophys_overrides: list[Mapping[str, Any]] | None = None,
    stim_target_ids: list[int] | None = None,
    run_notes: str = "",
    timeout_s: float = 900.0,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    paths = cached_session_paths(session_root)
    status = read_cache_status(paths.root) or {}
    if str(status.get("state") or "") != "ready":
        raise RuntimeError(f"Cached Phase 2 session is not ready: {status}")

    command_id = _next_command_id(paths)
    request_path = paths.requests_dir / f"request_{command_id:04d}.json"
    response_path = paths.responses_dir / f"response_{command_id:04d}.json"
    payload: dict[str, Any] = {
        "command_id": int(command_id),
        "action": "run",
        "run_id": str(run_id),
        "run_notes": str(run_notes or ""),
        "response_json": str(response_path),
        "runtime_overrides": dict(runtime_overrides or {}),
        "stim_overrides": dict(stim_overrides or {}),
        "record_overrides": dict(record_overrides or {}),
        "synapse_group_overrides": [dict(x) for x in list(synapse_group_overrides or [])],
        "gap_group_overrides": [dict(x) for x in list(gap_group_overrides or [])],
        "cell_biophys_overrides": [dict(x) for x in list(cell_biophys_overrides or [])],
    }
    if stim_target_ids is not None:
        payload["stim_target_ids"] = [int(x) for x in stim_target_ids]
    _write_json(request_path, payload)

    deadline = time.time() + float(timeout_s)
    while time.time() <= deadline:
        response = _safe_read_json(response_path)
        if response:
            if str(response.get("status") or "") == "ok":
                return response
            raise RuntimeError(f"Cached Phase 2 run failed: {response}")
        status = read_cache_status(paths.root) or {}
        if str(status.get("state") or "") == "error":
            raise RuntimeError(f"Cached Phase 2 session errored while running {run_id}: {status}")
        time.sleep(float(poll_s))
    raise TimeoutError(f"Timed out waiting for cached Phase 2 response: {response_path}")


def shutdown_cached_session(session_root: str | Path, *, timeout_s: float = 60.0) -> dict[str, Any] | None:
    paths = cached_session_paths(session_root)
    status = read_cache_status(paths.root) or {}
    if str(status.get("state") or "") in {"", "stopped"}:
        return status or None
    command_id = _next_command_id(paths)
    request_path = paths.requests_dir / f"request_{command_id:04d}.json"
    response_path = paths.responses_dir / f"response_{command_id:04d}.json"
    _write_json(
        request_path,
        {
            "command_id": int(command_id),
            "action": "shutdown",
            "response_json": str(response_path),
        },
    )
    deadline = time.time() + float(timeout_s)
    while time.time() <= deadline:
        response = _safe_read_json(response_path)
        if response:
            return response
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for cached Phase 2 shutdown response: {response_path}")


def run_cached_simulation(
    cfg: Mapping[str, Any],
    *,
    session_root: str | Path,
    run_id: str | None = None,
    nproc: int = 1,
    force_restart: bool = False,
    start_timeout_s: float = 900.0,
    run_timeout_s: float = 900.0,
    runtime_overrides: Mapping[str, Any] | None = None,
    stim_overrides: Mapping[str, Any] | None = None,
    record_overrides: Mapping[str, Any] | None = None,
    synapse_group_overrides: list[Mapping[str, Any]] | None = None,
    gap_group_overrides: list[Mapping[str, Any]] | None = None,
    cell_biophys_overrides: list[Mapping[str, Any]] | None = None,
    stim_target_ids: list[int] | None = None,
    run_notes: str = "",
    auto_install_python: bool = False,
) -> dict[str, Any]:
    if not _session_ready_for_config(session_root, cfg, nproc=nproc):
        status = read_cache_status(session_root) or {}
        if str(status.get("state") or "") == "ready" and not force_restart:
            raise RuntimeError(
                "A cached Phase 2 session is ready, but it was built for a different config. "
                "Pass force_restart=True or choose another session_root."
            )
        start_cached_session(
            cfg,
            session_root=session_root,
            nproc=nproc,
            force_restart=force_restart,
            wait_ready=True,
            timeout_s=start_timeout_s,
            auto_install_python=auto_install_python,
        )

    return submit_cached_run(
        session_root,
        run_id=str(run_id or cfg.get("run_id") or "cached_phase2_run"),
        runtime_overrides=runtime_overrides,
        stim_overrides=stim_overrides,
        record_overrides=record_overrides,
        synapse_group_overrides=synapse_group_overrides,
        gap_group_overrides=gap_group_overrides,
        cell_biophys_overrides=cell_biophys_overrides,
        stim_target_ids=stim_target_ids,
        run_notes=run_notes,
        timeout_s=run_timeout_s,
    )
