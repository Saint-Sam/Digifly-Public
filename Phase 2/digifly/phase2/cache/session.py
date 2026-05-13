from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _sanitize_neuron_python_env() -> None:
    app_python = "/Applications/NEURON/lib/python"
    raw_pythonpath = str(os.environ.get("PYTHONPATH", "") or "")
    if raw_pythonpath:
        kept = [part for part in raw_pythonpath.split(":") if part and not part.startswith(app_python)]
        if kept:
            os.environ["PYTHONPATH"] = ":".join(kept)
        else:
            os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.sys.path[:] = [entry for entry in os.sys.path if not str(entry).startswith(app_python)]


_sanitize_neuron_python_env()


PHASE2_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PHASE2_ROOT.parent
if str(PHASE2_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PHASE2_ROOT))

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

CACHE_PROTOCOL_VERSION = 1

from digifly.phase2.config.loader import _apply_user_friendly_overrides  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a reusable explicit-config cache session that builds once and serves repeated sim runs."
    )
    parser.add_argument("--case-json", required=True, help="Path to the full custom config JSON payload.")
    parser.add_argument("--session-root", required=True, help="Directory used for status, requests, responses, and logs.")
    return parser.parse_args()


def _backend_from_cfg(cfg: Dict[str, Any]) -> str:
    parallel_cfg = dict(cfg.get("parallel") or {})
    return str(parallel_cfg.get("build_backend") or "single_host").strip().lower()


def _preload_requested_mpi_library() -> None:
    mpi_lib = str(os.environ.get("MPI_LIB_NRN_PATH", "")).strip()
    if not mpi_lib:
        print("[phase2-cache] MPI_LIB_NRN_PATH not set before distributed run")
        return

    mpi_path = Path(mpi_lib).expanduser().resolve()
    if not mpi_path.exists():
        raise FileNotFoundError(f"MPI_LIB_NRN_PATH does not exist: {mpi_path}")

    import ctypes

    ctypes.CDLL(str(mpi_path), mode=ctypes.RTLD_GLOBAL)
    print(f"[phase2-cache] preloaded MPI library {mpi_path}")


def _maybe_init_neuron_mpi(backend: str) -> None:
    if backend != "distributed_gid":
        return
    _preload_requested_mpi_library()
    from neuron import h

    h.nrnmpi_init()
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())
    print(f"[phase2-cache] neuron mpi initialized rank={rank} nhost={nhost}")
    if nhost <= 1:
        raise RuntimeError(
            "Requested distributed_gid cached run, but NEURON MPI reports nhost=1. "
            "Launch under mpiexec and ensure MPI_LIB_NRN_PATH matches the active MPI runtime."
        )


def _maybe_quit_neuron_mpi(backend: str) -> None:
    if backend != "distributed_gid":
        return
    try:
        from neuron import h

        pc = h.ParallelContext()
        if int(pc.nhost()) > 1:
            pc.barrier()
            print("[phase2-cache] distributed session complete; calling h.quit() for clean MPI shutdown")
            h.quit()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[phase2-cache] warning: MPI shutdown was not clean: {exc}")


def _load_case(path_str: str) -> Dict[str, Any]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing case json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Case json must decode to an object: {path}")
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return out


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_json_retry(path: str | Path, *, attempts: int = 20, sleep_s: float = 0.25) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    last_exc: Exception | None = None
    for _ in range(max(int(attempts), 1)):
        try:
            return _load_json(resolved)
        except Exception as exc:
            last_exc = exc
            time.sleep(float(sleep_s))
    if last_exc is not None:
        raise last_exc
    return {}


def _session_paths(session_root: str | Path) -> Dict[str, Path]:
    root = Path(session_root).expanduser().resolve()
    paths = {
        "root": root,
        "status_json": root / "status.json",
        "requests_dir": root / "requests",
        "responses_dir": root / "responses",
        "session_info_json": root / "session_info.json",
    }
    root.mkdir(parents=True, exist_ok=True)
    paths["requests_dir"].mkdir(parents=True, exist_ok=True)
    paths["responses_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def _write_status(paths: Mapping[str, Path], payload: Mapping[str, Any]) -> Path:
    status = dict(payload)
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["pid"] = int(os.getpid())
    return _write_json(paths["status_json"], status)


def _append_rank_event(
    paths: Mapping[str, Path],
    *,
    rank: int,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    event_dir = Path(paths["root"]).expanduser().resolve() / "rank_events"
    event_dir.mkdir(parents=True, exist_ok=True)
    event_path = (event_dir / f"rank_{int(rank):03d}.jsonl").resolve()
    payload: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": int(os.getpid()),
        "rank": int(rank),
        "label": str(label),
    }
    if extra:
        payload.update(dict(extra))
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return event_path


def _write_command_status(
    paths: Mapping[str, Path],
    ctx: Mapping[str, Any],
    command: Mapping[str, Any],
    *,
    command_id: int,
    action: str,
    response_path: str | Path,
    phase: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: Dict[str, Any] = {
        "state": "running_command",
        "workflow_label": str(ctx["workflow_label"]),
        "backend": str(ctx["backend"]),
        "nproc": int(ctx["nproc"]),
        "command_id": int(command_id),
        "action": str(action),
        "request_json": str(command.get("request_json") or ""),
        "response_json": str(Path(response_path).expanduser().resolve()),
        "cache_fingerprint": str(ctx["cache_fingerprint"]),
        "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
        "run_id": str(command.get("run_id") or ""),
        "run_notes": str(command.get("run_notes") or ""),
        "phase": str(phase),
    }
    if extra:
        payload.update(dict(extra))
    return _write_status(paths, payload)


def _compact_parallel_state() -> Dict[str, int]:
    try:
        from neuron import h

        pc = h.ParallelContext()
        return {"rank": int(pc.id()), "nhost": int(pc.nhost())}
    except Exception:
        return {"rank": 0, "nhost": 1}


def _build_cache_fingerprint(cfg: Mapping[str, Any]) -> str:
    build_affecting = copy.deepcopy(dict(cfg))
    for key in [
        "run_id",
        "run_notes",
        "progress",
        "progress_chunk_ms",
        "use_tqdm",
        "tstop_ms",
        "dt_ms",
        "celsius_C",
        "iclamp_amp_nA",
        "iclamp_delay_ms",
        "iclamp_dur_ms",
        "iclamp_location",
        "pulse_train",
        "neg_pulse",
    ]:
        build_affecting.pop(key, None)
    build_affecting.pop("stim", None)
    build_affecting.pop("record", None)

    blob = json.dumps(build_affecting, sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _selection_ids(cfg: Mapping[str, Any]) -> list[int]:
    selection = dict(cfg.get("selection") or {})
    mode = str(selection.get("mode") or "").strip().lower()
    if mode == "single" and selection.get("neuron_id") is not None:
        return [int(selection["neuron_id"])]
    neuron_ids = selection.get("neuron_ids") or cfg.get("neuron_ids") or []
    return [int(x) for x in neuron_ids]


def _seed_ids(cfg: Mapping[str, Any]) -> list[int]:
    seeds = cfg.get("seeds")
    if seeds in (None, "", False):
        return []
    return [int(x) for x in (seeds or [])]


def _network_ids_from_net(net: Any, fallback: Iterable[int]) -> list[int]:
    candidates: list[int] = []
    ownership = getattr(net, "_ownership", None)
    if ownership is not None:
        try:
            candidates.extend(int(x) for x in getattr(ownership, "gids", ()) or ())
        except Exception:
            pass
    try:
        candidates.extend(int(x) for x in (getattr(net, "_swc_paths", {}) or {}).keys())
    except Exception:
        pass
    try:
        candidates.extend(int(x) for x in (getattr(net, "cells", {}) or {}).keys())
    except Exception:
        pass
    candidates.extend(int(x) for x in fallback)
    ordered: list[int] = []
    seen: set[int] = set()
    for nid in candidates:
        if int(nid) in seen:
            continue
        seen.add(int(nid))
        ordered.append(int(nid))
    return ordered


def _runtime_stim_target_ids(
    ctx: Mapping[str, Any],
    command: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[int]:
    raw_ids = command.get("stim_target_ids")
    if raw_ids in (None, "", False) or not list(raw_ids or []):
        raw_ids = cfg.get("seeds")
    if raw_ids in (None, "", False) or not list(raw_ids or []):
        raw_ids = ctx.get("seed_ids")
    if raw_ids in (None, "", False) or not list(raw_ids or []):
        raw_ids = ctx.get("final_network_ids")

    ordered: list[int] = []
    seen: set[int] = set()
    for value in list(raw_ids or []):
        nid = int(value)
        if nid in seen:
            continue
        seen.add(nid)
        ordered.append(nid)

    if not ordered:
        raise ValueError("Cached runs need at least one stimulation target id.")

    allowed = {int(x) for x in (ctx.get("final_network_ids") or [])}
    invalid = [nid for nid in ordered if nid not in allowed]
    if invalid:
        raise ValueError(
            "Cached stimulation targets must already exist in the built network. "
            f"Invalid ids: {invalid}"
        )
    return ordered


def _make_build_warm_cfg(base_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["run_id"] = f"{str(base_cfg.get('run_id') or 'explicit')}_cache_buildwarm"
    cfg["tstop_ms"] = 0.0
    cfg["progress"] = False
    cfg["use_tqdm"] = False
    # Keep the warm-build backend aligned with cached command runs.
    # Mixed CoreNEURON -> NEURON reuse has been a source of stalled smoke runs.
    cfg["enable_coreneuron"] = False
    cfg["coreneuron_gpu"] = False
    cfg["coreneuron_verbose"] = False
    cfg["coreneuron_nthread"] = 1
    cfg["record"] = {
        "soma_v": "none",
        "spikes": "none",
        "spike_thresh_mV": float(((base_cfg.get("record") or {}).get("spike_thresh_mV", 0.0))),
    }
    return cfg


def _make_cached_run_cfg(base_cfg: Mapping[str, Any], command: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    runtime_overrides = dict(command.get("runtime_overrides") or {})
    stim_overrides = dict(command.get("stim_overrides") or {})
    record_overrides = dict(command.get("record_overrides") or {})
    cell_biophys_overrides = _normalize_cell_biophys_overrides(command)

    allowed_runtime = {
        "tstop_ms",
        "dt_ms",
        "celsius_C",
        "progress",
        "use_tqdm",
        "progress_chunk_ms",
    }
    unexpected_runtime = sorted(set(runtime_overrides) - allowed_runtime)
    if unexpected_runtime:
        raise ValueError(
            "Cached runs only allow runtime overrides for "
            f"{sorted(allowed_runtime)}. Unexpected keys: {unexpected_runtime}"
        )
    for key, value in runtime_overrides.items():
        cfg[key] = copy.deepcopy(value)

    if record_overrides:
        cfg["record"] = _deep_merge(cfg.get("record") or {}, record_overrides)

    if stim_overrides:
        unexpected_stim = sorted(set(stim_overrides) - {"iclamp", "neg_pulse", "pulse_train"})
        if unexpected_stim:
            raise ValueError(
                "Cached runs only allow stim overrides for ['iclamp', 'neg_pulse', 'pulse_train']. "
                f"Unexpected keys: {unexpected_stim}"
            )
        cfg["stim"] = _deep_merge(cfg.get("stim") or {}, stim_overrides)
        cfg.pop("pulse_train", None)
        cfg.pop("neg_pulse", None)
        _apply_user_friendly_overrides(cfg)

    if command.get("run_id"):
        cfg["run_id"] = str(command["run_id"])
    if command.get("run_notes") is not None:
        cfg["run_notes"] = str(command.get("run_notes") or "")
    if cell_biophys_overrides:
        cfg["runtime_cell_biophys_overrides"] = copy.deepcopy(cell_biophys_overrides)
    return cfg


def _normalize_synapse_group_overrides(command: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw = command.get("synapse_group_overrides") or []
    if raw in (None, "", False):
        return []
    if not isinstance(raw, list):
        raise ValueError("synapse_group_overrides must be a list of group override mappings.")

    allowed_group_keys = {
        "name",
        "selectors",
        "weight_mult",
        "delay_mult",
        "tau1_mult",
        "tau2_mult",
        "e_rev_shift_mV",
    }
    allowed_selector_keys = {"pre_ids", "post_ids", "pairs"}
    groups: list[Dict[str, Any]] = []

    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"synapse_group_overrides[{idx}] must be a mapping.")
        unexpected = sorted(set(item) - allowed_group_keys)
        if unexpected:
            raise ValueError(
                f"synapse_group_overrides[{idx}] has unsupported keys: {unexpected}. "
                f"Allowed keys: {sorted(allowed_group_keys)}"
            )
        selectors = dict(item.get("selectors") or {})
        unexpected_selectors = sorted(set(selectors) - allowed_selector_keys)
        if unexpected_selectors:
            raise ValueError(
                f"synapse_group_overrides[{idx}].selectors has unsupported keys: {unexpected_selectors}. "
                f"Allowed selector keys: {sorted(allowed_selector_keys)}"
            )
        group: Dict[str, Any] = {
            "name": str(item.get("name") or f"group_{idx:02d}"),
            "selectors": {},
            "weight_mult": float(item.get("weight_mult", 1.0)),
            "delay_mult": float(item.get("delay_mult", 1.0)),
            "tau1_mult": float(item.get("tau1_mult", 1.0)),
            "tau2_mult": float(item.get("tau2_mult", 1.0)),
            "e_rev_shift_mV": float(item.get("e_rev_shift_mV", 0.0)),
        }
        if "pre_ids" in selectors:
            group["selectors"]["pre_ids"] = [int(x) for x in (selectors.get("pre_ids") or [])]
        if "post_ids" in selectors:
            group["selectors"]["post_ids"] = [int(x) for x in (selectors.get("post_ids") or [])]
        if "pairs" in selectors:
            group["selectors"]["pairs"] = [
                [int(pair[0]), int(pair[1])] for pair in (selectors.get("pairs") or [])
            ]
        groups.append(group)
    return groups


def _apply_runtime_synapse_group_overrides(net: Any, command: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if hasattr(net, "reset_synapse_parameters"):
        net.reset_synapse_parameters()
    groups = _normalize_synapse_group_overrides(command)
    if not groups:
        return []
    if not hasattr(net, "reset_synapse_parameters") or not hasattr(net, "apply_synapse_group_overrides"):
        raise RuntimeError(
            "Cached runtime synapse overrides were requested, but the live network "
            "does not expose reset/apply synapse override helpers."
    )
    return list(net.apply_synapse_group_overrides(groups) or [])


def _normalize_gap_group_overrides(command: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw = command.get("gap_group_overrides") or []
    if raw in (None, "", False):
        return []
    if not isinstance(raw, list):
        raise ValueError("gap_group_overrides must be a list of group override mappings.")

    allowed_group_keys = {"name", "selectors", "g_mult", "g_uS"}
    allowed_selector_keys = {"pairs", "a_ids", "b_ids", "source_ids", "target_ids", "modes"}
    groups: list[Dict[str, Any]] = []

    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"gap_group_overrides[{idx}] must be a mapping.")
        unexpected = sorted(set(item) - allowed_group_keys)
        if unexpected:
            raise ValueError(
                f"gap_group_overrides[{idx}] has unsupported keys: {unexpected}. "
                f"Allowed keys: {sorted(allowed_group_keys)}"
            )
        selectors = dict(item.get("selectors") or {})
        unexpected_selectors = sorted(set(selectors) - allowed_selector_keys)
        if unexpected_selectors:
            raise ValueError(
                f"gap_group_overrides[{idx}].selectors has unsupported keys: {unexpected_selectors}. "
                f"Allowed selector keys: {sorted(allowed_selector_keys)}"
            )
        group: Dict[str, Any] = {
            "name": str(item.get("name") or f"gap_group_{idx:02d}"),
            "selectors": {},
            "g_mult": float(item.get("g_mult", 1.0)),
        }
        if "g_uS" in item and item.get("g_uS") is not None:
            group["g_uS"] = float(item.get("g_uS"))
        if "pairs" in selectors:
            group["selectors"]["pairs"] = [
                [int(pair[0]), int(pair[1])] for pair in (selectors.get("pairs") or [])
            ]
        for key in ("a_ids", "b_ids", "source_ids", "target_ids"):
            if key in selectors:
                group["selectors"][key] = [int(x) for x in (selectors.get(key) or [])]
        if "modes" in selectors:
            group["selectors"]["modes"] = [str(x) for x in (selectors.get("modes") or [])]
        groups.append(group)
    return groups


def _apply_runtime_gap_group_overrides(net: Any, command: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if hasattr(net, "reset_gap_parameters"):
        net.reset_gap_parameters()
    groups = _normalize_gap_group_overrides(command)
    if not groups:
        return []
    if not hasattr(net, "reset_gap_parameters") or not hasattr(net, "apply_gap_group_overrides"):
        raise RuntimeError(
            "Cached runtime gap overrides were requested, but the live network "
            "does not expose reset/apply gap override helpers."
        )
    return list(net.apply_gap_group_overrides(groups) or [])


def _normalize_cell_biophys_overrides(command: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw = command.get("cell_biophys_overrides") or []
    if raw in (None, "", False):
        return []
    if not isinstance(raw, list):
        raise ValueError("cell_biophys_overrides must be a list of group override mappings.")

    allowed_group_keys = {
        "name",
        "ids",
        "neuron_ids",
        "passive_global",
        "soma_hh",
        "branch_hh",
        "active",
        "v_rest_mV",
        "v_init_mV",
        "ena_mV",
        "ek_mV",
        "el_mV",
    }
    groups: list[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"cell_biophys_overrides[{idx}] must be a mapping.")
        unexpected = sorted(set(item) - allowed_group_keys)
        if unexpected:
            raise ValueError(
                f"cell_biophys_overrides[{idx}] has unsupported keys: {unexpected}. "
                f"Allowed keys: {sorted(allowed_group_keys)}"
            )
        ids = item.get("ids")
        if ids is None:
            ids = item.get("neuron_ids")
        group: Dict[str, Any] = {
            "name": str(item.get("name") or f"cell_biophys_group_{idx:02d}"),
            "ids": [int(x) for x in list(ids or [])],
            "passive_global": dict(item.get("passive_global") or {}),
            "soma_hh": dict(item.get("soma_hh") or {}),
            "branch_hh": dict(item.get("branch_hh") or {}),
        }
        for key in ("active", "v_rest_mV", "v_init_mV", "ena_mV", "ek_mV", "el_mV"):
            if key in item and item.get(key) is not None:
                group[key] = item.get(key)
        groups.append(group)
    return groups


def _apply_runtime_cell_biophys_overrides(
    net: Any,
    cfg: Mapping[str, Any],
    command: Mapping[str, Any],
    *,
    seed_ids: Iterable[int],
) -> list[Dict[str, Any]]:
    if hasattr(net, "reset_cell_biophys_overrides"):
        net.reset_cell_biophys_overrides(seed_ids=[int(x) for x in seed_ids])
    groups = _normalize_cell_biophys_overrides(command)
    if not groups:
        return []
    if not hasattr(net, "reset_cell_biophys_overrides") or not hasattr(net, "apply_cell_biophys_overrides"):
        raise RuntimeError(
            "Cached runtime cell biophys overrides were requested, but the live network "
            "does not expose reset/apply cell biophys override helpers."
        )
    return list(net.apply_cell_biophys_overrides(groups, seed_ids=[int(x) for x in seed_ids]) or [])


def _build_timing_summary(
    *,
    cfg: Mapping[str, Any],
    phase_timings: Iterable[Mapping[str, Any]],
    sim_wall_s: float,
    total_wall_s: float,
    backend: str,
    integrator: str,
    cached_build_timing: Mapping[str, Any],
) -> Dict[str, Any]:
    phase_rows = [dict(row) for row in phase_timings]
    pre_sim_wall_s = 0.0
    for row in phase_rows:
        if str(row.get("label")) == "simulate":
            break
        try:
            pre_sim_wall_s += float(row.get("wall_s", 0.0))
        except Exception:
            pass
    save_wall_s = next(
        (float(row.get("wall_s", float("nan"))) for row in phase_rows if str(row.get("label")) == "save outputs"),
        float("nan"),
    )
    return {
        "run_id": str(cfg["run_id"]),
        "build_wall_s": 0.0,
        "cached_original_build_wall_s": float(cached_build_timing.get("build_wall_s", float("nan"))),
        "cache_session_reused": True,
        "pre_sim_wall_s": float(pre_sim_wall_s),
        "sim_wall_s": float(sim_wall_s),
        "post_sim_save_wall_s": float(save_wall_s),
        "total_wall_s": float(total_wall_s),
        "backend": str(backend),
        "integrator": str(integrator),
        "phase_rows": phase_rows,
    }


def _rebuild_iclamps_from_cfg(net: Any, cfg: Mapping[str, Any], seed_ids: Iterable[int]) -> int:
    from digifly.phase2.walking.runner import (
        _clamp_site_for_cell,
        _resolve_clamp_site,
        _resolve_neg_pulse,
        _resolve_pulse_train,
    )

    net.iclamps.clear()
    net._iclamp_meta.clear()

    clamp_site = _resolve_clamp_site(dict(cfg))
    neg_pulse = _resolve_neg_pulse(dict(cfg), clamp_site)
    pulse_train = _resolve_pulse_train(dict(cfg), clamp_site)

    created = 0
    for nid in [int(x) for x in seed_ids]:
        if bool(getattr(net, "is_distributed", False)) and not bool(net.is_local_gid(int(nid))):
            continue
        cell = net.ensure_cell(int(nid))
        site = _clamp_site_for_cell(cell, clamp_site)
        if pulse_train is None or bool(pulse_train.get("include_base_iclamp", False)):
            net.add_iclamp_site(
                int(nid),
                site,
                amp_nA=float(cfg["iclamp_amp_nA"]),
                delay_ms=float(cfg["iclamp_delay_ms"]),
                dur_ms=float(cfg["iclamp_dur_ms"]),
                kind="base",
            )
            created += 1
        if pulse_train is not None:
            tr_site = _clamp_site_for_cell(cell, str(pulse_train["site"]))
            for dly in pulse_train.get("delays", []):
                net.add_iclamp_site(
                    int(nid),
                    tr_site,
                    amp_nA=float(pulse_train["amp"]),
                    delay_ms=float(dly),
                    dur_ms=float(pulse_train["dur"]),
                    kind="pulse_train",
                )
                created += 1
        if neg_pulse is not None:
            neg_site = _clamp_site_for_cell(cell, str(neg_pulse["site"]))
            net.add_iclamp_site(
                int(nid),
                neg_site,
                amp_nA=float(neg_pulse["amp"]),
                delay_ms=float(neg_pulse["delay"]),
                dur_ms=float(neg_pulse["dur"]),
                kind="neg_pulse",
            )
            created += 1
    return int(created)


def _expected_iclamp_count(cfg: Mapping[str, Any], seed_ids: Iterable[int]) -> int:
    from digifly.phase2.walking.runner import _resolve_neg_pulse, _resolve_pulse_train

    target_count = len([int(x) for x in seed_ids])
    if target_count <= 0:
        return 0

    neg_pulse = _resolve_neg_pulse(dict(cfg), str(cfg.get("iclamp_location") or "ais"))
    pulse_train = _resolve_pulse_train(dict(cfg), str(cfg.get("iclamp_location") or "ais"))

    per_target = 0
    if pulse_train is None or bool(pulse_train.get("include_base_iclamp", False)):
        per_target += 1
    if pulse_train is not None:
        per_target += len(list(pulse_train.get("delays", []) or []))
    if neg_pulse is not None:
        per_target += 1
    return int(per_target * target_count)


def _force_neuron_runtime_for_cached_run(net: Any, cfg: Dict[str, Any]) -> bool:
    if not bool(getattr(net, "_coreneuron_on", False)):
        return False
    try:
        from digifly.phase2.neuron_build.network import _disable_coreneuron_state

        _disable_coreneuron_state()
    except Exception:
        pass
    net._coreneuron_on = False
    cfg["enable_coreneuron"] = False
    cfg["coreneuron_gpu"] = False
    return True


def _runtime_backend_label(net: Any) -> str:
    return "coreneuron" if bool(getattr(net, "_coreneuron_on", False)) else "neuron"


def _runtime_rank_label(net: Any) -> int:
    try:
        pc = getattr(net, "_pc", None)
        if pc is not None:
            return int(pc.id())
    except Exception:
        pass
    try:
        return int(getattr(net, "rank"))
    except Exception:
        return 0


def _execute_cached_run(ctx: Dict[str, Any], command: Mapping[str, Any]) -> Dict[str, Any]:
    from digifly.phase2.util.save import save_config, save_records, save_spikes
    from digifly.phase2.walking.runner import _dump_cell_biophys, _record_setup

    net = ctx["net"]
    session_paths = ctx["session_paths"]
    is_distributed = bool(getattr(net, "is_distributed", False))
    is_root_rank = (not is_distributed) or bool(getattr(net, "is_root_rank", False))
    rank_label = _runtime_rank_label(net)
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="command_execute_start",
        extra={
            "run_id": str(command.get("run_id") or ""),
            "command_id": int(command.get("command_id") or 0),
        },
    )
    cfg = _make_cached_run_cfg(ctx["base_cfg"], command)
    out_dir = (Path(cfg["runs_root"]).expanduser().resolve() / str(cfg["run_id"])).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _phase_update(phase: str, extra: Mapping[str, Any] | None = None) -> None:
        is_dist = bool(getattr(net, "is_distributed", False))
        is_root = (not is_dist) or bool(getattr(net, "is_root_rank", False))
        if not is_root:
            return
        _write_command_status(
            session_paths,
            ctx,
            command,
            command_id=int(command.get("command_id", 0)),
            action=str(command.get("action") or "run"),
            response_path=str(command.get("response_json") or ""),
            phase=str(phase),
            extra=extra,
        )

    phase_timings: list[Dict[str, Any]] = []
    total_start = time.perf_counter()
    net.cfg = copy.deepcopy(cfg)
    runtime_backend_forced = _force_neuron_runtime_for_cached_run(net, cfg)
    synapse_group_summary = _apply_runtime_synapse_group_overrides(net, command)
    gap_group_summary = _apply_runtime_gap_group_overrides(net, command)
    net.reset_run_artifacts()
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="post_reset",
        extra={"run_id": str(command.get("run_id") or "")},
    )
    if is_distributed:
        print(f"[cache][rank {rank_label}] post-reset barrier start")
        net._pc.barrier()
        print(f"[cache][rank {rank_label}] post-reset barrier done")
    runtime_seed_ids = _runtime_stim_target_ids(ctx, command, cfg)
    cfg["seeds"] = [int(x) for x in runtime_seed_ids]
    cell_biophys_group_summary = _apply_runtime_cell_biophys_overrides(
        net,
        cfg,
        command,
        seed_ids=runtime_seed_ids,
    )
    clamp_rebuilt = _rebuild_iclamps_from_cfg(net, cfg, runtime_seed_ids)
    local_clamp_count = int(clamp_rebuilt)
    expected_clamp_count = int(_expected_iclamp_count(cfg, runtime_seed_ids))
    _phase_update(
        "prepare_output",
        {
            "out_dir": str(out_dir),
            "stim_target_ids": [int(x) for x in runtime_seed_ids],
            "clamp_count_local": int(local_clamp_count),
            "clamp_count_expected": int(expected_clamp_count),
            "clamp_count": int(local_clamp_count),
            "runtime_backend": _runtime_backend_label(net),
            "tstop_ms": float(cfg["tstop_ms"]),
            "dt_ms": float(cfg["dt_ms"]),
            "synapse_override_group_count": int(len(synapse_group_summary)),
            "synapse_override_groups": [str(row.get("name") or "") for row in synapse_group_summary],
            "gap_override_group_count": int(len(gap_group_summary)),
            "gap_override_groups": [str(row.get("name") or "") for row in gap_group_summary],
            "cell_biophys_override_group_count": int(len(cell_biophys_group_summary)),
            "cell_biophys_override_groups": [str(row.get("name") or "") for row in cell_biophys_group_summary],
        },
    )
    if runtime_seed_ids and int(expected_clamp_count) <= 0:
        raise RuntimeError(
            "Runtime stimulation requested but the current config resolves to zero IClamps. "
            f"stim_target_ids={list(runtime_seed_ids)} "
            f"local_clamp_count={local_clamp_count} "
            f"expected_clamp_count={expected_clamp_count}"
        )

    t_prepare = time.perf_counter()
    buildwarm_biophys = (Path(ctx.get("build_out_dir", "")).expanduser().resolve() / "cell_biophys.csv").resolve()
    if is_root_rank:
        save_config(cfg, out_dir)
    if buildwarm_biophys.exists() and not cell_biophys_group_summary:
        if is_root_rank:
            import shutil

            shutil.copy2(buildwarm_biophys, out_dir / "cell_biophys.csv")
    else:
        _dump_cell_biophys(net, ctx["final_network_ids"], out_dir)
    if is_distributed:
        print(f"[cache][rank {rank_label}] post-prepare barrier start")
        net._pc.barrier()
        print(f"[cache][rank {rank_label}] post-prepare barrier done")
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="post_prepare",
        extra={"run_id": str(command.get("run_id") or "")},
    )
    prepare_wall_s = time.perf_counter() - t_prepare
    phase_timings.append({"label": "prepare output + save config", "wall_s": float(prepare_wall_s)})
    _phase_update(
        "record_setup",
        {
            "out_dir": str(out_dir),
            "prepare_wall_s": float(prepare_wall_s),
            "cell_biophys_override_group_count": int(len(cell_biophys_group_summary)),
        },
    )

    t_record = time.perf_counter()
    spike_map = _record_setup(net, cfg, runtime_seed_ids, ctx["final_network_ids"])
    if is_distributed:
        print(f"[cache][rank {rank_label}] post-record barrier start")
        net._pc.barrier()
        print(f"[cache][rank {rank_label}] post-record barrier done")
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="post_record_setup",
        extra={"run_id": str(command.get("run_id") or "")},
    )
    record_wall_s = time.perf_counter() - t_record
    phase_timings.append({"label": "record setup", "wall_s": float(record_wall_s)})
    _phase_update(
        "simulate",
        {
            "out_dir": str(out_dir),
            "record_setup_wall_s": float(record_wall_s),
            "stim_target_ids": [int(x) for x in runtime_seed_ids],
            "clamp_count_local": int(local_clamp_count),
            "clamp_count_expected": int(expected_clamp_count),
            "clamp_count": int(local_clamp_count),
            "runtime_backend": _runtime_backend_label(net),
            "tstop_ms": float(cfg["tstop_ms"]),
            "dt_ms": float(cfg["dt_ms"]),
            "synapse_override_group_count": int(len(synapse_group_summary)),
            "synapse_override_groups": [str(row.get("name") or "") for row in synapse_group_summary],
            "gap_override_group_count": int(len(gap_group_summary)),
            "gap_override_groups": [str(row.get("name") or "") for row in gap_group_summary],
            "cell_biophys_override_group_count": int(len(cell_biophys_group_summary)),
            "cell_biophys_override_groups": [str(row.get("name") or "") for row in cell_biophys_group_summary],
        },
    )

    print(f"[cache][rank {rank_label}] simulate start")
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="simulate_start",
        extra={"run_id": str(command.get("run_id") or "")},
    )
    sim_start = time.perf_counter()
    net.run(
        tstop_ms=float(cfg["tstop_ms"]),
        dt_ms=float(cfg["dt_ms"]),
        show_progress=bool(cfg.get("progress", True)),
    )
    sim_wall_s = time.perf_counter() - sim_start
    phase_timings.append({"label": "simulate", "wall_s": float(sim_wall_s)})

    cv_cfg = cfg.get("cvode", {}) or {}
    cv_enabled = bool(cv_cfg.get("enabled", False)) if isinstance(cv_cfg, dict) else False
    backend = "coreneuron" if bool(getattr(net, "_coreneuron_on", False)) else "neuron"
    integrator = "cvode" if cv_enabled else f"fixed-step(dt_ms={float(cfg['dt_ms'])})"
    print(
        f"[cache][rank {rank_label}] sim done wall_s={sim_wall_s:.3f} "
        f"backend={backend} integrator={integrator}"
    )
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="simulate_done",
        extra={
            "run_id": str(command.get("run_id") or ""),
            "sim_wall_s": float(sim_wall_s),
            "backend": str(backend),
        },
    )
    _phase_update(
        "save_outputs",
        {
            "out_dir": str(out_dir),
            "sim_wall_s": float(sim_wall_s),
            "backend_runtime": str(backend),
            "integrator": str(integrator),
            "cell_biophys_override_group_count": int(len(cell_biophys_group_summary)),
        },
    )

    t_save = time.perf_counter()
    if bool(getattr(net, "is_distributed", False)):
        save_records(net, out_dir)
        save_spikes(spike_map, out_dir, net=net)
        net._pc.barrier()
    else:
        save_records(net, out_dir)
        save_spikes(spike_map, out_dir)
    save_wall_s = time.perf_counter() - t_save
    phase_timings.append({"label": "save outputs", "wall_s": float(save_wall_s)})
    _phase_update(
        "finalizing_response",
        {
            "out_dir": str(out_dir),
            "sim_wall_s": float(sim_wall_s),
            "save_outputs_wall_s": float(save_wall_s),
            "backend_runtime": str(backend),
            "integrator": str(integrator),
        },
    )

    total_wall_s = time.perf_counter() - total_start
    timing_summary = _build_timing_summary(
        cfg=cfg,
        phase_timings=phase_timings,
        sim_wall_s=sim_wall_s,
        total_wall_s=total_wall_s,
        backend=backend,
        integrator=integrator,
        cached_build_timing=ctx.get("build_timing", {}),
    )
    if (not bool(getattr(net, "is_distributed", False))) or bool(getattr(net, "is_root_rank", False)):
        (out_dir / "_phase_timings.json").write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")
    _append_rank_event(
        session_paths,
        rank=rank_label,
        label="command_execute_done",
        extra={
            "run_id": str(command.get("run_id") or ""),
            "total_wall_s": float(total_wall_s),
        },
    )

    return {
        "status": "ok",
        "workflow_label": str(command.get("workflow_label") or ctx["workflow_label"]),
        "label": str(command.get("workflow_label") or ctx["workflow_label"]),
        "backend": str(command.get("backend") or ctx["backend"]),
        "nproc": int(command.get("nproc") or ctx["nproc"]),
        "returncode": 0,
        "build_cache_reused": True,
        "cached_original_build_wall_s": float(ctx.get("build_timing", {}).get("build_wall_s", float("nan"))),
        "baseline_out_dir": str(out_dir),
        "edges_path": str(ctx["base_cfg"].get("edges_path") or ""),
        "final_network_ids": [int(x) for x in ctx["final_network_ids"]],
        "seed_ids": [int(x) for x in runtime_seed_ids],
        "base_config": {
            "run_id": str(cfg["run_id"]),
            "runs_root": str(cfg["runs_root"]),
        },
        "synapse_group_summary": synapse_group_summary,
        "gap_group_summary": gap_group_summary,
        "cell_biophys_group_summary": cell_biophys_group_summary,
        "timing_summary": timing_summary,
        "out_dir": str(out_dir),
        "records_csv": str((out_dir / "records.csv").resolve()),
        "spikes_csv": str((out_dir / "spike_times.csv").resolve()),
        "phase_timings_json": str((out_dir / "_phase_timings.json").resolve()),
    }


def _wait_for_request(paths: Mapping[str, Path], *, last_command_id: int) -> tuple[int, Path, Dict[str, Any]]:
    requests_dir = paths["requests_dir"]
    while True:
        for path in sorted(requests_dir.glob("*.json")):
            try:
                payload = _load_json(path)
            except Exception:
                continue
            command_id = int(payload.get("command_id", 0))
            if command_id <= int(last_command_id):
                continue
            return command_id, path, payload
        time.sleep(1.0)


def _response_path(paths: Mapping[str, Path], command: Mapping[str, Any], command_id: int) -> Path:
    raw = command.get("response_json")
    if raw:
        return Path(raw).expanduser().resolve()
    return (paths["responses_dir"] / f"response_{int(command_id):04d}.json").resolve()


def main() -> int:
    args = parse_args()
    session_paths = _session_paths(args.session_root)
    case = _load_case(args.case_json)
    backend = _backend_from_cfg(case)

    _write_status(
        session_paths,
        {
            "state": "starting",
            "workflow_label": f"{backend}_explicit_cached",
            "backend": backend,
            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
        },
    )

    _maybe_init_neuron_mpi(backend)
    parallel_state = _compact_parallel_state()
    is_root = int(parallel_state["rank"]) == 0
    _append_rank_event(
        session_paths,
        rank=int(parallel_state["rank"]),
        label="main_start",
        extra={
            "backend": str(backend),
            "nhost": int(parallel_state["nhost"]),
        },
    )

    try:
        base_cfg = copy.deepcopy(case)
        final_network_ids = _selection_ids(base_cfg)
        seed_ids = _seed_ids(base_cfg)
        ctx: Dict[str, Any] = {
            "base_cfg": base_cfg,
            "final_network_ids": final_network_ids,
            "seed_ids": seed_ids,
            "backend": backend,
            "nproc": int(parallel_state["nhost"]) if backend == "distributed_gid" else 1,
            "workflow_label": f"{backend}_n{int(parallel_state['nhost'])}_explicit_cached"
            if backend == "distributed_gid"
            else "single_host_explicit_cached",
            "cache_fingerprint": _build_cache_fingerprint(base_cfg),
            "session_paths": session_paths,
        }

        warm_cfg = _make_build_warm_cfg(base_cfg)
        if is_root:
            _write_status(
                session_paths,
                {
                    "state": "building_cache",
                    "workflow_label": ctx["workflow_label"],
                    "backend": ctx["backend"],
                    "nproc": ctx["nproc"],
                    "cache_fingerprint": ctx["cache_fingerprint"],
                    "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                    "final_network_count": int(len(ctx["final_network_ids"])),
                    "seed_count": int(len(ctx["seed_ids"])),
                },
            )

        from digifly.phase2.walking.runner import run_walking_simulation

        _append_rank_event(
            session_paths,
            rank=int(parallel_state["rank"]),
            label="warm_build_start",
            extra={"final_network_count": int(len(final_network_ids))},
        )
        build_out_dir, net = run_walking_simulation(warm_cfg, return_net=True)
        _append_rank_event(
            session_paths,
            rank=int(parallel_state["rank"]),
            label="warm_build_done",
            extra={"build_out_dir": str(Path(build_out_dir).expanduser().resolve())},
        )
        ctx["net"] = net
        ctx["build_out_dir"] = str(Path(build_out_dir).expanduser().resolve())
        # run_walking_simulation expands user-facing configs with the full
        # Phase 2/default NEURON config before constructing Network. Cached
        # reruns must reuse that expanded config, not the app's raw JSON,
        # otherwise runtime resets can miss keys such as pre_soma_hh.
        ctx["base_cfg"] = copy.deepcopy(getattr(net, "cfg", None) or base_cfg)
        ctx["final_network_ids"] = _network_ids_from_net(net, ctx["final_network_ids"])
        if not ctx["seed_ids"]:
            ctx["seed_ids"] = [int(x) for x in ctx["final_network_ids"]]
        timing_path = Path(build_out_dir).expanduser().resolve() / "_phase_timings.json"
        if is_root:
            ctx["build_timing"] = _load_json_retry(timing_path) if timing_path.exists() else {}
        else:
            ctx["build_timing"] = {}
        net.reset_run_artifacts()
        if bool(getattr(net, "is_distributed", False)):
            net._pc.barrier()
        _append_rank_event(
            session_paths,
            rank=int(parallel_state["rank"]),
            label="post_warm_reset",
        )

        session_info = {
            "state": "ready",
            "workflow_label": ctx["workflow_label"],
            "backend": ctx["backend"],
            "nproc": ctx["nproc"],
            "cache_fingerprint": ctx["cache_fingerprint"],
            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
            "build_out_dir": ctx["build_out_dir"],
            "cached_original_build_wall_s": float(ctx.get("build_timing", {}).get("build_wall_s", float("nan"))),
            "final_network_count": int(len(ctx["final_network_ids"])),
            "seed_count": int(len(ctx["seed_ids"])),
            "session_root": str(session_paths["root"]),
        }
        if is_root:
            _write_json(session_paths["session_info_json"], session_info)
            _write_status(session_paths, session_info)
        _append_rank_event(
            session_paths,
            rank=int(parallel_state["rank"]),
            label="session_ready",
            extra={"nproc": int(ctx["nproc"])},
        )

        last_command_id = 0
        while True:
            _append_rank_event(
                session_paths,
                rank=int(parallel_state["rank"]),
                label="wait_for_request_start",
                extra={"last_command_id": int(last_command_id)},
            )
            command_id, request_path, command = _wait_for_request(session_paths, last_command_id=last_command_id)
            _append_rank_event(
                session_paths,
                rank=int(parallel_state["rank"]),
                label="wait_for_request_done",
                extra={"command_id": int(command_id)},
            )
            last_command_id = int(command_id)
            action = str(command.get("action", "")).strip().lower()
            response_path = _response_path(session_paths, command, command_id)
            command = dict(command)
            command["command_id"] = int(command_id)
            command["request_json"] = str(request_path)
            command["response_json"] = str(response_path)
            _append_rank_event(
                session_paths,
                rank=int(parallel_state["rank"]),
                label="command_received",
                extra={
                    "command_id": int(command_id),
                    "action": str(action),
                    "run_id": str(command.get("run_id") or ""),
                },
            )

            if is_root:
                _write_command_status(
                    session_paths,
                    ctx,
                    command,
                    command_id=int(command_id),
                    action=action,
                    response_path=response_path,
                    phase="dispatch",
                )

            if action == "shutdown":
                _append_rank_event(
                    session_paths,
                    rank=int(parallel_state["rank"]),
                    label="shutdown_received",
                    extra={"command_id": int(command_id)},
                )
                if is_root:
                    _write_json(
                        response_path,
                        {
                            "status": "ok",
                            "action": "shutdown",
                            "command_id": int(command_id),
                            "workflow_label": ctx["workflow_label"],
                            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                        },
                    )
                    _write_status(
                        session_paths,
                        {
                            "state": "stopped",
                            "workflow_label": ctx["workflow_label"],
                            "backend": ctx["backend"],
                            "nproc": ctx["nproc"],
                            "cache_fingerprint": ctx["cache_fingerprint"],
                            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                        },
                    )
                break

            if action != "run":
                _append_rank_event(
                    session_paths,
                    rank=int(parallel_state["rank"]),
                    label="unsupported_action",
                    extra={"command_id": int(command_id), "action": str(action)},
                )
                if is_root:
                    _write_json(
                        response_path,
                        {
                            "status": "error",
                            "command_id": int(command_id),
                            "error_type": "ValueError",
                            "error": f"Unsupported action: {action!r}",
                            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                        },
                    )
                    _write_status(
                        session_paths,
                        {
                            "state": "ready",
                            "workflow_label": ctx["workflow_label"],
                            "backend": ctx["backend"],
                            "nproc": ctx["nproc"],
                            "cache_fingerprint": ctx["cache_fingerprint"],
                            "last_error": f"Unsupported action: {action!r}",
                            "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                        },
                    )
                continue

            run_payload = _execute_cached_run(ctx, command)
            _append_rank_event(
                session_paths,
                rank=int(parallel_state["rank"]),
                label="command_returned",
                extra={
                    "command_id": int(command_id),
                    "run_id": str(command.get("run_id") or ""),
                },
            )
            if is_root:
                run_payload["command_id"] = int(command_id)
                run_payload["response_json"] = str(response_path)
                run_payload["cache_protocol_version"] = int(CACHE_PROTOCOL_VERSION)
                _write_json(response_path, run_payload)
                _write_status(
                    session_paths,
                    {
                        "state": "ready",
                        "workflow_label": ctx["workflow_label"],
                        "backend": ctx["backend"],
                        "nproc": ctx["nproc"],
                        "cache_fingerprint": ctx["cache_fingerprint"],
                        "last_run_id": str(run_payload["base_config"]["run_id"]),
                        "last_run_out_dir": str(run_payload["baseline_out_dir"]),
                        "cached_original_build_wall_s": float(run_payload["cached_original_build_wall_s"]),
                        "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                    },
                )
    except SystemExit:
        _append_rank_event(
            session_paths,
            rank=int(parallel_state.get("rank", 0)),
            label="system_exit",
        )
        raise
    except Exception as exc:
        _append_rank_event(
            session_paths,
            rank=int(parallel_state.get("rank", 0)),
            label="main_exception",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        if is_root:
            _write_status(
                session_paths,
                {
                    "state": "error",
                    "workflow_label": f"{backend}_explicit_cached",
                    "backend": backend,
                    "nproc": int(parallel_state["nhost"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "cache_protocol_version": int(CACHE_PROTOCOL_VERSION),
                },
            )
        raise
    finally:
        _append_rank_event(
            session_paths,
            rank=int(parallel_state.get("rank", 0)),
            label="finally_enter",
        )
        try:
            net = locals().get("ctx", {}).get("net")
            if net is not None:
                net.close(reset_parallel=False)
                _append_rank_event(
                    session_paths,
                    rank=int(parallel_state.get("rank", 0)),
                    label="net_close_done",
                )
        except Exception:
            pass
        try:
            _append_rank_event(
                session_paths,
                rank=int(parallel_state.get("rank", 0)),
                label="before_mpi_quit",
                extra={"backend": str(backend)},
            )
            _maybe_quit_neuron_mpi(backend)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
