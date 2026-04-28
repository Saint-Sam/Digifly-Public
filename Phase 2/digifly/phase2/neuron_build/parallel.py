from __future__ import annotations

from typing import Any, Dict, Iterable, List

from neuron import h

_PC = None
_THREAD_SIGNATURE: tuple[int, bool] | None = None
_THREAD_LOG_SIGNATURE: tuple[int, int, int] | None = None
_PARTITION_SIGNATURE: tuple[int, tuple[tuple[int, tuple[str, ...]], ...]] | None = None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def get_parallel_context():
    global _PC
    if _PC is None:
        _PC = h.ParallelContext()
    return _PC


def _set_nthread(pc, target_threads: int, *, sequential_threads: bool | None = None) -> None:
    if not sequential_threads:
        pc.nthread(int(target_threads))
        return
    try:
        # Documented NEURON variant: second arg 0 forces sequential thread execution,
        # which is useful only for race-debugging.
        pc.nthread(int(target_threads), 0)
    except TypeError:
        pc.nthread(int(target_threads))


def requested_threads(cfg: Dict[str, Any]) -> int | None:
    raw = cfg.get("threads", None)
    if raw in (None, "", False):
        parallel_cfg = cfg.get("parallel") or {}
        if isinstance(parallel_cfg, dict):
            raw = parallel_cfg.get("threads", None)
    if raw in (None, "", False):
        return None


def requested_build_backend(cfg: Dict[str, Any]) -> str:
    parallel_cfg = cfg.get("parallel") or {}
    if not isinstance(parallel_cfg, dict):
        parallel_cfg = {}
    backend = str(parallel_cfg.get("build_backend", "single_host") or "single_host").strip().lower()
    if backend in {"distributed", "pc_distributed", "distributed_gid"}:
        return "distributed_gid"
    return "single_host"


def distributed_gid_enabled(cfg: Dict[str, Any], *, state: Dict[str, Any] | None = None) -> bool:
    state_use = dict(state or get_parallel_state())
    return requested_build_backend(cfg) == "distributed_gid" and int(state_use.get("nhost", 1)) > 1
    try:
        return max(1, int(raw))
    except Exception:
        return None


def configure_parallel_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _THREAD_SIGNATURE, _THREAD_LOG_SIGNATURE

    pc = get_parallel_context()
    want_threads = requested_threads(cfg)
    sequential_threads = False
    parallel_cfg = cfg.get("parallel") or {}
    if isinstance(parallel_cfg, dict):
        if "sequential_threads" in parallel_cfg:
            sequential_threads = bool(parallel_cfg.get("sequential_threads"))
        elif "thread_workers" in parallel_cfg:
            # Backward compatibility: the old name implied "parallel workers",
            # but this NEURON API only controls whether thread execution is
            # parallel or sequential for debugging.
            sequential_threads = not bool(parallel_cfg.get("thread_workers"))

    target_threads = int(want_threads) if want_threads is not None else 1
    signature = (int(target_threads), bool(sequential_threads))
    if signature != _THREAD_SIGNATURE:
        _set_nthread(pc, int(target_threads), sequential_threads=sequential_threads)
        _THREAD_SIGNATURE = signature

    state = get_parallel_state()
    log_signature = (state["nhost"], state["id"], state["nthread"])
    if log_signature != _THREAD_LOG_SIGNATURE:
        msg = (
            "[parallel] ParallelContext active "
            f"host={state['id']}/{state['nhost']} threads={state['nthread']}"
        )
        if want_threads and state["nthread"] > 1:
            if sequential_threads:
                msg += " (sequential thread mode for debugging; this is not a performance mode)"
            else:
                msg += " (threaded simulation configured; single-host cell construction still runs in Python serial order)"
        print(msg)
        if want_threads and state["nthread"] > 1:
            gap_cfg = cfg.get("gap") or {}
            if isinstance(gap_cfg, dict) and bool(gap_cfg.get("enabled", False)):
                print("[parallel] warning: verify any custom MOD mechanisms are THREADSAFE before using NEURON threads.")
        if state["nhost"] > 1 and requested_build_backend(cfg) != "distributed_gid":
            print("[parallel] warning: MPI hosts are visible, but this builder still replicates cells per host; true build scaling needs GID ownership + gid_connect wiring.")
        elif distributed_gid_enabled(cfg, state=state):
            print("[parallel] distributed_gid backend enabled; cell ownership will be partitioned across MPI hosts.")
        _THREAD_LOG_SIGNATURE = log_signature
    return state


def reset_parallel_context(
    target_threads: int = 1,
    *,
    thread_workers: bool | None = False,
    clear_partition: bool = True,
) -> Dict[str, Any]:
    global _THREAD_SIGNATURE, _THREAD_LOG_SIGNATURE, _PARTITION_SIGNATURE

    pc = get_parallel_context()
    if clear_partition:
        try:
            pc.partition()
        except Exception:
            pass
    try:
        sequential_threads = None if thread_workers is None else (not bool(thread_workers))
        _set_nthread(pc, max(1, int(target_threads)), sequential_threads=sequential_threads)
    except Exception:
        pass

    _THREAD_SIGNATURE = None
    _THREAD_LOG_SIGNATURE = None
    _PARTITION_SIGNATURE = None
    return get_parallel_state()


def get_parallel_state() -> Dict[str, Any]:
    pc = get_parallel_context()
    try:
        nhost = _safe_int(pc.nhost(), 1)
    except Exception:
        nhost = 1
    try:
        host_id = _safe_int(pc.id(), 0)
    except Exception:
        host_id = 0
    try:
        nthread = _safe_int(pc.nthread(), 1)
    except Exception:
        nthread = 1
    return {
        "nhost": nhost,
        "id": host_id,
        "nthread": nthread,
    }


def _cell_root_sections(cell) -> List[Any]:
    roots: List[Any] = []
    for sec in list(getattr(cell, "_secs", []) or []):
        if sec is None:
            continue
        try:
            sref = h.SectionRef(sec=sec)
            if not bool(sref.has_parent()):
                roots.append(sec)
        except Exception:
            continue
    if roots:
        return roots
    soma_sec = getattr(cell, "soma_sec", None)
    return [soma_sec] if soma_sec is not None else []


def apply_thread_partitions(cells: Iterable[Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _PARTITION_SIGNATURE

    state = configure_parallel_context(cfg)
    nthread = int(state["nthread"])
    if nthread <= 1:
        try:
            get_parallel_context().partition()
        except Exception:
            pass
        _PARTITION_SIGNATURE = None
        return state

    cell_list = list(cells)
    if not cell_list:
        return state

    signature_rows: List[tuple[int, tuple[str, ...]]] = []
    thread_roots: List[List[Any]] = [[] for _ in range(nthread)]
    for idx, cell in enumerate(sorted(cell_list, key=lambda c: int(getattr(c, "gid", 0)))):
        roots = _cell_root_sections(cell)
        root_names = tuple(sec.name() for sec in roots if sec is not None)
        gid = int(getattr(cell, "gid", idx))
        signature_rows.append((gid, root_names))
        thread_roots[idx % nthread].extend(roots)

    signature = (nthread, tuple(signature_rows))
    if signature == _PARTITION_SIGNATURE:
        return state

    pc = get_parallel_context()
    pc.partition()
    for thread_id, roots in enumerate(thread_roots):
        sl = h.SectionList()
        for sec in roots:
            if sec is not None:
                sl.append(sec=sec)
        pc.partition(thread_id, sl)

    counts = ", ".join(f"t{idx}={len(roots)}" for idx, roots in enumerate(thread_roots))
    print(f"[parallel] partitioned cell roots across {nthread} threads ({counts})")
    _PARTITION_SIGNATURE = signature
    return state
