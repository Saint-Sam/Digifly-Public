"""Path resolution helpers for the VIP glia mutation launch notebooks."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


NOTEBOOK_DIR = Path(__file__).resolve().parent
PUBLIC_MANC_LABEL = "manc_v1.2.1"


def _load_local_config() -> dict[str, Any]:
    """Load ignored local launcher settings when notebooks run on one machine."""
    cfg_path = NOTEBOOK_DIR / "local_config.py"
    if not cfg_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("_digifly_vip_glia_local_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local config: {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: getattr(module, name) for name in dir(module) if name.isupper()}


LOCAL_CONFIG = _load_local_config()


def _config_value(*names: str) -> str | None:
    for name in names:
        env_value = os.environ.get(name, "").strip()
        if env_value:
            return env_value
        cfg_value = LOCAL_CONFIG.get(name)
        if cfg_value is not None and str(cfg_value).strip():
            return str(cfg_value)
    return None


def _config_paths(*names: str) -> list[Path]:
    """Return path-list settings from env/local config, split with os.pathsep."""
    out: list[Path] = []
    for name in names:
        env_value = os.environ.get(name, "").strip()
        if env_value:
            out.extend(Path(part) for part in env_value.split(os.pathsep) if part.strip())
        cfg_value = LOCAL_CONFIG.get(name)
        if cfg_value is None:
            continue
        if isinstance(cfg_value, (list, tuple, set)):
            out.extend(Path(str(part)) for part in cfg_value if str(part).strip())
        elif str(cfg_value).strip():
            out.extend(Path(part) for part in str(cfg_value).split(os.pathsep) if part.strip())
    return _dedupe_paths(out)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def find_app_root() -> Path:
    """Resolve the copied VIP_Glia_Sim app root."""
    candidates: list[Path] = []
    env_root = _config_value("DIGIFLY_VIP_GLIA_ROOT", "APP_ROOT", "WORK_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    cwd = Path.cwd()
    candidates.extend([NOTEBOOK_DIR.parent, cwd])
    for base in [cwd, *cwd.parents, NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:
        candidates.append(base / "Phase 2" / "apps" / "VIP_Glia_Sim")
        candidates.append(base / "apps" / "VIP_Glia_Sim")

    for candidate in _dedupe_paths(candidates):
        for root in [candidate, *candidate.parents]:
            if (root / "tools" / "morphology_mutation_app.py").exists():
                return root

    raise RuntimeError(
        "Could not locate VIP_Glia_Sim app root. Set DIGIFLY_VIP_GLIA_ROOT "
        "or APP_ROOT in notebooks/local_config.py."
    )


def resolve_phase2_root(app_root: Path) -> Path:
    """Resolve Phase 2 root from env/local config, otherwise the app location."""
    configured = _config_value("DIGIFLY_PHASE2_ROOT", "PHASE2_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(app_root).expanduser().resolve().parents[1]


def resolve_python_bin() -> Path:
    """Prefer the Python that has desktop VTK/PyVista dependencies installed."""
    candidates = [
        _config_value("VIP_PYTHON_BIN", "PYTHON_BIN"),
        "/opt/anaconda3/bin/python3.12",
        "/opt/anaconda3/bin/python3",
        "/opt/anaconda3/bin/python",
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def public_manc_swc_dir(phase2_root: Path) -> Path:
    """Return the local public-cache SWC root for MANC v1.2.1."""
    repo_root = Path(phase2_root).expanduser().resolve().parent
    return repo_root / "Phase 1" / PUBLIC_MANC_LABEL / "export_swc"


def public_manc_hemi_runs_dir(phase2_root: Path) -> Path:
    """Return the Docker-mounted public run-output root used by Phase 2."""
    return public_manc_swc_dir(phase2_root) / "hemi_runs"


_SWC_SUFFIXES = (
    "_healed_final.swc",
    "_healed.swc",
    "_axodendro_with_synapses.swc",
)


def _neuron_id_from_swc_name(filename: str) -> int | None:
    for suffix in _SWC_SUFFIXES:
        if filename.endswith(suffix):
            prefix = filename[: -len(suffix)]
            if prefix.isdigit():
                return int(prefix)
    return None


@lru_cache(maxsize=32)
def _available_swc_ids(root: str, wanted: tuple[int, ...]) -> frozenset[int]:
    found: set[int] = set()
    wanted_set = set(wanted)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for filename in filenames:
            nid = _neuron_id_from_swc_name(filename)
            if nid is None:
                continue
            if wanted_set and nid not in wanted_set:
                continue
            found.add(nid)
            if wanted_set and wanted_set.issubset(found):
                return frozenset(found)
    return frozenset(found)


def missing_neuron_ids(swc_dir: Path, neuron_ids: Sequence[int]) -> list[int]:
    """Return IDs without an exact SWC match under swc_dir."""
    root = Path(swc_dir).expanduser().resolve()
    ids = tuple(int(x) for x in neuron_ids)
    if not root.exists():
        return list(ids)
    available = _available_swc_ids(str(root), ids)
    return [nid for nid in ids if nid not in available]


def _public_swc_dirs(phase2_root: Path, app_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    configured = _config_value("DIGIFLY_SWC_DIR", "SWC_DIR")
    if configured:
        candidates.append(Path(configured))

    phase2_root = Path(phase2_root).expanduser().resolve()
    candidates.append(public_manc_swc_dir(phase2_root))
    candidates.append(phase2_root / "data" / "export_swc")

    roots = [phase2_root.parent]
    if app_root is not None:
        roots.append(Path(app_root).expanduser().resolve().parents[2])
    for repo_root in roots:
        phase1_root = repo_root / "Phase 1"
        candidates.append(phase1_root / "export_swc")
        if phase1_root.exists():
            candidates.extend(sorted(phase1_root.glob("*/export_swc")))

    return _dedupe_paths(candidates)


def _source_swc_dirs() -> list[Path]:
    """Return configured source exports used to seed the public MANC cache."""
    candidates: list[Path] = []
    configured = _config_value(
        "DIGIFLY_SOURCE_SWC_DIR",
        "DIGIFLY_FALLBACK_SWC_DIR",
        "SOURCE_SWC_DIR",
        "FALLBACK_SWC_DIR",
    )
    if configured:
        candidates.append(Path(configured))
    return _dedupe_paths(candidates)


def _score_source_neuron_dir(source_root: Path, neuron_dir: Path) -> tuple[int, int, str]:
    rel = neuron_dir.relative_to(source_root)
    first = rel.parts[0].lower() if rel.parts else ""
    if first == "by_id":
        top_level_penalty = 3
    elif first in {"unknown", "swc_circuits"}:
        top_level_penalty = 2
    else:
        top_level_penalty = 0
    return (top_level_penalty, len(rel.parts), str(rel))


def _find_source_neuron_dir(source_root: Path, neuron_id: int) -> Path | None:
    """Find the canonical source folder for one neuron ID."""
    nid = int(neuron_id)
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        path = Path(dirpath)
        if path.name != str(nid):
            continue
        has_swc = any(_neuron_id_from_swc_name(filename) == nid for filename in filenames)
        if has_swc:
            matches.append(path)
    if not matches:
        return None
    return min(matches, key=lambda p: _score_source_neuron_dir(source_root, p))


def _copy_file_if_needed(src: Path, dst: Path) -> bool:
    if dst.exists():
        try:
            src_stat = src.stat()
            dst_stat = dst.stat()
        except OSError:
            pass
        else:
            if src_stat.st_size == dst_stat.st_size and dst_stat.st_mtime >= src_stat.st_mtime:
                return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_neuron_dir(source_root: Path, target_root: Path, neuron_dir: Path) -> tuple[int, Path]:
    rel_dir = neuron_dir.relative_to(source_root)
    copied = 0
    for src in neuron_dir.rglob("*"):
        if not src.is_file():
            continue
        dst = target_root / rel_dir / src.relative_to(neuron_dir)
        if _copy_file_if_needed(src, dst):
            copied += 1
    return copied, target_root / rel_dir


def seed_public_manc_swc_cache(
    phase2_root: Path,
    neuron_ids: Sequence[int],
    *,
    target_root: Path | None = None,
) -> Path:
    """Copy missing MANC v1.2.1 neuron folders into Digifly Public."""
    ids = [int(x) for x in neuron_ids]
    target_root = public_manc_swc_dir(phase2_root) if target_root is None else Path(target_root)
    target_root = target_root.expanduser().resolve()
    missing = missing_neuron_ids(target_root, ids)
    if not missing:
        return target_root

    source_roots = [p for p in _source_swc_dirs() if p.exists()]
    if not source_roots:
        return target_root

    copied_by_id: dict[int, Path] = {}
    total_files = 0
    for nid in missing:
        for source_root in source_roots:
            neuron_dir = _find_source_neuron_dir(source_root, nid)
            if neuron_dir is None:
                continue
            copied, copied_dir = _copy_neuron_dir(source_root, target_root, neuron_dir)
            copied_by_id[nid] = copied_dir
            total_files += copied
            break

    if copied_by_id:
        _available_swc_ids.cache_clear()
        print(
            f"[launcher_paths] seeded {len(copied_by_id)} neuron folder(s) "
            f"({total_files} file(s) copied/updated) into {target_root}"
        )
        for nid, copied_dir in copied_by_id.items():
            print(f"[launcher_paths]   {nid}: {copied_dir.relative_to(target_root)}")
    return target_root


def _all_candidate_swc_dirs(phase2_root: Path, app_root: Path | None = None) -> list[Path]:
    return _dedupe_paths([*_public_swc_dirs(phase2_root, app_root=app_root), *_source_swc_dirs()])


def resolve_swc_dir(
    phase2_root: Path,
    neuron_ids: Sequence[int] | None = None,
    *,
    app_root: Path | None = None,
) -> Path:
    """Resolve an SWC root, requiring requested neuron IDs when provided."""
    candidates = _public_swc_dirs(phase2_root, app_root=app_root)
    ids = [int(x) for x in (neuron_ids or [])]

    for candidate in candidates:
        if not candidate.exists():
            continue
        if ids and missing_neuron_ids(candidate, ids):
            continue
        return candidate.resolve()

    if ids:
        seeded_dir = seed_public_manc_swc_cache(phase2_root, ids)
        if seeded_dir.exists() and not missing_neuron_ids(seeded_dir, ids):
            return seeded_dir

    candidates = _all_candidate_swc_dirs(phase2_root, app_root=app_root)
    default_dir = Path(phase2_root).expanduser().resolve() / "data" / "export_swc"
    checked = "\n".join(f"  - {p}" for p in candidates)
    if ids:
        raise FileNotFoundError(
            "Could not find an SWC export folder containing all requested neuron IDs "
            f"{ids}.\nChecked:\n{checked}\n\n"
            "Set DIGIFLY_SWC_DIR or create notebooks/local_config.py from "
            "local_config.example.py."
        )
    if not default_dir.exists():
        raise FileNotFoundError(
            f"Default SWC_DIR does not exist: {default_dir}\nChecked:\n{checked}\n\n"
            "Set DIGIFLY_SWC_DIR or create notebooks/local_config.py from "
            "local_config.example.py."
        )
    return default_dir


def candidate_flow_run_roots(
    phase2_root: Path,
    swc_dir: Path | None = None,
    *,
    app_root: Path | None = None,
) -> list[Path]:
    """Return run roots searched by the mutation app flow visualizer."""
    phase2_root = Path(phase2_root).expanduser().resolve()
    candidates: list[Path] = []
    candidates.extend(_config_paths("DIGIFLY_FLOW_RUNS_ROOTS", "FLOW_RUNS_ROOTS"))

    configured_root = _config_value("DIGIFLY_FLOW_RUNS_ROOT", "FLOW_RUNS_ROOT")
    if configured_root:
        candidates.append(Path(configured_root))

    if swc_dir is not None:
        candidates.append(Path(swc_dir).expanduser().resolve() / "hemi_runs")
    candidates.append(public_manc_hemi_runs_dir(phase2_root))
    candidates.append(phase2_root / "data" / "export_swc" / "hemi_runs")

    if app_root is not None:
        candidates.append(Path(app_root).expanduser().resolve() / "notebooks" / "debug" / "runs")
    candidates.append(phase2_root / "apps" / "VIP_Glia_Sim" / "notebooks" / "debug" / "runs")
    return _dedupe_paths(candidates)


def _records_neuron_ids(run_dir: Path) -> set[int]:
    rec_path = Path(run_dir) / "records.csv"
    if not rec_path.exists():
        return set()
    header = rec_path.open("r", encoding="utf-8").readline().strip().split(",")
    out: set[int] = set()
    for col in header:
        col = str(col).strip()
        if col.endswith("_soma_v"):
            nid = col[:-7]
            if nid.isdigit():
                out.add(int(nid))
    return out


def auto_pick_flow_run_dir(run_roots: Iterable[Path], neuron_ids: Sequence[int]) -> Path | None:
    """Pick the newest compatible simulation run for the requested neuron IDs."""
    wanted = {int(x) for x in neuron_ids}
    candidates: list[tuple[int, int, int, float, Path]] = []
    for root in _dedupe_paths(run_roots):
        if not root.exists():
            continue
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            cfg_path = run_dir / "config.json"
            rec_path = run_dir / "records.csv"
            if not cfg_path.exists() or not rec_path.exists():
                continue
            try:
                rec_ids = _records_neuron_ids(run_dir)
            except Exception:
                continue
            overlap = len(wanted & rec_ids)
            if wanted and overlap == 0:
                continue
            full_match = wanted.issubset(rec_ids) if wanted else True
            name = run_dir.name.lower()
            baseline_pref = int(("baseline" in name) and ("reduced" not in name) and ("coalesced" not in name))
            candidates.append((int(full_match), baseline_pref, overlap, run_dir.stat().st_mtime, run_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1].resolve()


def resolve_flow_run_dir(
    phase2_root: Path,
    swc_dir: Path,
    neuron_ids: Sequence[int],
    *,
    app_root: Path | None = None,
    explicit: str | Path | None = None,
) -> tuple[Path | None, list[Path]]:
    """Resolve an explicit or auto-picked flow run directory plus searched roots."""
    run_roots = candidate_flow_run_roots(phase2_root, swc_dir, app_root=app_root)
    configured = explicit or _config_value("DIGIFLY_FLOW_RUN_DIR", "FLOW_RUN_DIR")
    if configured:
        return Path(configured).expanduser().resolve(), run_roots
    return auto_pick_flow_run_dir(run_roots, neuron_ids), run_roots


def validate_swc_dir(swc_dir: Path, neuron_ids: Sequence[int]) -> None:
    """Fail before launching if the app would immediately miss required SWCs."""
    missing = missing_neuron_ids(swc_dir, neuron_ids)
    if missing:
        raise FileNotFoundError(
            f"SWC_DIR is missing SWCs for neuron IDs {missing}: {Path(swc_dir).resolve()}\n"
            "Set DIGIFLY_SWC_DIR/local_config.py or run Phase 1 to export these IDs."
        )
