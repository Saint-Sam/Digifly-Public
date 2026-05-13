from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from ctypes.util import find_library
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PYTHON_PACKAGE_IMPORTS: dict[str, str] = {
    "setuptools<81": "setuptools",
    "neuron>=8.2.6,<9": "neuron",
    "neuprint-python": "neuprint",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "openpyxl": "openpyxl",
    "plotly": "plotly",
    "ipywidgets": "ipywidgets",
    "jupyterlab": "jupyterlab",
    "notebook": "notebook",
    "tqdm": "tqdm",
    "pyarrow": "pyarrow",
    "PyYAML": "yaml",
    "pyvista": "pyvista",
    "pillow": "PIL",
}

PYTHON_PACKAGE_PROFILES: dict[str, tuple[str, ...]] = {
    "core": (
        "setuptools<81",
        "neuron>=8.2.6,<9",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "PyYAML",
    ),
    "notebook": (
        "matplotlib",
        "networkx",
        "openpyxl",
        "plotly",
        "ipywidgets",
        "jupyterlab",
        "notebook",
        "scikit-learn",
        "pyarrow",
    ),
    "neuprint": ("neuprint-python",),
    "mutation": ("pyvista", "pillow"),
    "dev": ("pytest",),
}


def phase2_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return phase2_root().parent


def is_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False
    for path in (Path("/proc/version"), Path("/proc/sys/kernel/osrelease")):
        try:
            if "microsoft" in path.read_text(encoding="utf-8", errors="ignore").lower():
                return True
        except Exception:
            pass
    return False


def is_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore").lower()
        return any(token in text for token in ("docker", "containerd", "kubepods"))
    except Exception:
        return False


def runtime_context() -> dict[str, Any]:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "is_wsl": is_wsl(),
        "is_docker": is_docker(),
        "phase2_root": str(phase2_root()),
        "workspace_root": str(workspace_root()),
    }


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _path_without_entries(raw: str, blocked: Iterable[str]) -> str:
    blocked_set = {str(item) for item in blocked if str(item or "").strip()}
    if not raw:
        return ""
    kept = [entry for entry in raw.split(os.pathsep) if entry and entry not in blocked_set]
    return os.pathsep.join(_dedupe(kept))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def python_packages_for_profiles(
    profiles: str | Sequence[str] = ("core", "notebook"),
    *,
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    if isinstance(profiles, str):
        requested = [profiles]
    else:
        requested = [str(profile) for profile in profiles]
    packages: list[str] = []
    for profile in requested:
        if profile == "all":
            for values in PYTHON_PACKAGE_PROFILES.values():
                packages.extend(values)
            continue
        if profile not in PYTHON_PACKAGE_PROFILES:
            raise ValueError(
                f"Unknown Phase 2 dependency profile {profile!r}. "
                f"Known profiles: {sorted(PYTHON_PACKAGE_PROFILES)}"
            )
        packages.extend(PYTHON_PACKAGE_PROFILES[profile])
    packages.extend(str(item) for item in extra)
    return tuple(_dedupe(packages))


def missing_python_packages(packages: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for package in packages:
        import_name = PYTHON_PACKAGE_IMPORTS.get(str(package), str(package).split("==")[0].split(">=")[0].split("<")[0])
        if importlib.util.find_spec(import_name) is None:
            missing.append(str(package))
    return missing


def install_python_packages(packages: Sequence[str], *, quiet: bool = False) -> None:
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", *[str(package) for package in packages]]
    if not quiet:
        print("[digifly-env] Installing missing Python packages:")
        print("[digifly-env] " + " ".join(cmd))
    subprocess.check_call(cmd)


def candidate_neuron_bin_dirs() -> list[Path]:
    candidates: list[Path] = []
    for raw in (
        os.environ.get("NEURON_BIN_DIR"),
        os.environ.get("NRNHOME") and str(Path(os.environ["NRNHOME"]) / "bin"),
        str(Path(sys.executable).resolve().parent),
        "/Applications/NEURON/bin",
        "/usr/local/bin",
        "/usr/bin",
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    return _dedupe_paths(candidates)


def _active_neuron_nrnivmodl_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
        import neuron  # type: ignore

        neuron_file = Path(neuron.__file__).expanduser().resolve()
        candidates.append(neuron_file.parent / ".data" / "bin" / "nrnivmodl")
        for parent in neuron_file.parents:
            candidates.append(parent / "bin" / "nrnivmodl")
    except Exception:
        pass
    return _dedupe_paths(candidates)


def resolve_executable(name: str, *, extra_dirs: Sequence[str | Path] = ()) -> str:
    candidates = [Path(p).expanduser() / name for p in extra_dirs]
    candidates.extend(path / name for path in candidate_neuron_bin_dirs())
    for candidate in _dedupe_paths(candidates):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(name)
    return str(found or "")


def resolve_nrnivmodl() -> str:
    env = os.environ.get("NRNIVMODL", "").strip()
    if env and Path(env).expanduser().exists():
        return str(Path(env).expanduser().resolve())
    for candidate in _active_neuron_nrnivmodl_candidates():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return resolve_executable("nrnivmodl")


def resolve_mpiexec() -> str:
    for raw in (
        os.environ.get("MPIEXEC"),
        "/opt/homebrew/bin/mpiexec",
        "/usr/local/bin/mpiexec",
        "/usr/bin/mpiexec",
    ):
        if raw and Path(raw).expanduser().exists():
            return str(Path(raw).expanduser().resolve())
    return resolve_executable("mpiexec")


def resolve_mpi_library() -> str:
    env = os.environ.get("MPI_LIB_NRN_PATH", "").strip()
    if env and Path(env).expanduser().exists():
        return str(Path(env).expanduser().resolve())

    candidates: list[Path] = [
        Path("/opt/homebrew/lib/libmpi.dylib"),
        Path("/usr/local/lib/libmpi.dylib"),
        Path("/usr/lib/x86_64-linux-gnu/libmpi.so"),
        Path("/usr/lib/aarch64-linux-gnu/libmpi.so"),
        Path("/usr/lib64/libmpi.so"),
    ]
    for pattern in (
        "/opt/homebrew/Cellar/open-mpi/*/lib/libmpi*.dylib",
        "/usr/lib/*/openmpi/lib/libmpi*.so*",
        "/usr/lib/*/libmpi*.so*",
        "/usr/local/lib/libmpi*.so*",
    ):
        candidates.extend(Path(path) for path in sorted(Path("/").glob(pattern.lstrip("/"))))

    for candidate in _dedupe_paths(candidates):
        if candidate.exists():
            return str(candidate)

    found = find_library("mpi")
    return str(found or "")


def build_phase2_launch_env(
    base_env: Mapping[str, str] | None = None,
    *,
    phase2_dir: str | Path | None = None,
    gap_mechanisms_dir: str | Path | None = None,
    include_mpi_library: bool = False,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.pop("PYTHONHOME", None)
    env["NEURON_MODULE_OPTIONS"] = str(env.get("NEURON_MODULE_OPTIONS") or "-nogui")

    p2 = Path(phase2_dir).expanduser().resolve() if phase2_dir else phase2_root()
    env.setdefault("DIGIFLY_PHASE2_ROOT", str(p2))
    env.setdefault("DIGIFLY_WORKSPACE", str(p2.parent))
    env.setdefault("DIGIFLY_GAP_MECH_DIR", str(Path(gap_mechanisms_dir).expanduser().resolve() if gap_mechanisms_dir else p2 / "data"))

    nrnivmodl = resolve_nrnivmodl()
    if nrnivmodl:
        env.setdefault("NRNIVMODL", nrnivmodl)

    mpi_lib = resolve_mpi_library()
    mpi_parent = ""
    if mpi_lib and Path(mpi_lib).expanduser().exists():
        mpi_parent = str(Path(mpi_lib).expanduser().resolve().parent)
    if include_mpi_library and mpi_lib and (Path(mpi_lib).expanduser().exists() or Path(mpi_lib).name.startswith("libmpi")):
        env.setdefault("MPI_LIB_NRN_PATH", mpi_lib)
    elif mpi_parent:
        # A global MPI DYLD/LD path can break Anaconda NumPy on macOS. Keep MPI
        # library search paths out of ordinary single-process notebook sessions.
        env["DYLD_LIBRARY_PATH"] = _path_without_entries(env.get("DYLD_LIBRARY_PATH", ""), [mpi_parent])
        env["LD_LIBRARY_PATH"] = _path_without_entries(env.get("LD_LIBRARY_PATH", ""), [mpi_parent])
        env["PATH"] = _path_without_entries(env.get("PATH", ""), [mpi_parent])

    path_entries: list[str] = []
    for path in candidate_neuron_bin_dirs():
        if path.exists():
            path_entries.append(str(path))
    mpiexec = resolve_mpiexec()
    if mpiexec:
        path_entries.append(str(Path(mpiexec).expanduser().resolve().parent))
    if include_mpi_library and mpi_parent:
        path_entries.append(mpi_parent)
    path_entries.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(_dedupe(path_entries))

    if include_mpi_library and mpi_parent:
        dyld_entries = [mpi_parent, env.get("DYLD_LIBRARY_PATH", "")]
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(dyld_entries))
        ld_entries = [mpi_parent, env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe(ld_entries))
    return env


def configure_phase2_environment(
    *,
    phase2_dir: str | Path | None = None,
    gap_mechanisms_dir: str | Path | None = None,
    include_mpi_library: bool = False,
) -> dict[str, str]:
    env = build_phase2_launch_env(
        os.environ,
        phase2_dir=phase2_dir,
        gap_mechanisms_dir=gap_mechanisms_dir,
        include_mpi_library=include_mpi_library,
    )
    os.environ.update(env)
    return env


def diagnose_phase2_environment(
    *,
    profiles: str | Sequence[str] = ("core", "notebook"),
    extra_packages: Sequence[str] = (),
    check_gap_mechanisms: bool = False,
) -> dict[str, Any]:
    packages = python_packages_for_profiles(profiles, extra=extra_packages)
    missing = missing_python_packages(packages)
    report: dict[str, Any] = {
        **runtime_context(),
        "profiles": list([profiles] if isinstance(profiles, str) else profiles),
        "python_packages": list(packages),
        "missing_python_packages": missing,
        "nrnivmodl": resolve_nrnivmodl(),
        "mpiexec": resolve_mpiexec(),
        "mpi_library": resolve_mpi_library(),
        "gap_mechanisms_dir": os.environ.get("DIGIFLY_GAP_MECH_DIR", str(phase2_root() / "data")),
        "warnings": [],
        "errors": [],
    }

    if not report["nrnivmodl"]:
        report["warnings"].append("nrnivmodl was not found; gap mechanism compilation may fail.")
    if not report["mpiexec"]:
        report["warnings"].append("mpiexec was not found; MPI cached sessions need OpenMPI or an equivalent MPI runtime.")
    if not report["mpi_library"]:
        report["warnings"].append("MPI library was not found; distributed NEURON launches may need MPI_LIB_NRN_PATH.")

    if check_gap_mechanisms and "neuron>=8.2.6,<9" not in missing:
        try:
            from digifly.phase2.neuron_build.gaps import ensure_gap_mechanism_available

            ensure_gap_mechanism_available(report["gap_mechanisms_dir"], require_rectifying=True, require_heterotypic=True)
            report["gap_mechanisms_available"] = True
        except Exception as exc:
            report["gap_mechanisms_available"] = False
            report["errors"].append(str(exc))
    return report


def ensure_phase2_environment(
    *,
    profiles: str | Sequence[str] = ("core", "notebook"),
    extra_packages: Sequence[str] = (),
    auto_install_python: bool = False,
    check_gap_mechanisms: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    configure_phase2_environment()
    packages = python_packages_for_profiles(profiles, extra=extra_packages)
    missing = missing_python_packages(packages)
    installed: list[str] = []
    if missing and auto_install_python:
        install_python_packages(missing, quiet=quiet)
        installed = list(missing)

    report = diagnose_phase2_environment(
        profiles=profiles,
        extra_packages=extra_packages,
        check_gap_mechanisms=check_gap_mechanisms,
    )
    report["installed_python_packages"] = installed
    if report["missing_python_packages"] and not auto_install_python:
        report["warnings"].append(
            "Missing Python packages were detected. Re-run with auto_install_python=True from a notebook "
            "or install the listed packages in the active Python environment."
        )
    if report["missing_python_packages"] and auto_install_python:
        report["errors"].append(
            "Some Python packages are still missing after pip install. Check the notebook output above for pip errors."
        )
    return report
