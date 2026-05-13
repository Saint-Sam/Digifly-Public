import pathlib
import sys


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


def test_runtime_env_build_launch_env_sets_phase2_defaults(monkeypatch, tmp_path):
    from digifly.phase2.runtime_env import build_phase2_launch_env

    mpi_dir = tmp_path / "mpi"
    mpi_dir.mkdir()
    mpi_lib = mpi_dir / "libmpi.so"
    mpi_lib.write_text("", encoding="utf-8")
    monkeypatch.setenv("MPI_LIB_NRN_PATH", str(mpi_lib))
    monkeypatch.setenv("DYLD_LIBRARY_PATH", str(mpi_dir))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(mpi_dir))
    monkeypatch.delenv("PYTHONHOME", raising=False)
    env = build_phase2_launch_env({"PATH": "/usr/bin"}, phase2_dir=_phase2_root())

    assert env["NEURON_MODULE_OPTIONS"] == "-nogui"
    assert env["DIGIFLY_PHASE2_ROOT"] == str(_phase2_root())
    assert env["DIGIFLY_WORKSPACE"] == str(_phase2_root().parent)
    assert env["DIGIFLY_GAP_MECH_DIR"] == str(_phase2_root() / "data")
    assert "PYTHONHOME" not in env
    assert str(mpi_dir) not in env.get("DYLD_LIBRARY_PATH", "")
    assert str(mpi_dir) not in env.get("LD_LIBRARY_PATH", "")


def test_runtime_env_profile_package_resolution_is_deduped():
    from digifly.phase2.runtime_env import python_packages_for_profiles

    packages = python_packages_for_profiles(("core", "core"), extra=("numpy", "plotly"))

    assert packages.count("numpy") == 1
    assert "neuron>=8.2.6,<9" in packages
    assert "plotly" in packages


def test_cache_prepare_config_enables_distributed_backend():
    from digifly.phase2.cache import prepare_cached_config

    cfg = {"parallel": {"build_backend": "single_host"}, "run_id": "demo"}
    out = prepare_cached_config(cfg, nproc=4)

    assert out["parallel"]["build_backend"] == "distributed_gid"
    assert cfg["parallel"]["build_backend"] == "single_host"


def test_cached_session_derives_single_selection_ids():
    from digifly.phase2.cache.session import _selection_ids

    cfg = {"selection": {"mode": "single", "neuron_id": 10000}}

    assert _selection_ids(cfg) == [10000]
