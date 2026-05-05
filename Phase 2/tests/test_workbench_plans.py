import csv
import pathlib
import sys


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _ensure_import_path() -> pathlib.Path:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _write_master_csv(path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bodyId", "hemilineage"])
        writer.writeheader()
        writer.writerow({"bodyId": 10000, "hemilineage": "09A"})
        writer.writerow({"bodyId": 10002, "hemilineage": "09A"})
        writer.writerow({"bodyId": 10004, "hemilineage": "10B"})


def _write_edges_csv(path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pre_id", "post_id"])
        writer.writeheader()
        writer.writerow({"pre_id": 10000, "post_id": 10002})


def _write_simple_swc(path: pathlib.Path, neuron_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# bodyId {int(neuron_id)}",
                "1 1 0.0 0.0 0.0 0.5 -1",
                "2 2 10.0 0.0 0.0 0.2 1",
                "3 2 20.0 0.0 0.0 0.2 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_custom_shared_plan_builds_from_preset(tmp_path):
    _ensure_import_path()
    from digifly.phase2.workbench.presets import apply_preset
    from digifly.phase2.workbench.runner import build_execution_plan
    from digifly.phase2.workbench.validation import validate_state

    swc_dir = tmp_path / "export_swc"
    swc_dir.mkdir()
    edges_csv = tmp_path / "custom_edges.csv"
    _write_edges_csv(edges_csv)

    state = apply_preset("custom-network-quick")
    state["swc_dir"] = str(swc_dir)
    state["edges_path"] = str(edges_csv)
    state["neuron_ids_text"] = "10000, 10002"

    report = validate_state(state)
    assert report.ok, report.format()

    plan = build_execution_plan(state, preset_slug="custom-network-quick")
    assert plan.runner_kind == "shared_runner"
    assert plan.payload["selection"]["mode"] == "custom"
    assert plan.payload["selection"]["neuron_ids"] == [10000, 10002]
    assert pathlib.Path(plan.payload["edges_path"]).resolve() == edges_csv.resolve()


def test_default_preset_is_public_single_neuron_smoke():
    _ensure_import_path()
    from digifly.phase2.workbench.presets import preset_options

    assert preset_options()[0] == ("Single Neuron Debug", "single-neuron-debug")


def test_single_neuron_preset_uses_four_pulse_demo():
    _ensure_import_path()
    from digifly.phase2.workbench.presets import apply_preset

    state = apply_preset("single-neuron-debug")

    assert float(state["tstop_ms"]) == 1000.0
    assert float(state["iclamp_amp_nA"]) == 0.0
    assert bool(state["pulse_train_enabled"]) is True
    assert float(state["pulse_train_freq_hz"]) == 4.0
    assert float(state["pulse_train_delay_ms"]) == 100.0
    assert float(state["pulse_train_stop_ms"]) == 900.0
    assert int(state["pulse_train_max_pulses"]) == 4
    assert bool(state["pulse_train_include_base"]) is False


def test_default_state_writes_shared_runs_under_swc_hemi_runs(tmp_path, monkeypatch):
    _ensure_import_path()
    from digifly.phase2.workbench.controls import default_state

    swc_dir = tmp_path / "Phase 1" / "manc_v1.2.1" / "export_swc"
    swc_dir.mkdir(parents=True)
    monkeypatch.setenv("DIGIFLY_SWC_DIR", str(swc_dir))

    state = default_state()

    assert pathlib.Path(state["swc_dir"]) == swc_dir.resolve()
    assert pathlib.Path(state["runs_root"]) == (swc_dir / "hemi_runs").resolve()


def test_default_state_infers_public_swc_dir_without_env(tmp_path, monkeypatch):
    _ensure_import_path()
    from digifly.phase2.workbench.controls import default_state

    monkeypatch.delenv("DIGIFLY_SWC_DIR", raising=False)
    monkeypatch.delenv("DIGIFLY_PHASE2_ROOT", raising=False)
    monkeypatch.delenv("DIGIFLY_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    state = default_state()
    swc_dir = pathlib.Path(state["swc_dir"])

    assert swc_dir.name == "export_swc"
    assert swc_dir.parent.name == "manc_v1.2.1"
    assert pathlib.Path(state["runs_root"]) == (swc_dir / "hemi_runs").resolve()


def test_mutation_launcher_auto_picks_public_hemi_run(tmp_path):
    repo_root = _ensure_import_path()
    launcher_dir = repo_root / "apps" / "VIP_Glia_Sim" / "notebooks"
    if str(launcher_dir) not in sys.path:
        sys.path.insert(0, str(launcher_dir))

    from launcher_paths import candidate_flow_run_roots, resolve_flow_run_dir

    phase2_root = tmp_path / "Phase 2"
    app_root = phase2_root / "apps" / "VIP_Glia_Sim"
    swc_dir = tmp_path / "Phase 1" / "manc_v1.2.1" / "export_swc"
    run_dir = swc_dir / "hemi_runs" / "single_neuron_debug"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "records.csv").write_text("t_ms,10000_soma_v\n0.0,-65.0\n", encoding="utf-8")

    roots = candidate_flow_run_roots(phase2_root, swc_dir, app_root=app_root)
    picked, searched = resolve_flow_run_dir(phase2_root, swc_dir, [10000], app_root=app_root)

    assert roots[0] == (swc_dir / "hemi_runs").resolve()
    assert searched == roots
    assert picked == run_dir.resolve()


def test_workbench_mutation_launch_plan_uses_recent_run(tmp_path):
    _ensure_import_path()
    from digifly.phase2.workbench.mutation_launcher import build_mutation_launch_plan

    phase2_root = tmp_path / "Phase 2"
    app_tools = phase2_root / "apps" / "VIP_Glia_Sim" / "tools"
    app_tools.mkdir(parents=True)
    app_script = app_tools / "morphology_mutation_app.py"
    app_script.write_text("print('mutation app placeholder')\n", encoding="utf-8")

    swc_dir = tmp_path / "Phase 1" / "manc_v1.2.1" / "export_swc"
    swc_dir.mkdir(parents=True)
    run_dir = swc_dir / "hemi_runs" / "single_neuron_debug"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "records.csv").write_text("t_ms,10000_soma_v\n0.0,-65.0\n", encoding="utf-8")

    plan = build_mutation_launch_plan(
        {"swc_dir": str(swc_dir), "mode": "single", "neuron_id": 10000},
        phase2_root=phase2_root,
        flow_run_dir=run_dir,
        python_bin=sys.executable,
    )

    assert plan.flow_run_dir == run_dir.resolve()
    assert plan.swc_dir == swc_dir.resolve()
    assert plan.neuron_ids == [10000]
    assert "--flow-run-dir" in plan.command
    assert str(run_dir.resolve()) in plan.command
    assert "--flow-duration-sec" in plan.command
    assert "20" in plan.command
    assert "--flow-speed-um-per-ms" in plan.command
    assert "25" in plan.command
    assert "--flow-pulse-sigma-ms" in plan.command
    assert "18" in plan.command
    assert "--flow-max-ms" in plan.command
    assert "0" in plan.command
    assert "--start-solo" in plan.command
    assert str(app_script.resolve()) in plan.command


def test_workbench_mutation_launch_blocks_headless_display(tmp_path, monkeypatch):
    _ensure_import_path()
    from digifly.phase2.workbench.mutation_launcher import MutationLaunchPlan, launch_mutation_app

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    plan = MutationLaunchPlan(
        command=[sys.executable, "--version"],
        app_root=tmp_path,
        app_path=tmp_path / "app.py",
        swc_dir=tmp_path,
        flow_run_dir=tmp_path,
        output_root=tmp_path / "outputs",
        log_path=tmp_path / "outputs" / "launch.log",
        neuron_ids=[10000],
        warning="headless",
    )

    result = launch_mutation_app(plan)

    assert result["blocked"] is True
    assert result["pid"] is None
    assert not plan.log_path.exists()


def test_notebook_ui_resolves_run_dir_and_time_column(tmp_path):
    _ensure_import_path()
    import pandas as pd

    from digifly.phase2.workbench.notebook_ui import _find_time_column, _result_run_dir

    out_dir = tmp_path / "hemi_runs" / "single_neuron_debug"
    result = {"result": {"output_dir": str(out_dir)}}

    assert _result_run_dir(result) == out_dir.resolve()
    assert _find_time_column(pd.DataFrame({"t_ms": [0.0], "10000_soma_v": [-65.0]})) == "t_ms"


def test_browser_visualizer_builds_plotly_flow_figure(tmp_path):
    _ensure_import_path()
    from digifly.phase2.workbench.browser_visualizer import (
        build_browser_flow_figure,
        find_swc_file,
        recorded_neuron_ids,
    )

    swc_dir = tmp_path / "export_swc"
    _write_simple_swc(swc_dir / "DN" / "DNp01" / "10000" / "10000_healed.swc", 10000)
    preferred = swc_dir / "DN" / "DNp01" / "10000" / "10000_axodendro_with_synapses.swc"
    _write_simple_swc(preferred, 10000)
    _write_simple_swc(swc_dir / "DN" / "DNp01" / "10000" / "10000_axodendro_with_synapses_OLD_test.swc", 10000)

    run_dir = swc_dir / "hemi_runs" / "single_neuron_debug"
    run_dir.mkdir(parents=True)
    with (run_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t_ms", "10000_soma_v"])
        writer.writeheader()
        for t_ms, v in [(0.0, -65.0), (1.0, -40.0), (2.0, 25.0), (3.0, -55.0)]:
            writer.writerow({"t_ms": t_ms, "10000_soma_v": v})

    assert recorded_neuron_ids(run_dir) == [10000]
    assert find_swc_file(swc_dir, 10000) == preferred.resolve()

    fig = build_browser_flow_figure(run_dir=run_dir, swc_dir=swc_dir, max_frames=4, playback_seconds=1.0)

    assert len(fig.frames) == 4
    assert any(trace.name == "browser flow" for trace in fig.data)
    assert "Play" in fig.layout.updatemenus[0].buttons[0].label


def test_hemi_project_plan_derives_core_ids_from_master_csv(tmp_path):
    _ensure_import_path()
    from digifly.phase2.workbench.presets import apply_preset
    from digifly.phase2.workbench.runner import build_execution_plan
    from digifly.phase2.workbench.validation import validate_state

    swc_dir = tmp_path / "export_swc"
    swc_dir.mkdir()
    master_csv = tmp_path / "all_neurons_neuroncriteria_template.csv"
    _write_master_csv(master_csv)
    projects_root = tmp_path / "projects"

    state = apply_preset("hemilineage-project-baseline")
    state["swc_dir"] = str(swc_dir)
    state["projects_root"] = str(projects_root)

    report = validate_state(state)
    assert report.ok, report.format()

    plan = build_execution_plan(state, preset_slug="hemilineage-project-baseline")
    assert plan.runner_kind == "hemilineage_project"
    assert plan.payload["hemilineage_label"] == "09A"
    assert plan.payload["neuron_ids"] == [10000, 10002]
    assert float(plan.payload["runtime"]["tstop_ms"]) == float(state["tstop_ms"])
