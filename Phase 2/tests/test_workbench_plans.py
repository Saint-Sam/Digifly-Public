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
