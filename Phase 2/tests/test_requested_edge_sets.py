import pathlib
import sys

import pandas as pd
import types
import json


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


from digifly.phase2.graph import (
    DEFAULT_NEUPRINT_DATASET,
    default_edges_registry_root,
    ensure_named_edge_set,
    expand_requested_network,
    normalize_neuprint_dataset,
    resolve_neuprint_dataset_choice,
)
from digifly.phase2.graph import requested_edge_sets as requested_edge_sets_module
from digifly.phase2.data.paths import (
    _find_swc,
    _syn_csv_path,
    invalidate_export_path_cache,
    refresh_export_path_index,
)


def test_graph_lazy_exports_resolve_requested_edge_helpers(tmp_path):
    root = default_edges_registry_root(tmp_path)

    assert callable(ensure_named_edge_set)
    assert root == tmp_path.resolve()
    assert normalize_neuprint_dataset("manc") == DEFAULT_NEUPRINT_DATASET
    assert resolve_neuprint_dataset_choice("male-cns", None) == "male-cns:v0.9"


def test_expand_requested_network_adds_only_immediate_motor_postsynaptic(tmp_path):
    master_csv = tmp_path / "master.csv"
    pd.DataFrame(
        [
            {"bodyId": 100, "class": "interneuron"},
            {"bodyId": 200, "class": "motor neuron"},
            {"bodyId": 300, "class": "interneuron"},
        ]
    ).to_csv(master_csv, index=False)

    raw_edges = pd.DataFrame(
        [
            {"pre_id": 100, "post_id": 200, "weight_uS": 1.0e-5},
            {"pre_id": 100, "post_id": 300, "weight_uS": 1.0e-5},
            {"pre_id": 300, "post_id": 200, "weight_uS": 1.0e-5},
        ]
    )

    expanded = expand_requested_network(
        [100, 300],
        raw_edges,
        expansion_mode="immediate_motor_postsynaptic",
        master_csv=master_csv,
    )

    assert expanded["selection_rule"] == "requested ids + immediate postsynaptic motor neurons"
    assert expanded["added_ids"] == [200]
    assert expanded["final_network_ids"] == [100, 300, 200]
    assert expanded["final_edges_df"][["pre_id", "post_id"]].to_dict(orient="records") == [
        {"pre_id": 100, "post_id": 200},
        {"pre_id": 100, "post_id": 300},
        {"pre_id": 300, "post_id": 200},
    ]


def test_phase1_export_forwarding_includes_parallel_export_knobs(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        requested_edge_sets_module,
        "missing_local_export_ids",
        lambda ids, **kwargs: [int(x) for x in ids],
    )

    def fake_ensure_phase2_neuron_exports(**kwargs):
        captured.update(kwargs)
        return {"status": "exported_missing"}

    fake_bridge = types.SimpleNamespace(ensure_phase2_neuron_exports=fake_ensure_phase2_neuron_exports)
    monkeypatch.setattr(requested_edge_sets_module, "_load_phase1_bridge", lambda: fake_bridge)

    report = requested_edge_sets_module.ensure_phase1_exports_if_needed(
        [101, 202],
        swc_dir=tmp_path,
        enabled=True,
        batch_size=3000,
        export_workers=4,
        progress_every=9,
    )

    assert report["status"] == "exported_missing"
    assert captured["batch_size"] == 3000
    assert captured["workers"] == 4
    assert captured["progress_every"] == 9


def test_export_index_falls_back_when_manifest_is_stale(tmp_path):
    by_id = tmp_path / "by_id"
    first_dir = by_id / "101"
    first_dir.mkdir(parents=True)
    (first_dir / "101_synapses_new.csv").write_text("post_id\n202\n", encoding="utf-8")
    (first_dir / "101_healed.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")

    refresh_export_path_index(tmp_path)
    invalidate_export_path_cache(tmp_path)

    second_dir = by_id / "202"
    second_dir.mkdir(parents=True)
    expected_syn = (second_dir / "202_synapses_new.csv").resolve()
    expected_swc = (second_dir / "202_healed.swc").resolve()
    expected_syn.write_text("post_id\n303\n", encoding="utf-8")
    expected_swc.write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")

    assert _syn_csv_path(tmp_path, 202) == expected_syn
    assert pathlib.Path(_find_swc(str(tmp_path), 202)).resolve() == expected_swc


def test_missing_local_export_ids_respects_indexed_dataset_metadata(tmp_path):
    root = tmp_path / "exports"
    by_id = root / "by_id" / "101"
    by_id.mkdir(parents=True)
    (by_id / "101_synapses_new.csv").write_text("post_id\n202\n", encoding="utf-8")
    (by_id / "101_healed.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
    (by_id / "101_export_meta.json").write_text(
        json.dumps({"neuprint_dataset": "manc:v1.2.1"}),
        encoding="utf-8",
    )

    refresh_export_path_index(root)

    assert requested_edge_sets_module.missing_local_export_ids(
        [101],
        swc_dir=root,
        neuprint_dataset="manc:v1.2.1",
    ) == []
    assert requested_edge_sets_module.missing_local_export_ids(
        [101],
        swc_dir=root,
        neuprint_dataset="manc:v1.2.3",
    ) == [101]
