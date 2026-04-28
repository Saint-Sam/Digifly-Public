import pathlib
import sys

import pandas as pd


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


from digifly.phase2.neuron_build.ownership import build_cell_ownership
from digifly.phase2.neuron_build.network import _build_pc_gid_maps
from digifly.phase2.neuron_build.wiring_plan import build_network_plan


def test_build_cell_ownership_round_robin_is_stable():
    plan = build_cell_ownership([7, 3, 5, 9], world_size=2, rank=1, strategy="round_robin")

    assert plan.gids == (3, 5, 7, 9)
    assert plan.owner_of(3) == 0
    assert plan.owner_of(5) == 1
    assert plan.owner_of(7) == 0
    assert plan.owner_of(9) == 1
    assert plan.local_gids == (5, 9)
    assert plan.remote_gids == (3, 7)


def test_build_cell_ownership_contiguous_groups_neighbors():
    plan = build_cell_ownership([10, 11, 12, 13, 14], world_size=2, rank=0, strategy="contiguous")

    assert plan.gids_by_owner[0] == (10, 11, 12)
    assert plan.gids_by_owner[1] == (13, 14)


def test_build_network_plan_tracks_local_posts_and_sites():
    df_wire = pd.DataFrame(
        [
            {
                "pre_id": 100,
                "post_id": 101,
                "weight_uS": 1.2e-5,
                "delay_ms": 0.8,
                "syn_e_rev_mV": 0.0,
                "tau1_ms": 0.3,
                "tau2_ms": 1.4,
                "post_syn_index": 12,
                "pre_x": 1.0,
                "pre_y": 2.0,
                "pre_z": 3.0,
            },
            {
                "pre_id": 101,
                "post_id": 102,
                "weight_uS": 2.5e-5,
                "delay_ms": 1.1,
                "syn_e_rev_mV": -70.0,
                "tau1_ms": 0.4,
                "tau2_ms": 5.0,
                "post_x": 4.0,
                "post_y": 5.0,
                "post_z": 6.0,
            },
        ]
    )

    ownership = build_cell_ownership([100, 101, 102], world_size=2, rank=0, strategy="round_robin")
    plan = build_network_plan(
        {"wire_force_soma": False, "use_geom_delay": True},
        node_ids=[100, 101, 102],
        df_wire=df_wire,
        swc_lookup=lambda gid: f"/tmp/{gid}.swc",
        drivers={100: {"site": "ais", "amp": 1.0}},
        active_posts=[101, 102],
        ownership=ownership,
    )

    assert plan.local_loaded_gids() == (100, 102)
    assert plan.remote_loaded_gids() == (101,)
    assert plan.local_driver_gids() == (100,)
    assert plan.local_active_post_gids() == (102,)
    assert [conn.post_id for conn in plan.local_connection_plans()] == [102]
    assert plan.connection_plans[0].post_site["kind"] == "catalog"
    assert plan.connection_plans[0].post_site["row"]["post_syn_index"] == 12
    assert plan.connection_plans[0].pre_site_hint == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_parallel_context_gid_map_compacts_large_external_ids():
    pc_gid_by_nid, nid_by_pc_gid = _build_pc_gid_maps([10000, 31386594729, 166265])

    assert pc_gid_by_nid == {
        10000: 0,
        166265: 1,
        31386594729: 2,
    }
    assert nid_by_pc_gid == {
        0: 10000,
        1: 166265,
        2: 31386594729,
    }


def test_suffix_run_id_sanitizes_user_suffix():
    from digifly.phase2.hemi.sim_project import _suffix_run_id

    assert _suffix_run_id("hemi_09a_baseline", None) == "hemi_09a_baseline"
    assert _suffix_run_id("hemi_09a_baseline", "1000 ms rerun") == "hemi_09a_baseline_1000-ms-rerun"


def test_maybe_enable_coreneuron_falls_back_when_runtime_missing(monkeypatch):
    from digifly.phase2.neuron_build import network as network_module

    monkeypatch.setattr(network_module, "_coreneuron_runtime_available", lambda: False)

    cfg = {
        "enable_coreneuron": True,
        "coreneuron_gpu": True,
        "cvode": {"enabled": False},
    }

    enabled = network_module.maybe_enable_coreneuron(cfg)

    assert enabled is False
    assert cfg["enable_coreneuron"] is False
    assert cfg["coreneuron_gpu"] is False


def test_network_sync_iclamp_parameters_from_cfg_updates_cached_clamps():
    from digifly.phase2.neuron_build.network import Network

    class DummyStim:
        def __init__(self, amp: float, delay: float, dur: float):
            self.amp = amp
            self.delay = delay
            self.dur = dur

    net = Network({"enable_coreneuron": False, "stim": {"iclamp": {"dur_ms": 90.0}}})
    try:
        base = DummyStim(0.6, 2.0, 0.3)
        pulse = DummyStim(0.6, 10.0, 0.3)
        neg = DummyStim(-1.0, 0.0, 50.0)
        net.iclamps = [base, pulse, neg]
        net._iclamp_meta = [
            {"stim": base, "nid": 10000, "kind": "base"},
            {"stim": pulse, "nid": 10000, "kind": "pulse_train"},
            {"stim": neg, "nid": 10000, "kind": "neg_pulse"},
        ]

        updated = net.sync_iclamp_parameters_from_cfg()

        assert updated == 1
        assert base.dur == 90.0
        assert pulse.dur == 0.3
        assert neg.dur == 50.0
    finally:
        net.close(reset_parallel=False)
