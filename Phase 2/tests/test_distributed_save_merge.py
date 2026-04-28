import pathlib
import sys


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


from digifly.phase2.util.save import _merge_record_payloads, _merge_spike_payloads


def test_merge_record_payloads_combines_rank_local_columns():
    df = _merge_record_payloads(
        [
            {
                "rank": 0,
                "t_ms": [0.0, 0.1, 0.2],
                "columns": {"10000_soma_v": [-65.0, -40.0, -55.0]},
            },
            {
                "rank": 1,
                "t_ms": [0.0, 0.1, 0.2],
                "columns": {"10002_soma_v": [-64.0, -39.5, -54.5]},
            },
        ]
    )

    assert list(df.columns) == ["t_ms", "10000_soma_v", "10002_soma_v"]
    assert df.shape == (3, 3)
    assert df["10000_soma_v"].tolist() == [-65.0, -40.0, -55.0]
    assert df["10002_soma_v"].tolist() == [-64.0, -39.5, -54.5]


def test_merge_spike_payloads_flattens_rank_local_spike_vectors():
    df = _merge_spike_payloads(
        [
            {"rank": 0, "spikes": {10000: [1.5, 2.5]}},
            {"rank": 1, "spikes": {10002: [3.0]}},
        ]
    )

    assert df.to_dict(orient="records") == [
        {"neuron_id": 10000, "spike_time_ms": 1.5},
        {"neuron_id": 10000, "spike_time_ms": 2.5},
        {"neuron_id": 10002, "spike_time_ms": 3.0},
    ]
