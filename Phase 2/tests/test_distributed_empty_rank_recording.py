import pathlib
import sys


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


from digifly.phase2.neuron_build.network import Network
from digifly.phase2.neuron_build.ownership import build_cell_ownership


def test_record_time_skips_distributed_rank_with_no_local_cells():
    ownership = build_cell_ownership(
        [10000, 10002, 10068, 10110, 11446, 11654],
        world_size=7,
        rank=6,
        strategy="round_robin",
    )

    net = Network(
        {"parallel": {"build_backend": "distributed_gid"}},
        ownership=ownership,
        parallel_state={"id": 6, "nhost": 7, "nthread": 1},
    )

    try:
        result = net.record_time()
        assert result is None
        assert net.records["t"] == []
        assert net.local_gid_count == 0
    finally:
        net.close(reset_parallel=False)
