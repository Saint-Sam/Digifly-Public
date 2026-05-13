import pathlib
import sys

import pandas as pd


def _phase2_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


if str(_phase2_root()) not in sys.path:
    sys.path.insert(0, str(_phase2_root()))


def test_pair_strength_equalization_targets_mean_sum():
    from digifly.phase2.graph.connection_equalization import (
        build_pair_strength_equalization_overrides,
    )

    edges = pd.DataFrame(
        [
            {"pre_id": 10000, "post_id": 10110, "weight_uS": 2.0},
            {"pre_id": 10000, "post_id": 10110, "weight_uS": 2.0},
            {"pre_id": 10002, "post_id": 10068, "weight_uS": 1.0},
            {"pre_id": 10068, "post_id": 10000, "weight_uS": 9.0},
        ]
    )

    groups, summary = build_pair_strength_equalization_overrides(
        edges,
        [(10000, 10110), (10002, 10068)],
        target="mean",
    )

    assert summary["target_weight_sum_uS"] == 2.5
    by_pair = {row["pair"]: row for row in summary["pairs"]}
    assert by_pair["10000->10110"]["weight_mult"] == 0.625
    assert by_pair["10002->10068"]["weight_mult"] == 2.5
    assert groups == [
        {
            "name": "pair_strength_equalized_10000_10110",
            "selectors": {"pairs": [[10000, 10110]]},
            "weight_mult": 0.625,
        },
        {
            "name": "pair_strength_equalized_10002_10068",
            "selectors": {"pairs": [[10002, 10068]]},
            "weight_mult": 2.5,
        },
    ]


def test_pair_strength_equalization_uses_default_weight_when_missing():
    from digifly.phase2.graph.connection_equalization import summarize_pair_strengths

    edges = pd.DataFrame(
        [
            {"pre_id": 10000, "post_id": 10110},
            {"pre_id": 10000, "post_id": 10110},
            {"pre_id": 10002, "post_id": 10068},
        ]
    )

    rows = summarize_pair_strengths(
        edges,
        [(10000, 10110), (10002, 10068)],
        default_weight_uS=0.25,
    )

    by_pair = {row["pair"]: row for row in rows}
    assert by_pair["10000->10110"]["synapse_count"] == 2
    assert by_pair["10000->10110"]["weight_sum_uS"] == 0.5
    assert by_pair["10002->10068"]["synapse_count"] == 1
    assert by_pair["10002->10068"]["weight_sum_uS"] == 0.25


def test_pair_threshold_equalization_boosts_high_threshold_pair():
    from digifly.phase2.graph.connection_equalization import (
        build_pair_threshold_equalization_overrides,
    )

    groups, summary = build_pair_threshold_equalization_overrides(
        {
            (10000, 10110): 8.5e-7,
            (10002, 10068): 1.5e-6,
        },
        target="min",
    )

    by_pair = {row["pair"]: row for row in summary["pairs"]}
    assert by_pair["10000->10110"]["weight_mult"] == 1.0
    assert round(by_pair["10002->10068"]["weight_mult"], 6) == round(1.5e-6 / 8.5e-7, 6)
    assert groups[1]["selectors"] == {"pairs": [[10002, 10068]]}


def test_pair_noise_overrides_are_reproducible_and_positive():
    from digifly.phase2.graph.connection_equalization import build_pair_noise_overrides

    groups_a, summary_a = build_pair_noise_overrides(
        [(10000, 10110), (10002, 10068)],
        seed=7,
        cv=0.25,
    )
    groups_b, summary_b = build_pair_noise_overrides(
        [(10000, 10110), (10002, 10068)],
        seed=7,
        cv=0.25,
    )

    assert groups_a == groups_b
    assert summary_a == summary_b
    assert all(row["weight_mult"] > 0.0 for row in summary_a["pairs"])
