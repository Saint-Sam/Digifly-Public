from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _list_named_dirs(root: Path | str) -> list[str]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and not path.name.startswith("_")
    )


def list_hemilineage_folders(root: Path | str) -> list[str]:
    return _list_named_dirs(root)


def list_run_folders(hemi_dir: Path | str) -> list[str]:
    hemi_dir = Path(hemi_dir).expanduser().resolve()
    return _list_named_dirs(hemi_dir / "runs")


def resolve_hemilineage_dir(root: Path | str, hemi_name: str) -> Path:
    hemi_name = str(hemi_name).strip()
    hemi_dir = (Path(root).expanduser().resolve() / hemi_name).resolve()
    if not hemi_name or not hemi_dir.exists():
        raise FileNotFoundError(f"Hemilineage folder not found: {hemi_dir}")
    return hemi_dir


def load_added_motor_neuron_ids(csv_path: Path | str, id_column: str = "neuron_id") -> pd.Series:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Added motor neuron CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise ValueError(f"{csv_path} is missing required column: {id_column}")

    ids = pd.to_numeric(df[id_column], errors="coerce").dropna().astype(int).drop_duplicates().sort_values()
    return ids.reset_index(drop=True)


def filter_spikes_to_neuron_ids(spikes: pd.DataFrame, neuron_ids: Iterable[int]) -> pd.DataFrame:
    keep_ids = sorted({int(x) for x in neuron_ids})
    if not keep_ids:
        return spikes.iloc[0:0].copy()

    out = spikes[spikes["neuron_id"].isin(keep_ids)].copy()
    return out.sort_values(["neuron_id", "spike_time_ms"]).reset_index(drop=True)


def summarize_focus_neuron_overlap(
    spikes: pd.DataFrame,
    mapping: pd.DataFrame,
    focus_neuron_ids: Iterable[int],
) -> Dict[str, int]:
    focus_ids = {int(x) for x in focus_neuron_ids}
    spike_ids = set(spikes["neuron_id"].astype(int).unique())
    map_ids = set(mapping["mn_id"].astype(int).unique())
    focus_spiking = focus_ids.intersection(spike_ids)
    focus_spiking_mapped = focus_spiking.intersection(map_ids)
    return {
        "focus_neuron_count": int(len(focus_ids)),
        "focus_spiking_neuron_count": int(len(focus_spiking)),
        "focus_spiking_mapped_count": int(len(focus_spiking_mapped)),
        "focus_spiking_unmapped_count": int(len(focus_spiking - map_ids)),
        "focus_spike_row_count": int(spikes[spikes["neuron_id"].isin(list(focus_ids))].shape[0]),
    }


def build_spike_mapping_summary(spikes: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if spikes.empty:
        return pd.DataFrame(
            columns=[
                "neuron_id",
                "spike_count",
                "first_spike_ms",
                "last_spike_ms",
                "mapped_actuator_names",
                "is_mapped",
            ]
        )

    spike_summary = (
        spikes.groupby("neuron_id")["spike_time_ms"]
        .agg(spike_count="size", first_spike_ms="min", last_spike_ms="max")
        .reset_index()
        .sort_values(["spike_count", "first_spike_ms", "neuron_id"], ascending=[False, True, True])
    )

    mapping_summary = (
        mapping.groupby("mn_id")["actuator_name"]
        .agg(lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()})))
        .reset_index()
        .rename(columns={"mn_id": "neuron_id", "actuator_name": "mapped_actuator_names"})
    )

    out = spike_summary.merge(mapping_summary, on="neuron_id", how="left")
    out["mapped_actuator_names"] = out["mapped_actuator_names"].fillna("")
    out["is_mapped"] = out["mapped_actuator_names"].ne("")
    return out.reset_index(drop=True)
