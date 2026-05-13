from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


Pair = Tuple[int, int]


def _normalize_pairs(pairs: Sequence[Sequence[int] | Pair]) -> list[Pair]:
    out: list[Pair] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Connection pair must have exactly two ids: {pair!r}")
        out.append((int(pair[0]), int(pair[1])))
    return out


def read_edges_table(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    sfx = p.suffix.lower()
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if sfx in {".feather", ".ftr"}:
        return pd.read_feather(p)
    return pd.read_csv(p)


def resolve_edges_for_cfg(cfg: Mapping[str, Any]) -> tuple[Path, list[int] | None]:
    cfg_use = dict(cfg or {})
    ecfg = dict(cfg_use.get("edge_cache") or {})
    sel = dict(cfg_use.get("selection") or {})
    mode = str(sel.get("mode") or "").strip().lower()

    if bool(ecfg.get("enabled", False)) and mode == "custom":
        from digifly.phase2.graph.edge_cache import resolve_custom_edges_from_cache

        loaded_ids = [int(x) for x in (sel.get("neuron_ids") or [])]
        seed_ids = [int(x) for x in (cfg_use.get("seeds") or [])]
        path, resolved_ids = resolve_custom_edges_from_cache(
            cfg_use,
            loaded_ids=loaded_ids,
            seed_ids=seed_ids,
        )
        return Path(path).expanduser().resolve(), [int(x) for x in resolved_ids]

    raw_path = cfg_use.get("edges_csv") or cfg_use.get("edges_path")
    if not raw_path:
        raise ValueError("Cannot resolve edges: config has no edge_cache, edges_csv, or edges_path.")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Edges file not found: {path}")
    return path, None


def summarize_pair_strengths(
    edges: pd.DataFrame,
    pairs: Sequence[Sequence[int] | Pair],
    *,
    weight_col: str = "weight_uS",
    default_weight_uS: float | None = None,
) -> list[Dict[str, Any]]:
    if not {"pre_id", "post_id"}.issubset(set(edges.columns)):
        raise ValueError("edges table must contain pre_id and post_id columns")

    pair_list = _normalize_pairs(pairs)
    df = edges.copy()
    df["pre_id"] = pd.to_numeric(df["pre_id"], errors="coerce")
    df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce")
    df = df.dropna(subset=["pre_id", "post_id"]).copy()
    df["pre_id"] = df["pre_id"].astype(int)
    df["post_id"] = df["post_id"].astype(int)

    if weight_col in df.columns:
        weights = pd.to_numeric(df[weight_col], errors="coerce")
        if default_weight_uS is not None:
            weights = weights.fillna(float(default_weight_uS))
    elif default_weight_uS is not None:
        weights = pd.Series(float(default_weight_uS), index=df.index, dtype=float)
    else:
        raise ValueError(f"edges table has no {weight_col!r} column and no default_weight_uS was provided")
    df["__equalization_weight_uS__"] = weights.astype(float)

    rows: list[Dict[str, Any]] = []
    for pre_id, post_id in pair_list:
        mask = (df["pre_id"] == int(pre_id)) & (df["post_id"] == int(post_id))
        sub = df.loc[mask]
        weight_sum = float(np.nansum(pd.to_numeric(sub["__equalization_weight_uS__"], errors="coerce")))
        rows.append(
            {
                "pre_id": int(pre_id),
                "post_id": int(post_id),
                "pair": f"{int(pre_id)}->{int(post_id)}",
                "synapse_count": int(len(sub)),
                "weight_sum_uS": float(weight_sum),
            }
        )
    return rows


def _target_sum(values: Sequence[float], target: str | float | int) -> float:
    valid = [float(v) for v in values if np.isfinite(float(v)) and float(v) > 0.0]
    if not valid:
        return 0.0
    if isinstance(target, (int, float)):
        return float(target)

    target_norm = str(target or "mean").strip().lower()
    if target_norm in {"mean", "average", "avg"}:
        return float(np.mean(valid))
    if target_norm in {"max", "maximum", "strongest"}:
        return float(np.max(valid))
    if target_norm in {"min", "minimum", "weakest"}:
        return float(np.min(valid))
    if target_norm in {"first", "reference"}:
        return float(valid[0])
    raise ValueError("target must be 'mean', 'max', 'min', 'first', or a numeric summed weight")


def build_pair_strength_equalization_overrides(
    edges: pd.DataFrame,
    pairs: Sequence[Sequence[int] | Pair],
    *,
    target: str | float | int = "mean",
    weight_col: str = "weight_uS",
    default_weight_uS: float | None = None,
    group_prefix: str = "pair_strength_equalized",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    summary_rows = summarize_pair_strengths(
        edges,
        pairs,
        weight_col=weight_col,
        default_weight_uS=default_weight_uS,
    )
    target_weight_sum = _target_sum([float(row["weight_sum_uS"]) for row in summary_rows], target)

    groups: list[Dict[str, Any]] = []
    for row in summary_rows:
        current = float(row["weight_sum_uS"])
        if current <= 0.0 or not np.isfinite(current) or target_weight_sum <= 0.0:
            multiplier = 1.0
            enabled = False
        else:
            multiplier = float(target_weight_sum / current)
            enabled = True
        row["target_weight_sum_uS"] = float(target_weight_sum)
        row["weight_mult"] = float(multiplier)
        row["enabled"] = bool(enabled)
        if enabled:
            groups.append(
                {
                    "name": f"{group_prefix}_{int(row['pre_id'])}_{int(row['post_id'])}",
                    "selectors": {"pairs": [[int(row["pre_id"]), int(row["post_id"])]]},
                    "weight_mult": float(multiplier),
                }
            )

    summary = {
        "target": str(target),
        "target_weight_sum_uS": float(target_weight_sum),
        "pairs": summary_rows,
        "groups": groups,
    }
    return groups, summary


def build_pair_strength_equalization_for_cfg(
    cfg: Mapping[str, Any],
    pairs: Sequence[Sequence[int] | Pair],
    *,
    target: str | float | int = "mean",
    weight_col: str = "weight_uS",
    default_weight_uS: float | None = None,
    group_prefix: str = "pair_strength_equalized",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    edges_path, resolved_ids = resolve_edges_for_cfg(cfg)
    edges = read_edges_table(edges_path)
    groups, summary = build_pair_strength_equalization_overrides(
        edges,
        pairs,
        target=target,
        weight_col=weight_col,
        default_weight_uS=default_weight_uS,
        group_prefix=group_prefix,
    )
    summary["edges_path"] = str(edges_path)
    if resolved_ids is not None:
        summary["resolved_ids"] = [int(x) for x in resolved_ids]
    return groups, summary


def build_pair_threshold_equalization_overrides(
    pair_threshold_uS: Mapping[Sequence[int] | Pair | str, float],
    *,
    target: str | float | int = "min",
    group_prefix: str = "pair_threshold_equalized",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for raw_pair, raw_threshold in dict(pair_threshold_uS or {}).items():
        if isinstance(raw_pair, str):
            parts = raw_pair.replace("->", ",").split(",")
            pair = (int(parts[0].strip()), int(parts[1].strip()))
        else:
            pair = _normalize_pairs([raw_pair])[0]
        threshold = float(raw_threshold)
        if threshold <= 0.0 or not np.isfinite(threshold):
            raise ValueError(f"Threshold must be a positive finite value for pair {pair}: {raw_threshold!r}")
        rows.append(
            {
                "pre_id": int(pair[0]),
                "post_id": int(pair[1]),
                "pair": f"{int(pair[0])}->{int(pair[1])}",
                "threshold_uS": float(threshold),
            }
        )

    target_threshold = _target_sum([float(row["threshold_uS"]) for row in rows], target)
    if target_threshold <= 0.0:
        return [], {"target": str(target), "target_threshold_uS": 0.0, "pairs": rows, "groups": []}

    groups: list[Dict[str, Any]] = []
    for row in rows:
        # Spike threshold scales approximately inversely with synaptic strength.
        multiplier = float(row["threshold_uS"]) / float(target_threshold)
        row["target_threshold_uS"] = float(target_threshold)
        row["weight_mult"] = float(multiplier)
        groups.append(
            {
                "name": f"{group_prefix}_{int(row['pre_id'])}_{int(row['post_id'])}",
                "selectors": {"pairs": [[int(row["pre_id"]), int(row["post_id"])]]},
                "weight_mult": float(multiplier),
            }
        )

    return groups, {
        "target": str(target),
        "target_threshold_uS": float(target_threshold),
        "pairs": rows,
        "groups": groups,
    }


def build_pair_noise_overrides(
    pairs: Sequence[Sequence[int] | Pair],
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    cv: float = 0.2,
    distribution: str = "lognormal",
    correlated: bool = False,
    group_prefix: str = "pair_synapse_noise",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    pair_list = _normalize_pairs(pairs)
    rng_use = rng if rng is not None else np.random.default_rng(seed)
    cv_use = max(0.0, float(cv))
    dist = str(distribution or "lognormal").strip().lower()

    def _draw_multiplier() -> float:
        if cv_use <= 0.0:
            return 1.0
        if dist in {"normal", "gaussian"}:
            return float(max(0.0, rng_use.normal(1.0, cv_use)))
        if dist in {"lognormal", "log-normal", "release"}:
            sigma = float(np.sqrt(np.log1p(cv_use * cv_use)))
            return float(rng_use.lognormal(mean=0.0, sigma=sigma))
        raise ValueError("distribution must be 'lognormal' or 'normal'")

    shared = _draw_multiplier() if correlated else None
    groups: list[Dict[str, Any]] = []
    rows: list[Dict[str, Any]] = []
    for pre_id, post_id in pair_list:
        mult = float(shared if shared is not None else _draw_multiplier())
        row = {
            "pre_id": int(pre_id),
            "post_id": int(post_id),
            "pair": f"{int(pre_id)}->{int(post_id)}",
            "weight_mult": float(mult),
        }
        rows.append(row)
        groups.append(
            {
                "name": f"{group_prefix}_{int(pre_id)}_{int(post_id)}",
                "selectors": {"pairs": [[int(pre_id), int(post_id)]]},
                "weight_mult": float(mult),
            }
        )

    return groups, {
        "cv": float(cv_use),
        "distribution": dist,
        "correlated": bool(correlated),
        "pairs": rows,
        "groups": groups,
    }
