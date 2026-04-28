from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _normalize_gids(values: Iterable[Any]) -> Tuple[int, ...]:
    gids = sorted({int(v) for v in values})
    return tuple(gids)


@dataclass(frozen=True)
class OwnershipPlan:
    gids: Tuple[int, ...]
    world_size: int
    rank: int
    strategy: str
    owner_by_gid: Dict[int, int]

    def owner_of(self, gid: int) -> int:
        return int(self.owner_by_gid[int(gid)])

    def is_local(self, gid: int) -> bool:
        return self.owner_of(int(gid)) == int(self.rank)

    @property
    def local_gids(self) -> Tuple[int, ...]:
        return tuple(gid for gid in self.gids if self.is_local(gid))

    @property
    def remote_gids(self) -> Tuple[int, ...]:
        return tuple(gid for gid in self.gids if not self.is_local(gid))

    @property
    def gids_by_owner(self) -> Dict[int, Tuple[int, ...]]:
        grouped: Dict[int, List[int]] = {owner: [] for owner in range(int(self.world_size))}
        for gid in self.gids:
            grouped[self.owner_of(gid)].append(int(gid))
        return {owner: tuple(vals) for owner, vals in grouped.items()}


def build_cell_ownership(
    gids: Sequence[Any],
    *,
    world_size: int = 1,
    rank: int = 0,
    strategy: str = "round_robin",
    explicit_owner_by_gid: Mapping[Any, Any] | None = None,
) -> OwnershipPlan:
    gids_norm = _normalize_gids(gids)
    if int(world_size) < 1:
        raise ValueError("world_size must be >= 1")
    if not (0 <= int(rank) < int(world_size)):
        raise ValueError("rank must be in [0, world_size)")

    strategy_norm = str(strategy or "round_robin").strip().lower()
    owner_by_gid: Dict[int, int] = {}

    if explicit_owner_by_gid is not None:
        for gid in gids_norm:
            if gid not in explicit_owner_by_gid:
                raise KeyError(f"Explicit owner map is missing gid={gid}")
            owner = int(explicit_owner_by_gid[gid])
            if owner < 0 or owner >= int(world_size):
                raise ValueError(f"gid={gid} owner={owner} is outside world_size={world_size}")
            owner_by_gid[int(gid)] = owner
    elif strategy_norm in {"round_robin", "rr"}:
        for idx, gid in enumerate(gids_norm):
            owner_by_gid[int(gid)] = int(idx % int(world_size))
        strategy_norm = "round_robin"
    elif strategy_norm in {"contiguous", "block"}:
        if not gids_norm:
            owner_by_gid = {}
        else:
            block_size = max(1, (len(gids_norm) + int(world_size) - 1) // int(world_size))
            for idx, gid in enumerate(gids_norm):
                owner_by_gid[int(gid)] = min(int(world_size) - 1, int(idx // block_size))
        strategy_norm = "contiguous"
    else:
        raise ValueError(
            f"Unsupported ownership strategy '{strategy}'. "
            "Use 'round_robin', 'contiguous', or provide explicit_owner_by_gid."
        )

    return OwnershipPlan(
        gids=gids_norm,
        world_size=int(world_size),
        rank=int(rank),
        strategy=strategy_norm,
        owner_by_gid=owner_by_gid,
    )


def ownership_from_cfg(
    gids: Sequence[Any],
    cfg: Mapping[str, Any] | None,
    *,
    world_size: int = 1,
    rank: int = 0,
) -> OwnershipPlan:
    cfg_use = dict(cfg or {})
    parallel_cfg = cfg_use.get("parallel") or {}
    if not isinstance(parallel_cfg, Mapping):
        parallel_cfg = {}

    strategy = parallel_cfg.get("ownership_strategy", parallel_cfg.get("cell_ownership", "round_robin"))
    explicit = parallel_cfg.get("owner_by_gid")
    if explicit is not None and not isinstance(explicit, Mapping):
        raise ValueError("CONFIG['parallel']['owner_by_gid'] must be a mapping when provided.")

    return build_cell_ownership(
        gids,
        world_size=int(world_size),
        rank=int(rank),
        strategy=str(strategy),
        explicit_owner_by_gid=explicit,
    )
