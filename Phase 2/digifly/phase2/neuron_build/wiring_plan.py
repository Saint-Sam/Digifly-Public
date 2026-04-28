from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from .ownership import OwnershipPlan, build_cell_ownership


def _normalize_gid_tuple(values: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(sorted({int(v) for v in values}))


def _normalize_drivers(drivers: Mapping[Any, Mapping[str, Any]] | None) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(drivers, Mapping):
        return out
    for gid, spec in drivers.items():
        out[int(gid)] = dict(spec or {})
    return out


def _pre_site_hint_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    if all(k in row for k in ("pre_x", "pre_y", "pre_z")):
        return {
            "x": float(row["pre_x"]),
            "y": float(row["pre_y"]),
            "z": float(row["pre_z"]),
        }
    if all(k in row for k in ("x_pre", "y_pre", "z_pre")):
        return {
            "x": float(row["x_pre"]),
            "y": float(row["y_pre"]),
            "z": float(row["z_pre"]),
        }
    return {}


@dataclass
class ConnectionPlan:
    pre_id: int
    post_id: int
    weight_uS: float
    delay_ms: float
    syn_e_rev_mV: float
    tau1_ms: float
    tau2_ms: float
    post_site: Dict[str, Any]
    geom_delay: bool = True
    pre_site_hint: Dict[str, float] = field(default_factory=dict)


@dataclass
class NetworkBuildPlan:
    ownership: OwnershipPlan
    swc_paths: Dict[int, str]
    loaded_gids: Tuple[int, ...]
    active_post_gids: Tuple[int, ...]
    driver_specs: Dict[int, Dict[str, Any]]
    connection_plans: Tuple[ConnectionPlan, ...]

    def local_loaded_gids(self) -> Tuple[int, ...]:
        return tuple(gid for gid in self.loaded_gids if self.ownership.is_local(gid))

    def remote_loaded_gids(self) -> Tuple[int, ...]:
        return tuple(gid for gid in self.loaded_gids if not self.ownership.is_local(gid))

    def local_active_post_gids(self) -> Tuple[int, ...]:
        return tuple(gid for gid in self.active_post_gids if self.ownership.is_local(gid))

    def local_driver_gids(self) -> Tuple[int, ...]:
        return tuple(sorted(gid for gid in self.driver_specs if self.ownership.is_local(gid)))

    def local_connection_plans(self) -> Tuple[ConnectionPlan, ...]:
        return tuple(cp for cp in self.connection_plans if self.ownership.is_local(cp.post_id))


def build_network_plan(
    cfg: Mapping[str, Any],
    *,
    node_ids: Sequence[Any],
    df_wire,
    swc_lookup: Callable[[int], str],
    drivers: Mapping[Any, Mapping[str, Any]] | None = None,
    active_posts: Sequence[Any] | None = None,
    ownership: OwnershipPlan | None = None,
) -> NetworkBuildPlan:
    loaded_gids = _normalize_gid_tuple(node_ids)
    driver_specs = _normalize_drivers(drivers)
    active_post_gids = _normalize_gid_tuple(active_posts or [])
    ownership_use = ownership or build_cell_ownership(loaded_gids, world_size=1, rank=0)

    swc_paths = {int(gid): str(swc_lookup(int(gid))) for gid in loaded_gids}

    force_soma = bool(cfg.get("wire_force_soma", False))
    use_geom_delay = bool(cfg.get("use_geom_delay", True))

    has_post_syn_index = "post_syn_index" in df_wire.columns
    has_post_xyz = all(c in df_wire.columns for c in ("post_x", "post_y", "post_z"))
    has_xpost = all(c in df_wire.columns for c in ("x_post", "y_post", "z_post"))
    use_post_x_keys = bool(has_post_xyz)
    use_xpost_keys = bool(has_xpost and not has_post_xyz)

    connection_plans = []
    for row in df_wire.itertuples(index=False):
        post_id = int(row.post_id)
        row_map: Dict[str, Any] = {"post_id": post_id}

        if force_soma:
            post_site = {"kind": "soma"}
        else:
            if has_post_syn_index:
                row_map["post_syn_index"] = getattr(row, "post_syn_index")
            if use_post_x_keys:
                row_map["post_x"] = getattr(row, "post_x")
                row_map["post_y"] = getattr(row, "post_y")
                row_map["post_z"] = getattr(row, "post_z")
            elif use_xpost_keys:
                row_map["x_post"] = getattr(row, "x_post")
                row_map["y_post"] = getattr(row, "y_post")
                row_map["z_post"] = getattr(row, "z_post")
            post_site = {"kind": "catalog", "row": row_map}

        row_dict = row._asdict()
        connection_plans.append(
            ConnectionPlan(
                pre_id=int(row.pre_id),
                post_id=post_id,
                weight_uS=float(getattr(row, "weight_uS")),
                delay_ms=float(getattr(row, "delay_ms")),
                syn_e_rev_mV=float(getattr(row, "syn_e_rev_mV")),
                tau1_ms=float(getattr(row, "tau1_ms")),
                tau2_ms=float(getattr(row, "tau2_ms")),
                post_site=post_site,
                geom_delay=use_geom_delay,
                pre_site_hint=_pre_site_hint_from_row(row_dict),
            )
        )

    return NetworkBuildPlan(
        ownership=ownership_use,
        swc_paths=swc_paths,
        loaded_gids=loaded_gids,
        active_post_gids=active_post_gids,
        driver_specs=driver_specs,
        connection_plans=tuple(connection_plans),
    )
