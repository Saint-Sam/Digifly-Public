from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .network import Network
from .biophys import make_passive, make_active, apply_biophys, set_hh
from .timing import geom_delay_ms, geom_delay_ms_from_xyz, timing_from_row, xyz_at_site
from .swc_cell import find_swc_with_fallback, pick_post_site
from .ownership import ownership_from_cfg
from .parallel import configure_parallel_context, distributed_gid_enabled
from .wiring_plan import NetworkBuildPlan, build_network_plan

def _maybe_tqdm(iterable, total=None, desc="", use_tqdm=True):
    if use_tqdm:
        try:
            from tqdm import tqdm
            return tqdm(iterable, total=total, desc=desc, leave=False)
        except Exception:
            pass
    return iterable


def _stage_log(msg: str) -> None:
    print(f"[build] {msg}")


def _biophys_override_index(cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    raw_groups = cfg.get("cell_biophys_overrides") or []
    if not isinstance(raw_groups, list):
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for item in raw_groups:
        if not isinstance(item, dict):
            continue
        ids = item.get("ids")
        if ids is None:
            ids = item.get("neuron_ids")
        try:
            gid_list = [int(x) for x in list(ids or [])]
        except Exception:
            gid_list = []
        if not gid_list:
            continue

        passive = dict(item.get("passive_global") or {})
        soma_hh = dict(item.get("soma_hh") or {})
        branch_hh = dict(item.get("branch_hh") or {})
        scalar_keys = [
            "active",
            "v_rest_mV",
            "v_init_mV",
            "ena_mV",
            "ek_mV",
            "el_mV",
        ]
        for gid in gid_list:
            merged = dict(out.get(int(gid)) or {})
            if passive:
                merged["passive_global"] = dict(merged.get("passive_global") or {})
                merged["passive_global"].update(passive)
            if soma_hh:
                merged["soma_hh"] = dict(merged.get("soma_hh") or {})
                merged["soma_hh"].update(soma_hh)
            if branch_hh:
                merged["branch_hh"] = dict(merged.get("branch_hh") or {})
                merged["branch_hh"].update(branch_hh)
            for key in scalar_keys:
                if key in item:
                    merged[key] = item[key]
            out[int(gid)] = merged
    return out


def _cell_biophys_spec(
    cfg: Dict[str, Any],
    gid: int,
    *,
    role: str,
    override_index: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
    cfg_use = dict(cfg)
    if str(role).strip().lower() == "pre":
        soma_hh = dict(cfg["pre_soma_hh"])
        branch_hh = dict(cfg["pre_branch_hh"])
        active_default = True
    else:
        soma_hh = dict(cfg.get("post_soma_hh", cfg["pre_soma_hh"]))
        branch_hh = dict(cfg.get("post_branch_hh", cfg["pre_branch_hh"]))
        active_default = bool(cfg.get("post_active", True))

    override = dict(override_index.get(int(gid)) or {})
    passive = dict(override.get("passive_global") or {})
    if "Ra" in passive:
        cfg_use["Ra"] = float(passive["Ra"])
    if "cm" in passive:
        cfg_use["cm"] = float(passive["cm"])
    if "g_pas" in passive:
        cfg_use["passive_g"] = float(passive["g_pas"])
    if "e_pas" in passive:
        cfg_use["passive_e"] = float(passive["e_pas"])

    if "v_rest_mV" in override:
        v_rest = float(override["v_rest_mV"])
        cfg_use["v_rest_mV"] = v_rest
        cfg_use["passive_e"] = v_rest
    if "v_init_mV" in override:
        cfg_use["v_init_mV"] = float(override["v_init_mV"])
    if "ena_mV" in override:
        cfg_use["ena_mV"] = float(override["ena_mV"])
    if "ek_mV" in override:
        cfg_use["ek_mV"] = float(override["ek_mV"])
    if "el_mV" in override:
        el_val = float(override["el_mV"])
        cfg_use["el_mV"] = el_val
        soma_hh["el"] = el_val
        branch_hh["el"] = el_val
    elif "v_rest_mV" in override:
        el_val = float(override["v_rest_mV"])
        soma_hh.setdefault("el", el_val)
        branch_hh.setdefault("el", el_val)

    soma_hh.update(dict(override.get("soma_hh") or {}))
    branch_hh.update(dict(override.get("branch_hh") or {}))
    active_flag = bool(override.get("active", active_default))
    return cfg_use, soma_hh, branch_hh, active_flag


def _first_present_col(df: pd.DataFrame, candidates: List[Optional[str]]) -> Optional[str]:
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _num_or_default(val: Any, default: Optional[float], *, required_msg: Optional[str] = None) -> float:
    try:
        x = float(val)
        if np.isnan(x):
            raise ValueError("nan")
        return x
    except Exception:
        if default is None:
            if required_msg:
                raise ValueError(required_msg)
            raise
        return float(default)


def _series_num(df: pd.DataFrame, col: Optional[str], default: Optional[float], *, required_msg: Optional[str] = None) -> pd.Series:
    if col is None:
        if default is None:
            if required_msg:
                raise ValueError(required_msg)
            raise ValueError("Missing required column with no default.")
        return pd.Series(float(default), index=df.index, dtype=float)

    s = pd.to_numeric(df[col], errors="coerce")
    if default is None:
        if s.isna().any():
            if required_msg:
                raise ValueError(required_msg)
            raise ValueError(f"Column {col} contains NaN and no default is provided.")
        return s.astype(float)
    return s.fillna(float(default)).astype(float)


def _make_site_key(df: pd.DataFrame, *, force_soma: bool, xyz_round: int = 3) -> pd.Series:
    if force_soma:
        return pd.Series("soma", index=df.index, dtype=object)

    key = pd.Series(np.nan, index=df.index, dtype=object)

    if "post_syn_index" in df.columns:
        sidx = pd.to_numeric(df["post_syn_index"], errors="coerce")
        ok = sidx.notna()
        if ok.any():
            key.loc[ok] = "i:" + sidx.loc[ok].astype(np.int64).astype(str)

    xyz_cols = None
    if all(c in df.columns for c in ("post_x", "post_y", "post_z")):
        xyz_cols = ("post_x", "post_y", "post_z")
    elif all(c in df.columns for c in ("x_post", "y_post", "z_post")):
        xyz_cols = ("x_post", "y_post", "z_post")

    if xyz_cols is not None:
        x = pd.to_numeric(df[xyz_cols[0]], errors="coerce")
        y = pd.to_numeric(df[xyz_cols[1]], errors="coerce")
        z = pd.to_numeric(df[xyz_cols[2]], errors="coerce")
        ok = key.isna() & x.notna() & y.notna() & z.notna()
        if ok.any():
            xr = x.round(xyz_round).astype(str)
            yr = y.round(xyz_round).astype(str)
            zr = z.round(xyz_round).astype(str)
            key.loc[ok] = "xyz:" + xr.loc[ok] + "," + yr.loc[ok] + "," + zr.loc[ok]

    fallback = pd.Series([f"row:{i}" for i in range(len(df))], index=df.index, dtype=object)
    return key.where(key.notna(), fallback)


def _prepare_wiring_df(
    df_sub: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    force_soma: bool,
    w_col: Optional[str],
    d_col: Optional[str],
    e_col: Optional[str],
    t1_col: Optional[str],
    t2_col: Optional[str],
    w_default: Optional[float],
    d_default: Optional[float],
    e_default: Optional[float],
    t1_default: Optional[float],
    t2_default: Optional[float],
    w_scale: float,
) -> pd.DataFrame:
    dfw = df_sub.copy()
    dfw["weight_uS"] = _series_num(
        dfw,
        w_col,
        w_default,
        required_msg="Edges row is missing weight and no CONFIG['default_weight_uS'] is set.",
    ) * float(w_scale)
    dfw["delay_ms"] = _series_num(dfw, d_col, d_default if d_default is not None else 0.0)
    dfw["syn_e_rev_mV"] = _series_num(
        dfw,
        e_col,
        e_default,
        required_msg="Edges row is missing Erev and no CONFIG['syn_e_rev_mV'] is set.",
    )
    dfw["tau1_ms"] = _series_num(
        dfw,
        t1_col,
        t1_default,
        required_msg="Edges row is missing tau1 and no CONFIG['syn_tau1_ms'] is set.",
    )
    dfw["tau2_ms"] = _series_num(
        dfw,
        t2_col,
        t2_default,
        required_msg="Edges row is missing tau2 and no CONFIG['syn_tau2_ms'] is set.",
    )

    w_floor = float(cfg.get("min_weight_uS", 0.0) or 0.0)
    if w_floor > 0:
        dfw["weight_uS"] = dfw["weight_uS"].clip(lower=w_floor)

    if not bool(cfg.get("coalesce_syns", False)):
        return dfw

    mode = str(cfg.get("coalesce_mode", "site")).strip().lower()
    if mode in {"none", "off", "false"}:
        return dfw

    site_cols = [c for c in ("post_syn_index", "post_x", "post_y", "post_z", "x_post", "y_post", "z_post") if c in dfw.columns]
    dfw["__site_key__"] = _make_site_key(dfw, force_soma=force_soma, xyz_round=int(cfg.get("coalesce_xyz_round", 3)))

    gkeys = ["pre_id", "post_id", "syn_e_rev_mV", "tau1_ms", "tau2_ms", "delay_ms"]
    if mode not in {"pair", "prepost"}:
        gkeys.append("__site_key__")

    agg: Dict[str, str] = {"weight_uS": "sum"}
    for c in site_cols:
        agg[c] = "first"

    before = len(dfw)
    grp = dfw.groupby(gkeys, sort=False, dropna=False).agg(agg).reset_index()
    after = len(grp)

    try:
        guard_drop = float(cfg.get("coalesce_guard_drop", 0.95))
    except Exception:
        guard_drop = 0.95
    try:
        guard_w = float(cfg.get("coalesce_guard_w_med_uS", 1e-8))
    except Exception:
        guard_w = 1e-8
    drop_frac = (before - after) / max(1, before)
    med_w = float(np.nanmedian(pd.to_numeric(grp["weight_uS"], errors="coerce"))) if after else 0.0
    if drop_frac >= guard_drop and med_w <= guard_w:
        _stage_log(
            f"coalesce guard fallback drop_frac={drop_frac:.3f} med_w={med_w:.3e} "
            "(keeping uncoalesced rows)"
        )
        return dfw

    if w_floor > 0:
        grp["weight_uS"] = pd.to_numeric(grp["weight_uS"], errors="coerce").fillna(w_floor).clip(lower=w_floor)

    _stage_log(
        f"coalesce mode={mode} rows {before:,}->{after:,} "
        f"drop={before - after:,} ({drop_frac*100.0:.1f}%)"
    )
    return grp


def _resolve_post_site_from_plan(net: Network, post_id: int, post_site: Dict[str, Any]) -> Tuple[Any, float]:
    post_cell = net.cells[int(post_id)]
    kind = str((post_site or {}).get("kind", "catalog")).strip().lower()
    if kind == "soma":
        return post_cell.soma_site()
    row = dict((post_site or {}).get("row") or {})
    return pick_post_site(post_cell, row, net.cfg["swc_dir"])


def _local_ais_xyz_map(net: Network, gids: Tuple[int, ...]) -> Dict[int, Tuple[float, float, float]]:
    out: Dict[int, Tuple[float, float, float]] = {}
    for gid in gids:
        cell = net.cells[int(gid)]
        xyz = xyz_at_site(cell.axon_ais_site())
        if xyz is not None:
            out[int(gid)] = xyz
    return out


def _merge_rank_dicts(items: List[Dict[int, Tuple[float, float, float]] | None]) -> Dict[int, Tuple[float, float, float]]:
    merged: Dict[int, Tuple[float, float, float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for gid, xyz in item.items():
            if xyz is None:
                continue
            merged[int(gid)] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return merged


def _materialize_serial_network_plan(cfg: Dict[str, Any], plan: NetworkBuildPlan) -> Network:
    net = Network(cfg, swc_paths=plan.swc_paths, ownership=plan.ownership)
    try:
        use_tqdm = bool(cfg.get("use_tqdm", True))
        local_loaded = plan.local_loaded_gids()
        override_index = _biophys_override_index(cfg)

        t_cells = time.perf_counter()
        for nid in _maybe_tqdm(
            local_loaded,
            total=len(local_loaded),
            desc="Loading cells",
            use_tqdm=use_tqdm,
        ):
            net.ensure_cell(int(nid))
        _stage_log(f"cell load done count={len(local_loaded)} wall_s={time.perf_counter() - t_cells:.3f}")

        t_passive = time.perf_counter()
        cells_all = [net.cells[int(gid)] for gid in local_loaded]
        for cell in _maybe_tqdm(
            cells_all,
            total=len(cells_all),
            desc="Applying passive",
            use_tqdm=use_tqdm,
        ):
            role = "pre" if int(cell.gid) in plan.driver_specs else "post"
            cell_cfg, _, _, _ = _cell_biophys_spec(
                cfg,
                int(cell.gid),
                role=role,
                override_index=override_index,
            )
            make_passive(cell, cell_cfg)
        _stage_log(f"passive biophys done count={len(cells_all)} wall_s={time.perf_counter() - t_passive:.3f}")

        t_active = time.perf_counter()
        activated = 0
        for pid in _maybe_tqdm(
            plan.local_active_post_gids(),
            total=len(plan.local_active_post_gids()),
            desc="Activating posts",
            use_tqdm=use_tqdm,
        ):
            if pid in net.cells:
                cell_cfg, soma_hh, branch_hh, active_flag = _cell_biophys_spec(
                    cfg,
                    int(pid),
                    role="post",
                    override_index=override_index,
                )
                if active_flag:
                    make_active(
                        net.cells[int(pid)],
                        cell_cfg,
                        soma_hh,
                        branch_hh,
                    )
                    activated += 1
        _stage_log(f"active biophys done count={activated} wall_s={time.perf_counter() - t_active:.3f}")

        t_wire = time.perf_counter()
        gt = cfg.get("global_timing", {}) or {}
        pre_ais_cache: Dict[int, Any] = {}
        connection_plans = plan.local_connection_plans()
        wired = 0
        wire_debug_every = int(cfg.get("wire_debug_every", 0) or 0)
        wire_log_first = bool(cfg.get("wire_log_first", False))
        _stage_log(f"serial synapse wiring start rows={len(connection_plans):,}")

        for idx, conn in enumerate(_maybe_tqdm(
            connection_plans,
            total=len(connection_plans),
            desc="Wiring synapses",
            use_tqdm=use_tqdm,
        )):
            if (wire_log_first and idx == 0) or (wire_debug_every > 0 and idx % wire_debug_every == 0):
                _stage_log(
                    "serial synapse wiring progress "
                    f"idx={idx:,}/{len(connection_plans):,} pre={int(conn.pre_id)} post={int(conn.post_id)}"
                )
            site = _resolve_post_site_from_plan(net, conn.post_id, conn.post_site)
            delay_ms = float(conn.delay_ms)
            if bool(conn.geom_delay):
                pre_ais = pre_ais_cache.get(int(conn.pre_id))
                if pre_ais is None:
                    pre_ais = net.cells[int(conn.pre_id)].axon_ais_site()
                    pre_ais_cache[int(conn.pre_id)] = pre_ais
                d_geom = geom_delay_ms(pre_ais, site, gt)
                if d_geom is not None and not np.isnan(d_geom):
                    delay_ms = float(d_geom)

            net.add_syn_exp2(
                int(conn.pre_id),
                int(conn.post_id),
                site,
                float(conn.weight_uS),
                delay_ms,
                e_rev=float(conn.syn_e_rev_mV),
                tau1=float(conn.tau1_ms),
                tau2=float(conn.tau2_ms),
            )
            wired += 1

        wire_s = time.perf_counter() - t_wire
        wire_rate = (wired / wire_s) if wire_s > 0 else float("nan")
        _stage_log(f"synapse wiring done rows={wired:,} wall_s={wire_s:.3f} rows_per_s={wire_rate:,.1f}")

        t_drv = time.perf_counter()
        n_drv = 0
        for pre_id in plan.local_driver_gids():
            spec = plan.driver_specs[int(pre_id)]
            cell = net.ensure_cell(int(pre_id))
            site = cell.axon_ais_site() if str(spec.get("site", "soma")).lower() == "ais" else cell.soma_site()
            cell_cfg, soma_hh, branch_hh, active_flag = _cell_biophys_spec(
                cfg,
                int(pre_id),
                role="pre",
                override_index=override_index,
            )
            if active_flag:
                make_active(cell, cell_cfg, soma_hh, branch_hh)
                try:
                    set_hh(
                        site[0],
                        **soma_hh,
                        Ra=cell_cfg["Ra"],
                        cm=cell_cfg["cm"],
                        ena=cell_cfg.get("ena_mV"),
                        ek=cell_cfg.get("ek_mV"),
                    )
                except Exception:
                    pass

            pulse_train = spec.get("pulse_train")
            train_delays = []
            train_enabled = False
            if isinstance(pulse_train, dict) and bool(pulse_train.get("enabled", False)):
                train_delays = [float(x) for x in (pulse_train.get("delays") or [])]
                train_enabled = len(train_delays) > 0

            if (not train_enabled) or bool((pulse_train or {}).get("include_base_iclamp", False)):
                net.add_iclamp_site(
                    int(pre_id),
                    site,
                    spec.get("amp", cfg["iclamp_amp_nA"]),
                    spec.get("delay", cfg["iclamp_delay_ms"]),
                    spec.get("dur", cfg["iclamp_dur_ms"]),
                    kind="base",
                )

            if train_enabled:
                tr_site = (
                    cell.axon_ais_site()
                    if str((pulse_train or {}).get("site", spec.get("site", "soma"))).lower() == "ais"
                    else cell.soma_site()
                )
                tr_amp = float((pulse_train or {}).get("amp", spec.get("amp", cfg["iclamp_amp_nA"])))
                tr_dur = float((pulse_train or {}).get("dur", spec.get("dur", cfg["iclamp_dur_ms"])))
                for tr_delay in train_delays:
                    net.add_iclamp_site(
                        int(pre_id),
                        tr_site,
                        tr_amp,
                        float(tr_delay),
                        tr_dur,
                        kind="pulse_train",
                    )

            neg_spec = spec.get("neg_pulse")
            if isinstance(neg_spec, dict) and bool(neg_spec.get("enabled", False)):
                neg_site = (
                    cell.axon_ais_site()
                    if str(neg_spec.get("site", spec.get("site", "soma"))).lower() == "ais"
                    else cell.soma_site()
                )
                net.add_iclamp_site(
                    int(pre_id),
                    neg_site,
                    neg_spec.get("amp", 0.0),
                    neg_spec.get("delay", 0.0),
                    neg_spec.get("dur", 0.0),
                    kind="neg_pulse",
                )
            n_drv += 1

        _stage_log(f"driver clamps done count={n_drv} wall_s={time.perf_counter() - t_drv:.3f}")
        return net
    except Exception:
        try:
            net.close()
        except Exception:
            pass
        raise


def _materialize_distributed_network_plan(
    cfg: Dict[str, Any],
    plan: NetworkBuildPlan,
    *,
    parallel_state: Dict[str, Any],
) -> Network:
    net = Network(
        cfg,
        swc_paths=plan.swc_paths,
        ownership=plan.ownership,
        parallel_state=parallel_state,
    )
    try:
        use_tqdm = bool(cfg.get("use_tqdm", True))
        local_loaded = plan.local_loaded_gids()
        override_index = _biophys_override_index(cfg)

        t_cells = time.perf_counter()
        for nid in _maybe_tqdm(
            local_loaded,
            total=len(local_loaded),
            desc="Loading local cells",
            use_tqdm=use_tqdm,
        ):
            net.ensure_cell(int(nid))
        _stage_log(
            "distributed cell load done "
            f"rank={plan.ownership.rank} count={len(local_loaded)} wall_s={time.perf_counter() - t_cells:.3f}"
        )

        t_passive = time.perf_counter()
        cells_all = [net.cells[int(gid)] for gid in local_loaded]
        for cell in _maybe_tqdm(
            cells_all,
            total=len(cells_all),
            desc="Applying passive",
            use_tqdm=use_tqdm,
        ):
            role = "pre" if int(cell.gid) in plan.driver_specs else "post"
            cell_cfg, _, _, _ = _cell_biophys_spec(
                cfg,
                int(cell.gid),
                role=role,
                override_index=override_index,
            )
            make_passive(cell, cell_cfg)
        _stage_log(
            "distributed passive biophys done "
            f"rank={plan.ownership.rank} count={len(cells_all)} wall_s={time.perf_counter() - t_passive:.3f}"
        )

        t_active = time.perf_counter()
        activated = 0
        local_active_posts = plan.local_active_post_gids()
        for pid in _maybe_tqdm(
            local_active_posts,
            total=len(local_active_posts),
            desc="Activating local posts",
            use_tqdm=use_tqdm,
        ):
            if pid in net.cells:
                cell_cfg, soma_hh, branch_hh, active_flag = _cell_biophys_spec(
                    cfg,
                    int(pid),
                    role="post",
                    override_index=override_index,
                )
                if active_flag:
                    make_active(
                        net.cells[int(pid)],
                        cell_cfg,
                        soma_hh,
                        branch_hh,
                    )
                    activated += 1
        _stage_log(
            "distributed active biophys done "
            f"rank={plan.ownership.rank} count={activated} wall_s={time.perf_counter() - t_active:.3f}"
        )

        t_gid = time.perf_counter()
        local_ais_xyz = _local_ais_xyz_map(net, local_loaded)
        gathered_xyz = net._pc.py_allgather(local_ais_xyz)
        global_ais_xyz = _merge_rank_dicts(gathered_xyz)
        for gid in local_loaded:
            net.register_output_gid(int(gid), site=net.cells[int(gid)].soma_site(), thresh=0.0)
        _stage_log(
            "distributed gid registration done "
            f"rank={plan.ownership.rank} local_gids={len(local_loaded)} "
            f"global_ais={len(global_ais_xyz)} wall_s={time.perf_counter() - t_gid:.3f}"
        )

        t_wire = time.perf_counter()
        gt = cfg.get("global_timing", {}) or {}
        connection_plans = plan.local_connection_plans()
        missing_geom = 0
        wired = 0
        wire_debug_every = int(cfg.get("wire_debug_every", 0) or 0)
        wire_log_first = bool(cfg.get("wire_log_first", False))
        _stage_log(
            "distributed synapse wiring start "
            f"rank={plan.ownership.rank} rows={len(connection_plans):,}"
        )
        for idx, conn in enumerate(_maybe_tqdm(
            connection_plans,
            total=len(connection_plans),
            desc="Wiring local synapses",
            use_tqdm=use_tqdm,
        )):
            if (wire_log_first and idx == 0) or (wire_debug_every > 0 and idx % wire_debug_every == 0):
                _stage_log(
                    "distributed synapse wiring progress "
                    f"rank={plan.ownership.rank} idx={idx:,}/{len(connection_plans):,} "
                    f"pre={int(conn.pre_id)} post={int(conn.post_id)}"
                )
            site = _resolve_post_site_from_plan(net, conn.post_id, conn.post_site)
            delay_ms = float(conn.delay_ms)
            if bool(conn.geom_delay):
                d_geom = geom_delay_ms_from_xyz(global_ais_xyz.get(int(conn.pre_id)), site, gt)
                if d_geom is not None and not np.isnan(d_geom):
                    delay_ms = float(d_geom)
                else:
                    missing_geom += 1

            net.add_syn_exp2_gid(
                int(conn.pre_id),
                int(conn.post_id),
                site,
                float(conn.weight_uS),
                delay_ms,
                e_rev=float(conn.syn_e_rev_mV),
                tau1=float(conn.tau1_ms),
                tau2=float(conn.tau2_ms),
            )
            wired += 1
        wire_s = time.perf_counter() - t_wire
        wire_rate = (wired / wire_s) if wire_s > 0 else float("nan")
        _stage_log(
            "distributed synapse wiring done "
            f"rank={plan.ownership.rank} rows={wired:,} missing_geom={missing_geom:,} "
            f"wall_s={wire_s:.3f} rows_per_s={wire_rate:,.1f}"
        )

        t_drv = time.perf_counter()
        n_drv = 0
        for pre_id in plan.local_driver_gids():
            spec = plan.driver_specs[int(pre_id)]
            cell = net.ensure_cell(int(pre_id))
            site = cell.axon_ais_site() if str(spec.get("site", "soma")).lower() == "ais" else cell.soma_site()
            cell_cfg, soma_hh, branch_hh, active_flag = _cell_biophys_spec(
                cfg,
                int(pre_id),
                role="pre",
                override_index=override_index,
            )
            if active_flag:
                make_active(cell, cell_cfg, soma_hh, branch_hh)
                try:
                    set_hh(
                        site[0],
                        **soma_hh,
                        Ra=cell_cfg["Ra"],
                        cm=cell_cfg["cm"],
                        ena=cell_cfg.get("ena_mV"),
                        ek=cell_cfg.get("ek_mV"),
                    )
                except Exception:
                    pass

            pulse_train = spec.get("pulse_train")
            train_delays = []
            train_enabled = False
            if isinstance(pulse_train, dict) and bool(pulse_train.get("enabled", False)):
                train_delays = [float(x) for x in (pulse_train.get("delays") or [])]
                train_enabled = len(train_delays) > 0

            if (not train_enabled) or bool((pulse_train or {}).get("include_base_iclamp", False)):
                net.add_iclamp_site(
                    int(pre_id),
                    site,
                    spec.get("amp", cfg["iclamp_amp_nA"]),
                    spec.get("delay", cfg["iclamp_delay_ms"]),
                    spec.get("dur", cfg["iclamp_dur_ms"]),
                    kind="base",
                )

            if train_enabled:
                tr_site = (
                    cell.axon_ais_site()
                    if str((pulse_train or {}).get("site", spec.get("site", "soma"))).lower() == "ais"
                    else cell.soma_site()
                )
                tr_amp = float((pulse_train or {}).get("amp", spec.get("amp", cfg["iclamp_amp_nA"])))
                tr_dur = float((pulse_train or {}).get("dur", spec.get("dur", cfg["iclamp_dur_ms"])))
                for tr_delay in train_delays:
                    net.add_iclamp_site(
                        int(pre_id),
                        tr_site,
                        tr_amp,
                        float(tr_delay),
                        tr_dur,
                        kind="pulse_train",
                    )

            neg_spec = spec.get("neg_pulse")
            if isinstance(neg_spec, dict) and bool(neg_spec.get("enabled", False)):
                neg_site = (
                    cell.axon_ais_site()
                    if str(neg_spec.get("site", spec.get("site", "soma"))).lower() == "ais"
                    else cell.soma_site()
                )
                net.add_iclamp_site(
                    int(pre_id),
                    neg_site,
                    neg_spec.get("amp", 0.0),
                    neg_spec.get("delay", 0.0),
                    neg_spec.get("dur", 0.0),
                    kind="neg_pulse",
                )
            n_drv += 1

        _stage_log(
            "distributed driver clamps done "
            f"rank={plan.ownership.rank} count={n_drv} wall_s={time.perf_counter() - t_drv:.3f}"
        )
        return net
    except Exception:
        try:
            net.close()
        except Exception:
            pass
        raise

def build_network_driven_subset(
    cfg: Dict[str, Any],
    drivers=None,
    active_posts=None,
    loaded_nodes=None,
    ignore_distance_filter: bool = False,
):
    t_total_start = time.perf_counter()
    drivers = drivers or {}
    pres = set(int(k) for k in drivers.keys())
    posts = set(int(x) for x in (active_posts or []))
    nodes = set(int(x) for x in (loaded_nodes or [])) | pres | posts
    if not nodes:
        raise ValueError("build_network_driven_subset requires at least one driver or active post neuron id")
    _stage_log(
        f"start nodes={len(nodes)} drivers={len(pres)} candidate_posts={len(posts)} edges_csv={cfg['edges_csv']}"
    )

    t_read = time.perf_counter()
    df = pd.read_csv(cfg["edges_csv"])
    _stage_log(f"read edges rows={len(df):,} wall_s={time.perf_counter() - t_read:.3f}")
    for c in (
        "pre_id", "post_id", "pre_syn_index", "post_syn_index",
        "pre_match_um", "post_match_um", "weight_uS", "weight", "delay_ms",
        "e_rev_mV", "tau1_ms", "tau2_ms",
        "post_x", "post_y", "post_z", "x_post", "y_post", "z_post",
        "pre_x", "pre_y", "pre_z", "x_pre", "y_pre", "z_pre",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"pre_match_um","post_match_um"}.issubset(df.columns) and not ignore_distance_filter:
        t_eps = time.perf_counter()
        before = len(df)
        eps = float(cfg["epsilon_um"])
        df = df[(df.pre_match_um <= eps) & (df.post_match_um <= eps)].copy()
        _stage_log(
            f"epsilon filter eps_um={eps} kept={len(df):,}/{before:,} wall_s={time.perf_counter() - t_eps:.3f}"
        )

    # Default to wiring all edges among loaded nodes (closer to Master behavior).
    # If needed, keep legacy seed-only presynaptic filtering via:
    #   cfg["custom_pre_filter"] = "drivers"
    pre_filter = str(cfg.get("custom_pre_filter", "all_loaded")).strip().lower()
    if pre_filter in {"drivers", "seed_only", "seeds", "drivers_only"}:
        allowed_pres = set(pres)
    else:
        allowed_pres = set(nodes)

    t_sub = time.perf_counter()
    df_sub = df[df["pre_id"].astype(int).isin(allowed_pres) & df["post_id"].astype(int).isin(nodes)].copy()
    _stage_log(
        f"subgraph filter kept={len(df_sub):,}/{len(df):,} rows wall_s={time.perf_counter() - t_sub:.3f}"
    )

    node_list = sorted(nodes)
    force_soma = bool(cfg.get("wire_force_soma", False))
    gt = cfg.get("global_timing", {}) or {}

    allow_legacy_weight = bool(
        cfg.get("allow_legacy_weight_column", False)
        or gt.get("allow_legacy_weight_column", False)
    )
    weight_candidates = [gt.get("weight_col"), "weight_uS", "w_uS"]
    if allow_legacy_weight:
        weight_candidates.append("weight")
    delay_candidates = [gt.get("delay_col"), "delay_ms", "delay", "d_ms"]
    erev_candidates = [gt.get("erev_col"), "syn_e_rev_mV", "erev_mV", "e_rev_mV", "syn_erev_mV"]
    tau1_candidates = [gt.get("tau1_col"), "tau1_ms", "tau1"]
    tau2_candidates = [gt.get("tau2_col"), "tau2_ms", "tau2"]

    w_col = _first_present_col(df_sub, weight_candidates)
    d_col = _first_present_col(df_sub, delay_candidates)
    e_col = _first_present_col(df_sub, erev_candidates)
    t1_col = _first_present_col(df_sub, tau1_candidates)
    t2_col = _first_present_col(df_sub, tau2_candidates)

    w_default = cfg.get("default_weight_uS", None)
    d_default = cfg.get("default_delay_ms", None)
    e_default = cfg.get("syn_e_rev_mV", None)
    t1_default = cfg.get("syn_tau1_ms", None)
    t2_default = cfg.get("syn_tau2_ms", None)

    if w_col is None and w_default is None:
        raise ValueError("Edges row is missing weight. Provide 'weight_uS' or set CONFIG['default_weight_uS'].")
    if e_col is None and e_default is None:
        raise ValueError("Edges row is missing Erev. Provide Erev column or set CONFIG['syn_e_rev_mV'].")
    if t1_col is None and t1_default is None:
        raise ValueError("Edges row is missing tau1. Provide 'tau1_ms' or set CONFIG['syn_tau1_ms'].")
    if t2_col is None and t2_default is None:
        raise ValueError("Edges row is missing tau2. Provide 'tau2_ms' or set CONFIG['syn_tau2_ms'].")

    try:
        w_scale = float(gt.get("global_weight_scale", 1.0))
    except Exception:
        w_scale = 1.0

    df_wire = _prepare_wiring_df(
        df_sub,
        cfg,
        force_soma=force_soma,
        w_col=w_col,
        d_col=d_col,
        e_col=e_col,
        t1_col=t1_col,
        t2_col=t2_col,
        w_default=w_default,
        d_default=d_default,
        e_default=e_default,
        t1_default=t1_default,
        t2_default=t2_default,
        w_scale=w_scale,
    )

    morph_root = cfg.get("morph_swc_dir")
    base_swc_root = cfg["swc_dir"]
    parallel_state = configure_parallel_context(cfg)
    if distributed_gid_enabled(cfg, state=parallel_state):
        ownership_world = int(parallel_state.get("nhost", 1))
        ownership_rank = int(parallel_state.get("id", 0))
    else:
        ownership_world = 1
        ownership_rank = 0
    ownership = ownership_from_cfg(
        node_list,
        cfg,
        world_size=ownership_world,
        rank=ownership_rank,
    )
    plan = build_network_plan(
        cfg,
        node_ids=node_list,
        df_wire=df_wire,
        swc_lookup=lambda nid: find_swc_with_fallback(morph_root, base_swc_root, int(nid)),
        drivers=drivers,
        active_posts=sorted(posts),
        ownership=ownership,
    )
    _stage_log(
        "cell ownership plan "
        f"strategy={plan.ownership.strategy} world={plan.ownership.world_size} "
        f"rank={plan.ownership.rank} local={len(plan.local_loaded_gids())} "
        f"remote={len(plan.remote_loaded_gids())} conns={len(plan.connection_plans):,}"
    )

    if distributed_gid_enabled(cfg, state=parallel_state):
        net = _materialize_distributed_network_plan(cfg, plan, parallel_state=parallel_state)
    else:
        net = _materialize_serial_network_plan(cfg, plan)
    setattr(net, "_build_plan", plan)
    _stage_log(
        "build_network_driven_subset total "
        f"rank={plan.ownership.rank} wall_s={time.perf_counter() - t_total_start:.3f}"
    )
    return net, df_wire

def expand_from_edges(net: Network, extra_edges_csv: str | Path):
    cfg = net.cfg
    df2 = pd.read_csv(extra_edges_csv)
    for c in (
        "pre_id", "post_id", "pre_syn_index", "post_syn_index",
        "pre_match_um", "post_match_um", "weight_uS", "weight", "delay_ms",
        "e_rev_mV", "tau1_ms", "tau2_ms",
        "post_x", "post_y", "post_z", "x_post", "y_post", "z_post",
        "pre_x", "pre_y", "pre_z", "x_pre", "y_pre", "z_pre",
    ):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    if {"pre_match_um","post_match_um"}.issubset(df2.columns):
        eps = float(cfg["epsilon_um"])
        df2 = df2[(df2.pre_match_um <= eps) & (df2.post_match_um <= eps)].copy()

    ids = np.unique(pd.concat([df2["pre_id"], df2["post_id"]], axis=0).dropna().astype(int))
    for nid in _maybe_tqdm([int(n) for n in ids], total=len(ids), desc="Loading new cells", use_tqdm=cfg.get("use_tqdm", True)):
        net.ensure_cell(nid)

    force_soma = bool(cfg.get("wire_force_soma", False))
    for r in _maybe_tqdm(list(df2.itertuples(index=False)), total=len(df2), desc="Wiring new synapses", use_tqdm=cfg.get("use_tqdm", True)):
        row = r._asdict()
        pre = int(row["pre_id"]); post = int(row["post_id"])
        site = net.cells[post].soma_site() if force_soma else pick_post_site(net.cells[post], row, cfg["swc_dir"])
        w = float(row["weight_uS"]) if "weight_uS" in row and pd.notna(row["weight_uS"]) else None
        d = float(row["delay_ms"]) if "delay_ms" in row and pd.notna(row["delay_ms"]) else None
        net.add_syn_exp2(pre, post, site, w, d)
    return df2

def build_pair_only(cfg: Dict[str, Any]):
    df = pd.read_csv(cfg["edges_csv"])
    for c in (
        "pre_id", "post_id", "pre_syn_index", "post_syn_index",
        "pre_match_um", "post_match_um", "weight_uS", "weight", "delay_ms",
        "e_rev_mV", "tau1_ms", "tau2_ms",
        "post_x", "post_y", "post_z", "x_post", "y_post", "z_post",
        "pre_x", "pre_y", "pre_z", "x_pre", "y_pre", "z_pre",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pre = int(cfg["pre_id"]); post = int(cfg["post_id"])
    df = df[(df.pre_id == pre) & (df.post_id == post)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for {pre}->{post} in {cfg['edges_csv']}")

    if {"pre_match_um","post_match_um"}.issubset(df.columns):
        df = df[(df.pre_match_um <= float(cfg["epsilon_um"])) & (df.post_match_um <= float(cfg["epsilon_um"]))].copy()

    net = Network(cfg)
    cpre = net.ensure_cell(pre)
    cpost = net.ensure_cell(post)
    apply_biophys(cpre, cpost, cfg)

    force_soma = bool(cfg.get("wire_force_soma", False))
    for r in _maybe_tqdm(list(df.itertuples(index=False)), total=len(df), desc="Wiring pair", use_tqdm=cfg.get("use_tqdm", True)):
        row = r._asdict()
        site = cpost.soma_site() if force_soma else pick_post_site(cpost, row, cfg["swc_dir"])
        w, d, e, t1, t2 = timing_from_row(row, cfg)
        if bool(cfg.get("use_geom_delay", True)):
            pre_ais = net.ensure_cell(int(row["pre_id"])).axon_ais_site()
            d_geom = geom_delay_ms(pre_ais, site, cfg.get("global_timing", {}))
            if d_geom is not None and not np.isnan(d_geom):
                d = float(d_geom)
        net.add_syn_exp2(int(row["pre_id"]), int(row["post_id"]), site, w, d, e_rev=e, tau1=t1, tau2=t2)

    return net, df

def run_pair_demo(cfg: Dict[str, Any]):
    net, df = build_pair_only(cfg)
    pre = int(cfg["pre_id"]); post = int(cfg["post_id"])

    cpre = net.ensure_cell(pre)
    ais_site = cpre.axon_ais_site()
    try:
        set_hh(ais_site[0], **cfg["pre_soma_hh"], Ra=cfg["Ra"], cm=cfg["cm"])
    except Exception:
        pass
    net.add_iclamp_site(pre, ais_site, cfg["iclamp_amp_nA"], cfg["iclamp_delay_ms"], cfg["iclamp_dur_ms"])

    net.record_time()
    net.record_soma(pre)
    net.record_soma(post)
    net.run(cfg["tstop_ms"], cfg["dt_ms"], show_progress=True)

    return net, df
