from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from neuron import h

from .swc_cell import pick_post_site, pick_pre_site


def gap_pair_ohmic(
    a_id: int,
    b_id: int,
    *,
    g_uS: float = 0.001,
    site_a: str = "ais",
    site_b: str = "ais",
) -> Dict[str, Any]:
    return {
        "mode": "ohmic",
        "a_id": int(a_id),
        "b_id": int(b_id),
        "g_uS": float(g_uS),
        "site_a": str(site_a),
        "site_b": str(site_b),
    }


def gap_pair_rectifying(
    a_id: int,
    b_id: int,
    *,
    direction: str = "a_to_b",
    g_uS: float = 0.001,
    site_a: str = "ais",
    site_b: str = "ais",
) -> Dict[str, Any]:
    return {
        "mode": "rectifying",
        "a_id": int(a_id),
        "b_id": int(b_id),
        "direction": str(direction),
        "g_uS": float(g_uS),
        "site_a": str(site_a),
        "site_b": str(site_b),
    }


def gap_pair_heterotypic_rectifying(
    a_id: int,
    b_id: int,
    *,
    preferred_direction: str = "a_to_b",
    g_uS: float = 0.001,
    g_closed_frac: float = 0.0,
    vhalf_mV: float = 0.0,
    vslope_mV: float = 5.0,
    site_a: str = "ais",
    site_b: str = "ais",
) -> Dict[str, Any]:
    return {
        "mode": "heterotypic_rectifying",
        "a_id": int(a_id),
        "b_id": int(b_id),
        "preferred_direction": str(preferred_direction),
        "g_uS": float(g_uS),
        "g_closed_frac": float(g_closed_frac),
        "vhalf_mV": float(vhalf_mV),
        "vslope_mV": float(vslope_mV),
        "site_a": str(site_a),
        "site_b": str(site_b),
    }


def gap_pair_directed(
    source_id: int,
    target_id: int,
    *,
    g_uS: float = 0.001,
    source_site: str = "ais",
    target_site: str = "ais",
) -> Dict[str, Any]:
    return {
        "mode": "rectifying",
        "source_id": int(source_id),
        "target_id": int(target_id),
        "g_uS": float(g_uS),
        "source_site": str(source_site),
        "target_site": str(target_site),
    }


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate_mechanism_roots(mechanisms_dir: str | None) -> List[Path]:
    roots: List[Path] = []
    if mechanisms_dir:
        roots.append(Path(mechanisms_dir).expanduser())

    env_dir = os.environ.get("DIGIFLY_GAP_MECH_DIR", "").strip()
    if env_dir:
        roots.append(Path(env_dir).expanduser())

    # Repo-local fallbacks to make migrated layouts work by default.
    try:
        phase2_root = Path(__file__).resolve().parents[3]
        roots.append(phase2_root / "data")
        roots.append(phase2_root)
    except Exception:
        pass

    out: List[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except Exception:
            out.append(root)
    return _dedupe_paths(out)


def _candidate_mechanism_dirs(mechanisms_dir: str | None) -> List[Path]:
    roots = _candidate_mechanism_roots(mechanisms_dir)
    dirs: List[Path] = []
    for root in roots:
        dirs.append(root)
        for sub in ("arm64", "x86_64"):
            p = root / sub
            if p.exists() and p.is_dir():
                dirs.append(p)
    return _dedupe_paths(dirs)


def _compile_gap_mechanisms(root: Path) -> str | None:
    """
    Best-effort compile in `root` if gap-related MOD sources exist.
    Returns a diagnostic message (or None if no source was found at this root).
    """
    mod_dir = root / "mod"
    mod_dir_mods = sorted(mod_dir.glob("*.mod")) if mod_dir.exists() else []
    root_mods = sorted(root.glob("*.mod"))
    if not mod_dir_mods and not root_mods:
        return None

    nrnivmodl: str | None = None
    runtime_data_root: Path | None = None

    try:
        import neuron  # type: ignore

        runtime_pkg_root = Path(neuron.__file__).resolve().parent
        runtime_data_root = (runtime_pkg_root / ".data").resolve()
        candidate = (runtime_data_root / "bin" / "nrnivmodl").resolve()
        if candidate.exists():
            nrnivmodl = str(candidate)
    except Exception:
        runtime_data_root = None

    if nrnivmodl is None:
        py_bin_candidate = (Path(sys.executable).resolve().parent / "nrnivmodl").resolve()
        if py_bin_candidate.exists():
            nrnivmodl = str(py_bin_candidate)

    if nrnivmodl is None:
        which_candidate = shutil.which("nrnivmodl")
        if which_candidate:
            nrnivmodl = which_candidate

    if nrnivmodl is None:
        app_candidate = Path("/Applications/NEURON/bin/nrnivmodl")
        if app_candidate.exists():
            nrnivmodl = str(app_candidate)

    if not nrnivmodl:
        return f"{root}: skipped compile (nrnivmodl not found)"

    compile_env = os.environ.copy()
    nrnivmodl_path = Path(str(nrnivmodl)).expanduser().resolve()
    wheel_data_root = runtime_data_root
    if wheel_data_root is None and nrnivmodl_path.name == "nrnivmodl":
        parent_root = nrnivmodl_path.parent.parent
        if (parent_root / "bin").exists():
            wheel_data_root = parent_root
    if wheel_data_root is not None and (wheel_data_root / "bin" / "nrnivmech_makefile").exists():
        compile_env["NRNHOME"] = str(wheel_data_root)
    elif wheel_data_root is not None and (wheel_data_root / "bin" / "nrnmech_makefile").exists():
        compile_env["NRNHOME"] = str(wheel_data_root)

    if mod_dir_mods:
        cmd = [nrnivmodl, "mod"]
    else:
        cmd = [nrnivmodl, *[p.name for p in root_mods]]
    try:
        subprocess.run(
            cmd,
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=compile_env,
        )
        return f"{root}: compiled via {' '.join(cmd)}"
    except Exception as e:
        return f"{root}: compile failed ({e})"


def ensure_gap_mechanism_available(
    mechanisms_dir: str | None = None,
    *,
    require_rectifying: bool = False,
    require_heterotypic: bool = False,
) -> None:
    required = ["Gap"]
    if require_rectifying:
        required.append("RectGap")
    if require_heterotypic:
        required.append("HeteroRectGap")

    if all(hasattr(h, mech) for mech in required):
        return

    try:
        from neuron import load_mechanisms
    except Exception as e:
        raise RuntimeError(
            f"NEURON load_mechanisms is unavailable; cannot load gap mechanisms: {e}"
        )

    candidates = _candidate_mechanism_dirs(mechanisms_dir)
    roots = _candidate_mechanism_roots(mechanisms_dir)
    errors: List[str] = []
    compile_msgs: List[str] = []

    def _try_load(cands: List[Path]) -> bool:
        for c in cands:
            if not c.exists() or not c.is_dir():
                continue
            try:
                load_mechanisms(str(c))
            except Exception as e:
                errors.append(f"{c}: {e}")
            if all(hasattr(h, mech) for mech in required):
                print(f"[gap] mechanisms loaded from {c}")
                return True
        return False

    # Best-effort source compile first. On mixed macOS NEURON installs, an
    # initial load attempt against a stale libnrnmech can mark the path as
    # "already loaded" for the rest of the process even if a later compile
    # fixes the binary on disk.
    for r in roots:
        msg = _compile_gap_mechanisms(r)
        if msg:
            compile_msgs.append(msg)

    if _try_load(candidates):
        return

    # Final pass after any successful compile may have created arch dirs.
    final_candidates = _candidate_mechanism_dirs(mechanisms_dir)
    if _try_load(final_candidates):
        return

    search_msg = ", ".join(str(p) for p in final_candidates) if final_candidates else "(none)"
    detail_parts: List[str] = []
    if errors:
        detail_parts.append("; ".join(errors))
    if compile_msgs:
        detail_parts.append("compile: " + " | ".join(compile_msgs))
    err_msg = " ; ".join(detail_parts) if detail_parts else "no load/compile attempt succeeded"
    raise RuntimeError(
        f"Required gap mechanism(s) {required} are not available in this NEURON session. "
        "Set cfg['gap']['mechanisms_dir'] (or DIGIFLY_GAP_MECH_DIR) to a folder containing compiled mechanisms "
        "or gap-related .mod source files (auto-compile will be attempted via nrnivmodl). "
        f"Searched: {search_msg}. Details: {err_msg}"
    )


def _norm_site(site: Any, default_site: str) -> str:
    return str(site if site is not None else default_site).strip().lower()


def _read_edges_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported edges format for gap placement: {path}")


def _load_edges_for_gap(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw = cfg.get("edges_csv") or cfg.get("edges_path")
    if not raw:
        raise ValueError(
            "Synapse-based gap placement requires edges data. "
            "Set CONFIG['edges_path'] (custom mode) or ensure cfg['edges_csv'] exists."
        )

    path = Path(str(raw)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gap placement edges file not found: {path}")

    df = _read_edges_table(path)
    for c in (
        "pre_id", "post_id", "pre_syn_index", "post_syn_index",
        "pre_match_um", "post_match_um",
        "post_x", "post_y", "post_z", "x_post", "y_post", "z_post",
        "pre_x", "pre_y", "pre_z", "x_pre", "y_pre", "z_pre",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"pre_match_um", "post_match_um"}.issubset(df.columns):
        eps = cfg.get("epsilon_um", None)
        if eps is not None:
            eps = float(eps)
            df = df[(df.pre_match_um <= eps) & (df.post_match_um <= eps)].copy()

    if "pre_id" not in df.columns or "post_id" not in df.columns:
        raise ValueError(
            f"Gap placement edges file must include pre_id/post_id columns: {path}"
        )
    return df


def _rows_for_pair(df: pd.DataFrame, a_id: int, b_id: int, include_reverse: bool) -> pd.DataFrame:
    a = int(a_id)
    b = int(b_id)
    m_ab = (df["pre_id"].astype(int) == a) & (df["post_id"].astype(int) == b)
    if include_reverse:
        m_ba = (df["pre_id"].astype(int) == b) & (df["post_id"].astype(int) == a)
        return df[m_ab | m_ba].copy()
    return df[m_ab].copy()


def _site_for_cell_from_row(net, cfg: Dict[str, Any], row: Dict[str, Any], cell_id: int, fallback_site: str):
    cid = int(cell_id)
    pre_id = int(row["pre_id"]) if "pre_id" in row and pd.notna(row["pre_id"]) else None
    post_id = int(row["post_id"]) if "post_id" in row and pd.notna(row["post_id"]) else None
    if bool(getattr(net, "is_distributed", False)):
        if pre_id == cid:
            return {
                "__gap_site_kind__": "pre_row",
                "fallback_site": str(fallback_site),
                "row": dict(row),
            }
        if post_id == cid:
            return {
                "__gap_site_kind__": "post_row",
                "fallback_site": str(fallback_site),
                "row": dict(row),
            }
        return {
            "__gap_site_kind__": "named_site",
            "site": str(fallback_site),
            "fallback_site": str(fallback_site),
        }

    cell = net.ensure_cell(cid)
    if pre_id == cid:
        return pick_pre_site(cell, row, cfg["swc_dir"])
    if post_id == cid:
        return pick_post_site(cell, row, cfg["swc_dir"])
    return net._resolve_site(cid, fallback_site)


def _pair_uses_synapse_placement(pair: Dict[str, Any]) -> bool:
    placement = str(pair.get("placement", "named_site")).strip().lower()
    if placement == "synapse":
        return True
    for k in ("site_a", "site_b", "source_site", "target_site"):
        if str(pair.get(k, "")).strip().lower() == "synapse":
            return True
    return False


def apply_gap_config(net, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    cfg['gap'] schema:
      enabled: bool
      mechanisms_dir: str | None
      default_site: 'ais' | 'soma'
      default_g_uS: float
      pairs: list[dict]

    Pair forms:
      1) Ohmic (bidirectional):
         {mode:'ohmic', a_id:int, b_id:int, g_uS:float, site_a:'ais|soma', site_b:'ais|soma'}

      2) Rectifying by a/b + direction:
         {mode:'rectifying', a_id:int, b_id:int, direction:'a_to_b|b_to_a', ...}

      3) Rectifying directed:
         {mode:'rectifying', source_id:int, target_id:int, ...}
    """
    gap_cfg = dict(cfg.get("gap") or {})
    if not bool(gap_cfg.get("enabled", False)):
        return []

    pairs = gap_cfg.get("pairs") or []
    if not isinstance(pairs, list):
        raise ValueError("cfg['gap']['pairs'] must be a list of pair dictionaries.")
    if not pairs:
        print("[gap] enabled=True but no pairs were provided; skipping.")
        return []

    require_rectifying = any(
        isinstance(pair, dict) and str(pair.get("mode", "ohmic")).strip().lower() == "rectifying"
        for pair in pairs
    )
    require_heterotypic = any(
        isinstance(pair, dict) and str(pair.get("mode", "ohmic")).strip().lower() == "heterotypic_rectifying"
        for pair in pairs
    )
    ensure_gap_mechanism_available(
        gap_cfg.get("mechanisms_dir"),
        require_rectifying=require_rectifying,
        require_heterotypic=require_heterotypic,
    )
    default_site = str(gap_cfg.get("default_site", "ais")).strip().lower()
    default_g_uS = float(gap_cfg.get("default_g_uS", 0.001))
    edges_df: pd.DataFrame | None = None

    added: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"gap.pairs[{i}] must be a dict, got: {type(pair).__name__}")

        mode = str(pair.get("mode", "ohmic")).strip().lower()
        g_uS = float(pair.get("g_uS", default_g_uS))
        use_synapse_sites = _pair_uses_synapse_placement(pair)
        all_synapses = bool(pair.get("all_synapses", False))
        max_synapses = int(pair.get("max_synapses", 1))
        max_synapses = max(1, max_synapses)

        if "source_id" in pair or "target_id" in pair:
            if "source_id" not in pair or "target_id" not in pair:
                raise ValueError(
                    f"gap.pairs[{i}] directed form requires both source_id and target_id."
                )
            src = int(pair["source_id"])
            dst = int(pair["target_id"])
            src_site = _norm_site(pair.get("source_site", pair.get("site_a")), default_site)
            dst_site = _norm_site(pair.get("target_site", pair.get("site_b")), default_site)
            syn_rows: list[Dict[str, Any]] = []
            if use_synapse_sites:
                if edges_df is None:
                    edges_df = _load_edges_for_gap(cfg)
                rows = _rows_for_pair(edges_df, src, dst, include_reverse=True)
                if not rows.empty:
                    if not all_synapses:
                        rows = rows.head(max_synapses)
                    syn_rows = [r._asdict() for r in rows.itertuples(index=False)]

            if mode == "ohmic":
                if syn_rows:
                    for row in syn_rows:
                        src_site_res = _site_for_cell_from_row(net, cfg, row, src, src_site)
                        dst_site_res = _site_for_cell_from_row(net, cfg, row, dst, dst_site)
                        net.add_gap_pair_ohmic(src, dst, g_uS=g_uS, site_a=src_site_res, site_b=dst_site_res)
                        added.append(
                            {
                                "mode": "ohmic",
                                "a_id": src,
                                "b_id": dst,
                                "placement": "synapse",
                                "row_pre_id": int(row["pre_id"]),
                                "row_post_id": int(row["post_id"]),
                                "g_uS": g_uS,
                            }
                        )
                    print(f"[gap] ohmic {src}<->{dst} placed at {len(syn_rows)} synapse coordinate(s), g={g_uS} uS")
                else:
                    net.add_gap_pair_ohmic(src, dst, g_uS=g_uS, site_a=src_site, site_b=dst_site)
                    print(f"[gap] ohmic {src}:{src_site} <-> {dst}:{dst_site} g={g_uS} uS")
                    added.append(
                        {
                            "mode": "ohmic",
                            "a_id": src,
                            "b_id": dst,
                            "placement": "named_site",
                            "site_a": src_site,
                            "site_b": dst_site,
                            "g_uS": g_uS,
                        }
                    )
            elif mode == "rectifying":
                if syn_rows:
                    for row in syn_rows:
                        src_site_res = _site_for_cell_from_row(net, cfg, row, src, src_site)
                        dst_site_res = _site_for_cell_from_row(net, cfg, row, dst, dst_site)
                        net.add_gap_rectifying(
                            source_id=src,
                            target_id=dst,
                            g_uS=g_uS,
                            source_site=src_site_res,
                            target_site=dst_site_res,
                            direction_label="source_to_target",
                        )
                        added.append(
                            {
                                "mode": "rectifying",
                                "source_id": src,
                                "target_id": dst,
                                "placement": "synapse",
                                "row_pre_id": int(row["pre_id"]),
                                "row_post_id": int(row["post_id"]),
                                "direction": "source_to_target",
                                "g_uS": g_uS,
                            }
                        )
                    print(f"[gap] rectifying {src}->{dst} placed at {len(syn_rows)} synapse coordinate(s), g={g_uS} uS")
                else:
                    net.add_gap_rectifying(
                        source_id=src,
                        target_id=dst,
                        g_uS=g_uS,
                        source_site=src_site,
                        target_site=dst_site,
                        direction_label="source_to_target",
                    )
                    print(f"[gap] rectifying {src}:{src_site} -> {dst}:{dst_site} g={g_uS} uS")
                    added.append(
                        {
                            "mode": "rectifying",
                            "source_id": src,
                            "target_id": dst,
                            "placement": "named_site",
                            "source_site": src_site,
                            "target_site": dst_site,
                            "direction": "source_to_target",
                            "g_uS": g_uS,
                        }
                    )
            else:
                raise ValueError(
                    f"gap.pairs[{i}] has unsupported mode '{mode}'. Use 'ohmic', 'rectifying', or 'heterotypic_rectifying'."
                )
            continue

        if "a_id" not in pair or "b_id" not in pair:
            raise ValueError(
                f"gap.pairs[{i}] must define either (a_id,b_id) or (source_id,target_id)."
            )

        a_id = int(pair["a_id"])
        b_id = int(pair["b_id"])
        site_a = _norm_site(pair.get("site_a"), default_site)
        site_b = _norm_site(pair.get("site_b"), default_site)
        syn_rows: list[Dict[str, Any]] = []
        if use_synapse_sites:
            if edges_df is None:
                edges_df = _load_edges_for_gap(cfg)
            rows = _rows_for_pair(edges_df, a_id, b_id, include_reverse=True)
            if not rows.empty:
                if not all_synapses:
                    rows = rows.head(max_synapses)
                syn_rows = [r._asdict() for r in rows.itertuples(index=False)]

        if mode == "ohmic":
            if syn_rows:
                for row in syn_rows:
                    site_a_res = _site_for_cell_from_row(net, cfg, row, a_id, site_a)
                    site_b_res = _site_for_cell_from_row(net, cfg, row, b_id, site_b)
                    net.add_gap_pair_ohmic(a_id, b_id, g_uS=g_uS, site_a=site_a_res, site_b=site_b_res)
                    added.append(
                        {
                            "mode": "ohmic",
                            "a_id": a_id,
                            "b_id": b_id,
                            "placement": "synapse",
                            "row_pre_id": int(row["pre_id"]),
                            "row_post_id": int(row["post_id"]),
                            "g_uS": g_uS,
                        }
                    )
                print(f"[gap] ohmic {a_id}<->{b_id} placed at {len(syn_rows)} synapse coordinate(s), g={g_uS} uS")
            else:
                net.add_gap_pair_ohmic(a_id, b_id, g_uS=g_uS, site_a=site_a, site_b=site_b)
                print(f"[gap] ohmic {a_id}:{site_a} <-> {b_id}:{site_b} g={g_uS} uS")
                added.append(
                    {
                        "mode": "ohmic",
                        "a_id": a_id,
                        "b_id": b_id,
                        "placement": "named_site",
                        "site_a": site_a,
                        "site_b": site_b,
                        "g_uS": g_uS,
                    }
                )
            continue

        if mode == "heterotypic_rectifying":
            preferred_direction = str(pair.get("preferred_direction", "a_to_b")).strip().lower()
            g_closed_frac = float(pair.get("g_closed_frac", 0.0))
            vhalf_mV = float(pair.get("vhalf_mV", 0.0))
            vslope_mV = float(pair.get("vslope_mV", 5.0))

            if syn_rows:
                for row in syn_rows:
                    site_a_res = _site_for_cell_from_row(net, cfg, row, a_id, site_a)
                    site_b_res = _site_for_cell_from_row(net, cfg, row, b_id, site_b)
                    net.add_gap_pair_heterotypic_rectifying(
                        a_id=a_id,
                        b_id=b_id,
                        g_uS=g_uS,
                        g_closed_frac=g_closed_frac,
                        preferred_direction=preferred_direction,
                        vhalf_mV=vhalf_mV,
                        vslope_mV=vslope_mV,
                        site_a=site_a_res,
                        site_b=site_b_res,
                    )
                    added.append(
                        {
                            "mode": "heterotypic_rectifying",
                            "a_id": a_id,
                            "b_id": b_id,
                            "source_id": a_id if preferred_direction == "a_to_b" else b_id,
                            "target_id": b_id if preferred_direction == "a_to_b" else a_id,
                            "placement": "synapse",
                            "row_pre_id": int(row["pre_id"]),
                            "row_post_id": int(row["post_id"]),
                            "preferred_direction": preferred_direction,
                            "g_uS": g_uS,
                            "g_closed_frac": g_closed_frac,
                            "vhalf_mV": vhalf_mV,
                            "vslope_mV": vslope_mV,
                        }
                    )
                print(
                    f"[gap] heterotypic_rectifying {a_id}<=>{b_id} "
                    f"({preferred_direction}) placed at {len(syn_rows)} synapse coordinate(s), "
                    f"g={g_uS} uS frac={g_closed_frac}"
                )
            else:
                net.add_gap_pair_heterotypic_rectifying(
                    a_id=a_id,
                    b_id=b_id,
                    g_uS=g_uS,
                    g_closed_frac=g_closed_frac,
                    preferred_direction=preferred_direction,
                    vhalf_mV=vhalf_mV,
                    vslope_mV=vslope_mV,
                    site_a=site_a,
                    site_b=site_b,
                )
                print(
                    f"[gap] heterotypic_rectifying {a_id}:{site_a} <=> {b_id}:{site_b} "
                    f"({preferred_direction}) g={g_uS} uS frac={g_closed_frac}"
                )
                added.append(
                    {
                        "mode": "heterotypic_rectifying",
                        "a_id": a_id,
                        "b_id": b_id,
                        "source_id": a_id if preferred_direction == "a_to_b" else b_id,
                        "target_id": b_id if preferred_direction == "a_to_b" else a_id,
                        "placement": "named_site",
                        "site_a": site_a,
                        "site_b": site_b,
                        "preferred_direction": preferred_direction,
                        "g_uS": g_uS,
                        "g_closed_frac": g_closed_frac,
                        "vhalf_mV": vhalf_mV,
                        "vslope_mV": vslope_mV,
                    }
                )
            continue

        if mode != "rectifying":
            raise ValueError(
                f"gap.pairs[{i}] has unsupported mode '{mode}'. Use 'ohmic', 'rectifying', or 'heterotypic_rectifying'."
            )

        direction = str(pair.get("direction", "a_to_b")).strip().lower()
        if direction == "a_to_b":
            source_id, target_id = a_id, b_id
            source_site, target_site = site_a, site_b
        elif direction == "b_to_a":
            source_id, target_id = b_id, a_id
            source_site, target_site = site_b, site_a
        else:
            raise ValueError(
                f"gap.pairs[{i}] invalid rectifying direction '{direction}'. "
                "Use 'a_to_b' or 'b_to_a'."
            )

        if syn_rows:
            for row in syn_rows:
                source_site_res = _site_for_cell_from_row(net, cfg, row, source_id, source_site)
                target_site_res = _site_for_cell_from_row(net, cfg, row, target_id, target_site)
                net.add_gap_rectifying(
                    source_id=source_id,
                    target_id=target_id,
                    g_uS=g_uS,
                    source_site=source_site_res,
                    target_site=target_site_res,
                    direction_label=direction,
                )
                added.append(
                    {
                        "mode": "rectifying",
                        "a_id": a_id,
                        "b_id": b_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "placement": "synapse",
                        "row_pre_id": int(row["pre_id"]),
                        "row_post_id": int(row["post_id"]),
                        "direction": direction,
                        "g_uS": g_uS,
                    }
                )
            print(f"[gap] rectifying {source_id}->{target_id} placed at {len(syn_rows)} synapse coordinate(s), g={g_uS} uS")
        else:
            net.add_gap_rectifying(
                source_id=source_id,
                target_id=target_id,
                g_uS=g_uS,
                source_site=source_site,
                target_site=target_site,
                direction_label=direction,
            )
            print(f"[gap] rectifying {source_id}:{source_site} -> {target_id}:{target_site} g={g_uS} uS")
            added.append(
                {
                    "mode": "rectifying",
                    "a_id": a_id,
                    "b_id": b_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "placement": "named_site",
                    "site_a": site_a,
                    "site_b": site_b,
                    "direction": direction,
                    "g_uS": g_uS,
                }
            )

    print(f"[gap] added {len(added)} configured pair entries.")
    return added
