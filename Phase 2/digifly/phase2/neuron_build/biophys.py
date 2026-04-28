from __future__ import annotations
from typing import Any, Dict
from neuron import h
from .swc_cell import SWCCell

def _safe_uninsert(sec, mech: str):
    try:
        sec.uninsert(mech)
    except Exception:
        pass

def set_pas(sec, g=1e-4, e=-65.0, Ra=100.0, cm=1.0):
    _safe_uninsert(sec, "hh")
    sec.insert("pas")
    sec.g_pas = g
    sec.e_pas = e
    sec.Ra = Ra
    sec.cm = cm

def set_hh(sec, gnabar=0.12, gkbar=0.036, gl=3e-4, el=-65.0, Ra=100.0, cm=1.0, ena=None, ek=None):
    _safe_uninsert(sec, "pas")
    sec.insert("hh")
    sec.gnabar_hh = gnabar
    sec.gkbar_hh = gkbar
    sec.gl_hh = gl
    sec.el_hh = el
    sec.Ra = Ra
    sec.cm = cm

    # Optional equilibrium potentials (mV)
    try:
        if ena is not None:
            sec.ena = float(ena)
    except Exception:
        pass
    try:
        if ek is not None:
            sec.ek = float(ek)
    except Exception:
        pass


def _make_passive(cell: SWCCell, cfg: Dict[str, Any]):
    for s in getattr(cell, "_secs", []):
        set_pas(s, g=cfg["passive_g"], e=cfg["passive_e"], Ra=cfg["Ra"], cm=cfg["cm"])

def _make_active(cell: SWCCell, cfg: Dict[str, Any], soma_hh: dict, branch_hh: dict):
    make_passive(cell, cfg)
    ena = cfg.get("ena_mV", None)
    ek  = cfg.get("ek_mV", None)
    try:
        set_hh(cell.soma_sec, **soma_hh, Ra=cfg["Ra"], cm=cfg["cm"], ena=ena, ek=ek)
    except Exception:
        pass
    for s in getattr(cell, "_secs", []):
        if s is getattr(cell, "soma_sec", None):
            continue
        try:
            set_hh(s, **branch_hh, Ra=cfg["Ra"], cm=cfg["cm"], ena=ena, ek=ek)
        except Exception:
            pass

def apply_biophys(pre: SWCCell, post: SWCCell, cfg: Dict[str, Any]):
    """
    Keeps your prior behavior: make both passive; activate pre; activate post only if cfg['post_active'].
    Override as needed in user code if you want different rules.
    """
    make_passive(pre, cfg)
    make_passive(post, cfg)
    make_active(pre, cfg, cfg["pre_soma_hh"], cfg["pre_branch_hh"])
    if bool(cfg.get("post_active", True)):
        make_active(post, cfg, cfg.get("post_soma_hh", cfg["pre_soma_hh"]), cfg.get("post_branch_hh", cfg["pre_branch_hh"]))

# ---- public aliases (match Phase 2 execution expectations) ----
make_passive = _make_passive
make_active  = _make_active
