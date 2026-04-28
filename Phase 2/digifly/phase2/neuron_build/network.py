from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from neuron import h

from .ownership import OwnershipPlan
from .swc_cell import SWCCell, find_swc_with_fallback, pick_post_site, pick_pre_site
from .ais import attach_ais_methods
from .parallel import (
    apply_thread_partitions,
    get_parallel_context,
    requested_build_backend,
    reset_parallel_context,
)

# Ensure AIS method is attached once
attach_ais_methods()


def _build_pc_gid_maps(values) -> tuple[Dict[int, int], Dict[int, int]]:
    """Build a stable compact gid map for NEURON ParallelContext.

    External neuron ids in this project are not guaranteed to fit inside the
    gid range accepted by NEURON's ParallelContext on all builds. We therefore
    keep Digifly ownership and outputs keyed by the original neuron ids, while
    mapping them to a compact 0..N-1 gid space only for pc.set_gid2node() and
    pc.gid_connect().
    """
    neuron_ids = tuple(sorted({int(v) for v in values}))
    pc_gid_by_nid = {int(nid): int(idx) for idx, nid in enumerate(neuron_ids)}
    nid_by_pc_gid = {int(idx): int(nid) for idx, nid in enumerate(neuron_ids)}
    return pc_gid_by_nid, nid_by_pc_gid


def _cvode_enabled(cfg: Dict[str, Any]) -> bool:
    cv = cfg.get("cvode", {}) or {}
    if isinstance(cv, dict):
        return bool(cv.get("enabled", False))
    return False


def _disable_coreneuron_state() -> None:
    try:
        from neuron import coreneuron as cnrn

        cnrn.enable = False
        try:
            cnrn.gpu = False
        except Exception:
            pass
    except Exception:
        pass


@lru_cache(maxsize=1)
def _coreneuron_runtime_candidates() -> tuple[str, ...]:
    roots: list[Path] = []
    try:
        import neuron as neuron_module

        neuron_file = Path(getattr(neuron_module, "__file__", "")).expanduser().resolve()
        roots.extend(
            [
                neuron_file.parent,
                neuron_file.parent.parent,
                neuron_file.parent.parent.parent,
            ]
        )
    except Exception:
        pass

    nrnhome = str(os.environ.get("NRNHOME", "")).strip()
    if nrnhome:
        roots.append(Path(nrnhome))

    patterns = ("*coreneuron*", "*corenrn*")
    lib_suffixes = {".so", ".dylib", ".dll", ".pyd"}
    hits: list[str] = []
    seen_roots: set[str] = set()
    seen_hits: set[str] = set()

    for root in roots:
        try:
            root = root.expanduser().resolve()
        except Exception:
            continue
        root_key = str(root)
        if root_key in seen_roots or not root.exists() or not root.is_dir():
            continue
        seen_roots.add(root_key)
        for pattern in patterns:
            try:
                matches = root.rglob(pattern)
            except Exception:
                continue
            for path in matches:
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix == ".py":
                    continue
                if suffix not in lib_suffixes and not path.name.lower().startswith("libcore"):
                    continue
                hit = str(path)
                if hit in seen_hits:
                    continue
                seen_hits.add(hit)
                hits.append(hit)
    return tuple(hits)


def _coreneuron_runtime_available() -> bool:
    return bool(_coreneuron_runtime_candidates())


def maybe_enable_coreneuron(cfg: Dict[str, Any]) -> bool:
    if not cfg.get("enable_coreneuron", False):
        # CoreNEURON state is global across runs in a live kernel.
        # Explicitly disable it when requested so a prior run cannot leak settings.
        _disable_coreneuron_state()
        return False
    if _cvode_enabled(cfg):
        raise ValueError(
            "Invalid config: CVODE and CoreNEURON are mutually exclusive. "
            "Set either cvode.enabled=true OR enable_coreneuron=true, not both."
        )
    if not _coreneuron_runtime_available():
        cfg["enable_coreneuron"] = False
        cfg["coreneuron_gpu"] = False
        _disable_coreneuron_state()
        print(
            "[CoreNEURON] runtime unavailable on this install; "
            "falling back to NEURON for this run."
        )
        return False
    try:
        from neuron import coreneuron as cnrn
        cnrn.enable = True
        cnrn.gpu = bool(cfg.get("coreneuron_gpu", False))
        nthread = cfg.get("coreneuron_nthread", None)
        if nthread not in (None, "", False):
            try:
                cnrn.nthread = max(1, int(nthread))
            except Exception as e:
                print(f"[CoreNEURON] warning: could not set nthread={nthread}: {e}")
        try:
            cnrn.verbose = bool(cfg.get("coreneuron_verbose", False))
        except Exception:
            pass
        print(
            f"[CoreNEURON] enabled (gpu={getattr(cnrn, 'gpu', False)}"
            f", nthread={getattr(cnrn, 'nthread', 'default')})"
        )
        return True
    except Exception as e:
        print(f"[CoreNEURON] not available: {e}")
        return False

class Network:
    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        swc_paths: Dict[int, str] | None = None,
        ownership: OwnershipPlan | None = None,
        parallel_state: Dict[str, Any] | None = None,
    ):
        self.cfg = dict(cfg)
        self._parallel_state: Dict[str, Any] = dict(parallel_state or {})
        self._ownership = ownership
        self._swc_paths: Dict[int, str] = {int(k): str(v) for k, v in dict(swc_paths or {}).items()}
        self.cells: Dict[int, SWCCell] = {}
        self.netcons = []
        self.syns = []
        self._synapse_meta = []
        self.gaps = []
        self.iclamps = []
        self._iclamp_meta = []
        self.records: Dict[str, list[Any]] = {"t": []}
        self._tvec = h.Vector()
        self._spike_src: Dict[int, tuple[Any, float]] = {}
        self._detectors = []
        self._gid_detectors: Dict[int, Any] = {}
        self._coreneuron_on = maybe_enable_coreneuron(self.cfg)
        self._pins = []
        self._cvode = h.CVode()
        self._closed = False
        self._root_rank = 0
        self._transfer_id_next = 0
        self._gap_transfer_dirty = False
        self._gap_transfer_ready = False
        gid_source = ()
        if self._ownership is not None:
            gid_source = tuple(int(gid) for gid in self._ownership.gids)
        elif self._swc_paths:
            gid_source = tuple(int(gid) for gid in self._swc_paths)
        self._pc_gid_by_nid, self._nid_by_pc_gid = _build_pc_gid_maps(gid_source)
        self._distributed = bool(
            self._ownership is not None
            and int(self._ownership.world_size) > 1
            and requested_build_backend(self.cfg) == "distributed_gid"
        )
        self._pc = get_parallel_context() if self._distributed else None

    @property
    def parallel_rank(self) -> int:
        if self._ownership is not None:
            return int(self._ownership.rank)
        return int((self._parallel_state or {}).get("id", 0))

    @property
    def parallel_world_size(self) -> int:
        if self._ownership is not None:
            return int(self._ownership.world_size)
        return int((self._parallel_state or {}).get("nhost", 1))

    @property
    def is_distributed(self) -> bool:
        return bool(self._distributed and self.parallel_world_size > 1)

    @property
    def is_root_rank(self) -> bool:
        return int(self.parallel_rank) == int(self._root_rank)

    @property
    def local_gid_count(self) -> int:
        if self._ownership is None:
            return int(len(self.cells))
        return int(len(self._ownership.local_gids))

    def owner_of(self, nid: int) -> int:
        if self._ownership is None:
            return int(self.parallel_rank)
        return int(self._ownership.owner_of(int(nid)))

    def pc_gid_for_nid(self, nid: int) -> int:
        nid = int(nid)
        if nid not in self._pc_gid_by_nid:
            raise KeyError(f"Missing ParallelContext gid mapping for neuron_id={nid}")
        return int(self._pc_gid_by_nid[nid])

    def nid_for_pc_gid(self, pc_gid: int) -> int:
        pc_gid = int(pc_gid)
        if pc_gid not in self._nid_by_pc_gid:
            raise KeyError(f"Missing neuron_id mapping for ParallelContext gid={pc_gid}")
        return int(self._nid_by_pc_gid[pc_gid])

    def is_local_gid(self, nid: int) -> bool:
        if self._ownership is None:
            return True
        return bool(self._ownership.is_local(int(nid)))

    def ensure_cell(self, nid: int) -> SWCCell:
        nid = int(nid)
        if nid in self.cells:
            return self.cells[nid]
        if self.is_distributed and not self.is_local_gid(nid):
            raise KeyError(
                f"gid={nid} is owned by rank {self.owner_of(nid)}; "
                f"rank {self.parallel_rank} cannot materialize it locally in distributed_gid mode."
            )
        swc_path = self._swc_paths.get(nid)
        if swc_path is None:
            swc_path = find_swc_with_fallback(
                self.cfg.get("morph_swc_dir"),
                self.cfg["swc_dir"],
                nid,
            )
        c = SWCCell(nid, swc_path, cfg=self.cfg)
        self.cells[nid] = c
        return c

    def set_spike_src(self, nid: int, site):
        sec, x = site
        self._spike_src[int(nid)] = (sec, float(x))

    def _get_spike_src(self, nid: int):
        nid = int(nid)
        if nid in self._spike_src:
            return self._spike_src[nid]
        return self.ensure_cell(nid).soma_site()

    def register_output_gid(self, nid: int, site=None, thresh: float = 0.0):
        if not self.is_distributed:
            return None
        nid = int(nid)
        pc_gid = self.pc_gid_for_nid(nid)
        if not self.is_local_gid(nid):
            raise KeyError(
                f"gid={nid} is owned by rank {self.owner_of(nid)} and cannot be registered on rank {self.parallel_rank}."
            )
        if site is None:
            site = self._get_spike_src(nid)
        sec, x = site
        src_nc = h.NetCon(sec(float(x))._ref_v, None, sec=sec)
        src_nc.threshold = float(thresh)
        self._pc.set_gid2node(int(pc_gid), int(self.parallel_rank))
        self._pc.cell(int(pc_gid), src_nc)
        self._gid_detectors[int(nid)] = src_nc
        self.set_spike_src(int(nid), (sec, float(x)))
        return src_nc

    def add_syn_exp2(self, pre_id: int, post_id: int, post_site, weight_uS=None, delay_ms=None,
                     e_rev=None, tau1=None, tau2=None):
        cfg = self.cfg
        w = float(cfg["default_weight_uS"] if weight_uS is None else weight_uS)
        d = float(cfg["default_delay_ms"] if delay_ms is None else delay_ms)
        e = float(cfg["syn_e_rev_mV"] if e_rev is None else e_rev)
        t1 = float(cfg["syn_tau1_ms"] if tau1 is None else tau1)
        t2 = float(cfg["syn_tau2_ms"] if tau2 is None else tau2)

        sec_post, xloc = post_site
        syn = h.Exp2Syn(sec_post(float(xloc)))
        syn.e = e
        syn.tau1 = t1
        syn.tau2 = t2

        src_sec, src_x = self._get_spike_src(pre_id)
        nc = h.NetCon(src_sec(float(src_x))._ref_v, syn, sec=src_sec)
        nc.threshold = 0.0
        nc.weight[0] = w
        nc.delay = d

        self._pins.append((syn, sec_post, src_sec))
        self.netcons.append(nc)
        self.syns.append(syn)
        self._synapse_meta.append(
            {
                "pre_id": int(pre_id),
                "post_id": int(post_id),
                "syn": syn,
                "nc": nc,
                "weight_uS_base": float(w),
                "delay_ms_base": float(d),
                "tau1_ms_base": float(t1),
                "tau2_ms_base": float(t2),
                "e_rev_mV_base": float(e),
            }
        )
        return syn, nc

    def add_syn_exp2_gid(self, pre_id: int, post_id: int, post_site, weight_uS=None, delay_ms=None,
                         e_rev=None, tau1=None, tau2=None):
        if not self.is_distributed:
            return self.add_syn_exp2(
                pre_id,
                post_id,
                post_site,
                weight_uS=weight_uS,
                delay_ms=delay_ms,
                e_rev=e_rev,
                tau1=tau1,
                tau2=tau2,
            )
        if not self.is_local_gid(int(post_id)):
            raise KeyError(
                f"post gid={post_id} is owned by rank {self.owner_of(post_id)}; "
                f"rank {self.parallel_rank} cannot attach a local target for it."
            )

        cfg = self.cfg
        w = float(cfg["default_weight_uS"] if weight_uS is None else weight_uS)
        d = float(cfg["default_delay_ms"] if delay_ms is None else delay_ms)
        e = float(cfg["syn_e_rev_mV"] if e_rev is None else e_rev)
        t1 = float(cfg["syn_tau1_ms"] if tau1 is None else tau1)
        t2 = float(cfg["syn_tau2_ms"] if tau2 is None else tau2)

        sec_post, xloc = post_site
        syn = h.Exp2Syn(sec_post(float(xloc)))
        syn.e = e
        syn.tau1 = t1
        syn.tau2 = t2

        nc = self._pc.gid_connect(self.pc_gid_for_nid(int(pre_id)), syn)
        nc.weight[0] = w
        nc.delay = d

        self._pins.append((syn, sec_post))
        self.netcons.append(nc)
        self.syns.append(syn)
        self._synapse_meta.append(
            {
                "pre_id": int(pre_id),
                "post_id": int(post_id),
                "syn": syn,
                "nc": nc,
                "weight_uS_base": float(w),
                "delay_ms_base": float(d),
                "tau1_ms_base": float(t1),
                "tau2_ms_base": float(t2),
                "e_rev_mV_base": float(e),
            }
        )
        return syn, nc

    def _resolve_site(self, nid: int, site_spec: Any):
        if isinstance(site_spec, dict) and str(site_spec.get("__gap_site_kind__", "")).strip():
            kind = str(site_spec.get("__gap_site_kind__", "")).strip().lower()
            fallback_site = str(site_spec.get("fallback_site", "soma")).strip().lower()
            row = dict(site_spec.get("row") or {})
            cell = self.ensure_cell(int(nid))
            if kind == "pre_row":
                return pick_pre_site(cell, row, self.cfg["swc_dir"])
            if kind == "post_row":
                return pick_post_site(cell, row, self.cfg["swc_dir"])
            if kind == "named_site":
                site_spec = str(site_spec.get("site", fallback_site))
            else:
                raise ValueError(f"Unsupported distributed gap site descriptor kind '{kind}' for gid={nid}.")

        # Explicit NEURON site tuple: (sec, xloc)
        if isinstance(site_spec, (tuple, list)) and len(site_spec) == 2:
            sec, x = site_spec
            return sec, float(x)

        cell = self.ensure_cell(int(nid))
        site = str(site_spec).strip().lower()
        if site == "ais":
            return cell.axon_ais_site()
        if site == "soma":
            return cell.soma_site()
        raise ValueError(
            f"Unsupported site '{site_spec}'. Use 'ais', 'soma', or an explicit (sec, xloc) tuple."
        )

    @staticmethod
    def _set_gap_pointer(gap_proc, target_sec, target_x: float):
        # Different NEURON builds accept different setpointer signatures.
        try:
            h.setpointer(target_sec(float(target_x))._ref_v, "vgap_ptr", gap_proc)
            return
        except Exception:
            pass
        h.setpointer(target_sec(float(target_x))._ref_v, gap_proc, "vgap_ptr")

    def _next_transfer_id(self) -> int:
        transfer_id = int(self._transfer_id_next)
        self._transfer_id_next += 1
        return transfer_id

    def _register_source_transfer(self, transfer_id: int, sec, xloc: float):
        if not self.is_distributed or self._pc is None:
            return
        try:
            self._pc.source_var(sec(float(xloc))._ref_v, int(transfer_id), sec=sec)
        except TypeError:
            self._pc.source_var(sec(float(xloc))._ref_v, int(transfer_id))
        self._gap_transfer_dirty = True
        self._gap_transfer_ready = False

    def _register_target_transfer(self, target_proc, transfer_id: int):
        if not self.is_distributed or self._pc is None:
            return
        try:
            self._pc.target_var(target_proc, target_proc._ref_vgap_xfer, int(transfer_id))
        except TypeError:
            self._pc.target_var(target_proc._ref_vgap_xfer, int(transfer_id))
        self._gap_transfer_dirty = True
        self._gap_transfer_ready = False

    def add_gap_pair_ohmic(
        self,
        a_id: int,
        b_id: int,
        g_uS: float = 0.001,
        site_a: Any = "ais",
        site_b: Any = "ais",
    ):
        if self.is_distributed:
            transfer_a_to_b = self._next_transfer_id()
            transfer_b_to_a = self._next_transfer_id()
            local_a = None
            local_b = None
            g_ns = float(g_uS) * 1000.0

            if self.is_local_gid(int(a_id)):
                sec_a, xa = self._resolve_site(int(a_id), site_a)
                local_a = h.Gap(float(xa), sec=sec_a)
                local_a.g = g_ns
                local_a.use_transfer = 1.0
                self._register_source_transfer(transfer_a_to_b, sec_a, xa)
                self._register_target_transfer(local_a, transfer_b_to_a)
                self._pins.append((local_a, sec_a))

            if self.is_local_gid(int(b_id)):
                sec_b, xb = self._resolve_site(int(b_id), site_b)
                local_b = h.Gap(float(xb), sec=sec_b)
                local_b.g = g_ns
                local_b.use_transfer = 1.0
                self._register_source_transfer(transfer_b_to_a, sec_b, xb)
                self._register_target_transfer(local_b, transfer_a_to_b)
                self._pins.append((local_b, sec_b))

            self.gaps.append(
                {
                    "mode": "ohmic",
                    "a_id": int(a_id),
                    "b_id": int(b_id),
                    "site_a": str(site_a),
                    "site_b": str(site_b),
                    "g_uS_base": float(g_uS),
                    "g_uS": float(g_uS),
                    "g_ns_base": g_ns,
                    "g_ns": g_ns,
                    "transfer_a_to_b": int(transfer_a_to_b),
                    "transfer_b_to_a": int(transfer_b_to_a),
                    "handles": tuple(x for x in (local_a, local_b) if x is not None),
                }
            )
            return local_a, local_b

        sec_a, xa = self._resolve_site(int(a_id), site_a)
        sec_b, xb = self._resolve_site(int(b_id), site_b)

        g_ns = float(g_uS) * 1000.0
        gj_ab = h.Gap(float(xa), sec=sec_a)
        gj_ba = h.Gap(float(xb), sec=sec_b)
        gj_ab.g = g_ns
        gj_ba.g = g_ns

        self._set_gap_pointer(gj_ab, sec_b, xb)
        self._set_gap_pointer(gj_ba, sec_a, xa)

        self._pins.append((gj_ab, gj_ba, sec_a, sec_b))
        self.gaps.append(
            {
                "mode": "ohmic",
                "a_id": int(a_id),
                "b_id": int(b_id),
                "site_a": str(site_a),
                "site_b": str(site_b),
                "g_uS_base": float(g_uS),
                "g_uS": float(g_uS),
                "g_ns_base": g_ns,
                "g_ns": g_ns,
                "handles": (gj_ab, gj_ba),
            }
        )
        return gj_ab, gj_ba

    def add_gap_rectifying(
        self,
        source_id: int,
        target_id: int,
        g_uS: float = 0.001,
        source_site: Any = "ais",
        target_site: Any = "ais",
        direction_label: str = "source_to_target",
    ):
        if self.is_distributed:
            transfer_source_to_target = self._next_transfer_id()
            local_gap = None
            g_ns = float(g_uS) * 1000.0

            if self.is_local_gid(int(source_id)):
                src_sec, src_x = self._resolve_site(int(source_id), source_site)
                self._register_source_transfer(transfer_source_to_target, src_sec, src_x)

            if self.is_local_gid(int(target_id)):
                dst_sec, dst_x = self._resolve_site(int(target_id), target_site)
                try:
                    local_gap = h.RectGap(float(dst_x), sec=dst_sec)
                except Exception as e:
                    raise RuntimeError(
                        "Rectifying gap mechanism 'RectGap' is not available. "
                        "Compile/load RectGap.mod before creating threaded rectifying gaps."
                    ) from e
                local_gap.gmax = g_ns
                local_gap.use_transfer = 1.0
                self._register_target_transfer(local_gap, transfer_source_to_target)
                self._pins.append((local_gap, dst_sec))

            self.gaps.append(
                {
                    "mode": "rectifying",
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "source_site": str(source_site),
                    "target_site": str(target_site),
                    "direction": str(direction_label),
                    "g_uS_base": float(g_uS),
                    "g_uS": float(g_uS),
                    "gmax_ns_base": g_ns,
                    "gmax_ns": g_ns,
                    "transfer_source_to_target": int(transfer_source_to_target),
                    "handles": (local_gap,) if local_gap is not None else (),
                }
            )
            return local_gap

        src_sec, src_x = self._resolve_site(int(source_id), source_site)
        dst_sec, dst_x = self._resolve_site(int(target_id), target_site)

        g_ns = float(g_uS) * 1000.0
        try:
            gj = h.RectGap(float(dst_x), sec=dst_sec)
        except Exception as e:
            raise RuntimeError(
                "Rectifying gap mechanism 'RectGap' is not available. "
                "Compile/load RectGap.mod before creating threaded rectifying gaps."
            ) from e
        gj.gmax = g_ns
        self._set_gap_pointer(gj, src_sec, src_x)

        self._pins.append((gj, src_sec, dst_sec))
        self.gaps.append(
            {
                "mode": "rectifying",
                "source_id": int(source_id),
                "target_id": int(target_id),
                "source_site": str(source_site),
                "target_site": str(target_site),
                "direction": str(direction_label),
                "g_uS_base": float(g_uS),
                "g_uS": float(g_uS),
                "gmax_ns_base": g_ns,
                "gmax_ns": g_ns,
                "handles": (gj,),
            }
        )
        return gj

    def add_gap_pair_heterotypic_rectifying(
        self,
        a_id: int,
        b_id: int,
        g_uS: float = 0.001,
        g_closed_frac: float = 0.0,
        site_a: Any = "ais",
        site_b: Any = "ais",
        preferred_direction: str = "a_to_b",
        vhalf_mV: float = 0.0,
        vslope_mV: float = 5.0,
    ):
        preferred = str(preferred_direction).strip().lower()
        if preferred == "a_to_b":
            orient_a = 1.0
            orient_b = -1.0
            source_id = int(a_id)
            target_id = int(b_id)
        elif preferred == "b_to_a":
            orient_a = -1.0
            orient_b = 1.0
            source_id = int(b_id)
            target_id = int(a_id)
        else:
            raise ValueError(
                f"Invalid preferred_direction '{preferred_direction}'. Use 'a_to_b' or 'b_to_a'."
            )

        g_open_ns = float(g_uS) * 1000.0
        g_closed_frac = max(0.0, float(g_closed_frac))
        g_closed_ns = g_open_ns * g_closed_frac

        if self.is_distributed:
            transfer_a_to_b = self._next_transfer_id()
            transfer_b_to_a = self._next_transfer_id()
            local_a = None
            local_b = None

            if self.is_local_gid(int(a_id)):
                sec_a, xa = self._resolve_site(int(a_id), site_a)
                try:
                    local_a = h.HeteroRectGap(float(xa), sec=sec_a)
                except Exception as e:
                    raise RuntimeError(
                        "Heterotypic rectifying gap mechanism 'HeteroRectGap' is not available. "
                        "Compile/load HeteroRectGap.mod before creating paired heterotypic gaps."
                    ) from e
                local_a.gmax_open = g_open_ns
                local_a.gmax_closed = g_closed_ns
                local_a.orientation = float(orient_a)
                local_a.vhalf = float(vhalf_mV)
                local_a.vslope = float(vslope_mV)
                local_a.use_transfer = 1.0
                self._register_source_transfer(transfer_a_to_b, sec_a, xa)
                self._register_target_transfer(local_a, transfer_b_to_a)
                self._pins.append((local_a, sec_a))

            if self.is_local_gid(int(b_id)):
                sec_b, xb = self._resolve_site(int(b_id), site_b)
                try:
                    local_b = h.HeteroRectGap(float(xb), sec=sec_b)
                except Exception as e:
                    raise RuntimeError(
                        "Heterotypic rectifying gap mechanism 'HeteroRectGap' is not available. "
                        "Compile/load HeteroRectGap.mod before creating paired heterotypic gaps."
                    ) from e
                local_b.gmax_open = g_open_ns
                local_b.gmax_closed = g_closed_ns
                local_b.orientation = float(orient_b)
                local_b.vhalf = float(vhalf_mV)
                local_b.vslope = float(vslope_mV)
                local_b.use_transfer = 1.0
                self._register_source_transfer(transfer_b_to_a, sec_b, xb)
                self._register_target_transfer(local_b, transfer_a_to_b)
                self._pins.append((local_b, sec_b))

            self.gaps.append(
                {
                    "mode": "heterotypic_rectifying",
                    "a_id": int(a_id),
                    "b_id": int(b_id),
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "site_a": str(site_a),
                    "site_b": str(site_b),
                    "preferred_direction": str(preferred),
                    "g_uS_base": float(g_uS),
                    "g_uS": float(g_uS),
                    "g_closed_frac_base": float(g_closed_frac),
                    "g_closed_frac": float(g_closed_frac),
                    "gmax_open_ns_base": float(g_open_ns),
                    "gmax_open_ns": float(g_open_ns),
                    "gmax_closed_ns_base": float(g_closed_ns),
                    "gmax_closed_ns": float(g_closed_ns),
                    "vhalf_mV": float(vhalf_mV),
                    "vslope_mV": float(vslope_mV),
                    "transfer_a_to_b": int(transfer_a_to_b),
                    "transfer_b_to_a": int(transfer_b_to_a),
                    "handles": tuple(x for x in (local_a, local_b) if x is not None),
                }
            )
            return local_a, local_b

        sec_a, xa = self._resolve_site(int(a_id), site_a)
        sec_b, xb = self._resolve_site(int(b_id), site_b)
        try:
            gj_a = h.HeteroRectGap(float(xa), sec=sec_a)
            gj_b = h.HeteroRectGap(float(xb), sec=sec_b)
        except Exception as e:
            raise RuntimeError(
                "Heterotypic rectifying gap mechanism 'HeteroRectGap' is not available. "
                "Compile/load HeteroRectGap.mod before creating paired heterotypic gaps."
            ) from e

        for gap, orient in ((gj_a, orient_a), (gj_b, orient_b)):
            gap.gmax_open = g_open_ns
            gap.gmax_closed = g_closed_ns
            gap.orientation = float(orient)
            gap.vhalf = float(vhalf_mV)
            gap.vslope = float(vslope_mV)

        self._set_gap_pointer(gj_a, sec_b, xb)
        self._set_gap_pointer(gj_b, sec_a, xa)

        self._pins.append((gj_a, gj_b, sec_a, sec_b))
        self.gaps.append(
            {
                "mode": "heterotypic_rectifying",
                "a_id": int(a_id),
                "b_id": int(b_id),
                "source_id": int(source_id),
                "target_id": int(target_id),
                "site_a": str(site_a),
                "site_b": str(site_b),
                "preferred_direction": str(preferred),
                "g_uS_base": float(g_uS),
                "g_uS": float(g_uS),
                "g_closed_frac_base": float(g_closed_frac),
                "g_closed_frac": float(g_closed_frac),
                "gmax_open_ns_base": float(g_open_ns),
                "gmax_open_ns": float(g_open_ns),
                "gmax_closed_ns_base": float(g_closed_ns),
                "gmax_closed_ns": float(g_closed_ns),
                "vhalf_mV": float(vhalf_mV),
                "vslope_mV": float(vslope_mV),
                "handles": (gj_a, gj_b),
            }
        )
        return gj_a, gj_b

    def record_time(self):
        # On this NEURON build, Vector.record(h._ref_t) can fail on MPI ranks
        # that own zero cells because there is no current section context.
        if self.is_distributed and self.local_gid_count <= 0:
            return None
        self._tvec = h.Vector()
        self._tvec.record(h._ref_t)
        self.records["t"].append(self._tvec)
        return self._tvec

    def record_soma(self, nid: int, label: Optional[str] = None):
        if self.is_distributed and not self.is_local_gid(int(nid)):
            return None
        sec, x = self.ensure_cell(nid).soma_site()
        v = h.Vector()
        v.record(sec(x)._ref_v)
        self.records.setdefault(label or f"{nid}_soma_v", []).append(v)
        return v

    def add_iclamp_site(
        self,
        nid: int,
        site,
        amp_nA: float,
        delay_ms: float,
        dur_ms: float,
        *,
        kind: str = "base",
    ):
        sec, x = site
        stim = h.IClamp(sec(x))
        stim.delay = float(delay_ms)
        stim.dur = float(dur_ms)
        stim.amp = float(amp_nA)
        self.iclamps.append(stim)
        self._iclamp_meta.append({"stim": stim, "nid": int(nid), "kind": str(kind)})
        self.set_spike_src(nid, site)
        return stim

    def count_spikes(self, nid: int, thresh: float = 0.0):
        if self.is_distributed and not self.is_local_gid(int(nid)):
            return None
        sec, x = self.ensure_cell(nid).soma_site()
        vt = h.Vector()
        nc = h.NetCon(sec(x)._ref_v, None, sec=sec)
        nc.threshold = float(thresh)
        nc.record(vt)
        self._detectors.append(nc)
        return vt

    def count_spikes_at_site(self, nid: int, site, thresh: float = 0.0):
        sec, x = site
        vt = h.Vector()
        nc = h.NetCon(sec(float(x))._ref_v, None, sec=sec)
        nc.threshold = float(thresh)
        nc.record(vt)
        self._detectors.append(nc)
        return vt

    def reset_run_artifacts(self):
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")
        self.records = {"t": []}
        self._tvec = h.Vector()
        self._detectors.clear()
        return True

    def reset_synapse_parameters(self) -> int:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        updated = 0
        for meta in self._synapse_meta:
            syn = meta.get("syn")
            nc = meta.get("nc")
            if syn is None or nc is None:
                continue
            nc.weight[0] = float(meta.get("weight_uS_base", 0.0))
            nc.delay = float(meta.get("delay_ms_base", 0.0))
            syn.tau1 = max(1e-6, float(meta.get("tau1_ms_base", 0.5)))
            syn.tau2 = max(1e-6, float(meta.get("tau2_ms_base", 1.0)))
            syn.e = float(meta.get("e_rev_mV_base", 0.0))
            updated += 1
        return updated

    def reset_gap_parameters(self) -> int:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        updated = 0
        for meta in self.gaps:
            mode = str(meta.get("mode", "")).strip().lower()
            handles = tuple(meta.get("handles") or ())
            if mode == "ohmic":
                g_ns = float(meta.get("g_ns_base", meta.get("g_ns", 0.0)))
                for gap in handles:
                    if gap is None:
                        continue
                    gap.g = g_ns
                    updated += 1
                meta["g_ns"] = g_ns
                meta["g_uS"] = float(meta.get("g_uS_base", g_ns / 1000.0))
            elif mode == "rectifying":
                g_ns = float(meta.get("gmax_ns_base", meta.get("gmax_ns", 0.0)))
                for gap in handles:
                    if gap is None:
                        continue
                    gap.gmax = g_ns
                    updated += 1
                meta["gmax_ns"] = g_ns
                meta["g_uS"] = float(meta.get("g_uS_base", g_ns / 1000.0))
            elif mode == "heterotypic_rectifying":
                g_open_ns = float(meta.get("gmax_open_ns_base", meta.get("gmax_open_ns", 0.0)))
                g_closed_ns = float(meta.get("gmax_closed_ns_base", meta.get("gmax_closed_ns", 0.0)))
                for gap in handles:
                    if gap is None:
                        continue
                    gap.gmax_open = g_open_ns
                    gap.gmax_closed = g_closed_ns
                    updated += 1
                meta["gmax_open_ns"] = g_open_ns
                meta["gmax_closed_ns"] = g_closed_ns
                meta["g_uS"] = float(meta.get("g_uS_base", g_open_ns / 1000.0))
        return updated

    def _apply_cell_biophys_groups(
        self,
        extra_groups: list[Dict[str, Any]] | None,
        *,
        seed_ids: list[int] | None = None,
    ) -> int:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        from .biophys import make_active, make_passive, set_hh
        from .builders import _biophys_override_index, _cell_biophys_spec

        cfg_use = dict(self.cfg)
        cfg_use["cell_biophys_overrides"] = [
            dict(x) for x in list(cfg_use.get("cell_biophys_overrides") or [])
        ] + [dict(x) for x in list(extra_groups or [])]
        override_index = _biophys_override_index(cfg_use)
        seed_set = {int(x) for x in list(seed_ids or cfg_use.get("seeds") or [])}

        updated = 0
        for gid, cell in list(self.cells.items()):
            gid = int(gid)
            if self.is_distributed and not self.is_local_gid(gid):
                continue
            role = "pre" if gid in seed_set else "post"
            cell_cfg, soma_hh, branch_hh, active_flag = _cell_biophys_spec(
                cfg_use,
                gid,
                role=role,
                override_index=override_index,
            )
            make_passive(cell, cell_cfg)
            if active_flag:
                make_active(cell, cell_cfg, soma_hh, branch_hh)
                if role == "pre":
                    try:
                        site = cell.axon_ais_site()
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
            updated += 1
        return int(updated)

    def reset_cell_biophys_overrides(self, *, seed_ids: list[int] | None = None) -> int:
        return int(self._apply_cell_biophys_groups([], seed_ids=seed_ids))

    def apply_cell_biophys_overrides(
        self,
        group_overrides: list[Dict[str, Any]] | None,
        *,
        seed_ids: list[int] | None = None,
    ) -> list[Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        groups = [dict(x) for x in list(group_overrides or [])]
        updated_cells = int(self._apply_cell_biophys_groups(groups, seed_ids=seed_ids))
        summary: list[Dict[str, Any]] = []
        for idx, raw_group in enumerate(groups):
            ids = raw_group.get("ids")
            if ids is None:
                ids = raw_group.get("neuron_ids")
            local_ids = [
                int(x)
                for x in list(ids or [])
                if (not self.is_distributed) or self.is_local_gid(int(x))
            ]
            summary.append(
                {
                    "name": str(raw_group.get("name") or f"cell_biophys_group_{idx:02d}"),
                    "matched_cells": int(len(local_ids)),
                    "updated_cells_total": int(updated_cells),
                    "ids": [int(x) for x in local_ids],
                }
            )
        return summary

    @staticmethod
    def _synapse_meta_matches(meta: Dict[str, Any], selectors: Dict[str, Any]) -> bool:
        if not selectors:
            return True

        pre_id = int(meta.get("pre_id", -1))
        post_id = int(meta.get("post_id", -1))

        pre_ids = selectors.get("pre_ids")
        if pre_ids is not None:
            allowed = {int(x) for x in pre_ids}
            if pre_id not in allowed:
                return False

        post_ids = selectors.get("post_ids")
        if post_ids is not None:
            allowed = {int(x) for x in post_ids}
            if post_id not in allowed:
                return False

        pairs = selectors.get("pairs")
        if pairs is not None:
            allowed_pairs = {(int(a), int(b)) for a, b in pairs}
            if (pre_id, post_id) not in allowed_pairs:
                return False

        return True

    @staticmethod
    def _gap_meta_matches(meta: Dict[str, Any], selectors: Dict[str, Any]) -> bool:
        if not selectors:
            return True

        mode = str(meta.get("mode", "")).strip().lower()
        modes = selectors.get("modes")
        if modes is not None:
            allowed_modes = {str(x).strip().lower() for x in modes}
            mode_aliases = {mode}
            if mode == "heterotypic_rectifying":
                mode_aliases.add("rectifying")
            if mode == "rectifying":
                mode_aliases.add("heterotypic_rectifying")
            if mode_aliases.isdisjoint(allowed_modes):
                return False

        pairs = selectors.get("pairs")
        if pairs is not None:
            allowed_pairs = {(int(a), int(b)) for a, b in pairs}
            if mode == "ohmic":
                a_id = int(meta.get("a_id", -1))
                b_id = int(meta.get("b_id", -1))
                if (a_id, b_id) not in allowed_pairs and (b_id, a_id) not in allowed_pairs:
                    return False
            else:
                source_id = int(meta.get("source_id", -1))
                target_id = int(meta.get("target_id", -1))
                if (source_id, target_id) not in allowed_pairs:
                    return False

        a_ids = selectors.get("a_ids")
        if a_ids is not None:
            allowed = {int(x) for x in a_ids}
            if int(meta.get("a_id", meta.get("source_id", -1))) not in allowed:
                return False

        b_ids = selectors.get("b_ids")
        if b_ids is not None:
            allowed = {int(x) for x in b_ids}
            if int(meta.get("b_id", meta.get("target_id", -1))) not in allowed:
                return False

        source_ids = selectors.get("source_ids")
        if source_ids is not None:
            allowed = {int(x) for x in source_ids}
            if int(meta.get("source_id", meta.get("a_id", -1))) not in allowed:
                return False

        target_ids = selectors.get("target_ids")
        if target_ids is not None:
            allowed = {int(x) for x in target_ids}
            if int(meta.get("target_id", meta.get("b_id", -1))) not in allowed:
                return False

        return True

    def apply_synapse_group_overrides(self, group_overrides: list[Dict[str, Any]] | None) -> list[Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        groups = list(group_overrides or [])
        if not groups:
            return []

        summary: list[Dict[str, Any]] = []
        for idx, raw_group in enumerate(groups):
            group = dict(raw_group or {})
            selectors = dict(group.get("selectors") or {})
            weight_mult = float(group.get("weight_mult", 1.0))
            delay_mult = float(group.get("delay_mult", 1.0))
            tau1_mult = float(group.get("tau1_mult", 1.0))
            tau2_mult = float(group.get("tau2_mult", 1.0))
            e_rev_shift_mV = float(group.get("e_rev_shift_mV", 0.0))
            matched = 0

            for meta in self._synapse_meta:
                if not self._synapse_meta_matches(meta, selectors):
                    continue
                syn = meta.get("syn")
                nc = meta.get("nc")
                if syn is None or nc is None:
                    continue
                nc.weight[0] = float(nc.weight[0]) * weight_mult
                nc.delay = float(nc.delay) * delay_mult
                syn.tau1 = max(1e-6, float(syn.tau1) * tau1_mult)
                syn.tau2 = max(1e-6, float(syn.tau2) * tau2_mult)
                syn.e = float(syn.e) + e_rev_shift_mV
                matched += 1

            summary.append(
                {
                    "name": str(group.get("name") or f"group_{idx:02d}"),
                    "matched_synapses": int(matched),
                    "weight_mult": float(weight_mult),
                    "delay_mult": float(delay_mult),
                    "tau1_mult": float(tau1_mult),
                    "tau2_mult": float(tau2_mult),
                    "e_rev_shift_mV": float(e_rev_shift_mV),
                    "selectors": selectors,
                }
            )
        return summary

    def apply_gap_group_overrides(self, group_overrides: list[Dict[str, Any]] | None) -> list[Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        groups = list(group_overrides or [])
        if not groups:
            return []

        summary: list[Dict[str, Any]] = []
        for idx, raw_group in enumerate(groups):
            group = dict(raw_group or {})
            selectors = dict(group.get("selectors") or {})
            g_mult = float(group.get("g_mult", 1.0))
            g_uS_abs = group.get("g_uS")
            g_uS_abs = None if g_uS_abs is None else float(g_uS_abs)
            matched = 0
            updated_handles = 0

            for meta in self.gaps:
                if not self._gap_meta_matches(meta, selectors):
                    continue
                mode = str(meta.get("mode", "")).strip().lower()
                handles = tuple(meta.get("handles") or ())
                base_g_uS = float(meta.get("g_uS_base", meta.get("g_uS", 0.0)))
                target_g_uS = float(g_uS_abs) if g_uS_abs is not None else (base_g_uS * g_mult)
                target_g_ns = float(target_g_uS) * 1000.0

                if mode == "ohmic":
                    for gap in handles:
                        if gap is None:
                            continue
                        gap.g = target_g_ns
                        updated_handles += 1
                    meta["g_ns"] = target_g_ns
                    meta["g_uS"] = float(target_g_uS)
                elif mode == "rectifying":
                    for gap in handles:
                        if gap is None:
                            continue
                        gap.gmax = target_g_ns
                        updated_handles += 1
                    meta["gmax_ns"] = target_g_ns
                    meta["g_uS"] = float(target_g_uS)
                elif mode == "heterotypic_rectifying":
                    closed_frac = float(meta.get("g_closed_frac_base", meta.get("g_closed_frac", 0.0)))
                    target_closed_ns = target_g_ns * max(0.0, closed_frac)
                    for gap in handles:
                        if gap is None:
                            continue
                        gap.gmax_open = target_g_ns
                        gap.gmax_closed = target_closed_ns
                        updated_handles += 1
                    meta["gmax_open_ns"] = target_g_ns
                    meta["gmax_closed_ns"] = target_closed_ns
                    meta["g_uS"] = float(target_g_uS)
                matched += 1

            summary.append(
                {
                    "name": str(group.get("name") or f"gap_group_{idx:02d}"),
                    "matched_gaps": int(matched),
                    "updated_gap_handles": int(updated_handles),
                    "g_mult": float(g_mult),
                    "g_uS": None if g_uS_abs is None else float(g_uS_abs),
                    "selectors": selectors,
                }
            )
        return summary

    def sync_iclamp_parameters_from_cfg(self) -> int:
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")

        stim_cfg = dict(self.cfg.get("stim") or {})
        kind_specs = {
            "base": dict(stim_cfg.get("iclamp") or {}),
            "neg_pulse": dict(stim_cfg.get("neg_pulse") or {}),
            "pulse_train": dict(stim_cfg.get("pulse_train") or {}),
        }

        base_spec = kind_specs["base"]
        if "amp_nA" not in base_spec and self.cfg.get("iclamp_amp_nA") is not None:
            base_spec["amp_nA"] = self.cfg.get("iclamp_amp_nA")
        if "delay_ms" not in base_spec and self.cfg.get("iclamp_delay_ms") is not None:
            base_spec["delay_ms"] = self.cfg.get("iclamp_delay_ms")
        if "dur_ms" not in base_spec and self.cfg.get("iclamp_dur_ms") is not None:
            base_spec["dur_ms"] = self.cfg.get("iclamp_dur_ms")

        updated = 0
        for meta in self._iclamp_meta:
            stim = meta.get("stim")
            kind = str(meta.get("kind", "base"))
            spec = kind_specs.get(kind) or {}
            if stim is None or not spec:
                continue
            if "amp_nA" in spec:
                stim.amp = float(spec["amp_nA"])
            elif "amp" in spec:
                stim.amp = float(spec["amp"])
            if "delay_ms" in spec:
                stim.delay = float(spec["delay_ms"])
            elif "delay" in spec:
                stim.delay = float(spec["delay"])
            if "dur_ms" in spec:
                stim.dur = float(spec["dur_ms"])
            elif "dur" in spec:
                stim.dur = float(spec["dur"])
            updated += 1
        return updated

    def run(self, tstop_ms: Optional[float] = None, dt_ms: Optional[float] = None, show_progress: bool = True):
        if self._closed:
            raise RuntimeError("Network has been closed and cannot be reused.")
        cfg = self.cfg
        self._parallel_state = apply_thread_partitions(self.cells.values(), cfg)
        base_dt = float(cfg["dt_ms"] if dt_ms is None else dt_ms)
        h.dt = base_dt

        cv_cfg = cfg.get("cvode", {}) or {}
        use_cvode = bool(cv_cfg.get("enabled", False)) if isinstance(cv_cfg, dict) else False
        if use_cvode and self._coreneuron_on:
            raise ValueError(
                "Invalid config: CVODE and CoreNEURON are mutually exclusive. "
                "Disable one and rerun."
            )

        try:
            self._cvode.active(1 if use_cvode else 0)
        except Exception:
            pass

        if self._coreneuron_on:
            # CoreNEURON requires cache-efficient data layout before psolve().
            try:
                self._cvode.cache_efficient(1)
            except Exception:
                pass

        if use_cvode and isinstance(cv_cfg, dict):
            atol = cv_cfg.get("atol", None)
            rtol = cv_cfg.get("rtol", None)
            maxstep = cv_cfg.get("maxstep_ms", None)
            if atol is not None:
                try:
                    self._cvode.atol(float(atol))
                except Exception:
                    pass
            if rtol is not None:
                try:
                    self._cvode.rtol(float(rtol))
                except Exception:
                    pass
            if maxstep is not None:
                try:
                    self._cvode.maxstep(float(maxstep))
                except Exception:
                    pass

        tstop = float(cfg["tstop_ms"] if tstop_ms is None else tstop_ms)
        chunk = float(cfg.get("progress_chunk_ms", 0.5))

        # Honor explicit simulation temperature if provided in config.
        celsius = cfg.get("celsius_C", cfg.get("celsius", None))
        if celsius is not None:
            try:
                h.celsius = float(celsius)
            except Exception:
                pass

        v0 = float(self.cfg.get("v_init_mV", -65.0))
        if self.is_distributed:
            parallel_cfg = cfg.get("parallel") or {}
            try:
                maxstep_default = float(parallel_cfg.get("maxstep_ms", max(chunk, base_dt)))
            except Exception:
                maxstep_default = max(chunk, base_dt)
            try:
                min_delay = float(self._pc.set_maxstep(maxstep_default))
                self._parallel_state = dict(self._parallel_state or {})
                self._parallel_state["pc_min_delay_ms"] = min_delay
            except Exception:
                pass
            if self._gap_transfer_dirty or not self._gap_transfer_ready:
                self._pc.setup_transfer()
                self._gap_transfer_dirty = False
                self._gap_transfer_ready = True
        h.finitialize(v0)

        h.tstop = tstop

        if not show_progress:
            if self.is_distributed:
                self._pc.psolve(tstop)
            else:
                h.run()
            return

        steps = int(math.ceil((tstop - h.t) / max(chunk, 1e-9)))
        if self.is_distributed and not self.is_root_rank:
            iterator = range(steps)
        else:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(steps), total=steps, desc="Simulating (ms)", leave=False)
            except Exception:
                iterator = range(steps)

        for _ in iterator:
            target = min(h.t + chunk, tstop)
            if self.is_distributed:
                self._pc.psolve(target)
            else:
                h.continuerun(target)
            if h.t >= tstop - 1e-12:
                break
        if h.t < tstop - 1e-12:
            if self.is_distributed:
                self._pc.psolve(tstop)
            else:
                h.continuerun(tstop)

    def close(self, *, reset_parallel: bool = True) -> None:
        if self._closed:
            return

        self._closed = True

        if reset_parallel:
            try:
                reset_parallel_context()
            except Exception:
                pass

        cells = list(self.cells.values())

        self.netcons.clear()
        self.syns.clear()
        self._synapse_meta.clear()
        self.gaps.clear()
        self.iclamps.clear()
        self._iclamp_meta.clear()
        self.records = {"t": []}
        self._spike_src.clear()
        self._detectors.clear()
        if self.is_distributed and self._pc is not None:
            try:
                self._pc.gid_clear()
            except Exception:
                pass
        self._gid_detectors.clear()
        self._pins.clear()
        self._parallel_state = None
        self._gap_transfer_dirty = False
        self._gap_transfer_ready = False

        for cell in cells:
            secs = list(getattr(cell, "_secs", []) or [])
            for sec in reversed(secs):
                try:
                    h.delete_section(sec=sec)
                except Exception:
                    pass
            try:
                cell._secs.clear()
            except Exception:
                pass
            for attr, value in (
                ("sections", {}),
                ("_cache_sec_pts", {}),
                ("_nodes", {}),
                ("_children", {}),
                ("_sec_for_node", {}),
                ("soma_sec", None),
                ("_ais_site", None),
            ):
                try:
                    setattr(cell, attr, value)
                except Exception:
                    pass

        self.cells.clear()
