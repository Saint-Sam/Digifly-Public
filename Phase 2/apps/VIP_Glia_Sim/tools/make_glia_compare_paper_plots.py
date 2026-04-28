#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parent
APP_ROOT = TOOLS_DIR.parent
OUTPUTS_ROOT = Path(
    os.environ.get(
        "DIGIFLY_VIP_GLIA_OUTPUTS_ROOT",
        APP_ROOT / "notebooks" / "debug" / "outputs",
    )
)

DEFAULT_COMPARE_ROOT = Path(
    os.environ.get(
        "DIGIFLY_GLIA_COMPARE_ROOT",
        OUTPUTS_ROOT / "glia_neuron_vs_arbor_compare_cache_bench",
    )
)
DEFAULT_PRIMARY_RUN = Path(
    os.environ.get(
        "DIGIFLY_GLIA_PRIMARY_RUN",
        DEFAULT_COMPARE_ROOT / "glia_neuron_vs_arbor_20260225T203804Z",
    )
)
DEFAULT_CACHE_COLD_RUN = Path(
    os.environ.get(
        "DIGIFLY_GLIA_CACHE_COLD_RUN",
        DEFAULT_COMPARE_ROOT / "glia_neuron_vs_arbor_20260225T203804Z",
    )
)
DEFAULT_CACHE_WARM_RUN = Path(
    os.environ.get(
        "DIGIFLY_GLIA_CACHE_WARM_RUN",
        DEFAULT_COMPARE_ROOT / "glia_neuron_vs_arbor_20260225T204700Z",
    )
)
DEFAULT_OUT_PARENT = Path(
    os.environ.get(
        "DIGIFLY_GLIA_FIGURE_OUTPUT_ROOT",
        OUTPUTS_ROOT / "glia_neuron_vs_arbor_paper_figures",
    )
)


COLORS = {
    "neuron": "#1f4e79",
    "arbor": "#b24c2e",
    "accent": "#2a9d8f",
    "muted": "#6b7280",
    "grid": "#d7dce2",
    "warn": "#8f2d56",
    "ok": "#3b7d2b",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fmt_s(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{float(v):.3f} s"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{100.0*float(v):.1f}%"


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _find_run_output_dir(run_dir: Path, key: str) -> Path:
    root = run_dir / key
    if not root.exists():
        raise FileNotFoundError(root)
    subs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subs:
        raise RuntimeError(f"No run subdirectories under {root}")
    if len(subs) == 1:
        return subs[0]
    # Prefer the most recently modified directory.
    return max(subs, key=lambda p: p.stat().st_mtime)


def _parse_worker_stdout_tail(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"raw": text or ""}
    if not text:
        return out

    def _m(pattern: str, flags: int = 0):
        return re.search(pattern, text, flags)

    def _f(pattern: str) -> Optional[float]:
        m = _m(pattern)
        return _safe_float(m.group(1)) if m else None

    def _i(pattern: str) -> Optional[int]:
        m = _m(pattern)
        if not m:
            return None
        try:
            return int(str(m.group(1)).replace(",", ""))
        except Exception:
            return None

    out["loaded_neuron_ids_line"] = (_m(r"loaded neuron_ids\s*=\s*(\[[^\n]+\])") or [None, None])[1]
    out["stim_seeds_line"] = (_m(r"stim seeds\s*=\s*(\[[^\n]+\])") or [None, None])[1]
    out["selector_json"] = (_m(r"selector_json\s*=\s*([^\n]+)") or [None, None])[1]
    out["edge_cache_mode"] = (_m(r"edge_cache\.mode\s*=\s*([^\n]+)") or [None, None])[1]
    out["gap_pairs"] = _i(r"gap\.pairs\s*=\s*([0-9,]+)")
    tdt = _m(r"tstop_ms / dt_ms=\s*([0-9.]+)\s*/\s*([0-9.]+)")
    if tdt:
        out["tstop_ms"] = _safe_float(tdt.group(1))
        out["dt_ms"] = _safe_float(tdt.group(2))
    out["spike_match_tol_ms"] = None

    # Phase timings.
    phase_patterns = {
        "phase_swc_check_s": r"\[phase\] swc availability check: done wall_s=([0-9.]+)",
        "phase_build_subset_s": r"\[phase\] build driven subset: done wall_s=([0-9.]+)",
        "phase_apply_gaps_s": r"\[phase\] apply gaps: done wall_s=([0-9.]+)",
        "phase_prepare_output_s": r"\[phase\] prepare output \+ save config: done wall_s=([0-9.]+)",
        "phase_sim_done_s": r"\[phase\] simulate: done wall_s=([0-9.]+)",
        "phase_save_outputs_s": r"\[phase\] save outputs: done wall_s=([0-9.]+)",
        "build_cell_load_s": r"\[build\] cell load done count=\d+ wall_s=([0-9.]+)",
        "build_passive_s": r"\[build\] passive biophys done count=\d+ wall_s=([0-9.]+)",
        "build_active_s": r"\[build\] active biophys done count=\d+ wall_s=([0-9.]+)",
        "build_synapse_wiring_s": r"\[build\] synapse wiring done rows=[0-9,]+ wall_s=([0-9.]+)",
        "build_driver_clamps_s": r"\[build\] driver clamps done count=\d+ wall_s=([0-9.]+)",
        "build_total_s": r"\[build\] build_network_driven_subset total wall_s=([0-9.]+)",
        "sim_wall_s": r"\[timing\] sim_wall_s=([0-9.]+)",
    }
    for k, p in phase_patterns.items():
        out[k] = _f(p)

    m_backend = _m(r"\[timing\] sim_wall_s=[0-9.]+\s+backend=([^\s]+)")
    out["sim_backend"] = m_backend.group(1) if m_backend else None

    cache_line_match = _m(r"(\[arbor\] native cache: [^\n]+)")
    out["native_cache_line"] = cache_line_match.group(1) if cache_line_match else None

    cache_counts: dict[str, int] = {}
    if cache_line_match:
        for token in cache_line_match.group(1).split(":", 1)[-1].split(","):
            token = token.strip()
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            try:
                cache_counts[k.strip()] = int(v.strip())
            except Exception:
                continue
    out["native_cache_counts"] = cache_counts

    # Gap workaround diagnostics.
    out["rect_raw_pairs"] = _i(r"aggregated duplicate gap site-pairs\s+([0-9,]+)\s*->")
    out["rect_agg_pairs"] = _i(r"aggregated duplicate gap site-pairs\s+[0-9,]+\s*->\s*([0-9,]+)")
    out["rect_compact_before"] = _i(r"compacted rectifying->ohmic contacts across [0-9,]+ cell-pairs\s+([0-9,]+)\s*->")
    out["rect_compact_after"] = _i(r"compacted rectifying->ohmic contacts across [0-9,]+ cell-pairs\s+[0-9,]+\s*->\s*([0-9,]+)")
    out["rect_anchor_contacts"] = _i(r"anchored rectifying->ohmic native gap sites to '[^']+'\s+for\s+([0-9,]+)\s+contact")
    out["rect_demoted_contacts"] = _i(r"demoted\s+([0-9,]+)\s+rectifying->ohmic contact")
    out["rect_post_corr_contacts"] = _i(r"applied post-native surrogate-ohmic correction for\s+([0-9,]+)\s+contact")
    out["rect_post_corr_neurons"] = _i(r"applied post-native surrogate-ohmic correction for\s+[0-9,]+\s+contact\(s\) across\s+([0-9,]+)\s+neuron")

    # Dataset summary from edge-cache log.
    out["edge_cache_loaded"] = _i(r"\[edge-cache\].*loaded=([0-9,]+)")
    out["edge_cache_seeds"] = _i(r"\[edge-cache\].*seeds=([0-9,]+)")
    out["edge_cache_resolved_nodes"] = _i(r"\[edge-cache\].*resolved_nodes=([0-9,]+)")
    out["edge_cache_edges"] = _i(r"\[edge-cache\].*edges=([0-9,]+)")
    return out


@dataclass
class CompareRunData:
    run_dir: Path
    compare_payload: dict[str, Any]
    comparison_summary: dict[str, Any]
    worker_results: dict[str, Any]
    backend_runs: pd.DataFrame
    spike_summary: pd.DataFrame
    spike_per_neuron: pd.DataFrame
    trace_summary: pd.DataFrame
    trace_per_signal: pd.DataFrame
    timing_comparison: pd.DataFrame
    neuron_records: pd.DataFrame
    arbor_records: pd.DataFrame
    parsed_neuron: dict[str, Any]
    parsed_arbor: dict[str, Any]


def load_compare_run(run_dir: Path) -> CompareRunData:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    compare_payload = _read_json(run_dir / "compare_payload.json")
    comparison_summary = _read_json(run_dir / "comparison_summary.json")
    worker_results = _read_json(run_dir / "worker_results.json")
    backend_runs = _read_csv(run_dir / "backend_runs.csv")
    spike_summary = _read_csv(run_dir / "spike_comparison_summary.csv")
    spike_per_neuron = _read_csv(run_dir / "spike_comparison_per_neuron.csv")
    trace_summary = _read_csv(run_dir / "trace_comparison_summary.csv")
    trace_per_signal = _read_csv(run_dir / "trace_comparison_per_signal.csv")
    timing_comparison = _read_csv(run_dir / "timing_comparison.csv")

    neuron_out = Path(comparison_summary["neuron_out_dir"]).expanduser().resolve()
    arbor_out = Path(comparison_summary["arbor_out_dir"]).expanduser().resolve()
    neuron_records = _read_csv(neuron_out / "records.csv")
    arbor_records = _read_csv(arbor_out / "records.csv")

    parsed_neuron = _parse_worker_stdout_tail(str(worker_results.get("neuron", {}).get("stdout_tail", "")))
    parsed_arbor = _parse_worker_stdout_tail(str(worker_results.get("arbor", {}).get("stdout_tail", "")))

    return CompareRunData(
        run_dir=run_dir,
        compare_payload=compare_payload,
        comparison_summary=comparison_summary,
        worker_results=worker_results,
        backend_runs=backend_runs,
        spike_summary=spike_summary,
        spike_per_neuron=spike_per_neuron,
        trace_summary=trace_summary,
        trace_per_signal=trace_per_signal,
        timing_comparison=timing_comparison,
        neuron_records=neuron_records,
        arbor_records=arbor_records,
        parsed_neuron=parsed_neuron,
        parsed_arbor=parsed_arbor,
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str, manifest: list[dict[str, Any]], caption: str) -> None:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    manifest.append({"id": stem, "png": str(png), "svg": str(svg), "caption": caption})
    plt.close(fig)


def _draw_box(ax, x, y, w, h, text, fc="#f7f8fa", ec="#4b5563", lw=1.0, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color="#111827", wrap=True)
    return box


def _arrow(ax, x1, y1, x2, y2, color="#4b5563", lw=1.2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=lw, color=color)
    ax.add_patch(arr)
    return arr


def fig_methods_overview(primary: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    payload_args = primary.compare_payload.get("args", {}) or {}
    neuron_row = primary.worker_results.get("neuron", {}).get("row", {}) or {}
    arbor_row = primary.worker_results.get("arbor", {}).get("row", {}) or {}
    pn, pa = primary.parsed_neuron, primary.parsed_arbor

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1.0], height_ratios=[0.95, 1.05], hspace=0.28, wspace=0.22)

    # Panel A: pipeline diagram.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_title("A. Comparison pipeline (shared scenario, two backend branches)", loc="left", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _draw_box(ax, 0.05, 0.72, 0.38, 0.18, "glia_simulation scenario\n(single_compare)\nselector + forced chem-only edges", fc="#eef4fb", ec=COLORS["neuron"])
    _draw_box(ax, 0.57, 0.72, 0.38, 0.18, "Common controls\nsame loaded neuron set\nsame seeds, dt, tstop\nsame recorded soma traces", fc="#eefaf8", ec=COLORS["accent"])

    _draw_box(ax, 0.05, 0.42, 0.38, 0.18, "NEURON / CoreNEURON worker\nPhase 2 (original backend)", fc="#eef4fb", ec=COLORS["neuron"])
    _draw_box(ax, 0.57, 0.42, 0.38, 0.18, "Arbor worker\nbackends/arbor_phase2\n(native Arbor path enabled)", fc="#fbf2ee", ec=COLORS["arbor"])
    _draw_box(ax, 0.31, 0.10, 0.38, 0.18, "Post-run comparison\nruntime + worker wall\nspikes (tol match)\ntrace RMSE / correlation", fc="#f7f7fb", ec="#5b5f97")

    _arrow(ax, 0.24, 0.72, 0.24, 0.60)
    _arrow(ax, 0.76, 0.72, 0.76, 0.60)
    _arrow(ax, 0.24, 0.42, 0.43, 0.28)
    _arrow(ax, 0.76, 0.42, 0.57, 0.28)
    _arrow(ax, 0.43, 0.81, 0.57, 0.81)

    # Panel B: methods and parameter summary.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.set_title("B. Controlled conditions used for the comparison", loc="left", pad=6)
    methods_lines = [
        f"Scenario: {neuron_row.get('scenario', 'single_compare')} (glia notebook helper path)",
        f"Selector JSON: {Path(str(neuron_row.get('selector_json', ''))).name}",
        f"Loaded neurons: {pn.get('edge_cache_resolved_nodes', 'NA')}  |  Seeds: {pn.get('edge_cache_seeds', 'NA')}",
        f"Edge subset rows (cache view): {pn.get('edge_cache_edges', 'NA')}",
        f"Recorded soma traces: {neuron_row.get('recorded_soma_traces', 'NA')}",
        f"Gap pairs configured: {pn.get('gap_pairs', 'NA')}  |  Glia state: {neuron_row.get('run_state_tag', 'NA')}",
        f"dt = {pn.get('dt_ms', 'NA')} ms,  tstop = {pn.get('tstop_ms', 'NA')} ms",
        f"Spike match tolerance: {primary.compare_payload.get('spike_match_tol_ms', 'NA')} ms",
        f"Launch mode: {'parallel' if primary.compare_payload.get('parallel_launch') else 'sequential'}",
        f"Arbor rectifying-gap policy: {payload_args.get('arbor_rect_gap_policy', 'NA')}",
        f"Gap-unsafe handling target(s): {payload_args.get('arbor_gap_exclude_nids', 'NA')} (demoted to surrogate-ohmic correction)",
        f"Arbor worker python: {Path(str(payload_args.get('arbor_worker_python', ''))).name if payload_args.get('arbor_worker_python') else 'system'}",
    ]
    y = 0.94
    for line in methods_lines:
        ax2.text(0.01, y, line, ha="left", va="top", fontsize=9.5, family="DejaVu Sans Mono")
        y -= 0.075

    # Panel C: Arbor native gap handling transformations.
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("C. Arbor-native rectifying-gap handling used in the comparison (current workaround path)", loc="left", pad=6)
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    raw_pairs = pa.get("rect_raw_pairs")
    agg_pairs = pa.get("rect_agg_pairs")
    c_before = pa.get("rect_compact_before")
    c_after = pa.get("rect_compact_after")
    demoted = pa.get("rect_demoted_contacts")
    anchored = pa.get("rect_anchor_contacts")
    postcorr = pa.get("rect_post_corr_contacts")

    _draw_box(ax3, 0.03, 0.58, 0.20, 0.22, f"Rectifying gaps from glia workflow\nraw contact entries\n{raw_pairs or 'NA'}", fc="#fff4f0", ec=COLORS["arbor"])
    _draw_box(ax3, 0.28, 0.58, 0.20, 0.22, f"Duplicate site-pair aggregation\n{raw_pairs or 'NA'} -> {agg_pairs or 'NA'}", fc="#fff9ef", ec="#d08c2c")
    _draw_box(ax3, 0.53, 0.58, 0.20, 0.22, f"Per-cell-pair compaction\n{c_before or 'NA'} -> {c_after or 'NA'}\n(conductance preserved)", fc="#f4fbf2", ec=COLORS["ok"])
    _draw_box(ax3, 0.78, 0.58, 0.19, 0.22, f"Site anchoring + native gj build\nanchored: {anchored or 'NA'}", fc="#eef7fb", ec=COLORS["neuron"])

    _arrow(ax3, 0.23, 0.69, 0.28, 0.69)
    _arrow(ax3, 0.48, 0.69, 0.53, 0.69)
    _arrow(ax3, 0.73, 0.69, 0.78, 0.69)

    _draw_box(ax3, 0.16, 0.18, 0.30, 0.22, f"Gap-unsafe neuron workaround\ncontacts involving nid 11654\ndemoted: {demoted or 'NA'}", fc="#fff5f8", ec=COLORS["warn"])
    _draw_box(ax3, 0.54, 0.18, 0.30, 0.22, f"Post-native surrogate-ohmic correction\napplied contacts: {postcorr or 'NA'}", fc="#f8f7ff", ec="#5b5f97")
    _arrow(ax3, 0.78, 0.58, 0.68, 0.40)
    _arrow(ax3, 0.62, 0.58, 0.31, 0.40)
    ax3.text(
        0.02, 0.03,
        "All timings and parity results shown in subsequent figures use this exact configuration and workflow.",
        fontsize=9.5, color="#374151",
    )

    fig.suptitle("NEURON vs Arbor glia comparison: methods and execution path (paper-style summary)", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig01_methods_overview",
        manifest,
        "Methods overview figure showing the shared glia comparison pipeline, controlled conditions, and the Arbor-native rectifying-gap workaround path used in the runs.",
    )


def fig_timing_decomposition(primary: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    trow = primary.timing_comparison.iloc[0]
    pn = primary.parsed_neuron
    pa = primary.parsed_arbor

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # Panel A: headline runtime metrics.
    ax = fig.add_subplot(gs[0, 0])
    metrics = ["runtime_s", "worker_wall_s", "sim_wall_s"]
    labels = ["Run metric\n(runtime_s)", "Worker wall\n(end-to-end worker)", "Simulation phase\n(sim_wall_s)"]
    neuron_vals = [float(trow["neuron_runtime_s"]), float(trow["neuron_worker_wall_s"]), float(pn.get("sim_wall_s") or np.nan)]
    arbor_vals = [float(trow["arbor_runtime_s"]), float(trow["arbor_worker_wall_s"]), float(pa.get("sim_wall_s") or np.nan)]
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, neuron_vals, width=w, color=COLORS["neuron"], label="NEURON/CoreNEURON")
    ax.bar(x + w/2, arbor_vals, width=w, color=COLORS["arbor"], label="Arbor (native)")
    for i, (nv, av) in enumerate(zip(neuron_vals, arbor_vals)):
        if np.isfinite(nv):
            ax.text(i - w/2, nv, f"{nv:.1f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(av):
            ax.text(i + w/2, av, f"{av:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_title("A. Headline timing metrics")
    ax.legend(loc="upper left", frameon=False)

    # Panel B: phase timing decomposition.
    ax2 = fig.add_subplot(gs[0, 1])
    phase_keys = [
        ("phase_swc_check_s", "SWC availability"),
        ("phase_build_subset_s", "Build driven subset"),
        ("phase_apply_gaps_s", "Apply gaps"),
        ("phase_prepare_output_s", "Prepare output"),
        ("phase_sim_done_s", "Simulate"),
        ("phase_save_outputs_s", "Save outputs"),
    ]
    phase_labels = [lab for _, lab in phase_keys]
    nvals = [pn.get(k) or 0.0 for k, _ in phase_keys]
    avals = [pa.get(k) or 0.0 for k, _ in phase_keys]
    y = np.arange(len(phase_labels))
    ax2.barh(y - 0.18, nvals, height=0.32, color=COLORS["neuron"], label="NEURON/CoreNEURON")
    ax2.barh(y + 0.18, avals, height=0.32, color=COLORS["arbor"], label="Arbor (native)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(phase_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Seconds")
    ax2.set_title("B. Parsed worker phase timings")
    ax2.legend(loc="lower right", frameon=False)

    # Panel C: build substeps that often dominate.
    ax3 = fig.add_subplot(gs[1, 0])
    sub_keys = [
        ("build_cell_load_s", "Cell load"),
        ("build_synapse_wiring_s", "Synapse wiring"),
        ("build_passive_s", "Passive biophys"),
        ("build_active_s", "Active biophys"),
        ("build_driver_clamps_s", "Driver clamps"),
    ]
    xs = np.arange(len(sub_keys))
    nsub = [pn.get(k) or 0.0 for k, _ in sub_keys]
    asub = [pa.get(k) or 0.0 for k, _ in sub_keys]
    ax3.bar(xs - w/2, nsub, width=w, color=COLORS["neuron"])
    ax3.bar(xs + w/2, asub, width=w, color=COLORS["arbor"])
    ax3.set_xticks(xs)
    ax3.set_xticklabels([lab for _, lab in sub_keys], rotation=20, ha="right")
    ax3.set_ylabel("Seconds")
    ax3.set_title("C. Build substeps (parsed from worker stdout tail)")

    # Panel D: ratios and notes.
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    rt_ratio = float(trow["arbor_runtime_div_neuron_runtime"])
    sim_ratio = (float(pa.get("sim_wall_s")) / float(pn.get("sim_wall_s"))) if (pn.get("sim_wall_s") and pa.get("sim_wall_s")) else float("nan")
    worker_diff = float(trow["arbor_worker_wall_s"]) - float(trow["neuron_worker_wall_s"])
    txt = [
        "D. Timing interpretation (primary run)",
        "",
        f"Arbor runtime / NEURON runtime: {rt_ratio:.3f}x",
        f"Arbor sim_wall / NEURON sim_wall: {sim_ratio:.3f}x",
        f"Arbor worker wall - NEURON worker wall: {worker_diff:+.3f} s",
        "",
        "Notes:",
        "- `runtime_s` comes from the scenario row written by each backend worker.",
        "- `worker_wall_s` includes worker setup/build/sim/save plus notebook helper overhead.",
        "- `sim_wall_s` is the backend's timed integration/simulation phase only.",
        "- Arbor run shown here used native execution with the rectifying-gap workaround.",
    ]
    y0 = 0.97
    for i, line in enumerate(txt):
        fs = 11 if i == 0 else 9.5
        family = "DejaVu Serif" if i == 0 else "DejaVu Sans"
        ax4.text(0.01, y0, line, ha="left", va="top", fontsize=fs, family=family, color="#111827")
        y0 -= 0.085 if line else 0.045

    fig.suptitle("Timing results for the primary NEURON vs Arbor glia comparison run", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig02_timing_decomposition",
        manifest,
        "Timing decomposition for the primary comparison run, including headline runtime metrics, worker phase timings, and build substeps parsed from worker logs.",
    )


def fig_spike_parity(primary: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    ss = primary.spike_summary.iloc[0]
    spn = primary.spike_per_neuron.copy()
    spn = spn.sort_values(["count_delta_arbor_minus_neuron", "arbor_spike_count", "neuron_id"], ascending=[False, False, True]).reset_index(drop=True)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.9, 1.25], hspace=0.30, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    totals = ["Total spikes", "Neurons with spikes"]
    neuron_vals = [float(ss["neuron_total_spikes"]), float(ss["neurons_with_spikes_neuron"])]
    arbor_vals = [float(ss["arbor_total_spikes"]), float(ss["neurons_with_spikes_arbor"])]
    x = np.arange(len(totals))
    w = 0.35
    ax.bar(x - w/2, neuron_vals, width=w, color=COLORS["neuron"], label="NEURON")
    ax.bar(x + w/2, arbor_vals, width=w, color=COLORS["arbor"], label="Arbor")
    ax.set_xticks(x)
    ax.set_xticklabels(totals)
    ax.set_ylabel("Count")
    ax.set_title("A. Aggregate spike counts")
    ax.legend(frameon=False, loc="upper left")
    for i, (a, b) in enumerate(zip(neuron_vals, arbor_vals)):
        ax.text(i - w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    qual_labels = ["Recall\n(vs NEURON)", "Precision\n(vs Arbor)", "F1"]
    qual_vals = [float(ss["recall_vs_neuron"]), float(ss["precision_vs_arbor"]), float(ss["f1"])]
    bars = ax2.bar(np.arange(3), qual_vals, color=[COLORS["accent"], "#8aa1c1", COLORS["warn"]], width=0.55)
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(qual_labels)
    ax2.set_ylabel("Metric (0-1)")
    ax2.set_title(f"B. Spike matching quality (tolerance = {float(ss['tol_ms']):.2f} ms)")
    for b, v in zip(bars, qual_vals):
        ax2.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax3 = fig.add_subplot(gs[1, :])
    plot_df = spn.copy()
    # Keep order readable; plot lollipop with counts side-by-side.
    y = np.arange(len(plot_df))
    ax3.hlines(y, xmin=plot_df["neuron_spike_count"], xmax=plot_df["arbor_spike_count"], color="#c7ccd4", lw=1.2)
    ax3.scatter(plot_df["neuron_spike_count"], y, color=COLORS["neuron"], s=45, label="NEURON count", zorder=3)
    ax3.scatter(plot_df["arbor_spike_count"], y, color=COLORS["arbor"], s=45, label="Arbor count", zorder=3)
    for yi, nid in zip(y, plot_df["neuron_id"]):
        ax3.text(-0.15, yi, str(int(nid)), ha="right", va="center", fontsize=8, color="#374151")
    ax3.set_yticks([])
    ax3.set_xlabel("Spike count in run")
    ax3.set_title("C. Per-neuron spike count comparison (labels at left = neuron IDs)")
    ax3.legend(frameon=False, loc="upper right")
    ax3.set_xlim(left=-0.6)
    ax3.text(
        0.0,
        -0.12,
        f"Matched spikes = {int(ss['matched_spikes'])} of {int(ss['neuron_total_spikes'])} NEURON spikes "
        f"(Δ total spikes Arbor-NEURON = {int(ss['total_spike_delta_arbor_minus_neuron'])}).",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )

    fig.suptitle("Spike-level parity between NEURON and Arbor (primary run)", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig03_spike_parity",
        manifest,
        "Spike-parity figure for the primary run, showing aggregate spike counts, spike-matching quality metrics, and per-neuron spike count differences.",
    )


def fig_trace_quality(primary: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    ts = primary.trace_summary.iloc[0]
    tps = primary.trace_per_signal.copy().sort_values("rmse", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    # Distribution of RMSE/MAE (same units mV).
    bins = min(8, max(4, len(tps)//2))
    ax.hist(tps["rmse"].dropna(), bins=bins, alpha=0.75, color=COLORS["arbor"], label="RMSE")
    ax.hist(tps["mae"].dropna(), bins=bins, alpha=0.55, color=COLORS["neuron"], label="MAE")
    ax.axvline(float(ts["mean_rmse"]), color=COLORS["warn"], ls="--", lw=1.3, label=f"Mean RMSE = {float(ts['mean_rmse']):.3f}")
    ax.axvline(float(ts["median_rmse"]), color=COLORS["accent"], ls=":", lw=1.5, label=f"Median RMSE = {float(ts['median_rmse']):.3f}")
    ax.set_xlabel("Error (mV)")
    ax.set_ylabel("Trace count")
    ax.set_title("A. Distribution of trace errors across recorded soma signals")
    ax.legend(frameon=False, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(
        tps["rmse"],
        tps["corr"],
        c=tps["max_abs_err"],
        cmap="magma",
        s=70,
        edgecolor="white",
        linewidth=0.4,
    )
    for _, row in tps.head(5).iterrows():
        ax2.text(float(row["rmse"]) + 0.02, float(row["corr"]), str(row["signal"]), fontsize=8)
    ax2.set_xlabel("RMSE (mV)")
    ax2.set_ylabel("Correlation")
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_title("B. Per-signal RMSE vs correlation (color = max abs error)")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Max abs error (mV)")

    ax3 = fig.add_subplot(gs[1, :])
    top = tps.head(min(12, len(tps))).copy()
    y = np.arange(len(top))
    ax3.barh(y, top["rmse"], color=COLORS["arbor"], alpha=0.85, label="RMSE")
    ax3.plot(top["mae"], y, "o-", color=COLORS["neuron"], lw=1.2, ms=4, label="MAE")
    ax3.set_yticks(y)
    ax3.set_yticklabels(top["signal"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Error (mV)")
    ax3.set_title("C. Worst-case signals by RMSE (primary run)")
    ax3.legend(frameon=False, loc="lower right")

    fig.text(
        0.02,
        0.01,
        (
            f"Trace summary: status={ts['status']}, NEURON samples={int(ts['neuron_samples'])}, "
            f"Arbor samples={int(ts['arbor_samples'])}, overlap samples used by comparator={int(ts['overlap_samples'])}, "
            f"common traces={int(ts['common_trace_columns'])}, mean corr={float(ts['mean_corr']):.3f}."
        ),
        fontsize=9,
        color="#374151",
    )
    fig.suptitle("Voltage-trace comparison quality metrics (primary run)", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig04_trace_quality",
        manifest,
        "Trace-quality figure summarizing per-signal voltage error metrics and correlation for the primary comparison run.",
    )


def _pick_trace_signals(primary: CompareRunData, n_top: int = 4, n_bottom: int = 2) -> list[str]:
    tps = primary.trace_per_signal.copy().sort_values(["rmse", "signal"], ascending=[False, True]).reset_index(drop=True)
    top = list(tps.head(min(n_top, len(tps)))["signal"])
    bottom = list(tps.tail(min(n_bottom, max(0, len(tps)-len(top))))["signal"])
    # Preserve unique order.
    out: list[str] = []
    for s in top + bottom:
        if s not in out:
            out.append(str(s))
    return out


def fig_trace_overlays(primary: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    signals = _pick_trace_signals(primary, n_top=4, n_bottom=2)
    n = len(signals)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2*nrows), squeeze=False)

    rec_n = primary.neuron_records
    rec_a = primary.arbor_records
    t_n = rec_n["t_ms"].to_numpy(dtype=float) if "t_ms" in rec_n.columns else np.arange(len(rec_n), dtype=float)
    t_a = rec_a["t_ms"].to_numpy(dtype=float) if "t_ms" in rec_a.columns else np.arange(len(rec_a), dtype=float)
    metric_by_signal = {str(r["signal"]): r for _, r in primary.trace_per_signal.iterrows()}

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, sig in zip(axes.ravel(), signals):
        ax.set_visible(True)
        if sig not in rec_n.columns or sig not in rec_a.columns:
            ax.text(0.5, 0.5, f"Missing signal {sig}", ha="center", va="center")
            continue
        ax.plot(t_n, rec_n[sig].to_numpy(dtype=float), color=COLORS["neuron"], lw=1.5, label="NEURON")
        ax.plot(t_a, rec_a[sig].to_numpy(dtype=float), color=COLORS["arbor"], lw=1.2, alpha=0.9, label="Arbor")
        m = metric_by_signal.get(sig)
        if m is not None:
            rmse = float(m["rmse"])
            corr = float(m["corr"]) if np.isfinite(m["corr"]) else float("nan")
            ax.set_title(f"{sig} | RMSE={rmse:.3f} mV, corr={corr:.3f}", fontsize=10)
        else:
            ax.set_title(sig, fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Soma V (mV)")
        # Add zoom guide for early dynamics/spikes.
        ax.axvspan(0, 6, color="#f4f6fa", alpha=0.35, lw=0)
    handles = [
        Line2D([0], [0], color=COLORS["neuron"], lw=1.6, label="NEURON/CoreNEURON"),
        Line2D([0], [0], color=COLORS["arbor"], lw=1.3, label="Arbor (native)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle("Representative soma-voltage overlays (highest and lowest RMSE signals, primary run)", x=0.02, ha="left", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_fig(
        fig,
        out_dir,
        "fig05_trace_overlays",
        manifest,
        "Representative soma-voltage overlays comparing NEURON and Arbor traces for high-RMSE and low-RMSE signals in the primary run.",
    )


def fig_cache_benchmark(cold: CompareRunData, warm: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    pc, pw = cold.parsed_arbor, warm.parsed_arbor
    t_c = cold.timing_comparison.iloc[0]
    t_w = warm.timing_comparison.iloc[0]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # Panel A: cache event counts.
    ax = fig.add_subplot(gs[0, 0])
    keys = [
        "branch_disk_miss",
        "branch_disk_write",
        "branch_disk_hit",
        "swcfix_disk_miss",
        "swcfix_disk_write",
        "swcfix_disk_hit",
    ]
    labels = ["branch miss", "branch write", "branch hit", "swcfix miss", "swcfix write", "swcfix hit"]
    cold_counts = [int((pc.get("native_cache_counts") or {}).get(k, 0)) for k in keys]
    warm_counts = [int((pw.get("native_cache_counts") or {}).get(k, 0)) for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, cold_counts, width=w, color="#9ca3af", label="Cold run")
    ax.bar(x + w/2, warm_counts, width=w, color=COLORS["accent"], label="Warm run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("A. Persistent cache activity (Arbor native prebuild)")
    ax.legend(frameon=False, loc="upper right")

    # Panel B: Arbor-specific timing comparison.
    ax2 = fig.add_subplot(gs[0, 1])
    timing_keys = [
        ("build_cell_load_s", "Cell load"),
        ("phase_build_subset_s", "Build subset"),
        ("phase_apply_gaps_s", "Apply gaps"),
        ("sim_wall_s", "Sim wall"),
        ("phase_sim_done_s", "Phase simulate"),
    ]
    x2 = np.arange(len(timing_keys))
    cold_vals = [float(pc.get(k) or np.nan) for k, _ in timing_keys]
    warm_vals = [float(pw.get(k) or np.nan) for k, _ in timing_keys]
    ax2.bar(x2 - w/2, cold_vals, width=w, color=COLORS["arbor"], alpha=0.55, label="Cold")
    ax2.bar(x2 + w/2, warm_vals, width=w, color=COLORS["arbor"], alpha=0.90, label="Warm")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lab for _, lab in timing_keys], rotation=20, ha="right")
    ax2.set_ylabel("Seconds")
    ax2.set_title("B. Arbor worker phase timings (cold vs warm)")
    ax2.legend(frameon=False, loc="upper left")

    # Panel C: high-level run metrics across both backends for context.
    ax3 = fig.add_subplot(gs[1, :])
    metric_names = [
        ("neuron_runtime_s", "NEURON runtime"),
        ("arbor_runtime_s", "Arbor runtime"),
        ("neuron_worker_wall_s", "NEURON worker wall"),
        ("arbor_worker_wall_s", "Arbor worker wall"),
    ]
    xm = np.arange(len(metric_names))
    cvals = [float(t_c[k]) for k, _ in metric_names]
    wvals = [float(t_w[k]) for k, _ in metric_names]
    ax3.bar(xm - w/2, cvals, width=w, color="#b6bec8", label="Cold comparison run")
    ax3.bar(xm + w/2, wvals, width=w, color="#5f6b7a", label="Warm comparison run")
    ax3.set_xticks(xm)
    ax3.set_xticklabels([lab for _, lab in metric_names], rotation=12, ha="right")
    ax3.set_ylabel("Seconds")
    ax3.set_title("C. Context: whole comparison run variability can dominate cache gains")
    for i, (cv, wv) in enumerate(zip(cvals, wvals)):
        delta = wv - cv
        ax3.text(i, max(cv, wv) + 2, f"{delta:+.1f}s", ha="center", va="bottom", fontsize=8, color="#374151")
    ax3.legend(frameon=False, loc="upper left")
    ax3.text(
        0.01,
        -0.18,
        (
            "Interpretation: persistent Arbor caching is confirmed (all 15 branch caches + 15 repaired SWCs hit on the warm run), "
            "but end-to-end comparison wall time remains noisy because synapse wiring/build phases vary substantially between runs."
        ),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
    )

    fig.suptitle("Persistent Arbor cache benchmark (cold vs warm comparison runs)", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig06_cache_benchmark",
        manifest,
        "Cold-versus-warm Arbor cache benchmark showing persistent cache activity counts and timing variability across comparison reruns.",
    )


def fig_summary_dashboard(primary: CompareRunData, cold: CompareRunData, warm: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    ss = primary.spike_summary.iloc[0]
    ts = primary.trace_summary.iloc[0]
    tc = primary.timing_comparison.iloc[0]
    pc, pw = cold.parsed_arbor, warm.parsed_arbor

    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    # 1: timing ratio
    ax = fig.add_subplot(gs[0, 0])
    vals = [float(tc["arbor_runtime_div_neuron_runtime"]), float((primary.parsed_arbor.get("sim_wall_s") or np.nan) / (primary.parsed_neuron.get("sim_wall_s") or np.nan))]
    labels = ["Runtime ratio\n(Arbor/NEURON)", "Sim wall ratio\n(Arbor/NEURON)"]
    bars = ax.bar(np.arange(2), vals, color=[COLORS["arbor"], COLORS["warn"]], width=0.55)
    ax.axhline(1.0, color="#374151", lw=1.0, ls="--")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ratio")
    ax.set_title("A. Speed ratios")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)

    # 2: spike metrics
    ax2 = fig.add_subplot(gs[0, 1])
    spike_vals = [float(ss["recall_vs_neuron"]), float(ss["precision_vs_arbor"]), float(ss["f1"])]
    ax2.bar(np.arange(3), spike_vals, color=[COLORS["accent"], "#8aa1c1", COLORS["warn"]], width=0.55)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(["Recall", "Precision", "F1"])
    ax2.set_title("B. Spike parity")
    ax2.set_ylabel("0-1")

    # 3: trace metrics
    ax3 = fig.add_subplot(gs[0, 2])
    trace_vals = [float(ts["mean_rmse"]), float(ts["median_rmse"]), float(ts["mean_corr"])]
    ax3.bar([0, 1], trace_vals[:2], color=COLORS["arbor"], width=0.55, alpha=0.85)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["Mean RMSE\n(mV)", "Median RMSE\n(mV)", "Mean corr"])
    ax3.bar([2], [trace_vals[2]], color=COLORS["neuron"], width=0.55)
    ax3.set_title("C. Trace parity")

    # 4: cache state
    ax4 = fig.add_subplot(gs[1, 0])
    cold_hits = int((pc.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    warm_hits = int((pw.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    cold_writes = int((pc.get("native_cache_counts") or {}).get("branch_disk_write", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_write", 0))
    warm_writes = int((pw.get("native_cache_counts") or {}).get("branch_disk_write", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_write", 0))
    ax4.bar([0, 1], [cold_writes, warm_writes], color=["#9ca3af", COLORS["accent"]], width=0.55)
    ax4.plot([0, 1], [cold_hits, warm_hits], "o-", color=COLORS["arbor"], lw=1.5, label="Disk hits")
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Cold", "Warm"])
    ax4.set_ylabel("Count")
    ax4.set_title("D. Arbor cache behavior")
    ax4.legend(frameon=False, loc="upper left")

    # 5-6: methods/caveats text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    lines = [
        "E. Summary and caveats for manuscript-style reporting",
        "",
        f"Primary run: {primary.run_dir.name}",
        f"NEURON phase2 root: {Path(str(primary.worker_results['neuron']['row'].get('phase2_root',''))).name}",
        f"Arbor phase2 root: {Path(str(primary.worker_results['arbor']['row'].get('phase2_root',''))).name}",
        f"Shared scenario: {primary.worker_results['neuron']['row'].get('scenario')} | run_state_tag={primary.worker_results['neuron']['row'].get('run_state_tag')}",
        "Arbor native rectifying gaps were approximated as symmetric ohmic; contacts involving nid 11654 were demoted to a post-native surrogate-ohmic correction.",
        "Persistent Arbor cache stores repaired SWCs and branch geometry per SWC fingerprint (path+mtime+size+node-count).",
        "Cache improved Arbor prebuild reuse (cold writes -> warm hits), but comparison-level wall time remained sensitive to build/synapse-wiring variability.",
        "Result parity remains the gating issue for replacement in this glia workflow (spike mismatch is still large).",
    ]
    y = 0.98
    for i, line in enumerate(lines):
        fs = 11 if i == 0 else 9.5
        ax5.text(0.01, y, line, ha="left", va="top", fontsize=fs, color="#111827")
        y -= 0.11 if line == "" else 0.085

    fig.suptitle("Paper-style comparison dashboard (results + methodology context)", x=0.02, ha="left", fontsize=15, y=0.99)
    _save_fig(
        fig,
        out_dir,
        "fig07_summary_dashboard",
        manifest,
        "Paper-style dashboard summarizing timing, spike and trace parity, cache behavior, and experimental caveats across the comparison runs.",
    )


def _signal_to_neuron_id(signal: Any) -> Optional[int]:
    s = str(signal)
    m = re.match(r"^\s*(\d+)_", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def fig_abstract_two_panel(primary: CompareRunData, cold: CompareRunData, warm: CompareRunData, out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    trow = primary.timing_comparison.iloc[0]
    ss = primary.spike_summary.iloc[0]
    ts = primary.trace_summary.iloc[0]
    pn, pa = primary.parsed_neuron, primary.parsed_arbor
    pc, pw = cold.parsed_arbor, warm.parsed_arbor

    sim_ratio = (
        float(pa.get("sim_wall_s")) / float(pn.get("sim_wall_s"))
        if (pn.get("sim_wall_s") is not None and pa.get("sim_wall_s") is not None and float(pn.get("sim_wall_s")) > 0)
        else np.nan
    )
    runtime_ratio = float(trow["arbor_runtime_div_neuron_runtime"])
    worker_ratio = float(trow["arbor_worker_wall_s"]) / float(trow["neuron_worker_wall_s"]) if float(trow["neuron_worker_wall_s"]) > 0 else np.nan

    # Merge per-trace and per-neuron spike deltas to annotate parity by neuron.
    trace_df = primary.trace_per_signal.copy()
    spike_df = primary.spike_per_neuron.copy()
    trace_df["neuron_id"] = trace_df["signal"].map(_signal_to_neuron_id)
    merged = trace_df.merge(
        spike_df[["neuron_id", "count_delta_arbor_minus_neuron", "neuron_spike_count", "arbor_spike_count"]],
        on="neuron_id",
        how="left",
    )
    merged["count_delta_arbor_minus_neuron"] = merged["count_delta_arbor_minus_neuron"].fillna(0.0)
    merged["spike_mismatch"] = merged["count_delta_arbor_minus_neuron"] != 0

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.96, 1.20], wspace=0.22)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.84, bottom=0.22, wspace=0.24)

    # Panel A: performance summary in normalized form.
    ax = fig.add_subplot(gs[0, 0])
    ratios = np.array([runtime_ratio, worker_ratio, sim_ratio], dtype=float)
    ratio_labels = ["Scenario runtime\n(runtime_s)", "Worker wall\n(worker_wall_s)", "Simulation phase\n(sim_wall_s)"]
    colors = [COLORS["arbor"] if v >= 1.0 else COLORS["ok"] for v in ratios]
    x = np.arange(len(ratios))
    bars = ax.bar(x, ratios, color=colors, width=0.60)
    ax.axhline(1.0, color="#374151", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels)
    ax.set_ylabel("Arbor / NEURON ratio")
    ax.set_title("A. Performance summary (primary comparison run)", pad=10)
    ax.set_ylim(0, max(1.55, float(np.nanmax(ratios)) + 0.20))
    for b, v in zip(bars, ratios):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.03,
            f"{v:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    cold_sim = float(pc.get("sim_wall_s") or np.nan)
    warm_sim = float(pw.get("sim_wall_s") or np.nan)
    delta_sim = warm_sim - cold_sim if np.isfinite(cold_sim) and np.isfinite(warm_sim) else np.nan
    cold_hits = int((pc.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    warm_hits = int((pw.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    ax.text(
        0.02,
        0.98,
        "Ratios shown for primary NEURON-vs-Arbor run (dashed line = 1.0 parity).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#4b5563",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#e5e7eb", alpha=0.95),
    )
    # Panel B: parity summary scatter with spike mismatch encoding.
    ax2 = fig.add_subplot(gs[0, 1])
    cvals = merged["count_delta_arbor_minus_neuron"].to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(cvals)))) if len(cvals) else 1.0
    sizes = np.where(merged["spike_mismatch"].to_numpy(dtype=bool), 95, 55)
    sc = ax2.scatter(
        merged["rmse"].to_numpy(dtype=float),
        merged["corr"].to_numpy(dtype=float),
        c=cvals,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=sizes,
        alpha=0.9,
        edgecolor=np.where(merged["spike_mismatch"].to_numpy(dtype=bool), "#111827", "white"),
        linewidth=0.8,
    )
    ax2.axhline(0.95, color=COLORS["grid"], lw=1.0, ls="--")
    ax2.axvline(0.5, color=COLORS["grid"], lw=1.0, ls="--")
    ax2.set_xlabel("Trace RMSE (mV)")
    ax2.set_ylabel("Trace correlation")
    ax2.set_ylim(-1.05, 1.02)
    # Give a bit more headroom on the right for labels.
    x_min = float(np.nanmin(merged["rmse"].to_numpy(dtype=float))) if len(merged) else 0.0
    x_max = float(np.nanmax(merged["rmse"].to_numpy(dtype=float))) if len(merged) else 1.0
    ax2.set_xlim(min(-0.15, x_min - 0.15), x_max + 0.55)
    ax2.set_title("B. Parity summary across recorded neurons (trace quality + spike mismatch)", pad=10)
    for _, row in merged.sort_values("rmse", ascending=False).head(min(4, len(merged))).iterrows():
        x0 = float(row["rmse"])
        y0 = float(row["corr"])
        dy = -8 if y0 > 0.92 else 4
        va = "top" if y0 > 0.92 else "bottom"
        ax2.annotate(
            str(row["signal"]),
            (x0, y0),
            xytext=(5, dy),
            textcoords="offset points",
            ha="left",
            va=va,
            fontsize=8,
            clip_on=True,
        )

    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("Spike delta (Arbor − NEURON)")

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="white", markersize=7, label="No spike count mismatch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#fdae6b", markeredgecolor="#111827", markersize=8, label="Spike count mismatch"),
    ]
    ax2.legend(handles=legend_elems, frameon=False, loc="lower right", borderpad=0.2, handletextpad=0.5)

    # Move dense explanatory text outside the plotting region to avoid covering data.
    left_note = (
        f"Cache benchmark (separate Arbor-only reruns): warm cache hits={warm_hits} (cold hits={cold_hits}), "
        f"sim_wall_s {cold_sim:.2f}\u2192{warm_sim:.2f} s (\u0394={delta_sim:+.2f} s)."
    )
    right_note = (
        f"Spikes: NEURON={int(ss['neuron_total_spikes'])}, Arbor={int(ss['arbor_total_spikes'])} | "
        f"Recall={float(ss['recall_vs_neuron']):.2f}, Precision={float(ss['precision_vs_arbor']):.2f}, F1={float(ss['f1']):.2f}\n"
        f"Traces: mean RMSE={float(ts['mean_rmse']):.3f} mV, mean corr={float(ts['mean_corr']):.3f} | "
        f"Common traces={int(ts['common_trace_columns'])}, overlap samples={int(ts['overlap_samples'])}"
    )
    fig.text(
        0.075,
        0.125,
        left_note,
        ha="left",
        va="top",
        fontsize=8.9,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.97),
    )
    fig.text(
        0.515,
        0.125,
        right_note,
        ha="left",
        va="top",
        fontsize=8.8,
        family="DejaVu Sans Mono",
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.97),
    )

    fig.suptitle(
        "Figure Abstract: Arbor is promising, but not yet a drop-in replacement for this glia workflow",
        x=0.02,
        ha="left",
        fontsize=14.5,
        y=0.94,
    )
    fig.text(
        0.02,
        0.035,
        (
            "Takeaway: Arbor native execution runs and benefits from persistent SWC/branch caching, "
            "but in the current glia comparison it remains slower in the timed simulation phase and shows poor spike-level parity vs NEURON/CoreNEURON."
        ),
        fontsize=9.4,
        color="#374151",
    )

    _save_fig(
        fig,
        out_dir,
        "fig08_abstract_two_panel",
        manifest,
        "Two-panel figure-abstract summary: Panel A shows performance ratios (Arbor/NEURON) and cache rerun context; Panel B shows trace parity and spike mismatch across recorded neurons, supporting the conclusion that Arbor is not yet a drop-in replacement for this glia workflow.",
    )


def write_caption_files(out_dir: Path, manifest: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    (out_dir / "figure_manifest.json").write_text(json.dumps({"figures": manifest, "meta": meta}, indent=2), encoding="utf-8")
    lines = ["# Glia NEURON vs Arbor Paper-Style Figures", ""]
    for fig in manifest:
        lines.append(f"## {fig['id']}")
        lines.append("")
        lines.append(fig["caption"])
        lines.append("")
        lines.append(f"- PNG: `{fig['png']}`")
        lines.append(f"- SVG: `{fig['svg']}`")
        lines.append("")
    (out_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def _default_out_dir(parent: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"glia_neuron_vs_arbor_paper_figures_{ts}"


def build_figures(primary_run: Path, cache_cold_run: Path, cache_warm_run: Path, out_dir: Path) -> dict[str, Any]:
    _setup_style()
    out_dir = _ensure_out_dir(out_dir)

    primary = load_compare_run(primary_run)
    cold = load_compare_run(cache_cold_run)
    warm = load_compare_run(cache_warm_run)

    manifest: list[dict[str, Any]] = []

    fig_methods_overview(primary, out_dir, manifest)
    fig_timing_decomposition(primary, out_dir, manifest)
    fig_spike_parity(primary, out_dir, manifest)
    fig_trace_quality(primary, out_dir, manifest)
    fig_trace_overlays(primary, out_dir, manifest)
    fig_cache_benchmark(cold, warm, out_dir, manifest)
    fig_summary_dashboard(primary, cold, warm, out_dir, manifest)
    fig_abstract_two_panel(primary, cold, warm, out_dir, manifest)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_run_dir": str(primary.run_dir),
        "cache_cold_run_dir": str(cold.run_dir),
        "cache_warm_run_dir": str(warm.run_dir),
        "primary_compare_payload_args": primary.compare_payload.get("args", {}),
        "primary_summary": primary.comparison_summary,
        "cache_cold_arbor_cache_line": primary.parsed_arbor.get("native_cache_line") if primary.run_dir == cold.run_dir else cold.parsed_arbor.get("native_cache_line"),
        "cache_warm_arbor_cache_line": warm.parsed_arbor.get("native_cache_line"),
    }
    write_caption_files(out_dir, manifest, meta)
    return {"out_dir": str(out_dir), "figures": manifest, "meta": meta}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate paper-style plots for NEURON vs Arbor glia comparison artifacts.")
    ap.add_argument("--primary-run", default=str(DEFAULT_PRIMARY_RUN), help="Path to a comparison run directory used for the main parity/timing figures.")
    ap.add_argument("--cache-cold-run", default=str(DEFAULT_CACHE_COLD_RUN), help="Path to the cold-cache comparison run directory.")
    ap.add_argument("--cache-warm-run", default=str(DEFAULT_CACHE_WARM_RUN), help="Path to the warm-cache comparison run directory.")
    ap.add_argument("--out-dir", default="", help="Output directory for figures (default: timestamped folder under the paper-figures output root).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    primary_run = Path(args.primary_run).expanduser().resolve()
    cache_cold_run = Path(args.cache_cold_run).expanduser().resolve()
    cache_warm_run = Path(args.cache_warm_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(DEFAULT_OUT_PARENT)

    result = build_figures(primary_run=primary_run, cache_cold_run=cache_cold_run, cache_warm_run=cache_warm_run, out_dir=out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
