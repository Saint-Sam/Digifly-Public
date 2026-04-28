#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import make_glia_compare_paper_plots as base  # noqa: E402


NOGLIA_COMPARE_ROOT = Path(
    os.environ.get(
        "DIGIFLY_NOGLIA_COMPARE_ROOT",
        base.OUTPUTS_ROOT / "glia_neuron_vs_arbor_compare_noglia_precached",
    )
)
DEFAULT_PRIMARY_RUN = Path(
    os.environ.get(
        "DIGIFLY_NOGLIA_PRIMARY_RUN",
        NOGLIA_COMPARE_ROOT / "glia_neuron_vs_arbor_20260226T174526Z",
    )
)
DEFAULT_CACHE_COLD_RUN = Path(
    os.environ.get(
        "DIGIFLY_NOGLIA_CACHE_COLD_RUN",
        NOGLIA_COMPARE_ROOT / "glia_neuron_vs_arbor_20260226T173255Z",
    )
)
DEFAULT_CACHE_WARM_RUN = DEFAULT_PRIMARY_RUN
DEFAULT_OUT_PARENT = Path(
    os.environ.get(
        "DIGIFLY_NOGLIA_FIGURE_OUTPUT_ROOT",
        base.OUTPUTS_ROOT / "glia_neuron_vs_arbor_paper_figures_noglia",
    )
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_out_dir(parent: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"glia_neuron_vs_arbor_noglia_figures_{ts}"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _signal_to_neuron_id(signal: Any) -> Optional[int]:
    return base._signal_to_neuron_id(signal)


def load_arbor_cache_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    wr = _read_json(run_dir / "worker_results.json")
    arbor = wr.get("arbor", {}) or {}
    row = (arbor.get("row") or {}) if isinstance(arbor, dict) else {}
    parsed_arbor = base._parse_worker_stdout_tail(str(arbor.get("stdout_tail", "")))
    return {
        "run_dir": run_dir,
        "worker_results": wr,
        "arbor": arbor,
        "row": row,
        "parsed_arbor": parsed_arbor,
    }


def _merged_trace_spike(primary: base.CompareRunData) -> pd.DataFrame:
    trace_df = primary.trace_per_signal.copy()
    spike_df = primary.spike_per_neuron.copy()
    trace_df["neuron_id"] = trace_df["signal"].map(_signal_to_neuron_id)
    merged = trace_df.merge(
        spike_df[
            [
                "neuron_id",
                "count_delta_arbor_minus_neuron",
                "neuron_spike_count",
                "arbor_spike_count",
                "matched_spikes_tol_ms",
                "match_recall_vs_neuron",
                "match_precision_vs_arbor",
            ]
        ],
        on="neuron_id",
        how="left",
    )
    merged["count_delta_arbor_minus_neuron"] = merged["count_delta_arbor_minus_neuron"].fillna(0.0)
    merged["spike_mismatch"] = merged["count_delta_arbor_minus_neuron"] != 0
    return merged


def _common_summary(primary: base.CompareRunData) -> dict[str, Any]:
    trow = primary.timing_comparison.iloc[0]
    ss = primary.spike_summary.iloc[0]
    ts = primary.trace_summary.iloc[0]
    pn, pa = primary.parsed_neuron, primary.parsed_arbor
    sim_ratio = (
        float(pa.get("sim_wall_s")) / float(pn.get("sim_wall_s"))
        if (pn.get("sim_wall_s") is not None and pa.get("sim_wall_s") is not None and float(pn.get("sim_wall_s")) > 0)
        else float("nan")
    )
    worker_ratio = (
        float(trow["arbor_worker_wall_s"]) / float(trow["neuron_worker_wall_s"])
        if float(trow["neuron_worker_wall_s"]) > 0
        else float("nan")
    )
    return {
        "trow": trow,
        "ss": ss,
        "ts": ts,
        "pn": pn,
        "pa": pa,
        "runtime_ratio": float(trow["arbor_runtime_div_neuron_runtime"]),
        "worker_ratio": worker_ratio,
        "sim_ratio": sim_ratio,
    }


def fig_noglia_abstract_two_panel(
    primary: base.CompareRunData,
    cold_cache: dict[str, Any],
    warm_cache: dict[str, Any],
    out_dir: Path,
    manifest: list[dict[str, Any]],
) -> None:
    s = _common_summary(primary)
    trow, ss, ts, pn, pa = s["trow"], s["ss"], s["ts"], s["pn"], s["pa"]
    merged = _merged_trace_spike(primary)
    pc = cold_cache["parsed_arbor"]
    pw = warm_cache["parsed_arbor"]

    ratios = np.array([s["runtime_ratio"], s["worker_ratio"], s["sim_ratio"]], dtype=float)
    ratio_labels = ["Total run time", "All-in backend time", "Simulation compute time"]

    fig = plt.figure(figsize=(16.0, 9.4))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.95, 1.25], wspace=0.24)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.84, bottom=0.23)

    # Panel A: performance ratios.
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ratios))
    colors = [base.COLORS["arbor"] if v >= 1.0 else base.COLORS["ok"] for v in ratios]
    bars = ax.bar(x, ratios, color=colors, width=0.60)
    ax.axhline(1.0, color="#374151", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels)
    ax.set_ylabel("Arbor / NEURON ratio")
    ax.set_ylim(0.0, max(1.55, float(np.nanmax(ratios)) + 0.25))
    ax.set_title("A. Performance summary (same neurons, glia disabled)", pad=10)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "Primary run ratios (dashed line = parity). Lower is better for Arbor in this panel.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#4b5563",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e5e7eb", alpha=0.95),
    )

    # Panel B: trace parity scatter + spike mismatch encoding.
    ax2 = fig.add_subplot(gs[0, 1])
    cvals = merged["count_delta_arbor_minus_neuron"].to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(cvals)))) if len(cvals) else 1.0
    mismatch_mask = merged["spike_mismatch"].to_numpy(dtype=bool)
    sc = ax2.scatter(
        merged["rmse"].to_numpy(dtype=float),
        merged["corr"].to_numpy(dtype=float),
        c=cvals,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=np.where(mismatch_mask, 95, 55),
        alpha=0.9,
        edgecolor=np.where(mismatch_mask, "#111827", "white"),
        linewidth=0.8,
    )
    ax2.axhline(0.95, color=base.COLORS["grid"], lw=1.0, ls="--")
    ax2.axvline(0.5, color=base.COLORS["grid"], lw=1.0, ls="--")
    ax2.set_xlabel("Trace RMSE (mV)")
    ax2.set_ylabel("Trace correlation")
    ax2.set_ylim(-1.05, 1.02)
    rmse_vals = merged["rmse"].to_numpy(dtype=float)
    ax2.set_xlim(min(-0.15, float(np.nanmin(rmse_vals)) - 0.15), float(np.nanmax(rmse_vals)) + 0.7)
    ax2.set_title("B. Parity across recorded neurons (trace quality + spike mismatch)", pad=10)

    for _, row in merged.sort_values("rmse", ascending=False).head(min(4, len(merged))).iterrows():
        y0 = float(row["corr"])
        ax2.annotate(
            str(row["signal"]),
            (float(row["rmse"]), y0),
            xytext=(6, -8 if y0 > 0.92 else 4),
            textcoords="offset points",
            ha="left",
            va="top" if y0 > 0.92 else "bottom",
            fontsize=8,
            clip_on=True,
        )

    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("Spike delta (Arbor − NEURON)")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="white", markersize=7, label="No spike count mismatch"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#fdae6b", markeredgecolor="#111827", markersize=8, label="Spike count mismatch"),
        ],
        frameon=False,
        loc="lower right",
        borderpad=0.2,
        handletextpad=0.5,
    )

    cold_sim = _safe_float(pc.get("sim_wall_s"))
    warm_sim = _safe_float(pw.get("sim_wall_s"))
    delta_sim = warm_sim - cold_sim if np.isfinite(cold_sim) and np.isfinite(warm_sim) else float("nan")
    cold_hits = int((pc.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    warm_hits = int((pw.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))

    fig.text(
        0.075,
        0.13,
        (
            f"Cache benchmark (Arbor-only warmup vs measured run): warm cache hits={warm_hits} (cold hits={cold_hits}), "
            f"simulation compute time {cold_sim:.2f}\u2192{warm_sim:.2f} s (\u0394={delta_sim:+.2f} s)."
        ),
        ha="left",
        va="top",
        fontsize=8.9,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.97),
    )
    fig.text(
        0.515,
        0.13,
        (
            f"Spikes: NEURON={int(ss['neuron_total_spikes'])}, Arbor={int(ss['arbor_total_spikes'])} | "
            f"Recall={float(ss['recall_vs_neuron']):.2f}, Precision={float(ss['precision_vs_arbor']):.2f}, F1={float(ss['f1']):.2f}\n"
            f"Traces: mean RMSE={float(ts['mean_rmse']):.3f} mV, mean corr={float(ts['mean_corr']):.3f} | "
            f"Common traces={int(ts['common_trace_columns'])}, overlap samples={int(ts['overlap_samples'])}"
        ),
        ha="left",
        va="top",
        fontsize=8.7,
        family="DejaVu Sans Mono",
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.97),
    )

    resolved = pn.get("edge_cache_resolved_nodes")
    edges = pn.get("edge_cache_edges")
    fig.suptitle(
        "Figure Abstract (No Glia): NEURON vs Arbor on the same 15-neuron subset with Arbor pre-caching",
        x=0.02,
        ha="left",
        fontsize=14.2,
        y=0.94,
    )
    fig.text(
        0.02,
        0.035,
        (
            f"Takeaway: In this no-glia rerun (resolved_nodes={resolved}, edges={edges}), Arbor shows lower total run time and all-in backend time, "
            f"but its core simulation compute time is still slower than NEURON/CoreNEURON and spike-level parity remains poor."
        ),
        fontsize=9.35,
        color="#374151",
    )

    base._save_fig(
        fig,
        out_dir,
        "fig_noglia_abstract_two_panel",
        manifest,
        "Two-panel no-glia abstract summary using the warmed-cache NEURON vs Arbor comparison on the same 15-neuron subset. Panel A shows performance ratios (Arbor/NEURON), and Panel B shows trace parity with spike mismatch encoding. Footer annotations document Arbor cache warmup behavior and summary parity metrics.",
    )


def fig_noglia_merged_four_panel(
    primary: base.CompareRunData,
    cold_cache: dict[str, Any],
    warm_cache: dict[str, Any],
    out_dir: Path,
    manifest: list[dict[str, Any]],
) -> None:
    s = _common_summary(primary)
    trow, ss, ts, pn, pa = s["trow"], s["ss"], s["ts"], s["pn"], s["pa"]
    merged = _merged_trace_spike(primary)
    pc = cold_cache["parsed_arbor"]
    pw = warm_cache["parsed_arbor"]

    fig = plt.figure(figsize=(15.8, 11.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.89, bottom=0.14)

    # A. Ratio overview
    axA = fig.add_subplot(gs[0, 0])
    ratio_vals = [s["runtime_ratio"], s["worker_ratio"], s["sim_ratio"]]
    ratio_labels = ["Total run time", "All-in backend time", "Simulation compute time"]
    x = np.arange(3)
    bars = axA.bar(
        x,
        ratio_vals,
        color=[base.COLORS["ok"] if v < 1 else base.COLORS["arbor"] for v in ratio_vals],
        width=0.58,
    )
    axA.axhline(1.0, color="#374151", ls="--", lw=1.0)
    axA.set_xticks(x)
    axA.set_xticklabels(ratio_labels)
    axA.set_ylabel("Arbor / NEURON")
    axA.set_title("A. Performance ratios (primary no-glia run)")
    axA.set_ylim(0, max(1.6, float(np.nanmax(ratio_vals)) + 0.22))
    for b, v in zip(bars, ratio_vals):
        axA.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # B. Absolute timing decomposition
    axB = fig.add_subplot(gs[0, 1])
    metric_labels = ["Total run time", "All-in backend time", "Simulation compute"]
    neuron_vals = [float(trow["neuron_runtime_s"]), float(trow["neuron_worker_wall_s"]), float(pn.get("sim_wall_s") or np.nan)]
    arbor_vals = [float(trow["arbor_runtime_s"]), float(trow["arbor_worker_wall_s"]), float(pa.get("sim_wall_s") or np.nan)]
    idx = np.arange(len(metric_labels))
    w = 0.35
    axB.bar(idx - w/2, neuron_vals, w, label="NEURON", color=base.COLORS["neuron"])
    axB.bar(idx + w/2, arbor_vals, w, label="Arbor", color=base.COLORS["arbor"])
    axB.set_xticks(idx)
    axB.set_xticklabels([f"{m}\n(s)" for m in metric_labels])
    axB.set_ylabel("Seconds")
    axB.set_title("B. Time breakdown")
    axB.legend(frameon=False, loc="upper right")
    ymax = max([v for v in neuron_vals + arbor_vals if np.isfinite(v)])
    axB.set_ylim(0, ymax * 1.18)

    # C. Trace parity + spike mismatch scatter
    axC = fig.add_subplot(gs[1, 0])
    cvals = merged["count_delta_arbor_minus_neuron"].to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(cvals)))) if len(cvals) else 1.0
    mismatch_mask = merged["spike_mismatch"].to_numpy(dtype=bool)
    sc = axC.scatter(
        merged["rmse"].to_numpy(dtype=float),
        merged["corr"].to_numpy(dtype=float),
        c=cvals,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=np.where(mismatch_mask, 90, 52),
        edgecolor=np.where(mismatch_mask, "#111827", "white"),
        linewidth=0.8,
        alpha=0.9,
    )
    axC.axhline(0.95, color=base.COLORS["grid"], ls="--", lw=1.0)
    axC.axvline(0.5, color=base.COLORS["grid"], ls="--", lw=1.0)
    axC.set_xlabel("Trace RMSE (mV)")
    axC.set_ylabel("Trace correlation")
    axC.set_ylim(-1.05, 1.02)
    axC.set_xlim(min(-0.15, float(np.nanmin(merged["rmse"])) - 0.15), float(np.nanmax(merged["rmse"])) + 0.8)
    axC.set_title("C. Trace parity with spike mismatch encoding")
    for _, row in merged.sort_values("rmse", ascending=False).head(min(5, len(merged))).iterrows():
        y0 = float(row["corr"])
        axC.annotate(
            str(row["signal"]),
            (float(row["rmse"]), y0),
            xytext=(6, -8 if y0 > 0.92 else 4),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="top" if y0 > 0.92 else "bottom",
            clip_on=True,
        )
    cb = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.02)
    cb.set_label("Spike delta (Arbor − NEURON)")

    # D. Spike parity summary + cache verification
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.42, height_ratios=[0.54, 0.46])
    axD1 = fig.add_subplot(sub[0, 0])
    spike_bars = [float(ss["neuron_total_spikes"]), float(ss["arbor_total_spikes"]), float(ss["matched_spikes"])]
    axD1.bar([0, 1, 2], spike_bars, color=[base.COLORS["neuron"], base.COLORS["arbor"], base.COLORS["accent"]], width=0.58)
    axD1.set_xticks([0, 1, 2])
    axD1.set_xticklabels(["NEURON\nspikes", "Arbor\nspikes", "Matched\n(tol)"])
    axD1.set_ylabel("Count")
    axD1.set_title("D1. Spike parity summary")
    for i, v in enumerate(spike_bars):
        axD1.text(i, v + max(0.3, 0.04 * max(spike_bars)), f"{int(v)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axD1.text(
        0.98,
        0.93,
        f"Recall={float(ss['recall_vs_neuron']):.2f}\nPrecision={float(ss['precision_vs_arbor']):.2f}\nF1={float(ss['f1']):.2f}",
        transform=axD1.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#d1d5db', alpha=0.96),
    )

    axD2 = fig.add_subplot(sub[1, 0])
    cold_sim = _safe_float(pc.get("sim_wall_s"))
    warm_sim = _safe_float(pw.get("sim_wall_s"))
    cold_hits = int((pc.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    warm_hits = int((pw.get("native_cache_counts") or {}).get("branch_disk_hit", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_hit", 0))
    cold_writes = int((pc.get("native_cache_counts") or {}).get("branch_disk_write", 0) + (pc.get("native_cache_counts") or {}).get("swcfix_disk_write", 0))
    warm_writes = int((pw.get("native_cache_counts") or {}).get("branch_disk_write", 0) + (pw.get("native_cache_counts") or {}).get("swcfix_disk_write", 0))
    xx = np.arange(2)
    axD2.bar(xx, [cold_sim, warm_sim], color=["#9ca3af", base.COLORS["accent"]], width=0.55)
    axD2.set_xticks(xx)
    axD2.set_xticklabels(["Cold cache\n(warmup)", "Warm cache\n(primary Arbor)"])
    axD2.set_ylabel("Arbor simulation compute time (s)")
    axD2.set_title("D2. Arbor pre-cache verification")
    for i, v in enumerate([cold_sim, warm_sim]):
        axD2.text(i, v + 0.6, f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axD2r = axD2.twinx()
    axD2r.plot(xx, [cold_hits, warm_hits], "o-", color=base.COLORS["arbor"], lw=1.6, label="Disk hits")
    axD2r.plot(xx, [cold_writes, warm_writes], "s--", color="#4b5563", lw=1.2, label="Disk writes")
    axD2r.set_ylabel("Cache events")
    lines1, labels1 = axD2.get_legend_handles_labels()
    lines2, labels2 = axD2r.get_legend_handles_labels()
    # axD2 has no legend entries currently; keep right-axis legend only.
    axD2r.legend(lines2, labels2, frameon=False, loc="upper left")

    selector_name = Path(str(primary.worker_results["neuron"]["row"].get("selector_json", ""))).name
    fig.suptitle(
        "NEURON vs Arbor (No Glia): Combined comparison summary on the same neuron subset",
        x=0.02,
        ha="left",
        fontsize=15,
        y=0.97,
    )
    fig.text(
        0.02,
        0.93,
        (
            f"Conditions: selector={selector_name} | glia_off | resolved_nodes={pn.get('edge_cache_resolved_nodes')} | "
            f"edges={pn.get('edge_cache_edges')} | dt={pn.get('dt_ms')} ms | tstop={pn.get('tstop_ms')} ms | "
            f"Arbor backend={pa.get('sim_backend')} | NEURON backend={pn.get('sim_backend')}"
        ),
        fontsize=9.3,
        color="#374151",
    )
    fig.text(
        0.02,
        0.045,
        (
            f"Overall: Arbor is faster in total run time ({float(trow['arbor_runtime_s']):.1f}s vs {float(trow['neuron_runtime_s']):.1f}s) "
            f"and all-in backend time ({float(trow['arbor_worker_wall_s']):.1f}s vs {float(trow['neuron_worker_wall_s']):.1f}s), "
            f"but slower in core simulation compute time ({float(pa.get('sim_wall_s')):.1f}s vs {float(pn.get('sim_wall_s')):.1f}s) "
            f"and not yet spike-parity matched."
        ),
        fontsize=9.2,
        color="#374151",
    )

    base._save_fig(
        fig,
        out_dir,
        "fig_noglia_merged_four_panel",
        manifest,
        "Four-panel no-glia comparison summary combining performance ratios, absolute timings, trace parity with spike mismatch encoding, and Arbor pre-cache verification from a cold warmup vs the warmed primary run.",
    )


def write_manifest(out_dir: Path, manifest: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    (out_dir / "figure_manifest.json").write_text(json.dumps({"figures": manifest, "meta": meta}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate no-glia NEURON-vs-Arbor summary figures (2-panel abstract + merged 4-panel).")
    ap.add_argument("--primary-run", default=str(DEFAULT_PRIMARY_RUN), help="Primary full comparison run (NEURON + Arbor).")
    ap.add_argument("--cache-cold-run", default=str(DEFAULT_CACHE_COLD_RUN), help="Arbor warmup run used to populate cache (cold cache baseline).")
    ap.add_argument("--cache-warm-run", default=str(DEFAULT_CACHE_WARM_RUN), help="Warm-cache run (typically the primary run).")
    ap.add_argument("--out-dir", default="", help="Output directory (default: timestamped folder under no-glia paper figures root).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    primary_run = Path(args.primary_run).expanduser().resolve()
    cache_cold_run = Path(args.cache_cold_run).expanduser().resolve()
    cache_warm_run = Path(args.cache_warm_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(DEFAULT_OUT_PARENT)

    base._setup_style()
    _ensure_dir(out_dir)

    primary = base.load_compare_run(primary_run)
    cold_cache = load_arbor_cache_run(cache_cold_run)
    warm_cache = load_arbor_cache_run(cache_warm_run)

    manifest: list[dict[str, Any]] = []
    fig_noglia_abstract_two_panel(primary, cold_cache, warm_cache, out_dir, manifest)
    fig_noglia_merged_four_panel(primary, cold_cache, warm_cache, out_dir, manifest)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_run_dir": str(primary_run),
        "cache_cold_run_dir": str(cache_cold_run),
        "cache_warm_run_dir": str(cache_warm_run),
        "primary_summary": primary.comparison_summary,
        "cache_cold_arbor_cache_line": cold_cache["parsed_arbor"].get("native_cache_line"),
        "cache_warm_arbor_cache_line": warm_cache["parsed_arbor"].get("native_cache_line"),
    }
    write_manifest(out_dir, manifest, meta)
    print(json.dumps({"out_dir": str(out_dir), "figures": manifest, "meta": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
