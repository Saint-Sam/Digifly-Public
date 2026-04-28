from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuron import h

from .network import Network
from .ais import _xyz_of_section_site

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _inv_node_map(cell):
    return {sec: nid for nid, sec in getattr(cell, "_sec_for_node", {}).items()}

def _choose_swc_soma_rostral(cell):
    nodes = getattr(cell, "_nodes", {})
    soma_nodes = [n for n, r in nodes.items() if int(r.get("type", 0)) == 1]
    if not soma_nodes:
        return None, None
    pick = max(soma_nodes, key=lambda n: (nodes[n]["z"], nodes[n]["r"]))
    return pick, nodes[pick]

def _bind_neuron_soma_to_swc(cell, soma_node_id: int):
    if soma_node_id is None:
        return
    sec = getattr(cell, "_sec_for_node", {}).get(int(soma_node_id))
    if sec is None:
        return
    cell._soma_id = int(soma_node_id)
    cell.soma_sec = sec

def _euclid(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))

def _save_fig_current(cfg: Dict[str, Any], fig, name: str):
    try:
        out_dir = Path(cfg["edges_csv"]).parent
    except Exception:
        out_dir = Path.cwd()
    out = (out_dir / name).with_suffix(".pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[FIG] saved -> {out}")

def _write_ais_report_csv(report_csv: str | Path, csv_hdr: list[str], csv_rows: list[list[Any]]) -> None:
    if not csv_rows:
        return
    with open(report_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(csv_hdr)
        for r in csv_rows:
            w.writerow(r)
    print(f"[REPORT] saved -> {report_csv}")

def build_ais_strict_figure(neuron_ids, cfg: Dict[str, Any], title="AIS visualization (STRICT)",
                            enforce=True, tol_um=None, report_csv=None):
    """
    Build the strict AIS visualization using the same logic as visualize_ais_strict,
    but keep the figure open and return plot/report artifacts for interactive notebooks.
    """
    tol = float(cfg.get("viz_tol_um", 0.75)) if tol_um is None else float(tol_um)
    enforce = bool(cfg.get("viz_strict_enforce_swc_soma", True)) if enforce is None else bool(enforce)

    if report_csv is None:
        try:
            report_csv = str(Path(cfg["edges_csv"]).parent / "ais_visualization_report.csv")
        except Exception:
            report_csv = "ais_visualization_report.csv"

    net = Network(cfg)
    for nid in neuron_ids:
        net.ensure_cell(int(nid))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
    except Exception:
        pass

    csv_rows = []
    csv_hdr = ["gid",
               "swc_soma_node", "swc_soma_x", "swc_soma_y", "swc_soma_z",
               "nrn_soma_sec", "nrn_soma_x", "nrn_soma_y", "nrn_soma_z",
               "nrn_soma_vs_swc_um",
               "ais_sec", "ais_xloc", "ais_x", "ais_y", "ais_z", "ais_on_axon"]

    for nid in neuron_ids:
        nid = int(nid)
        cell = net.ensure_cell(nid)

        for sec, pts in cell._cache_sec_pts.items():
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.6, alpha=0.7)

        soma_node_id, srec = _choose_swc_soma_rostral(cell)
        if soma_node_id is None:
            nrn_soma_sec, nrn_soma_x = cell.soma_site()
            nx, ny, nz = _xyz_of_section_site(nrn_soma_sec, nrn_soma_x)
            ax.scatter([nx], [ny], [nz], s=70, marker="o", color="red", label=f"{nid} soma (no SWC)")
            ais_sec, ais_x = cell.axon_ais_site()
            axx, axy, axz = _xyz_of_section_site(ais_sec, ais_x)
            inv = _inv_node_map(cell)
            ais_is_axon = int(cell._nodes.get(inv.get(ais_sec, -999), {}).get("type", 0)) == 2
            csv_rows.append([nid, "", "", "", "",
                             nrn_soma_sec.name(), nx, ny, nz, "",
                             ais_sec.name(), float(ais_x), axx, axy, axz, bool(ais_is_axon)])
            continue

        if enforce:
            _bind_neuron_soma_to_swc(cell, soma_node_id)

        sx, sy, sz = float(srec["x"]), float(srec["y"]), float(srec["z"])
        ax.scatter([sx], [sy], [sz], s=70, marker="o", color="red", label=f"{nid} soma (SWC)")

        nrn_soma_sec, nrn_soma_x = cell.soma_site()
        nx, ny, nz = _xyz_of_section_site(nrn_soma_sec, nrn_soma_x)

        ais_sec, ais_x = cell.axon_ais_site()
        axx, axy, axz = _xyz_of_section_site(ais_sec, ais_x)
        inv = _inv_node_map(cell)
        ais_node = inv.get(ais_sec, None)
        ais_is_axon = (ais_node is not None) and (int(cell._nodes[ais_node]["type"]) == 2)

        d_soma = _euclid((sx, sy, sz), (nx, ny, nz))
        if d_soma > tol:
            ax.scatter([nx], [ny], [nz], s=60, marker="o", color="orange", label=f"{nid} soma (NEURON)")

        ax.scatter([axx], [axy], [axz], s=70, marker="^", color="blue", label=f"{nid} AIS")

        csv_rows.append([nid,
                         int(soma_node_id), sx, sy, sz,
                         nrn_soma_sec.name(), nx, ny, nz, d_soma,
                         ais_sec.name(), float(ais_x), axx, axy, axz, bool(ais_is_axon)])

    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "net": net,
        "csv_rows": csv_rows,
        "csv_hdr": csv_hdr,
        "report_csv": report_csv,
        "tol_um": tol,
        "enforce": enforce,
        "neuron_ids": [int(n) for n in neuron_ids],
    }

def visualize_ais_strict(neuron_ids, cfg: Dict[str, Any], title="AIS visualization (STRICT)",
                         save_name="ais_visualization_strict.pdf", show=True, enforce=True, tol_um=None, report_csv=None):
    built = build_ais_strict_figure(
        neuron_ids,
        cfg=cfg,
        title=title,
        enforce=enforce,
        tol_um=tol_um,
        report_csv=report_csv,
    )
    fig = built["fig"]
    _save_fig_current(cfg, fig, save_name)
    _write_ais_report_csv(built["report_csv"], built["csv_hdr"], built["csv_rows"])

    if show:
        plt.show()
    plt.close(fig)

def fix_and_visualize_soma_ais(neuron_ids, cfg: Dict[str, Any], enforce=True, tol_um=2.0,
                               save_name="ais_visualization_strict.pdf",
                               report_csv="_ais_visualization_report.csv",
                               title="AIS visualization (STRICT)"):
    return visualize_ais_strict(
        neuron_ids,
        cfg=cfg,
        title=title,
        save_name=save_name,
        show=True,
        enforce=enforce,
        tol_um=tol_um,
        report_csv=report_csv,
    )

def visualize_ais(neuron_ids, cfg: Dict[str, Any], title="AIS visualization", save_name="ais_visualization.pdf", show=True):
    net = Network(cfg)
    for nid in neuron_ids:
        net.ensure_cell(int(nid))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
    except Exception:
        pass

    for nid in neuron_ids:
        cell = net.ensure_cell(int(nid))
        for sec, pts in cell._cache_sec_pts.items():
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.6, alpha=0.7)
        ais_site = cell.axon_ais_site()
        axx, axy, axz = _xyz_of_section_site(ais_site[0], ais_site[1])
        ax.scatter([axx], [axy], [axz], s=70, marker="^", color="blue", label=f"{nid} AIS")

    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()
    _save_fig_current(cfg, fig, save_name)
    if show:
        plt.show()
    plt.close(fig)
