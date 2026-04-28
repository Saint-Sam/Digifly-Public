from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency in runtime env
    go = None

try:
    import ipywidgets as widgets
except Exception:  # pragma: no cover - optional dependency in runtime env
    widgets = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional dependency in runtime env
    display = None


SWC_TYPE_LABELS: Dict[int, str] = {
    1: "soma",
    2: "axon",
    3: "dendrite",
    4: "apical",
}

SWC_TYPE_COLORS: Dict[int, str] = {
    1: "#f6aa1c",
    2: "#1b998b",
    3: "#2d3047",
    4: "#ff6b6b",
}

SELECTED_COLOR = "#ef476f"


def _require_interactive_dependencies() -> None:
    missing = []
    if go is None:
        missing.append("plotly")
    if widgets is None:
        missing.append("ipywidgets")
    if display is None:
        missing.append("ipython")
    if missing:
        names = ", ".join(missing)
        raise ImportError(
            f"Missing interactive dependencies: {names}. "
            "Install in your notebook kernel, then rerun."
        )


def _figurewidget_backend_available() -> bool:
    if go is None or widgets is None or display is None:
        return False
    force_static = str(os.environ.get("SWC_SELECTOR_FORCE_STATIC", "")).strip().lower()
    if force_static in {"1", "true", "yes", "on"}:
        return False
    try:
        import plotly

        major = int(str(plotly.__version__).split(".", maxsplit=1)[0])
    except Exception:
        major = 0
    if major >= 6:
        try:
            import anywidget  # noqa: F401
        except Exception:
            return False
    return True


def _default_cell_cfg(swc_dir: str | Path, cfg_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    swc_root = str(Path(swc_dir).expanduser().resolve())
    cfg = {
        "swc_dir": swc_root,
        "morph_swc_dir": swc_root,
        "Ra": 35.4,
        "cm": 1.0,
        "celsius_C": 22.0,
        "ais_min_xloc": 0.05,
        "ais_strict_axon_map": True,
        "ais_override_filename": "ais_overrides.csv",
        "ais_cache_csv": str(Path(swc_root) / "_ais_cache.csv"),
    }
    if cfg_overrides:
        cfg.update(dict(cfg_overrides))
    return cfg


def load_swc_cell_for_selection(
    neuron_id: int,
    swc_dir: str | Path,
    cfg_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Build a lightweight SWCCell for standalone interactive compartment picking.
    """
    from digifly.phase2.neuron_build.swc_cell import SWCCell, find_swc

    cfg = _default_cell_cfg(swc_dir, cfg_overrides=cfg_overrides)
    swc_path = find_swc(cfg.get("morph_swc_dir") or cfg["swc_dir"], int(neuron_id))
    return SWCCell(int(neuron_id), swc_path, cfg=cfg)


def section_metadata_table(cell: Any) -> pd.DataFrame:
    """
    Return one row per section for display/export.
    """
    inv = {sec: int(nid) for nid, sec in getattr(cell, "_sec_for_node", {}).items()}
    rows: List[Dict[str, Any]] = []

    for sec, pts in getattr(cell, "_cache_sec_pts", {}).items():
        if pts is None or len(pts) < 1:
            continue

        node_id = inv.get(sec)
        swc_type = int(getattr(cell, "_nodes", {}).get(node_id, {}).get("type", 0)) if node_id is not None else 0
        swc_label = SWC_TYPE_LABELS.get(swc_type, f"type_{swc_type}")
        color = SWC_TYPE_COLORS.get(swc_type, "#8d99ae")
        length_um = float(pts[-1, 3] - pts[0, 3]) if pts.shape[1] >= 4 else float("nan")
        center = np.nanmean(pts[:, :3], axis=0)

        rows.append(
            {
                "section_name": str(sec.name()),
                "node_id": int(node_id) if node_id is not None else None,
                "swc_type": int(swc_type),
                "swc_label": str(swc_label),
                "color": str(color),
                "length_um": float(length_um),
                "center_x_um": float(center[0]),
                "center_y_um": float(center[1]),
                "center_z_um": float(center[2]),
                "_sec_obj": sec,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "section_name",
                "node_id",
                "swc_type",
                "swc_label",
                "color",
                "length_um",
                "center_x_um",
                "center_y_um",
                "center_z_um",
                "_sec_obj",
            ]
        )

    df = pd.DataFrame(rows)
    if "node_id" in df.columns:
        df = df.sort_values(["node_id", "section_name"], na_position="last").reset_index(drop=True)
    return df


def _ko_to_ek_mV(ko_mM: float, ki_mM: float, celsius_C: float) -> float:
    """
    Nernst estimate for K+ reversal potential in mV (z = +1).
    """
    t_k = 273.15 + float(celsius_C)
    return 26.640 * (t_k / 310.15) * np.log(max(float(ko_mM), 1e-9) / max(float(ki_mM), 1e-9))


def _set_ko_and_ek_on_section(
    sec: Any,
    *,
    ko_mM: float,
    ki_mM: float,
    celsius_C: float,
    update_ek: bool = True,
) -> int:
    touched = 0
    ek_target = _ko_to_ek_mV(ko_mM, ki_mM, celsius_C) if update_ek else None
    for seg in sec:
        changed = False
        try:
            if hasattr(seg, "ko"):
                seg.ko = float(ko_mM)
                changed = True
        except Exception:
            pass
        if update_ek and ek_target is not None:
            try:
                if hasattr(seg, "ek"):
                    seg.ek = float(ek_target)
                    changed = True
            except Exception:
                pass
        if changed:
            touched += 1
    return int(touched)


def _sec_lookup(cell: Any) -> Dict[str, Any]:
    out = {}
    for sec in getattr(cell, "_secs", []):
        out[str(sec.name())] = sec
    out.update({str(k): v for k, v in getattr(cell, "sections", {}).items()})
    return out


def _inv_node_map(cell: Any) -> Dict[Any, int]:
    return {sec: int(node_id) for node_id, sec in getattr(cell, "_sec_for_node", {}).items()}


@dataclass
class AISAssignment:
    neuron_id: int
    section_name: str
    node_id: Optional[int]
    xloc: float
    persisted: bool


class InteractiveSWCSelector:
    """
    Part 1: interactive 3D SWC section picker for one neuron.

    `backend="widget"` supports click-to-select with Plotly FigureWidget.
    `backend="static"` renders a plain Plotly 3D figure and uses a list picker.
    """

    def __init__(self, cell: Any, title: Optional[str] = None, backend: str = "auto"):
        _require_interactive_dependencies()
        backend = str(backend).strip().lower()
        if backend not in {"auto", "widget", "static"}:
            raise ValueError("backend must be one of: auto, widget, static")
        self.cell = cell
        self.neuron_id = int(getattr(cell, "gid", -1))
        self.title = title or f"Neuron {self.neuron_id}: SWC Compartments"
        if backend == "auto":
            self.backend = "widget" if _figurewidget_backend_available() else "static"
        else:
            self.backend = backend

        self.section_df = section_metadata_table(cell)
        self.selected_section_names: Set[str] = set()
        self.last_clicked_section_name: Optional[str] = None

        self._rows_by_section = {
            str(r["section_name"]): r for _, r in self.section_df.iterrows()
        }
        self._base_colors = {
            str(r["section_name"]): str(r["color"]) for _, r in self.section_df.iterrows()
        }
        self._traces_by_section: Dict[str, Any] = {}
        self._updating_picker = False

        self.figure = None
        self.figure_output = widgets.Output(layout=widgets.Layout(width="980px", height="780px"))
        self.mode_widget = widgets.HTML()
        self.section_picker_widget = widgets.SelectMultiple(
            options=tuple(sorted(self._rows_by_section.keys())),
            value=(),
            description="Pick",
            rows=12,
            layout=widgets.Layout(width="430px"),
        )
        self.section_picker_widget.observe(self._on_picker_changed, names="value")
        self.selected_widget = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Selected",
            rows=10,
            layout=widgets.Layout(width="430px"),
        )
        self.status_widget = widgets.HTML()
        self.output_widget = widgets.Output(layout=widgets.Layout(width="430px", max_height="200px"))
        self.clear_button = widgets.Button(description="Clear Selection", button_style="")
        self.refresh_button = widgets.Button(description="Refresh Marker", button_style="")
        self.clear_button.on_click(self._on_clear_clicked)
        self.refresh_button.on_click(self._on_refresh_clicked)

        self._ais_trace = None
        self._soma_trace = None
        self._build()

    def _build(self) -> None:
        if self.backend == "widget":
            self._build_widget_figure()
        else:
            self._render_static_figure()
        self._update_status()

    def _make_section_trace(self, row: pd.Series, *, selected: bool = False):
        sec = row["_sec_obj"]
        pts = getattr(self.cell, "_cache_sec_pts", {}).get(sec)
        if pts is None or pts.shape[0] < 2:
            return None

        section_name = str(row["section_name"])
        node_id = row["node_id"]
        swc_label = str(row["swc_label"])
        length_um = float(row["length_um"]) if pd.notna(row["length_um"]) else float("nan")
        color = SELECTED_COLOR if selected else str(row["color"])
        width = 6 if selected else 3
        n = int(pts.shape[0])
        customdata = np.column_stack(
            [
                np.full(n, section_name, dtype=object),
                np.full(n, node_id if node_id is not None else -1, dtype=object),
                np.full(n, swc_label, dtype=object),
                np.full(n, length_um, dtype=float),
            ]
        )

        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line={"color": color, "width": width},
            name=section_name,
            customdata=customdata,
            hovertemplate=(
                "section=%{customdata[0]}<br>"
                "node_id=%{customdata[1]}<br>"
                "swc=%{customdata[2]}<br>"
                "length_um=%{customdata[3]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )

    def _apply_layout(self, fig) -> None:
        fig.update_layout(
            title=self.title,
            height=760,
            width=980,
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
            scene={
                "aspectmode": "data",
                "xaxis": {"title": "X (um)"},
                "yaxis": {"title": "Y (um)"},
                "zaxis": {"title": "Z (um)"},
            },
        )

    def _build_widget_figure(self) -> None:
        fig = go.FigureWidget()

        for _, row in self.section_df.iterrows():
            section_name = str(row["section_name"])
            tr = self._make_section_trace(row, selected=section_name in self.selected_section_names)
            if tr is None:
                continue
            tr.on_click(self._make_trace_click_handler(section_name))
            fig.add_trace(tr)
            self._traces_by_section[section_name] = tr

        self._soma_trace = self._build_soma_trace()
        if self._soma_trace is not None:
            fig.add_trace(self._soma_trace)

        self._ais_trace = self._build_ais_trace()
        if self._ais_trace is not None:
            fig.add_trace(self._ais_trace)

        self._apply_layout(fig)
        self.figure = fig

    def _build_static_figure(self):
        fig = go.Figure()
        for _, row in self.section_df.iterrows():
            section_name = str(row["section_name"])
            tr = self._make_section_trace(row, selected=section_name in self.selected_section_names)
            if tr is not None:
                fig.add_trace(tr)

        soma_trace = self._build_soma_trace()
        if soma_trace is not None:
            fig.add_trace(soma_trace)

        ais_trace = self._build_ais_trace()
        if ais_trace is not None:
            fig.add_trace(ais_trace)

        self._apply_layout(fig)
        return fig

    def _render_static_figure(self) -> None:
        self.figure = self._build_static_figure()
        with self.figure_output:
            self.figure_output.clear_output(wait=True)
            if display is not None:
                display(self.figure)

    def _build_soma_trace(self):
        from digifly.phase2.neuron_build.ais import _xyz_of_section_site

        try:
            sec, xloc = self.cell.soma_site()
        except Exception:
            return None
        x, y, z = _xyz_of_section_site(sec, xloc)
        return go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers+text",
            marker={"size": 7, "color": "#f6aa1c"},
            text=["soma"],
            textposition="top center",
            name="soma",
            hovertemplate="soma<extra></extra>",
            showlegend=False,
        )

    def _build_ais_trace(self):
        from digifly.phase2.neuron_build.ais import _xyz_of_section_site

        try:
            sec, xloc = self.cell.axon_ais_site()
        except Exception:
            return None
        x, y, z = _xyz_of_section_site(sec, xloc)
        return go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers+text",
            marker={"size": 7, "color": "#118ab2"},
            text=["AIS"],
            textposition="top center",
            name="AIS",
            hovertemplate="AIS<extra></extra>",
            showlegend=False,
        )

    def _make_trace_click_handler(self, section_name: str):
        def _handler(trace, points, _selector):
            if not points.point_inds:
                return
            self.toggle_section(section_name)

        return _handler

    def _on_clear_clicked(self, _btn) -> None:
        self.clear_selection()

    def _on_refresh_clicked(self, _btn) -> None:
        self.refresh_ais_marker()
        with self.output_widget:
            print("Refreshed AIS marker from current cell state.")

    def _on_picker_changed(self, change) -> None:
        if self._updating_picker:
            return
        picked = set(change.get("new", ()) or ())
        self.selected_section_names = picked
        if self.last_clicked_section_name not in self.selected_section_names:
            self.last_clicked_section_name = sorted(self.selected_section_names)[-1] if self.selected_section_names else None
        if self.backend == "widget":
            for section_name, tr in self._traces_by_section.items():
                if section_name in self.selected_section_names:
                    tr.line.color = SELECTED_COLOR
                    tr.line.width = 6
                else:
                    tr.line.color = self._base_colors.get(section_name, "#8d99ae")
                    tr.line.width = 3
        else:
            self._render_static_figure()
        self._update_status()

    def _update_status(self) -> None:
        picked = sorted(self.selected_section_names)
        self.selected_widget.options = picked
        self.selected_widget.value = tuple(picked)
        self._updating_picker = True
        try:
            self.section_picker_widget.value = tuple(picked)
        finally:
            self._updating_picker = False
        last = self.last_clicked_section_name if self.last_clicked_section_name else "None"
        mode_note = (
            "click traces to select"
            if self.backend == "widget"
            else "select from Pick list (static fallback)"
        )
        self.status_widget.value = (
            f"<b>Neuron:</b> {self.neuron_id} | "
            f"<b>Selected:</b> {len(picked)} | "
            f"<b>Last clicked:</b> {last} | "
            f"<b>Mode:</b> {mode_note}"
        )
        if self.backend == "static":
            self.mode_widget.value = (
                "<i>Static mode avoids Plotly FigureWidget/anywidget JS errors. "
                "Use the Pick list for selection and hover in 3D to inspect names.</i>"
            )
        else:
            self.mode_widget.value = ""

    def toggle_section(self, section_name: str) -> None:
        if section_name not in self._base_colors:
            return

        if section_name in self.selected_section_names:
            self.selected_section_names.remove(section_name)
        else:
            self.selected_section_names.add(section_name)

        if self.backend == "widget":
            tr = self._traces_by_section.get(section_name)
            if tr is not None:
                if section_name in self.selected_section_names:
                    tr.line.color = SELECTED_COLOR
                    tr.line.width = 6
                else:
                    tr.line.color = self._base_colors.get(section_name, "#8d99ae")
                    tr.line.width = 3
        else:
            self._render_static_figure()

        self.last_clicked_section_name = section_name
        self._update_status()

    def clear_selection(self) -> None:
        if self.backend == "widget":
            for section_name, tr in self._traces_by_section.items():
                tr.line.color = self._base_colors.get(section_name, "#8d99ae")
                tr.line.width = 3
        self.selected_section_names.clear()
        self.last_clicked_section_name = None
        if self.backend == "static":
            self._render_static_figure()
        self._update_status()

    def selected_sections(self) -> List[Any]:
        lookup = _sec_lookup(self.cell)
        out: List[Any] = []
        for nm in sorted(self.selected_section_names):
            sec = lookup.get(nm)
            if sec is not None:
                out.append(sec)
        return out

    def selected_node_ids(self) -> List[int]:
        rows = self.selected_table()
        if rows.empty:
            return []
        return [int(x) for x in pd.to_numeric(rows["node_id"], errors="coerce").dropna().astype(int).tolist()]

    def selected_table(self) -> pd.DataFrame:
        if not self.selected_section_names:
            return self.section_df.iloc[0:0].drop(columns=["_sec_obj"], errors="ignore").copy()
        df = self.section_df[self.section_df["section_name"].isin(sorted(self.selected_section_names))].copy()
        return df.drop(columns=["_sec_obj"], errors="ignore").reset_index(drop=True)

    def show(self):
        panel_items = [self.status_widget, self.mode_widget, widgets.HBox([self.clear_button, self.refresh_button])]
        if self.backend == "static":
            panel_items.append(self.section_picker_widget)
        panel_items.extend([self.selected_widget, self.output_widget])
        panel = widgets.VBox(panel_items, layout=widgets.Layout(width="440px"))
        left_obj = self.figure if self.backend == "widget" else self.figure_output
        ui = widgets.HBox([left_obj, panel])
        if display is not None:
            display(ui)
        return ui

    def refresh_ais_marker(self) -> None:
        from digifly.phase2.neuron_build.ais import _xyz_of_section_site

        if self.backend != "widget":
            self._render_static_figure()
            return
        if self._ais_trace is None:
            return
        try:
            sec, xloc = self.cell.axon_ais_site()
            x, y, z = _xyz_of_section_site(sec, xloc)
            self._ais_trace.x = [x]
            self._ais_trace.y = [y]
            self._ais_trace.z = [z]
        except Exception:
            pass


def apply_glia_loss_to_selected_sections(
    cell: Any,
    selected_sections: Sequence[Any] | InteractiveSWCSelector,
    *,
    ko_mM: Optional[float] = None,
    default_ko_mM: float = 3.0,
    ko_scale: float = 1.0,
    ko_delta_mM: float = 0.0,
    ki_mM: float = 140.0,
    celsius_C: Optional[float] = None,
    update_ek: bool = True,
) -> pd.DataFrame:
    """
    Part 2: apply glia-loss-like local potassium changes to selected compartments.
    """
    if isinstance(selected_sections, InteractiveSWCSelector):
        sections = selected_sections.selected_sections()
        section_names = sorted(selected_sections.selected_section_names)
    else:
        sections = list(selected_sections)
        section_names = [str(sec.name()) for sec in sections]

    if celsius_C is None:
        celsius_C = float(getattr(cell, "cfg", {}).get("celsius_C", 22.0))

    target_ko = (
        float(ko_mM)
        if ko_mM is not None
        else float(default_ko_mM) * float(ko_scale) + float(ko_delta_mM)
    )
    target_ek = _ko_to_ek_mV(target_ko, float(ki_mM), float(celsius_C)) if update_ek else float("nan")

    inv = _inv_node_map(cell)
    rows: List[Dict[str, Any]] = []

    for sec, sec_name in zip(sections, section_names):
        seg_updates = _set_ko_and_ek_on_section(
            sec,
            ko_mM=float(target_ko),
            ki_mM=float(ki_mM),
            celsius_C=float(celsius_C),
            update_ek=bool(update_ek),
        )
        rows.append(
            {
                "neuron_id": int(getattr(cell, "gid", -1)),
                "section_name": sec_name,
                "node_id": int(inv.get(sec)) if inv.get(sec) is not None else None,
                "segment_updates": int(seg_updates),
                "ko_mM": float(target_ko),
                "ki_mM": float(ki_mM),
                "ek_mV": float(target_ek) if update_ek else np.nan,
                "ek_updated": bool(update_ek),
            }
        )

    return pd.DataFrame(rows)


def make_glia_loss_spec_from_selection(
    selector: InteractiveSWCSelector,
    *,
    ko_mM: Optional[float] = None,
    ko_scale: Optional[float] = None,
    ko_delta_mM: Optional[float] = None,
    ki_mM: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Export selection to the GLIA_LOSS_SPEC shape used in glia_simulation.ipynb.
    """
    out: List[Dict[str, Any]] = []
    for section_name in sorted(selector.selected_section_names):
        item: Dict[str, Any] = {
            "neuron_id": int(selector.neuron_id),
            "compartment": f"sec:{section_name}",
        }
        if ko_mM is not None:
            item["ko_mM"] = float(ko_mM)
        if ko_scale is not None:
            item["ko_scale"] = float(ko_scale)
        if ko_delta_mM is not None:
            item["ko_delta_mM"] = float(ko_delta_mM)
        if ki_mM is not None:
            item["ki_mM"] = float(ki_mM)
        out.append(item)
    return out


def set_ais_from_selection(
    cell: Any,
    selector: InteractiveSWCSelector,
    *,
    xloc: float = 0.5,
    section_name: Optional[str] = None,
    persist_override: bool = False,
    source: str = "interactive_selector",
) -> AISAssignment:
    """
    Part 3: set selected compartment as AIS (in-memory, optional persistent override).
    """
    target_section = section_name
    if target_section is None:
        if selector.last_clicked_section_name:
            target_section = selector.last_clicked_section_name
        elif selector.selected_section_names:
            target_section = sorted(selector.selected_section_names)[0]
    if not target_section:
        raise ValueError("No selected section. Select one compartment first.")

    sec_map = _sec_lookup(cell)
    sec = sec_map.get(str(target_section))
    if sec is None:
        raise KeyError(f"Section not found on cell: {target_section}")

    xloc = float(np.clip(float(xloc), 0.0, 1.0))
    cell._ais_site = (sec, xloc)

    inv = _inv_node_map(cell)
    node_id = inv.get(sec)
    persisted = False

    if persist_override:
        if node_id is None:
            raise RuntimeError("Cannot persist AIS override: selected section has no node_id mapping.")
        try:
            from digifly.phase2.neuron_build.ais import _ais_override_write

            _ais_override_write(
                cfg=getattr(cell, "cfg", {}),
                gid=int(getattr(cell, "gid", -1)),
                node_id=int(node_id),
                xloc=float(xloc),
                cell=cell,
                src=str(source),
            )
            persisted = True
        except Exception as exc:  # pragma: no cover - depends on runtime files/permissions
            raise RuntimeError(f"Failed to persist AIS override: {exc}") from exc

    selector.refresh_ais_marker()
    return AISAssignment(
        neuron_id=int(getattr(cell, "gid", -1)),
        section_name=str(target_section),
        node_id=int(node_id) if node_id is not None else None,
        xloc=float(xloc),
        persisted=bool(persisted),
    )
