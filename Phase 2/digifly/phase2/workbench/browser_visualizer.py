from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SWC_SUFFIX_PRIORITY = (
    "_axodendro_with_synapses.swc",
    "_healed_final.swc",
    "_healed.swc",
)
SCENE_BG = "#070b16"
PAPER_BG = "#05070d"
PANEL_BG = "#080d18"
GRID_COLOR = "#263247"
AXIS_COLOR = "#cbd5e1"


@dataclass(frozen=True)
class SwcMorphology:
    neuron_id: int
    path: Path
    nodes: pd.DataFrame


@dataclass(frozen=True)
class BrowserFlowInputs:
    run_dir: Path
    swc_dir: Path
    neuron_ids: tuple[int, ...]


def recorded_neuron_ids(run_dir: str | Path) -> list[int]:
    """Return neuron IDs with soma voltage traces in a Phase 2 records.csv."""

    records_path = Path(run_dir).expanduser().resolve() / "records.csv"
    if not records_path.exists():
        return []
    header = records_path.open("r", encoding="utf-8").readline().strip().split(",")
    ids: set[int] = set()
    for column in header:
        name = str(column).strip()
        if not name.endswith("_soma_v"):
            continue
        raw = name[: -len("_soma_v")]
        if raw.isdigit():
            ids.add(int(raw))
    return sorted(ids)


def find_swc_file(swc_dir: str | Path, neuron_id: int) -> Path:
    """Find the preferred public SWC for one neuron under an export_swc tree."""

    root = Path(swc_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SWC root not found: {root}")

    nid = int(neuron_id)
    matches = [p for p in root.rglob(f"{nid}_*.swc") if "_OLD_" not in p.name and not p.name.startswith(".")]
    if not matches:
        raise FileNotFoundError(f"No SWC found for neuron {nid} under {root}")

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name
        for idx, suffix in enumerate(SWC_SUFFIX_PRIORITY):
            if name == f"{nid}{suffix}":
                return (idx, len(path.parts), str(path))
        return (len(SWC_SUFFIX_PRIORITY), len(path.parts), str(path))

    return sorted(matches, key=score)[0].resolve()


def load_swc(path: str | Path, *, neuron_id: int | None = None) -> SwcMorphology:
    """Parse an SWC file into a typed node table."""

    swc_path = Path(path).expanduser().resolve()
    rows: list[tuple[int, int, float, float, float, float, int]] = []
    with swc_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    (
                        int(float(parts[0])),
                        int(float(parts[1])),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        int(float(parts[6])),
                    )
                )
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"SWC contains no parseable nodes: {swc_path}")
    df = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "radius", "parent"])
    resolved_id = int(neuron_id) if neuron_id is not None else _neuron_id_from_path(swc_path)
    return SwcMorphology(neuron_id=resolved_id, path=swc_path, nodes=df)


def resolve_browser_flow_inputs(
    *,
    run_dir: str | Path,
    swc_dir: str | Path,
    neuron_ids: Sequence[int] | None = None,
) -> BrowserFlowInputs:
    """Resolve and validate the run/SWC inputs needed by the browser visualizer."""

    run_root = Path(run_dir).expanduser().resolve()
    swc_root = Path(swc_dir).expanduser().resolve()
    missing = [name for name in ("records.csv",) if not (run_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run directory is missing {', '.join(missing)}: {run_root}")
    ids = tuple(int(x) for x in (neuron_ids or recorded_neuron_ids(run_root)))
    if not ids:
        raise ValueError(f"No recorded neuron IDs found in {run_root / 'records.csv'}")
    for nid in ids:
        find_swc_file(swc_root, int(nid))
    return BrowserFlowInputs(run_dir=run_root, swc_dir=swc_root, neuron_ids=ids)


def build_browser_flow_figure(
    *,
    run_dir: str | Path,
    swc_dir: str | Path,
    neuron_ids: Sequence[int] | None = None,
    max_frames: int = 240,
    playback_seconds: float = 20.0,
    flow_speed_um_per_ms: float = 25.0,
    pulse_sigma_ms: float = 18.0,
    max_nodes_per_neuron: int = 1000,
    max_edges_per_neuron: int = 3200,
):
    """Build a Plotly figure that animates Phase 2 voltage flow in-browser."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    inputs = resolve_browser_flow_inputs(run_dir=run_dir, swc_dir=swc_dir, neuron_ids=neuron_ids)
    records = pd.read_csv(inputs.run_dir / "records.csv")
    time_col = _find_time_column(records)
    t_ms = pd.to_numeric(records[time_col], errors="coerce").to_numpy(dtype=float)
    if t_ms.size < 2 or not np.all(np.isfinite(t_ms)):
        raise ValueError(f"records.csv has an invalid time column: {time_col}")

    frame_times = _frame_times(t_ms, max_frames=max_frames)
    morphs = [load_swc(find_swc_file(inputs.swc_dir, nid), neuron_id=nid) for nid in inputs.neuron_ids]
    colors = _palette([m.neuron_id for m in morphs])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.05,
    )

    all_point_x: list[float] = []
    all_point_y: list[float] = []
    all_point_z: list[float] = []
    all_point_size: list[float] = []
    all_point_custom: list[list[Any]] = []
    intensity_by_frame_parts: list[np.ndarray] = []

    voltage_cols = []
    y_min = float("inf")
    y_max = float("-inf")

    for morph in morphs:
        nid = int(morph.neuron_id)
        line_x, line_y, line_z = _edge_line_coords(morph.nodes, max_edges=max_edges_per_neuron)
        fig.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": colors[nid], "width": 6},
                opacity=0.62,
                name=f"{nid} morphology",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        sample = _sample_nodes(morph.nodes, max_nodes=max_nodes_per_neuron)
        dist = _distance_from_source(morph.nodes, _source_node_id(morph.nodes))
        dists = np.asarray([dist.get(int(node_id), np.nan) for node_id in sample["id"]], dtype=float)
        dists = np.where(np.isfinite(dists), dists, 0.0)
        trace_col = _trace_column(records, nid)
        voltage = pd.to_numeric(records[trace_col], errors="coerce").to_numpy(dtype=float)
        voltage_cols.append((nid, trace_col, voltage))
        y_min = min(y_min, float(np.nanmin(voltage)))
        y_max = max(y_max, float(np.nanmax(voltage)))
        spike_times = _spike_times_from_trace(t_ms, voltage, thresh_mV=0.0)
        base_norm = _voltage_norm(t_ms, voltage, frame_times)
        intensities = _node_flow_intensity(
            frame_times,
            base_norm=base_norm,
            spike_times=spike_times,
            dists_um=dists,
            flow_speed_um_per_ms=flow_speed_um_per_ms,
            pulse_sigma_ms=pulse_sigma_ms,
        )
        intensity_by_frame_parts.append(intensities)

        all_point_x.extend(sample["x"].astype(float).tolist())
        all_point_y.extend(sample["y"].astype(float).tolist())
        all_point_z.extend(sample["z"].astype(float).tolist())
        radius = np.clip(sample["radius"].astype(float).to_numpy(), 0.04, 1.2)
        all_point_size.extend((5.5 + radius * 10.0).tolist())
        all_point_custom.extend([[nid, int(node_id)] for node_id in sample["id"]])

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -80.0, 40.0
    y_pad = max(5.0, (y_max - y_min) * 0.08)
    y_range = [y_min - y_pad, y_max + y_pad]
    initial_intensity = (
        np.concatenate([part[0] for part in intensity_by_frame_parts])
        if intensity_by_frame_parts
        else np.array([], dtype=float)
    )
    flow_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=all_point_x,
            y=all_point_y,
            z=all_point_z,
            mode="markers",
            marker={
                "size": all_point_size,
                "color": initial_intensity,
                "cmin": 0.0,
                "cmax": 1.0,
                "colorscale": _flow_colorscale(),
                "opacity": 0.94,
                "colorbar": {
                    "title": {"text": "activity", "font": {"color": AXIS_COLOR}},
                    "len": 0.72,
                    "tickfont": {"color": AXIS_COLOR},
                    "outlinecolor": GRID_COLOR,
                },
            },
            customdata=all_point_custom,
            name="browser flow",
            hovertemplate="neuron %{customdata[0]}<br>node %{customdata[1]}<br>activity %{marker.color:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for nid, _trace_col, voltage in voltage_cols:
        fig.add_trace(
            go.Scatter(
                x=t_ms,
                y=voltage,
                mode="lines",
                line={"width": 1.8, "color": colors[nid]},
                name=f"{nid} soma V",
            ),
            row=1,
            col=2,
        )

    cursor_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[float(frame_times[0]), float(frame_times[0])],
            y=y_range,
            mode="lines",
            line={"color": "#f8fafc", "width": 2, "dash": "dot"},
            name="time",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    frame_ms = max(20, int(round((float(playback_seconds) * 1000.0) / max(1, len(frame_times)))))
    frames = []
    for idx, frame_t in enumerate(frame_times):
        color = np.concatenate([part[idx] for part in intensity_by_frame_parts])
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter3d(marker={"color": color}),
                    go.Scatter(x=[float(frame_t), float(frame_t)], y=y_range),
                ],
                traces=[flow_trace_idx, cursor_trace_idx],
                layout=go.Layout(title_text=f"Browser flow visualizer: sim t={float(frame_t):.1f} ms"),
            )
        )
    fig.frames = frames

    steps = [
        {
            "args": [
                [str(i)],
                {"frame": {"duration": frame_ms, "redraw": True}, "mode": "immediate"},
            ],
            "label": f"{float(t):.0f}",
            "method": "animate",
        }
        for i, t in enumerate(frame_times)
    ]
    fig.update_layout(
        title=f"Browser flow visualizer: {inputs.run_dir.name}",
        height=720,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": AXIS_COLOR},
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "bgcolor": SCENE_BG,
            "xaxis": _scene_axis(),
            "yaxis": _scene_axis(),
            "zaxis": _scene_axis(),
        },
        hoverlabel={"bgcolor": "#111827", "font": {"color": "#f8fafc"}, "bordercolor": GRID_COLOR},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08, "xanchor": "left", "x": 0.0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.18,
                "y": 1.08,
                "len": 0.76,
                "xanchor": "left",
                "yanchor": "top",
                "currentvalue": {"prefix": "sim ms: ", "font": {"color": AXIS_COLOR}},
                "font": {"color": AXIS_COLOR},
                "bgcolor": "#111827",
                "activebgcolor": "#2563eb",
                "bordercolor": GRID_COLOR,
                "pad": {"t": 0, "b": 0},
            }
        ],
    )
    fig.update_xaxes(
        title_text="time (ms)",
        row=1,
        col=2,
        color=AXIS_COLOR,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )
    fig.update_yaxes(
        title_text="soma voltage (mV)",
        range=y_range,
        row=1,
        col=2,
        color=AXIS_COLOR,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )
    return fig


class BrowserFlowVisualizerApp:
    """Small ipywidgets app for building the Plotly browser visualizer."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        swc_dir: str | Path,
        neuron_ids: Sequence[int] | None = None,
    ):
        try:
            import ipywidgets as widgets
        except Exception as exc:  # pragma: no cover - notebook runtime dependency
            raise RuntimeError("ipywidgets is required for the browser visualizer app.") from exc

        self.widgets = widgets
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.swc_dir = Path(swc_dir).expanduser().resolve()
        self.neuron_ids = tuple(int(x) for x in (neuron_ids or recorded_neuron_ids(self.run_dir)))

        self.frames = widgets.IntSlider(value=240, min=60, max=420, step=30, description="Frames")
        self.playback = widgets.FloatSlider(value=20.0, min=5.0, max=45.0, step=1.0, description="Seconds")
        self.speed = widgets.FloatText(value=25.0, description="um/ms")
        self.sigma = widgets.FloatText(value=18.0, description="Glow ms")
        self.max_nodes = widgets.IntSlider(value=1000, min=200, max=2500, step=100, description="Nodes")
        self.render_button = widgets.Button(description="Render Browser Visualizer", button_style="success")
        self.output = widgets.Output()
        self.render_button.on_click(lambda _button: self.render())

    def display(self) -> None:
        from IPython.display import display

        controls = self.widgets.VBox(
            [
                self.widgets.HTML(
                    "<h3>Browser Flow Visualizer</h3>"
                    "<p>Runs entirely in JupyterLab using Plotly. Use this path in Docker/Windows sessions.</p>"
                ),
                self.widgets.HTML(f"<b>Run:</b> {self.run_dir}<br><b>SWC root:</b> {self.swc_dir}"),
                self.widgets.HBox([self.frames, self.playback, self.speed, self.sigma, self.max_nodes]),
                self.render_button,
            ]
        )
        display(self.widgets.VBox([controls, self.output]))
        self.render()

    def render(self) -> None:
        from IPython.display import display

        with self.output:
            self.output.clear_output()
            fig = build_browser_flow_figure(
                run_dir=self.run_dir,
                swc_dir=self.swc_dir,
                neuron_ids=self.neuron_ids,
                max_frames=int(self.frames.value),
                playback_seconds=float(self.playback.value),
                flow_speed_um_per_ms=float(self.speed.value),
                pulse_sigma_ms=float(self.sigma.value),
                max_nodes_per_neuron=int(self.max_nodes.value),
            )
            display(fig)


def launch_browser_flow_visualizer(
    *,
    run_dir: str | Path,
    swc_dir: str | Path,
    neuron_ids: Sequence[int] | None = None,
) -> BrowserFlowVisualizerApp:
    app = BrowserFlowVisualizerApp(run_dir=run_dir, swc_dir=swc_dir, neuron_ids=neuron_ids)
    app.display()
    return app


def _neuron_id_from_path(path: Path) -> int:
    raw = path.name.split("_", 1)[0]
    if raw.isdigit():
        return int(raw)
    for parent in path.parents:
        if parent.name.isdigit():
            return int(parent.name)
    raise ValueError(f"Could not infer neuron ID from SWC path: {path}")


def _find_time_column(records: pd.DataFrame) -> str:
    for name in ("t_ms", "time_ms", "time", "t"):
        if name in records.columns:
            return str(name)
    raise ValueError(f"records.csv has no recognizable time column. Columns: {list(records.columns)[:12]}")


def _trace_column(records: pd.DataFrame, neuron_id: int) -> str:
    preferred = f"{int(neuron_id)}_soma_v"
    if preferred in records.columns:
        return preferred
    prefix = f"{int(neuron_id)}"
    for column in records.columns:
        name = str(column)
        if name.startswith(prefix) and name.endswith("_soma_v"):
            return name
    raise ValueError(f"No soma voltage column found for neuron {int(neuron_id)}")


def _frame_times(t_ms: np.ndarray, *, max_frames: int) -> np.ndarray:
    vals = np.asarray(t_ms, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= max(2, int(max_frames)):
        return vals
    return np.linspace(float(vals[0]), float(vals[-1]), max(2, int(max_frames)), dtype=float)


def _palette(neuron_ids: Iterable[int]) -> dict[int, str]:
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    )
    return {int(nid): colors[i % len(colors)] for i, nid in enumerate(sorted(set(int(x) for x in neuron_ids)))}


def _flow_colorscale() -> list[list[Any]]:
    return [
        [0.00, "#0b1020"],
        [0.15, "#12356f"],
        [0.35, "#1455d9"],
        [0.58, "#06b6d4"],
        [0.78, "#e0f7ff"],
        [1.00, "#ffffff"],
    ]


def _scene_axis() -> dict[str, Any]:
    return {
        "backgroundcolor": SCENE_BG,
        "gridcolor": GRID_COLOR,
        "zerolinecolor": GRID_COLOR,
        "showbackground": True,
        "color": AXIS_COLOR,
    }


def _edge_line_coords(nodes: pd.DataFrame, *, max_edges: int) -> tuple[list[float | None], list[float | None], list[float | None]]:
    pos = {
        int(row.id): (float(row.x), float(row.y), float(row.z))
        for row in nodes.itertuples(index=False)
    }
    edges = [
        (int(row.id), int(row.parent))
        for row in nodes.itertuples(index=False)
        if int(row.parent) in pos and int(row.parent) != -1
    ]
    stride = max(1, int(np.ceil(len(edges) / max(1, int(max_edges)))))
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for child, parent in edges[::stride]:
        x0, y0, z0 = pos[parent]
        x1, y1, z1 = pos[child]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        zs.extend([z0, z1, None])
    return xs, ys, zs


def _sample_nodes(nodes: pd.DataFrame, *, max_nodes: int) -> pd.DataFrame:
    if len(nodes) <= int(max_nodes):
        return nodes.copy()
    root_like = nodes[(nodes["parent"] == -1) | (nodes["type"] == 1)]
    stride = max(1, int(np.ceil(len(nodes) / max(1, int(max_nodes)))))
    sampled = nodes.iloc[::stride].copy()
    if not root_like.empty:
        sampled = pd.concat([root_like, sampled], ignore_index=True).drop_duplicates("id")
    return sampled.head(int(max_nodes)).copy()


def _source_node_id(nodes: pd.DataFrame) -> int:
    soma = nodes[nodes["type"] == 1]
    if not soma.empty:
        return int(soma.iloc[0]["id"])
    root = nodes[nodes["parent"] == -1]
    if not root.empty:
        return int(root.iloc[0]["id"])
    return int(nodes.iloc[0]["id"])


def _distance_from_source(nodes: pd.DataFrame, source_node_id: int) -> dict[int, float]:
    pos = {
        int(row.id): np.asarray([float(row.x), float(row.y), float(row.z)], dtype=float)
        for row in nodes.itertuples(index=False)
    }
    adj: dict[int, list[tuple[int, float]]] = {nid: [] for nid in pos}
    for row in nodes.itertuples(index=False):
        child = int(row.id)
        parent = int(row.parent)
        if parent == -1 or parent not in pos:
            continue
        dist = float(np.linalg.norm(pos[child] - pos[parent]))
        adj[child].append((parent, dist))
        adj[parent].append((child, dist))

    source = int(source_node_id)
    out: dict[int, float] = {source: 0.0}
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        cur_dist, cur = heapq.heappop(heap)
        if cur_dist > out.get(cur, float("inf")):
            continue
        for nxt, edge_dist in adj.get(cur, []):
            cand = cur_dist + edge_dist
            if cand < out.get(nxt, float("inf")):
                out[nxt] = cand
                heapq.heappush(heap, (cand, nxt))
    return out


def _spike_times_from_trace(t_ms: np.ndarray, v_mV: np.ndarray, thresh_mV: float) -> np.ndarray:
    tt = np.asarray(t_ms, dtype=float)
    vv = np.asarray(v_mV, dtype=float)
    if tt.size != vv.size or tt.size < 2:
        return np.array([], dtype=float)
    idx = np.where((vv[:-1] < float(thresh_mV)) & (vv[1:] >= float(thresh_mV)))[0]
    if idx.size == 0:
        return np.array([], dtype=float)
    t0 = tt[idx]
    t1 = tt[idx + 1]
    v0 = vv[idx]
    v1 = vv[idx + 1]
    denom = np.where(np.abs(v1 - v0) < 1e-12, 1e-12, v1 - v0)
    return t0 + ((float(thresh_mV) - v0) / denom) * (t1 - t0)


def _voltage_norm(t_ms: np.ndarray, voltage: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    vals = np.asarray(voltage, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(frame_times, dtype=float)
    lo = min(float(np.percentile(finite, 5)), -70.0)
    hi = max(float(np.percentile(finite, 98)), 20.0)
    if hi <= lo:
        hi = lo + 1.0
    interp = np.interp(frame_times, t_ms, vals)
    return np.clip((interp - lo) / (hi - lo), 0.0, 1.0)


def _node_flow_intensity(
    frame_times: np.ndarray,
    *,
    base_norm: np.ndarray,
    spike_times: np.ndarray,
    dists_um: np.ndarray,
    flow_speed_um_per_ms: float,
    pulse_sigma_ms: float,
) -> np.ndarray:
    n_frames = len(frame_times)
    n_nodes = len(dists_um)
    out = np.repeat((0.04 + 0.44 * base_norm[:, None]), n_nodes, axis=1)
    if spike_times.size == 0:
        return np.clip(out, 0.0, 1.0)
    arrivals = spike_times[:, None] + dists_um[None, :] / max(1e-6, float(flow_speed_um_per_ms))
    rise = max(1e-6, float(pulse_sigma_ms) * 0.75)
    decay = max(1e-6, float(pulse_sigma_ms) * 1.8)
    for frame_idx, frame_t in enumerate(frame_times):
        dt = float(frame_t) - arrivals
        pulse = np.zeros_like(dt, dtype=float)
        before = dt < 0.0
        if np.any(before):
            pulse[before] = np.exp(-0.5 * (dt[before] / rise) ** 2)
        after = ~before
        if np.any(after):
            pulse[after] = np.exp(-dt[after] / decay)
        out[frame_idx, :] = np.maximum(out[frame_idx, :], np.max(0.08 + 0.92 * pulse, axis=0))
    return np.clip(out, 0.0, 1.0)
