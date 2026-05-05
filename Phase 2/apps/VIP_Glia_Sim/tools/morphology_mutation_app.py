from __future__ import annotations

import argparse
import copy
from datetime import datetime
import heapq
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyvista as pv

try:
    from PIL import Image
except Exception:
    Image = None

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.morphology_mutation import MorphologyMutationProject


SelectionKey = Tuple[int, int]  # (neuron_id, child_node_id)
FocusPair = Tuple[int, int]


def _hex_from_rgb01(rgb: Sequence[float]) -> str:
    r = int(np.clip(round(float(rgb[0]) * 255.0), 0, 255))
    g = int(np.clip(round(float(rgb[1]) * 255.0), 0, 255))
    b = int(np.clip(round(float(rgb[2]) * 255.0), 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb01_from_hex(color: str) -> Tuple[float, float, float]:
    s = str(color).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return (0.298, 0.471, 0.659)
    try:
        return (
            int(s[0:2], 16) / 255.0,
            int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0,
        )
    except Exception:
        return (0.298, 0.471, 0.659)


def _mix_rgb01(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float, float]:
    tt = float(np.clip(float(t), 0.0, 1.0))
    return tuple(
        float((1.0 - tt) * float(x) + tt * float(y))
        for x, y in zip(a[:3], b[:3])
    )


def _volume_cmap_for_color(color: str):
    base = _rgb01_from_hex(color)
    dark = _mix_rgb01((0.03, 0.04, 0.06), base, 0.40)
    mid = _mix_rgb01((0.08, 0.10, 0.14), base, 0.82)
    high = _mix_rgb01(base, (1.0, 1.0, 1.0), 0.35)
    try:
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            f"ng_like_{_hex_from_rgb01(base).lstrip('#')}",
            [dark, mid, high],
        )
    except Exception:
        return [_hex_from_rgb01(dark), _hex_from_rgb01(mid), _hex_from_rgb01(high)]


def _color_map_for_ids(ids: Sequence[int]) -> Dict[int, str]:
    # Matplotlib tab20 categorical palette (fixed, deterministic).
    palette = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ]
    out: Dict[int, str] = {}
    for i, nid in enumerate(ids):
        out[int(nid)] = palette[int(i % len(palette))]
    return out


def _rgba_from_any(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        rgb = np.stack([a, a, a], axis=-1)
        alpha = np.full((a.shape[0], a.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([rgb.astype(np.uint8), alpha], axis=-1)
    if a.ndim == 3 and a.shape[2] == 3:
        alpha = np.full((a.shape[0], a.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([a.astype(np.uint8), alpha], axis=-1)
    if a.ndim == 3 and a.shape[2] >= 4:
        return a[..., :4].astype(np.uint8)
    raise ValueError(f"Unsupported image shape for RGBA conversion: {a.shape}")



def _parse_neuron_ids(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).replace(";", ",").split(","):
        s = token.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("No neuron ids provided")
    return sorted(set(int(x) for x in out))


def _parse_focus_pair(raw: Optional[str]) -> Optional[FocusPair]:
    if raw is None:
        return None
    vals = _parse_neuron_ids(str(raw))
    if len(vals) != 2:
        raise ValueError("flow focus pair must contain exactly two neuron IDs")
    return int(vals[0]), int(vals[1])


def _parse_neuron_color_overrides(raw: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if raw is None:
        return out
    for token in re.split(r"[;,]", str(raw)):
        s = token.strip()
        if not s:
            continue
        if ":" in s:
            nid_s, color_s = s.split(":", 1)
        elif "=" in s:
            nid_s, color_s = s.split("=", 1)
        else:
            raise ValueError(
                "Neuron color overrides must look like '10000:#1d4ed8,10068:#b91c1c'"
            )
        nid = int(str(nid_s).strip())
        color = str(color_s).strip()
        if not color:
            raise ValueError(f"Missing color for neuron_id={nid}")
        out[int(nid)] = color
    return out


def _flow_movie_frame_times(
    t_ms: Sequence[float],
    *,
    flow_max_ms: Optional[float],
    frame_stride: int,
    fps: int,
    duration_sec: Optional[float],
) -> np.ndarray:
    """Choose simulation times for movie frames.

    When duration_sec is positive, the full selected simulation interval is
    resampled to fps * duration_sec frames. This compresses long runs, such as
    1000 ms simulations, into a smooth fixed-duration visualization.
    """

    vals = np.asarray(t_ms, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    if flow_max_ms is not None:
        vals = vals[vals <= float(flow_max_ms)]
    if vals.size < 2:
        return vals

    duration = 0.0 if duration_sec is None else float(duration_sec)
    if duration > 0.0:
        n_frames = max(2, int(round(duration * max(1, int(fps)))))
        return np.linspace(float(vals[0]), float(vals[-1]), n_frames, dtype=float)

    frame_times = vals[:: max(1, int(frame_stride))]
    return frame_times if frame_times.size >= 2 else vals


def _asymmetric_flow_pulse(dt_ms: np.ndarray, *, rise_ms: float, decay_ms: float) -> np.ndarray:
    """Return a soft pre-arrival ramp and slower post-arrival fade."""

    dt = np.asarray(dt_ms, dtype=float)
    pulse = np.zeros_like(dt, dtype=float)
    before = dt < 0.0
    if np.any(before):
        pulse[before] = np.exp(-0.5 * (dt[before] / max(1e-6, float(rise_ms))) ** 2)
    after = ~before
    if np.any(after):
        pulse[after] = np.exp(-dt[after] / max(1e-6, float(decay_ms)))
    return pulse


def _resolve_flow_cmap(style: str):
    mode = str(style).strip().lower()
    if mode == "neuron_yellow":
        try:
            from matplotlib.colors import LinearSegmentedColormap
            return LinearSegmentedColormap.from_list(
                "neuron_yellow_flow",
                ["#111111", "#3b3000", "#8a6b00", "#ffd23f", "#fff59d"],
            )
        except Exception:
            return "autumn"
    if mode == "electric_cyan":
        try:
            from matplotlib.colors import LinearSegmentedColormap
            return LinearSegmentedColormap.from_list(
                "electric_cyan_flow",
                ["#080808", "#0b2545", "#134074", "#00b4d8", "#90e0ef", "#f1fdff"],
            )
        except Exception:
            return "winter"
    return "viridis"


def _syn_csv_path(swc_root: str | Path, nid: int) -> Optional[Path]:
    p = Path(swc_root)
    nid = int(nid)
    for pat in (
        f"**/{nid}/{nid}_synapses_new.csv",
        f"**/{nid}_synapses_new.csv",
        f"**/*{nid}*synapses_new*.csv",
    ):
        hits = list(p.glob(pat))
        if hits:
            hits.sort(key=lambda q: len(str(q)))
            return hits[0]
    return None


def _load_syn_catalog_local(swc_root: str | Path, nid: int) -> Dict[str, np.ndarray]:
    path = _syn_csv_path(swc_root, int(nid))
    if not path or not path.exists():
        return {"pre": np.empty((0, 3), dtype=float), "post": np.empty((0, 3), dtype=float)}
    df = pd.read_csv(path)
    for c in ("x", "y", "z", "type"):
        if c not in df.columns:
            df[c] = np.nan
    xyz = df[["x", "y", "z"]].to_numpy(float)
    if np.nanmedian(np.abs(xyz)) > 200.0:
        df[["x", "y", "z"]] = df[["x", "y", "z"]] * 0.001
    t = df["type"].astype(str).str.lower()
    pre = df.loc[t.eq("pre"), ["x", "y", "z"]].to_numpy(float)
    post = df.loc[t.eq("post"), ["x", "y", "z"]].to_numpy(float)
    return {"pre": pre, "post": post}


def _resolve_connection_cmap(style: str):
    mode = str(style).strip().lower()
    if mode == "neuron_yellow":
        try:
            from matplotlib.colors import LinearSegmentedColormap
            return LinearSegmentedColormap.from_list(
                "neuron_yellow_conn",
                ["#141414", "#4a3400", "#c77d00", "#ffe082"],
            )
        except Exception:
            return "autumn"
    if mode == "electric_cyan":
        try:
            from matplotlib.colors import LinearSegmentedColormap
            return LinearSegmentedColormap.from_list(
                "electric_cyan_conn",
                ["#101010", "#12304a", "#1d5f8a", "#38bdf8", "#dff8ff"],
            )
        except Exception:
            return "cool"
    return "autumn"



def _extract_selected_segment_indices(picked: Any) -> Set[int]:
    out: Set[int] = set()
    if picked is None:
        return out

    blocks: Sequence[Any]
    if isinstance(picked, pv.MultiBlock):
        blocks = [b for b in picked if b is not None]
    else:
        blocks = [picked]

    for b in blocks:
        try:
            arr = b.cell_data.get("segment_idx")
            if arr is None:
                arr = b.point_data.get("segment_idx")
            if arr is None:
                continue
            out.update(int(x) for x in np.asarray(arr).astype(int).tolist())
        except Exception:
            continue
    return out



def _as_xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    try:
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.size >= 3 and np.all(np.isfinite(p[:3])):
            return p[:3]
    except Exception:
        pass
    return None



class MorphologyMutationApp:
    def __init__(
        self,
        *,
        project: MorphologyMutationProject,
        output_root: Path,
        tag: str,
        selected_color: str,
        thin_factor: float,
        thick_factor: float,
        grow_length_um: float,
        grow_segments: int,
        grow_radius_scale: float,
        include_subtree_radius: bool,
        move_step_um: float,
        translate_subtree: bool,
        connection_chemical: int,
        connection_gap: int,
        connection_gap_mode: str,
        connection_gap_direction: Optional[str],
        require_single_component: bool,
        render_mode: str,
        skeleton_line_width: float,
        screenshot_scale: float,
        jpeg_quality: int,
        screenshot_dpi: int,
        visual_style: str,
        flow_run_dir: Optional[Path],
        flow_focus_pair: Optional[FocusPair],
        flow_fps: int,
        flow_frame_stride: int,
        flow_speed_um_per_ms: float,
        flow_pulse_sigma_ms: float,
        flow_syn_delay_ms: Optional[float],
        flow_threshold_mV: float,
        flow_max_ms: Optional[float],
        flow_duration_sec: Optional[float],
        neuron_color_overrides: Optional[Dict[int, str]],
        flow_preserve_camera: bool,
        flow_overlay_style: str,
        start_solo_mode: bool,
        start_neuron_id: Optional[int],
        neuroglancer_quality: str,
        neuroglancer_voxel_um: Optional[float],
        neuroglancer_max_dim: Optional[int],
        neuroglancer_max_voxels: Optional[int],
    ):
        self.project = project
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.tag = str(tag).strip().replace(" ", "_")

        self.selected_color = str(selected_color)
        self.thin_factor = float(thin_factor)
        self.thick_factor = float(thick_factor)
        self.grow_length_um = float(grow_length_um)
        self.grow_segments = max(1, int(grow_segments))
        self.grow_radius_scale = float(grow_radius_scale)
        self.include_subtree_radius = bool(include_subtree_radius)
        self.move_step_um = float(move_step_um)
        self.translate_subtree = bool(translate_subtree)

        self.connection_chemical = max(0, int(connection_chemical))
        self.connection_gap = max(0, int(connection_gap))
        self.connection_gap_mode = str(connection_gap_mode).strip().lower()
        self.connection_gap_direction = (
            str(connection_gap_direction).strip() if connection_gap_direction is not None else None
        )

        self.require_single_component = bool(require_single_component)
        self.render_mode = str(render_mode).strip().lower()
        if self.render_mode in {"volume", "neuroglancer_like", "neuroglancer-like", "ng"}:
            self.render_mode = "neuroglancer"
        if self.render_mode not in {"tube", "skeleton", "neuroglancer"}:
            self.render_mode = "tube"
        self._standard_render_mode = (
            self.render_mode if self.render_mode in {"tube", "skeleton"} else "tube"
        )
        self.skeleton_line_width = max(1.0, float(skeleton_line_width))

        self.screenshot_scale = max(1.0, float(screenshot_scale))
        self.jpeg_quality = int(np.clip(int(jpeg_quality), 50, 100))
        self.screenshot_dpi = int(np.clip(int(screenshot_dpi), 72, 1200))
        self.visual_style = "classic"
        self.ng_quality_mode = str(neuroglancer_quality).strip().lower() or "auto"
        if self.ng_quality_mode not in {"auto", "balanced", "high", "ultra"}:
            self.ng_quality_mode = "auto"
        self.ng_voxel_um_override = (
            None if neuroglancer_voxel_um is None else max(0.03, float(neuroglancer_voxel_um))
        )
        self.ng_max_dim_override = (
            None if neuroglancer_max_dim is None else max(64, int(neuroglancer_max_dim))
        )
        self.ng_max_voxels_override = (
            None if neuroglancer_max_voxels is None else max(1_000_000, int(neuroglancer_max_voxels))
        )
        self._volume_cache: Dict[Tuple[Any, ...], Tuple[pv.ImageData, float]] = {}
        self._last_ng_volume_stats: Dict[int, Tuple[float, Tuple[int, int, int]]] = {}

        self.flow_run_dir = (None if flow_run_dir is None else Path(flow_run_dir).expanduser().resolve())
        self.flow_focus_pair = flow_focus_pair
        self.flow_fps = max(1, int(flow_fps))
        self.flow_frame_stride = max(1, int(flow_frame_stride))
        self.flow_speed_um_per_ms = max(1e-6, float(flow_speed_um_per_ms))
        self.flow_pulse_sigma_ms = max(1e-6, float(flow_pulse_sigma_ms))
        self.flow_syn_delay_ms = (
            None if flow_syn_delay_ms is None else max(0.0, float(flow_syn_delay_ms))
        )
        self.flow_threshold_mV = float(flow_threshold_mV)
        self.flow_max_ms = (
            None if (flow_max_ms is None or float(flow_max_ms) <= 0.0) else float(flow_max_ms)
        )
        self.flow_duration_sec = (
            None if (flow_duration_sec is None or float(flow_duration_sec) <= 0.0)
            else float(flow_duration_sec)
        )
        self.flow_preserve_camera = bool(flow_preserve_camera)
        self.flow_overlay_style = str(flow_overlay_style).strip().lower()
        if self.flow_overlay_style not in {"neuron_yellow", "viridis", "electric_cyan"}:
            self.flow_overlay_style = "electric_cyan"
        self.camera_preset_path = self.output_root / f"morphology_mutation_camera_{self.tag}.json"

        self.mesh: Optional[pv.PolyData] = None
        self.segment_df = None
        self.root_points: Optional[pv.PolyData] = None
        self.neuron_colors: Dict[int, str] = _color_map_for_ids(self.project.neuron_ids)
        for _nid, _color in dict(neuron_color_overrides or {}).items():
            self.neuron_colors[int(_nid)] = str(_color)
        self._distance_cache: Dict[Tuple[int, int], Dict[int, float]] = {}
        self._syn_catalog_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self._synapse_overlay_cache: Dict[str, Any] = {}
        self.synapse_overlay_enabled: bool = False
        self.synapse_pair_index: int = 0
        _ids = [int(x) for x in self.project.neuron_ids]
        self.solo_mode = bool(start_solo_mode) and bool(_ids)
        if _ids:
            if start_neuron_id is not None and int(start_neuron_id) in _ids:
                self.solo_neuron_index = _ids.index(int(start_neuron_id))
            else:
                self.solo_neuron_index = 0
        else:
            self.solo_neuron_index = -1

        self.selected_keys: Set[SelectionKey] = set()
        self.selection_order: List[SelectionKey] = []
        self.pending_draw_anchor: Optional[SelectionKey] = None

        self.undo_stack: List[Dict[str, Any]] = []
        self.last_action: str = "none"

        self.plotter = pv.Plotter(window_size=(1600, 980))
        self._setup_scene()

    def _display_geom_from_cells(self, ds: Any) -> pv.PolyData:
        """Preserve line geometry for skeleton rendering across PyVista versions."""
        if isinstance(ds, pv.PolyData):
            try:
                return ds.clean()
            except Exception:
                return ds
        try:
            geom = ds.extract_geometry()
            if isinstance(geom, pv.PolyData) and int(geom.n_cells) > 0:
                try:
                    return geom.clean()
                except Exception:
                    return geom
        except Exception:
            pass
        try:
            geom = ds.extract_surface()
            if isinstance(geom, pv.PolyData) and int(geom.n_cells) > 0:
                try:
                    return geom.clean()
                except Exception:
                    return geom
        except Exception:
            pass
        try:
            edges = ds.extract_all_edges()
            if isinstance(edges, pv.PolyData) and int(edges.n_cells) > 0:
                try:
                    return edges.clean()
                except Exception:
                    return edges
        except Exception:
            pass
        return pv.PolyData()

    def _build_mesh(self) -> Tuple[pv.PolyData, Any, Optional[pv.PolyData]]:
        points: List[np.ndarray] = []
        lines: List[int] = []

        point_segment_idx: List[int] = []
        point_neuron_id: List[int] = []
        point_child_id: List[int] = []
        point_parent_id: List[int] = []
        point_swc_type: List[int] = []
        point_radius_um: List[float] = []

        cell_segment_idx: List[int] = []
        cell_neuron_id: List[int] = []
        cell_child_id: List[int] = []
        cell_parent_id: List[int] = []
        cell_swc_type: List[int] = []
        cell_radius_um: List[float] = []

        seg_rows: List[Dict[str, Any]] = []
        root_xyz: List[np.ndarray] = []

        p_off = 0
        s_idx = 0

        for nid in self.project.neuron_ids:
            df = self.project.table(nid)
            id_to_row = {int(r["id"]): r for _, r in df.iterrows()}

            for _, r in df.iterrows():
                child = int(r["id"])
                parent = int(r["parent"])
                if parent == -1 or parent not in id_to_row:
                    root_xyz.append(np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float))
                    continue

                prow = id_to_row[parent]
                pxyz = np.array([float(prow["x"]), float(prow["y"]), float(prow["z"])], dtype=float)
                cxyz = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)

                points.extend([pxyz, cxyz])
                lines.extend([2, p_off, p_off + 1])
                p_off += 2

                swc_type = int(r["type"])
                key = (int(nid), int(child))
                seg_rows.append(
                    {
                        "segment_idx": int(s_idx),
                        "neuron_id": int(nid),
                        "child_node_id": int(child),
                        "parent_node_id": int(parent),
                        "swc_type": int(swc_type),
                        "radius_um": float(r["r"]),
                        "key": key,
                    }
                )

                for _ in range(2):
                    point_segment_idx.append(int(s_idx))
                    point_neuron_id.append(int(nid))
                    point_child_id.append(int(child))
                    point_parent_id.append(int(parent))
                    point_swc_type.append(int(swc_type))
                    point_radius_um.append(float(r["r"]))

                cell_segment_idx.append(int(s_idx))
                cell_neuron_id.append(int(nid))
                cell_child_id.append(int(child))
                cell_parent_id.append(int(parent))
                cell_swc_type.append(int(swc_type))
                cell_radius_um.append(float(r["r"]))
                s_idx += 1

        if not points:
            raise RuntimeError("No drawable SWC segments found for selected neurons.")

        # Build line-only PolyData. Using pv.PolyData(points) implicitly adds vertex
        # cells for each point, which makes cell_data lengths mismatch line segments.
        mesh = pv.PolyData()
        mesh.points = np.vstack(points)
        mesh.lines = np.asarray(lines, dtype=np.int64)

        mesh.point_data["segment_idx"] = np.asarray(point_segment_idx, dtype=np.int64)
        mesh.point_data["neuron_id"] = np.asarray(point_neuron_id, dtype=np.int64)
        mesh.point_data["child_node_id"] = np.asarray(point_child_id, dtype=np.int64)
        mesh.point_data["parent_node_id"] = np.asarray(point_parent_id, dtype=np.int64)
        mesh.point_data["swc_type"] = np.asarray(point_swc_type, dtype=np.int32)
        mesh.point_data["radius_um"] = np.asarray(point_radius_um, dtype=float)

        mesh.cell_data["segment_idx"] = np.asarray(cell_segment_idx, dtype=np.int64)
        mesh.cell_data["neuron_id"] = np.asarray(cell_neuron_id, dtype=np.int64)
        mesh.cell_data["child_node_id"] = np.asarray(cell_child_id, dtype=np.int64)
        mesh.cell_data["parent_node_id"] = np.asarray(cell_parent_id, dtype=np.int64)
        mesh.cell_data["swc_type"] = np.asarray(cell_swc_type, dtype=np.int32)
        mesh.cell_data["radius_um"] = np.asarray(cell_radius_um, dtype=float)

        root_poly = None
        if root_xyz:
            root_poly = pv.PolyData(np.vstack(root_xyz))

        import pandas as pd

        seg_df = pd.DataFrame(seg_rows)
        return mesh, seg_df, root_poly

    def _apply_visual_style(self) -> None:
        try:
            if self.render_mode == "neuroglancer":
                self.plotter.set_background("#0b0f14", top="#182231")
            else:
                self.plotter.set_background("#646b75")
        except Exception:
            pass
        try:
            self.plotter.disable_eye_dome_lighting()
        except Exception:
            pass

    def _table_signature(self, neuron_id: int) -> Tuple[Any, ...]:
        df = self.project.table(int(neuron_id))
        return (
            int(neuron_id),
            int(len(df)),
            int(np.asarray(df["id"], dtype=np.int64).sum()) if "id" in df.columns else 0,
            int(np.asarray(df["parent"], dtype=np.int64).sum()) if "parent" in df.columns else 0,
            float(np.round(np.asarray(df["x"], dtype=float).sum(), 3)) if "x" in df.columns else 0.0,
            float(np.round(np.asarray(df["y"], dtype=float).sum(), 3)) if "y" in df.columns else 0.0,
            float(np.round(np.asarray(df["z"], dtype=float).sum(), 3)) if "z" in df.columns else 0.0,
            float(np.round(np.asarray(df["r"], dtype=float).sum(), 3)) if "r" in df.columns else 0.0,
        )

    def _effective_neuroglancer_settings(self) -> Dict[str, float]:
        visible_count = max(1, len(self._visible_neuron_ids()))
        if self.ng_quality_mode == "ultra":
            if visible_count <= 1:
                base = {
                    "voxel_um": 0.05,
                    "max_dim": 1536.0,
                    "max_voxels": 160_000_000.0,
                    "texture_strength": 0.28,
                    "shell_strength": 1.62,
                    "shell_frac": 0.045,
                    "core_weight": 0.020,
                    "density_gamma": 1.60,
                    "shell_power": 1.40,
                    "sample_distance_scale": 1.05,
                }
            elif visible_count <= 3:
                base = {
                    "voxel_um": 0.065,
                    "max_dim": 1280.0,
                    "max_voxels": 112_000_000.0,
                    "texture_strength": 0.25,
                    "shell_strength": 1.48,
                    "shell_frac": 0.050,
                    "core_weight": 0.025,
                    "density_gamma": 1.48,
                    "shell_power": 1.30,
                    "sample_distance_scale": 1.08,
                }
            else:
                base = {
                    "voxel_um": 0.085,
                    "max_dim": 1024.0,
                    "max_voxels": 84_000_000.0,
                    "texture_strength": 0.22,
                    "shell_strength": 1.34,
                    "shell_frac": 0.055,
                    "core_weight": 0.030,
                    "density_gamma": 1.38,
                    "shell_power": 1.22,
                    "sample_distance_scale": 1.10,
                }
        elif self.ng_quality_mode == "high":
            if visible_count <= 1:
                base = {
                    "voxel_um": 0.07,
                    "max_dim": 1280.0,
                    "max_voxels": 112_000_000.0,
                    "texture_strength": 0.24,
                    "shell_strength": 1.42,
                    "shell_frac": 0.055,
                    "core_weight": 0.028,
                    "density_gamma": 1.42,
                    "shell_power": 1.28,
                    "sample_distance_scale": 1.10,
                }
            elif visible_count <= 3:
                base = {
                    "voxel_um": 0.09,
                    "max_dim": 1024.0,
                    "max_voxels": 76_000_000.0,
                    "texture_strength": 0.21,
                    "shell_strength": 1.30,
                    "shell_frac": 0.060,
                    "core_weight": 0.034,
                    "density_gamma": 1.30,
                    "shell_power": 1.20,
                    "sample_distance_scale": 1.12,
                }
            else:
                base = {
                    "voxel_um": 0.12,
                    "max_dim": 768.0,
                    "max_voxels": 56_000_000.0,
                    "texture_strength": 0.19,
                    "shell_strength": 1.20,
                    "shell_frac": 0.070,
                    "core_weight": 0.040,
                    "density_gamma": 1.20,
                    "shell_power": 1.12,
                    "sample_distance_scale": 1.16,
                }
        elif self.ng_quality_mode == "balanced":
            base = {
                "voxel_um": 0.14 if visible_count <= 3 else 0.17,
                "max_dim": 768.0 if visible_count <= 3 else 640.0,
                "max_voxels": 60_000_000.0 if visible_count <= 3 else 40_000_000.0,
                "texture_strength": 0.18,
                "shell_strength": 1.16,
                "shell_frac": 0.075,
                "core_weight": 0.050,
                "density_gamma": 1.14,
                "shell_power": 1.08,
                "sample_distance_scale": 1.18,
            }
        else:
            if visible_count <= 1:
                base = {
                    "voxel_um": 0.06,
                    "max_dim": 1408.0,
                    "max_voxels": 132_000_000.0,
                    "texture_strength": 0.26,
                    "shell_strength": 1.50,
                    "shell_frac": 0.050,
                    "core_weight": 0.024,
                    "density_gamma": 1.50,
                    "shell_power": 1.34,
                    "sample_distance_scale": 1.08,
                }
            elif visible_count <= 3:
                base = {
                    "voxel_um": 0.08,
                    "max_dim": 1152.0,
                    "max_voxels": 88_000_000.0,
                    "texture_strength": 0.22,
                    "shell_strength": 1.34,
                    "shell_frac": 0.058,
                    "core_weight": 0.030,
                    "density_gamma": 1.34,
                    "shell_power": 1.22,
                    "sample_distance_scale": 1.10,
                }
            else:
                base = {
                    "voxel_um": 0.11,
                    "max_dim": 832.0,
                    "max_voxels": 60_000_000.0,
                    "texture_strength": 0.20,
                    "shell_strength": 1.24,
                    "shell_frac": 0.068,
                    "core_weight": 0.036,
                    "density_gamma": 1.22,
                    "shell_power": 1.14,
                    "sample_distance_scale": 1.14,
                }

        if self.ng_voxel_um_override is not None:
            base["voxel_um"] = float(self.ng_voxel_um_override)
        if self.ng_max_dim_override is not None:
            base["max_dim"] = float(self.ng_max_dim_override)
        if self.ng_max_voxels_override is not None:
            base["max_voxels"] = float(self.ng_max_voxels_override)
        return base

    def _build_neuroglancer_volume(self, neuron_id: int) -> Tuple[pv.ImageData, float]:
        nid = int(neuron_id)
        ng = self._effective_neuroglancer_settings()
        cache_key = (
            self._table_signature(nid),
            round(float(ng["voxel_um"]), 4),
            int(ng["max_dim"]),
            round(float(ng["texture_strength"]), 4),
            round(float(ng["shell_strength"]), 4),
            round(float(ng["shell_frac"]), 4),
            round(float(ng.get("core_weight", 0.05)), 4),
            round(float(ng.get("density_gamma", 1.0)), 4),
            round(float(ng.get("shell_power", 1.0)), 4),
            int(ng["max_voxels"]),
        )
        cached = self._volume_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self.project.table(nid)
        id_to_row = {int(r["id"]): r for _, r in df.iterrows()}

        all_xyz: List[np.ndarray] = []
        all_r: List[float] = []
        segments: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for _, r in df.iterrows():
            parent = int(r["parent"])
            cxyz = np.asarray([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
            cr = max(0.10, float(r["r"]))
            all_xyz.append(cxyz)
            all_r.append(cr)
            if parent == -1 or parent not in id_to_row:
                continue
            prow = id_to_row[parent]
            pxyz = np.asarray([float(prow["x"]), float(prow["y"]), float(prow["z"])], dtype=float)
            pr = max(0.10, float(prow["r"]))
            segments.append((pxyz, cxyz, max(0.10, 0.5 * (pr + cr))))

        if not segments or not all_xyz:
            raise RuntimeError(f"Neuron {nid} has no drawable segments for neuroglancer mode")

        xyz = np.vstack(all_xyz)
        max_r = float(np.max(np.asarray(all_r, dtype=float))) if all_r else 1.0
        pad = max(6.0, 2.5 * max_r)
        mins = xyz.min(axis=0) - pad
        maxs = xyz.max(axis=0) + pad
        span = np.maximum(maxs - mins, 1.0)

        spacing = max(float(ng["voxel_um"]), float(np.max(span)) / float(max(32, int(ng["max_dim"]))))
        spacing = float(np.clip(spacing, 0.03, 3.0))
        dims = np.maximum(np.ceil(span / spacing).astype(int) + 1, 24)
        while int(np.prod(dims)) > int(ng["max_voxels"]):
            spacing *= 1.12
            dims = np.maximum(np.ceil(span / spacing).astype(int) + 1, 24)

        nx, ny, nz = [int(v) for v in dims.tolist()]
        xs = mins[0] + np.arange(nx, dtype=float) * spacing
        ys = mins[1] + np.arange(ny, dtype=float) * spacing
        zs = mins[2] + np.arange(nz, dtype=float) * spacing
        field = np.zeros((nx, ny, nz), dtype=np.float32)

        for p0, p1, radius in segments:
            vec = np.asarray(p1 - p0, dtype=float)
            seg_len = float(np.linalg.norm(vec))
            if not np.isfinite(seg_len) or seg_len <= 0.0:
                continue
            n_samples = max(2, int(np.ceil(seg_len / max(spacing * 0.32, 0.06))) + 1)
            ts = np.linspace(0.0, 1.0, n_samples, dtype=float)
            extent = float(radius + max(0.95 * spacing, radius * 0.24))
            core_sigma = max(spacing * 0.28, radius * 0.16)
            shell_sigma = max(spacing * 0.22, radius * float(ng["shell_frac"]))
            for t in ts:
                center = p0 + (vec * float(t))
                imin = max(0, int(np.floor((center[0] - extent - mins[0]) / spacing)))
                imax = min(nx - 1, int(np.ceil((center[0] + extent - mins[0]) / spacing)))
                jmin = max(0, int(np.floor((center[1] - extent - mins[1]) / spacing)))
                jmax = min(ny - 1, int(np.ceil((center[1] + extent - mins[1]) / spacing)))
                kmin = max(0, int(np.floor((center[2] - extent - mins[2]) / spacing)))
                kmax = min(nz - 1, int(np.ceil((center[2] + extent - mins[2]) / spacing)))
                if imin > imax or jmin > jmax or kmin > kmax:
                    continue

                xloc = xs[imin : imax + 1]
                yloc = ys[jmin : jmax + 1]
                zloc = zs[kmin : kmax + 1]
                dx2 = (xloc - center[0])[:, None, None] ** 2
                dy2 = (yloc - center[1])[None, :, None] ** 2
                dz2 = (zloc - center[2])[None, None, :] ** 2
                dist = np.sqrt(dx2 + dy2 + dz2).astype(np.float32, copy=False)

                core = np.exp(-0.5 * (dist / float(core_sigma)) ** 2).astype(np.float32, copy=False)
                shell = np.exp(-0.5 * ((dist - float(radius)) / float(shell_sigma)) ** 2).astype(np.float32, copy=False)
                shell = np.power(shell, float(ng.get("shell_power", 1.0))).astype(np.float32, copy=False)
                phase = (
                    (xloc[:, None, None] * 0.065)
                    + (yloc[None, :, None] * 0.083)
                    + (zloc[None, None, :] * 0.057)
                )
                texture = 1.0 + float(ng["texture_strength"]) * (
                    0.60 * np.sin(phase) + 0.40 * np.sin((phase * 1.7) + 1.1)
                )
                contrib = np.maximum(float(ng.get("core_weight", 0.05)) * core, float(ng["shell_strength"]) * shell)
                contrib = np.clip(contrib * texture.astype(np.float32), 0.0, 1.0)
                block = field[imin : imax + 1, jmin : jmax + 1, kmin : kmax + 1]
                np.add(block, (1.0 - block) * contrib.astype(np.float32), out=block)

        gamma = max(1.0, float(ng.get("density_gamma", 1.0)))
        if gamma > 1.001:
            field = np.power(np.clip(field, 0.0, 1.0), gamma).astype(np.float32, copy=False)
        density = np.asarray(np.clip(np.round(field * 255.0), 0.0, 255.0), dtype=np.uint8)
        image = pv.ImageData()
        image.dimensions = (nx, ny, nz)
        image.origin = (float(mins[0]), float(mins[1]), float(mins[2]))
        image.spacing = (float(spacing), float(spacing), float(spacing))
        image.point_data["density"] = density.ravel(order="F")
        image.set_active_scalars("density")
        self._last_ng_volume_stats[nid] = (float(spacing), (nx, ny, nz))

        self._volume_cache[cache_key] = (image, float(spacing))
        if len(self._volume_cache) > 24:
            self._volume_cache = dict(list(self._volume_cache.items())[-24:])
        return image, float(spacing)

    def _add_neuroglancer_volume_actor(self, neuron_id: int) -> bool:
        nid = int(neuron_id)
        try:
            ng = self._effective_neuroglancer_settings()
            volume, spacing = self._build_neuroglancer_volume(nid)
            actor = self.plotter.add_volume(
                volume,
                scalars="density",
                cmap=_volume_cmap_for_color(self.neuron_colors.get(nid, "#4c78a8")),
                opacity=[0.0, 0.0, 0.0, 0.0, 0.02, 0.10, 0.30, 0.72, 1.0],
                clim=[0, 255],
                name=f"morphology_{nid}",
                pickable=False,
                show_scalar_bar=False,
                blending="composite",
                opacity_unit_distance=max(0.55, float(spacing) * 0.72),
                shade=True,
                ambient=0.14,
                diffuse=0.86,
                specular=0.34,
                specular_power=18.0,
                mapper="smart",
            )
            try:
                actor.prop.SetInterpolationTypeToLinear()
            except Exception:
                try:
                    actor.prop.interpolation_type = "linear"
                except Exception:
                    pass
            try:
                mapper = actor.mapper
                mapper.SetAutoAdjustSampleDistances(False)
                mapper.SetSampleDistance(max(0.018, float(spacing) * float(ng.get("sample_distance_scale", 1.1))))
                mapper.SetUseJittering(True)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[render][warn] neuroglancer-like render failed for neuron {nid}; using tube fallback ({e})")
            return False

    def _overlay_text_color(self) -> str:
        return "#f5f7fb"

    def _node_xyz(self, neuron_id: int, node_id: int) -> np.ndarray:
        df = self.project.table(int(neuron_id))
        row = df.loc[df["id"] == int(node_id)]
        if row.empty:
            raise KeyError(f"node_id={int(node_id)} not found for neuron_id={int(neuron_id)}")
        rr = row.iloc[0]
        return np.asarray([float(rr["x"]), float(rr["y"]), float(rr["z"])], dtype=float)

    def _fallback_source_node_id(self, neuron_id: int) -> int:
        df = self.project.table(int(neuron_id))
        soma_rows = df.loc[df["type"] == 1]
        if not soma_rows.empty:
            return int(soma_rows.iloc[0]["id"])
        root_rows = df.loc[df["parent"] == -1]
        if not root_rows.empty:
            return int(root_rows.iloc[0]["id"])
        return int(df.iloc[0]["id"])

    def _primary_source_node_id(self, neuron_id: int) -> int:
        pol = self.project.ais_policy(int(neuron_id))
        if pol.primary_node_id is not None:
            return int(pol.primary_node_id)
        return self._fallback_source_node_id(int(neuron_id))

    def _connection_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in self.project.connections:
            out.append(
                {
                    "pre_neuron_id": int(c.pre_neuron_id),
                    "pre_node_id": int(c.pre_node_id),
                    "post_neuron_id": int(c.post_neuron_id),
                    "post_node_id": int(c.post_node_id),
                    "chemical_synapses": int(c.chemical_synapses),
                    "gap_junctions": int(c.gap_junctions),
                    "gap_mode": str(c.gap_mode),
                    "gap_direction": c.gap_direction,
                    "note": str(c.note),
                }
            )
        return out

    def _current_synapse_pair(self) -> Optional[FocusPair]:
        if not self.synapse_overlay_enabled:
            return None
        ds = self._synapse_overlay_dataset(quiet=True)
        pairs = list(ds.get("pair_order", []) or [])
        if not pairs:
            return None
        idx = int(self.synapse_pair_index) % len(pairs)
        self.synapse_pair_index = idx
        pair = pairs[idx]
        return int(pair[0]), int(pair[1])

    def _visible_neuron_ids(self) -> List[int]:
        ids = [int(x) for x in self.project.neuron_ids]
        if (not self.solo_mode) or (not ids):
            return ids
        idx = int(self.solo_neuron_index) % len(ids)
        self.solo_neuron_index = idx
        return [int(ids[idx])]

    def _visible_neuron_set(self) -> Set[int]:
        return set(int(x) for x in self._visible_neuron_ids())

    def _preferred_neuron_index(self) -> int:
        ids = [int(x) for x in self.project.neuron_ids]
        if not ids:
            return -1
        idx_by_id = {int(nid): i for i, nid in enumerate(ids)}
        for key in reversed(self.selection_order):
            nid = int(key[0])
            if nid in idx_by_id:
                return int(idx_by_id[nid])
        return int(np.clip(self.solo_neuron_index, 0, len(ids) - 1))

    def _update_connection_actor(self) -> None:
        visible = self._visible_neuron_set()
        recs = [
            r for r in self._connection_records()
            if int(r["pre_neuron_id"]) in visible and int(r["post_neuron_id"]) in visible
        ]
        if not recs:
            try:
                self.plotter.remove_actor("connections", render=False)
            except Exception:
                pass
            return

        pts: List[np.ndarray] = []
        lines: List[int] = []
        weights: List[float] = []
        p_off = 0
        for r in recs:
            try:
                p0 = self._node_xyz(int(r["pre_neuron_id"]), int(r["pre_node_id"]))
                p1 = self._node_xyz(int(r["post_neuron_id"]), int(r["post_node_id"]))
            except Exception:
                continue
            pts.extend([p0, p1])
            lines.extend([2, p_off, p_off + 1])
            p_off += 2
            weights.append(0.7)

        if not pts:
            return

        mesh = pv.PolyData()
        mesh.points = np.vstack(pts)
        mesh.lines = np.asarray(lines, dtype=np.int64)
        mesh.cell_data["conn_weight"] = np.asarray(weights, dtype=float)
        self.plotter.add_mesh(
            mesh,
            name="connections",
            color="#666666",
            line_width=1.5,
            render_lines_as_tubes=True,
            opacity=0.35,
            pickable=False,
        )

    def _update_ais_actors(self) -> None:
        primary_pts: List[np.ndarray] = []
        extra_pts: List[np.ndarray] = []
        visible = self._visible_neuron_set()
        for nid in self.project.neuron_ids:
            if int(nid) not in visible:
                continue
            pol = self.project.ais_policy(int(nid))
            if pol.primary_node_id is not None:
                try:
                    primary_pts.append(self._node_xyz(int(nid), int(pol.primary_node_id)))
                except Exception:
                    pass
            for node_id in pol.extra_node_ids:
                try:
                    extra_pts.append(self._node_xyz(int(nid), int(node_id)))
                except Exception:
                    pass

        if primary_pts:
            self.plotter.add_points(
                np.vstack(primary_pts),
                name="ais_primary",
                color="#ffd23f",
                point_size=14,
                render_points_as_spheres=True,
                pickable=False,
            )
        if extra_pts:
            self.plotter.add_points(
                np.vstack(extra_pts),
                name="ais_extra",
                color="#ff8c42",
                point_size=10,
                render_points_as_spheres=True,
                pickable=False,
            )

    def _update_recording_actors(self) -> None:
        probe_pts: List[np.ndarray] = []
        visible = self._visible_neuron_set()
        for nid in self.project.neuron_ids:
            if int(nid) not in visible:
                continue
            pol = self.project.recording_policies.get(int(nid), {})
            for node_id in sorted(int(x) for x in pol.get("probe", set())):
                try:
                    probe_pts.append(self._node_xyz(int(nid), int(node_id)))
                except Exception:
                    pass
        if not probe_pts:
            try:
                self.plotter.remove_actor("recording_probes", render=False)
            except Exception:
                pass
            return
        self.plotter.add_points(
            np.vstack(probe_pts),
            name="recording_probes",
            color="#00b894",
            point_size=12,
            render_points_as_spheres=True,
            pickable=False,
        )

    def _synapse_overlay_dataset(self, *, quiet: bool = False) -> Dict[str, Any]:
        if self.flow_run_dir is None:
            raise ValueError("No flow directory configured. Relaunch with --flow-run-dir.")
        cache_key = str(self.flow_run_dir)
        cached = self._synapse_overlay_cache.get(cache_key)
        if isinstance(cached, dict) and cached:
            return cached

        cfg_path = Path(self.flow_run_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in flow run dir: {self.flow_run_dir}")
        cfg = json.loads(cfg_path.read_text())
        edges_csv = cfg.get("edges_csv") or cfg.get("edges_path")
        if not edges_csv:
            raise ValueError("Flow run config does not define edges_csv/edges_path")
        edges_path = Path(str(edges_csv)).expanduser().resolve()
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges table not found: {edges_path}")
        swc_root = Path(
            str(cfg.get("swc_dir") or cfg.get("morph_swc_dir") or self.project.swc_root)
        ).expanduser().resolve()
        df = pd.read_csv(edges_path)
        if df.empty:
            raise ValueError(f"No edge rows found in {edges_path}")
        loaded = set(int(x) for x in self.project.neuron_ids)
        if "pre_id" not in df.columns or "post_id" not in df.columns:
            raise ValueError(f"Edges table missing pre_id/post_id columns: {edges_path}")
        df = df.loc[df["pre_id"].isin(list(loaded)) & df["post_id"].isin(list(loaded))].copy()
        if df.empty:
            raise ValueError("No edge rows between the currently loaded neurons")

        pair_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for _, row in df.iterrows():
            try:
                pre_id = int(row["pre_id"])
                post_id = int(row["post_id"])
            except Exception:
                continue
            pair = (int(pre_id), int(post_id))
            rec = pair_data.setdefault(
                pair,
                {
                    "pair": pair,
                    "rows": 0,
                    "pre_points": [],
                    "post_points": [],
                },
            )
            rec["rows"] += 1
            for side in ("pre", "post"):
                xyz = self._resolve_edge_synapse_xyz(row, side=side, swc_root=swc_root)
                if xyz is None:
                    continue
                rec[f"{side}_points"].append(np.asarray(xyz, dtype=float))

        pair_order: List[FocusPair] = []
        for pair in sorted(pair_data.keys()):
            rec = pair_data[pair]
            for key in ("pre_points", "post_points"):
                pts = rec[key]
                if pts:
                    arr = np.vstack(pts)
                    keep = np.unique(np.round(arr, 4), axis=0)
                    rec[key] = [np.asarray(v, dtype=float) for v in keep]
                else:
                    rec[key] = []
            pair_order.append((int(pair[0]), int(pair[1])))
        if not pair_order:
            raise ValueError("No synapse coordinates could be resolved for the loaded neuron pairs")

        ds = {
            "cfg": cfg,
            "edges_csv": str(edges_path),
            "swc_root": str(swc_root),
            "pair_order": pair_order,
            "pair_data": pair_data,
        }
        self._synapse_overlay_cache[cache_key] = ds
        if not quiet:
            print(f"[synapse overlay] loaded {len(pair_order)} pairs from {edges_path}")
        return ds

    def _load_syn_catalog_for_neuron(self, swc_root: Path, nid: int) -> Dict[str, np.ndarray]:
        key = (str(swc_root), int(nid))
        if key not in self._syn_catalog_cache:
            self._syn_catalog_cache[key] = _load_syn_catalog_local(swc_root, int(nid))
        return self._syn_catalog_cache[key]

    def _resolve_edge_synapse_xyz(
        self,
        row: Any,
        *,
        side: str,
        swc_root: Path,
    ) -> Optional[np.ndarray]:
        side = str(side).strip().lower()
        if side not in {"pre", "post"}:
            raise ValueError(f"Unsupported synapse side: {side}")
        xyz_sets = (
            (f"{side}_x", f"{side}_y", f"{side}_z"),
            (f"x_{side}", f"y_{side}", f"z_{side}"),
        )
        for xk, yk, zk in xyz_sets:
            if all(k in row and pd.notna(row[k]) for k in (xk, yk, zk)):
                try:
                    x = float(row[xk])
                    y = float(row[yk])
                    z = float(row[zk])
                    if max(abs(x), abs(y), abs(z)) > 200.0:
                        x *= 0.001
                        y *= 0.001
                        z *= 0.001
                    return np.asarray([x, y, z], dtype=float)
                except Exception:
                    pass
        try:
            nid = int(row[f"{side}_id"])
        except Exception:
            return None
        try:
            syn_idx = row.get(f"{side}_syn_index", None)
            jj = None if pd.isna(syn_idx) else int(float(syn_idx))
        except Exception:
            jj = None
        cat = self._load_syn_catalog_for_neuron(swc_root, int(nid))
        pts = np.asarray(cat.get(side, np.empty((0, 3), dtype=float)), dtype=float)
        if pts.size == 0:
            return None
        if jj is None or jj < 0 or jj >= len(pts):
            return None
        return np.asarray(pts[int(jj)], dtype=float)

    def _update_synapse_overlay_actor(self) -> None:
        names = ("synapse_pre_points", "synapse_post_points")
        if not self.synapse_overlay_enabled:
            for name in names:
                try:
                    self.plotter.remove_actor(name, render=False)
                except Exception:
                    pass
            return
        try:
            ds = self._synapse_overlay_dataset(quiet=True)
        except Exception as e:
            for name in names:
                try:
                    self.plotter.remove_actor(name, render=False)
                except Exception:
                    pass
            self.last_action = f"synapse overlay unavailable: {e}"
            return
        pair = self._current_synapse_pair()
        if pair is None:
            for name in names:
                try:
                    self.plotter.remove_actor(name, render=False)
                except Exception:
                    pass
            return
        rec = dict(ds.get("pair_data", {}).get(tuple(pair), {}))
        pre_pts = rec.get("pre_points", []) or []
        post_pts = rec.get("post_points", []) or []
        if pre_pts:
            self.plotter.add_points(
                np.vstack(pre_pts),
                name="synapse_pre_points",
                color=self.neuron_colors.get(int(pair[0]), "#1f77b4"),
                point_size=8,
                render_points_as_spheres=False,
                pickable=False,
                opacity=1.0,
            )
        else:
            try:
                self.plotter.remove_actor("synapse_pre_points", render=False)
            except Exception:
                pass
        if post_pts:
            self.plotter.add_points(
                np.vstack(post_pts),
                name="synapse_post_points",
                color=self.neuron_colors.get(int(pair[1]), "#d62728"),
                point_size=8,
                render_points_as_spheres=False,
                pickable=False,
                opacity=1.0,
            )
        else:
            try:
                self.plotter.remove_actor("synapse_post_points", render=False)
            except Exception:
                pass

    def _setup_scene(self) -> None:
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes(viewport=(0.88, 0.02, 0.995, 0.16))
        # Rectangle picking in the selector conflicts with the VTK slider
        # widget on this PyVista build, so keep line width fixed while editing.
        self.skeleton_width_slider = None

        self._rebuild_scene(first_time=True)
        self._focus_camera_on_neurons()
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        _loaded_cam = self._load_camera_preset(quiet=True)
        if _loaded_cam is not None:
            self.last_action = f"loaded_camera_preset={_loaded_cam.name}"

        self.plotter.enable_cell_picking(
            callback=self._on_pick_cells,
            through=True,
            # Use the app's own selected-segment actor for highlighting rather
            # than PyVista's frustum/box overlay.
            show=False,
            show_message=False,
            style="wireframe",
            color="magenta",
            line_width=3,
            start=False,
            show_frustum=False,
        )

        try:
            self.plotter.enable_point_picking(
                callback=self._on_pick_point,
                use_mesh=True,
                show_message=False,
                # Keep left-drag free for PyVista's built-in "r" rectangle
                # selection mode. Use right-click for draw-to-point instead.
                left_clicking=False,
                picker="hardware",
            )
        except Exception:
            # Fallback for older pyvista builds.
            try:
                self.plotter.enable_point_picking(
                    callback=self._on_pick_point,
                    use_mesh=True,
                    show_message=False,
                    left_clicking=False,
                )
            except Exception:
                pass

        self.plotter.add_key_event("c", self._clear_selection)
        self.plotter.add_key_event("p", self._print_selection)
        self.plotter.add_key_event("t", self._thin_selected)
        self.plotter.add_key_event("y", self._thicken_selected)
        self.plotter.add_key_event("g", self._grow_selected)
        self.plotter.add_key_event("d", self._arm_draw_to_click)
        self.plotter.add_key_event("a", self._reparent_last_pair)
        self.plotter.add_key_event("x", self._detach_selected)
        self.plotter.add_key_event("j", self._add_connection_from_last_pair)
        self.plotter.add_key_event("f", self._mark_selected_passive)
        self.plotter.add_key_event("q", self._mark_selected_active)
        self.plotter.add_key_event("e", self._mark_selected_recording)
        self.plotter.add_key_event("5", self._mark_selected_ais)

        self.plotter.add_key_event("i", lambda: self._translate_selected(dx=self.move_step_um, dy=0.0, dz=0.0))
        self.plotter.add_key_event("k", lambda: self._translate_selected(dx=-self.move_step_um, dy=0.0, dz=0.0))
        self.plotter.add_key_event("l", lambda: self._translate_selected(dx=0.0, dy=self.move_step_um, dz=0.0))
        self.plotter.add_key_event("o", lambda: self._translate_selected(dx=0.0, dy=-self.move_step_um, dz=0.0))
        self.plotter.add_key_event("u", lambda: self._translate_selected(dx=0.0, dy=0.0, dz=self.move_step_um))
        self.plotter.add_key_event("n", lambda: self._translate_selected(dx=0.0, dy=0.0, dz=-self.move_step_um))

        self.plotter.add_key_event("z", self._undo_last)
        self.plotter.add_key_event("v", self._print_validation)
        self.plotter.add_key_event("s", self._save_bundle)
        self.plotter.add_key_event("w", self._toggle_render_mode)
        self.plotter.add_key_event("b", self._split_selected_edges)
        self.plotter.add_key_event("h", self._print_help)
        self.plotter.add_key_event("m", self._save_jpeg_screenshot)
        self.plotter.add_key_event("0", self._export_flow_movie)
        self.plotter.add_key_event("1", self._toggle_neuroglancer_mode)
        self.plotter.add_key_event("3", self._save_synapse_overlay_screenshot)
        self.plotter.add_key_event("7", lambda: self._cycle_synapse_pair(-1))
        self.plotter.add_key_event("8", lambda: self._cycle_synapse_pair(1))
        self.plotter.add_key_event("9", self._refocus_camera)
        self.plotter.add_key_event("[", lambda: self._cycle_visible_neuron(-1))
        self.plotter.add_key_event("]", lambda: self._cycle_visible_neuron(1))
        self.plotter.add_key_event("4", lambda: self._cycle_visible_neuron(-1))
        self.plotter.add_key_event("6", lambda: self._cycle_visible_neuron(1))
        self.plotter.add_key_event("-", self._toggle_solo_mode)
        self.plotter.add_key_event("2", self._toggle_solo_mode)

        self._update_overlay()

    def _rebuild_scene(self, *, first_time: bool = False) -> None:
        cam = None if first_time else self.plotter.camera_position
        self.mesh, self.segment_df, self.root_points = self._build_mesh()
        self._distance_cache.clear()

        try:
            self.plotter.remove_actor("morphology", render=False)
        except Exception:
            pass
        for nid in self.project.neuron_ids:
            try:
                self.plotter.remove_actor(f"morphology_{int(nid)}", render=False)
            except Exception:
                pass
        try:
            self.plotter.remove_actor("roots", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("selected", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("connections", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("ais_primary", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("ais_extra", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("recording_probes", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("synapse_pre_points", render=False)
        except Exception:
            pass
        try:
            self.plotter.remove_actor("synapse_post_points", render=False)
        except Exception:
            pass

        self._apply_visual_style()
        visible = self._visible_neuron_set()
        vis_seg_idxs = self.segment_df.loc[
            self.segment_df["neuron_id"].isin(list(visible)), "segment_idx"
        ].to_numpy(dtype=np.int64)
        pick_mesh = self.mesh if len(visible) == len(self.project.neuron_ids) else self.mesh.extract_cells(vis_seg_idxs)

        # Hidden-ish pick mesh (line segments) keeps robust cell-picking behavior.
        self.plotter.add_mesh(
            pick_mesh,
            color="#ffffff",
            line_width=1,
            render_lines_as_tubes=False,
            name="morphology",
            pickable=True,
            show_scalar_bar=False,
            opacity=0.03,
        )

        # Render morphology with either 3D tubes, colored skeleton lines, or a
        # synthetic volume pass that approximates Neuroglancer's textured look.
        for nid in self._visible_neuron_ids():
            idxs = self.segment_df.loc[self.segment_df["neuron_id"] == int(nid), "segment_idx"].to_numpy(dtype=np.int64)
            if idxs.size == 0:
                continue
            sub_raw = self.mesh.extract_cells(idxs)
            sub = self._display_geom_from_cells(sub_raw)
            if int(sub.n_cells) <= 0:
                continue
            if self.render_mode == "skeleton":
                self.plotter.add_mesh(
                    sub,
                    color=self.neuron_colors.get(int(nid), "#4c78a8"),
                    name=f"morphology_{int(nid)}",
                    pickable=False,
                    render_lines_as_tubes=False,
                    line_width=float(self.skeleton_line_width),
                    opacity=0.98,
                )
            elif self.render_mode == "neuroglancer":
                if self._add_neuroglancer_volume_actor(int(nid)):
                    continue
                try:
                    geom = sub.tube(
                        scalars="radius_um",
                        absolute=True,
                        radius=None,
                        radius_factor=1.0,
                        n_sides=18,
                        capping=True,
                        preference="cell",
                    )
                except Exception:
                    geom = sub
                self.plotter.add_mesh(
                    geom,
                    color=self.neuron_colors.get(int(nid), "#4c78a8"),
                    name=f"morphology_{int(nid)}",
                    pickable=False,
                    smooth_shading=True,
                    ambient=0.2,
                    diffuse=0.75,
                    specular=0.15,
                    specular_power=10.0,
                    opacity=0.98,
                )
            else:
                try:
                    geom = sub.tube(
                        scalars="radius_um",
                        absolute=True,
                        radius=None,
                        radius_factor=1.0,
                        n_sides=18,
                        capping=True,
                        preference="cell",
                    )
                except Exception:
                    geom = sub
                self.plotter.add_mesh(
                    geom,
                    color=self.neuron_colors.get(int(nid), "#4c78a8"),
                    name=f"morphology_{int(nid)}",
                    pickable=False,
                    smooth_shading=True,
                    ambient=0.2,
                    diffuse=0.75,
                    specular=0.15,
                    specular_power=10.0,
                    opacity=0.98,
                )

        if self.root_points is not None and int(self.root_points.n_points) > 0:
            self.plotter.add_mesh(
                self.root_points,
                color="#ffd166",
                point_size=8,
                render_points_as_spheres=True,
                name="roots",
                pickable=False,
            )

        self._sanitize_selection()
        self._update_connection_actor()
        self._update_ais_actors()
        self._update_recording_actors()
        self._update_selected_actor()
        self._update_synapse_overlay_actor()

        if cam is not None:
            self.plotter.camera_position = cam

    def _sanitize_selection(self) -> None:
        valid = set((int(r["neuron_id"]), int(r["child_node_id"])) for _, r in self.segment_df.iterrows())
        if self.solo_mode:
            visible = self._visible_neuron_set()
            valid = set(k for k in valid if int(k[0]) in visible)
        self.selected_keys = set(k for k in self.selected_keys if k in valid)
        self.selection_order = [k for k in self.selection_order if k in valid]
        if self.pending_draw_anchor not in valid:
            self.pending_draw_anchor = None

    def _update_selected_actor(self) -> None:
        if not self.selected_keys:
            try:
                self.plotter.remove_actor("selected", render=False)
            except Exception:
                pass
            return

        mask = self.segment_df["key"].isin(list(self.selected_keys)).to_numpy(dtype=bool)
        seg_idxs = self.segment_df.loc[mask, "segment_idx"].to_numpy(dtype=np.int64)
        if seg_idxs.size == 0:
            return

        picked_raw = self.mesh.extract_cells(seg_idxs)
        picked = self._display_geom_from_cells(picked_raw)
        if int(picked.n_cells) <= 0:
            return
        if self.render_mode == "skeleton":
            self.plotter.add_mesh(
                picked,
                color=self.selected_color,
                pickable=False,
                name="selected",
                render_lines_as_tubes=False,
                line_width=max(3.0, float(self.skeleton_line_width) + 3.0),
                opacity=1.0,
            )
        else:
            radii = np.asarray(picked.cell_data.get("radius_um", np.array([], dtype=float)), dtype=float)
            med_r = float(np.nanmedian(radii)) if radii.size else 0.25
            tube_r = max(0.25, min(4.0, med_r * 2.3))
            try:
                picked_geom = picked.tube(radius=tube_r, n_sides=20, capping=True)
            except Exception:
                picked_geom = picked
            self.plotter.add_mesh(
                picked_geom,
                color=self.selected_color,
                pickable=False,
                name="selected",
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.2,
                opacity=1.0,
            )

    def _last_two_selected(self) -> Optional[Tuple[SelectionKey, SelectionKey]]:
        uniq = []
        seen = set()
        for k in self.selection_order:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        if len(uniq) < 2:
            return None
        return uniq[-2], uniq[-1]

    def _selected_by_neuron(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for nid, child in sorted(self.selected_keys):
            out.setdefault(int(nid), []).append(int(child))
        for nid in list(out.keys()):
            out[nid] = sorted(set(out[nid]))
        return out

    def _push_undo(self) -> None:
        snapshot = {
            "tables": {int(nid): df.copy(deep=True) for nid, df in self.project.tables.items()},
            "operations": list(self.project.operations),
            "connections": list(self.project.connections),
            "biophys_policies": {
                int(nid): {
                    "passive": set(int(x) for x in pol.get("passive", set())),
                    "active": set(int(x) for x in pol.get("active", set())),
                }
                for nid, pol in self.project.biophys_policies.items()
            },
            "ais_policies": {
                int(nid): {
                    "primary_node_id": pol.get("primary_node_id"),
                    "primary_xloc": float(pol.get("primary_xloc", 0.5)),
                    "extra_node_ids": set(int(x) for x in pol.get("extra_node_ids", set())),
                }
                for nid, pol in self.project.ais_policies.items()
            },
            "recording_policies": {
                int(nid): {
                    "probe": set(int(x) for x in pol.get("probe", set())),
                }
                for nid, pol in self.project.recording_policies.items()
            },
            "selected_keys": set(self.selected_keys),
            "selection_order": list(self.selection_order),
            "last_action": str(self.last_action),
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 25:
            self.undo_stack = self.undo_stack[-25:]

    def _undo_last(self) -> None:
        if not self.undo_stack:
            print("[undo] no snapshot available")
            return
        snap = self.undo_stack.pop()
        self.project.tables = {int(nid): df.copy(deep=True) for nid, df in snap["tables"].items()}
        self.project.operations = list(snap["operations"])
        self.project.connections = list(snap["connections"])
        self.project.biophys_policies = {
            int(nid): {
                "passive": set(int(x) for x in pol.get("passive", set())),
                "active": set(int(x) for x in pol.get("active", set())),
            }
            for nid, pol in snap.get("biophys_policies", {}).items()
        }
        self.project.ais_policies = {
            int(nid): {
                "primary_node_id": (
                    None if pol.get("primary_node_id") is None else int(pol.get("primary_node_id"))
                ),
                "primary_xloc": float(pol.get("primary_xloc", 0.5)),
                "extra_node_ids": set(int(x) for x in pol.get("extra_node_ids", set())),
            }
            for nid, pol in snap.get("ais_policies", {}).items()
        }
        self.project.recording_policies = {
            int(nid): {
                "probe": set(int(x) for x in pol.get("probe", set())),
            }
            for nid, pol in snap.get("recording_policies", {}).items()
        }
        self.selected_keys = set(snap["selected_keys"])
        self.selection_order = list(snap["selection_order"])
        self.last_action = str(snap["last_action"])
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print("[undo] restored previous mutation state")

    def _update_overlay(self) -> None:
        nsel = len(self.selected_keys)
        ndraw = "none" if self.pending_draw_anchor is None else f"{self.pending_draw_anchor}"
        visible_ids = self._visible_neuron_ids()
        if self.solo_mode and visible_ids:
            visible_label = (
                f"solo neuron {visible_ids[0]} "
                f"({int(self.solo_neuron_index) + 1}/{len(self.project.neuron_ids)})"
            )
        else:
            visible_label = f"all neurons ({len(visible_ids)})"
        passive_total = 0
        active_total = 0
        ais_primary_total = 0
        ais_extra_total = 0
        probe_total = 0
        synapse_label = "off"
        for pol in self.project.biophys_policies.values():
            passive_total += len(pol.get("passive", set()))
            active_total += len(pol.get("active", set()))
        for pol in self.project.ais_policies.values():
            if pol.get("primary_node_id") is not None:
                ais_primary_total += 1
            ais_extra_total += len(pol.get("extra_node_ids", set()))
        for pol in self.project.recording_policies.values():
            probe_total += len(pol.get("probe", set()))
        if self.synapse_overlay_enabled:
            try:
                _pair = self._current_synapse_pair()
                _ds = self._synapse_overlay_dataset(quiet=True)
                _rec = dict(_ds.get("pair_data", {}).get(tuple(_pair), {})) if _pair is not None else {}
                synapse_label = (
                    f"on pair={_pair[0]}->{_pair[1]} rows={int(_rec.get('rows', 0))} "
                    f"pre_pts={len(_rec.get('pre_points', []) or [])} "
                    f"post_pts={len(_rec.get('post_points', []) or [])}"
                ) if _pair is not None else "on (no pair)"
            except Exception as _e:
                synapse_label = f"error: {_e}"
        ng_render_label = ""
        if self.render_mode == "neuroglancer":
            ng = self._effective_neuroglancer_settings()
            actual_ng_label = ""
            visible_ids = self._visible_neuron_ids()
            if len(visible_ids) == 1:
                actual_stats = self._last_ng_volume_stats.get(int(visible_ids[0]))
                if actual_stats is not None:
                    actual_spacing, actual_dims = actual_stats
                    actual_ng_label = (
                        f"Actual voxel: {float(actual_spacing):.3f} um | "
                        f"dims={int(actual_dims[0])}x{int(actual_dims[1])}x{int(actual_dims[2])}\n\n"
                    )
            ng_render_label = (
                f"Neuroglancer quality: {self.ng_quality_mode} | "
                f"target voxel={float(ng['voxel_um']):.2f} um | "
                f"max_dim={int(ng['max_dim'])} | "
                f"max_voxels={int(ng['max_voxels']):,}\n\n"
                f"{actual_ng_label}"
            )
        status = (
            "Morphology Mutation (multi-SWC)\n\n"
            f"Loaded neuron IDs: {self.project.neuron_ids}\n\n"
            f"Visible/editable: {visible_label}\n\n"
            f"Selected segments: {nsel}\n\n"
            f"Connection specs: {len(self.project.connections)}\n\n"
            f"Biophys policy nodes: passive={passive_total}, active={active_total}\n\n"
            f"AIS regions: primary={ais_primary_total}, extra={ais_extra_total}\n\n"
            f"Recording probe nodes: {probe_total}\n\n"
            f"Synapse overlay: {synapse_label}\n\n"
            f"Render mode: {self.render_mode} (line width {self.skeleton_line_width:.1f}; slider disabled for stable selection)\n\n"
            f"{ng_render_label}"
            f"Visual style: {self.visual_style}\n\n"
            f"Flow run dir: {self.flow_run_dir if self.flow_run_dir is not None else 'none'}\n\n"
            f"Pending draw anchor: {ndraw}\n\n"
            f"Last action: {self.last_action}"
        )

        keys = (
            "Keys\n\n"
            "r: drag-box select (additive)\n"
            "c: clear selection\n"
            "p: print selection\n"
            "t/y: thin/thicken selected\n"
            "g: grow branch along tangent\n"
            "d: draw branch to clicked 3D point (right click)\n"
            "b: split selected edges\n"
            "a: reparent using last two selected\n"
            "x: detach selected nodes (split)\n"
            "i/k: +X / -X translate\n"
            "l/o: +Y / -Y translate\n"
            "u/n: +Z / -Z translate\n"
            "j: add pre->post connection (last two selected)\n"
            "f/q: mark selected passive/active\n"
            "e: mark selected recording probe sites\n"
            "5: assign selected AIS regions (last selected per neuron = primary)\n"
            "3: toggle synapse-dot overlay for current pair\n"
            "7/8: previous/next synapse pair\n"
            "w: toggle skeleton <-> tube | 1: toggle neuroglancer-like volume\n"
            "[: previous neuron | ]: next neuron | 4/6: previous/next neuron | -/2: toggle solo/all\n"
            "z: undo\n"
            "v: validate\n"
            "s: save bundle + camera preset\n"
            "m: save cropped photo export using current background (png+jpg+pdf)\n"
            "0: export flow movie from configured run dir\n"
            "9: refocus camera on loaded neurons\n"
            "h: help"
        )

        self.plotter.add_text(
            status,
            position=(0.02, 0.93),
            viewport=True,
            name="status",
            font_size=11,
            color=self._overlay_text_color(),
        )
        self.plotter.add_text(
            keys,
            position=(0.02, 0.03),
            viewport=True,
            name="keys",
            font_size=10,
            color=self._overlay_text_color(),
        )

    def _on_skeleton_width_slider(self, value: float) -> None:
        self.skeleton_line_width = max(1.0, float(value))
        if self.render_mode == "skeleton":
            self._rebuild_scene()
            self._update_overlay()
            self.plotter.render()
        else:
            # Keep slider state updated in overlay even when in tube mode.
            self._update_overlay()
            self.plotter.render()

    def _toggle_render_mode(self) -> None:
        if self.render_mode == "neuroglancer":
            self.render_mode = self._standard_render_mode
        else:
            self.render_mode = "skeleton" if self.render_mode == "tube" else "tube"
            self._standard_render_mode = self.render_mode
        self.last_action = f"render_mode={self.render_mode}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[view] render mode -> {self.render_mode}")

    def _toggle_neuroglancer_mode(self) -> None:
        if self.render_mode == "neuroglancer":
            self.render_mode = self._standard_render_mode
        else:
            if self.render_mode in {"tube", "skeleton"}:
                self._standard_render_mode = self.render_mode
            self.render_mode = "neuroglancer"
        self.last_action = f"render_mode={self.render_mode}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[view] render mode -> {self.render_mode}")

    def _save_synapse_overlay_screenshot(self) -> None:
        try:
            ds = self._synapse_overlay_dataset(quiet=False)
        except Exception as e:
            self.last_action = f"synapse screenshot unavailable: {e}"
            self._update_overlay()
            self.plotter.render()
            print(f"[synapse screenshot] unavailable: {e}")
            return

        pair_data = dict(ds.get("pair_data", {}))
        pre_pts = []
        post_pts = []
        for rec in pair_data.values():
            pre_pts.extend(list(rec.get("pre_points", []) or []))
            post_pts.extend(list(rec.get("post_points", []) or []))
        if not pre_pts and not post_pts:
            print("[synapse screenshot] no resolved synapse points to draw")
            return

        out_dir = self.output_root
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_png = out_dir / f"morphology_mutation_synapses_{stamp}.png"
        out_pdf = out_dir / f"morphology_mutation_synapses_{stamp}.pdf"

        win_w, win_h = [int(v) for v in self.plotter.window_size]
        tgt_w = max(2200, int(round(win_w * self.screenshot_scale)))
        tgt_h = max(1400, int(round(win_h * self.screenshot_scale)))

        cam_before = self.plotter.camera_position
        bg_before = None
        try:
            try:
                bg_before = tuple(float(x) for x in self.plotter.renderer.GetBackground())
            except Exception:
                bg_before = None
            for actor_name in ("status", "keys"):
                try:
                    self.plotter.remove_actor(actor_name, render=False)
                except Exception:
                    pass
            try:
                self.plotter.hide_axes()
            except Exception:
                pass
            try:
                self.plotter.set_background("white")
            except Exception:
                pass

            pre_name = "synapse_pre_capture"
            post_name = "synapse_post_capture"
            if pre_pts:
                self.plotter.add_points(
                    np.vstack(pre_pts),
                    name=pre_name,
                    color="#00e5ff",
                    point_size=22,
                    render_points_as_spheres=False,
                    pickable=False,
                    opacity=1.0,
                )
            if post_pts:
                self.plotter.add_points(
                    np.vstack(post_pts),
                    name=post_name,
                    color="#ffb000",
                    point_size=22,
                    render_points_as_spheres=False,
                    pickable=False,
                    opacity=1.0,
                )
            self.plotter.render()
            img = self.plotter.screenshot(
                return_img=True,
                window_size=(tgt_w, tgt_h),
                transparent_background=False,
            )
            try:
                from PIL import Image
                pil = Image.fromarray(np.asarray(img))
                pil.save(out_png, format="PNG", optimize=True, dpi=(self.screenshot_dpi, self.screenshot_dpi))
                try:
                    import matplotlib.pyplot as _plt
                    h, w = np.asarray(img).shape[:2]
                    fig = _plt.figure(figsize=(float(w) / float(self.screenshot_dpi), float(h) / float(self.screenshot_dpi)), dpi=float(self.screenshot_dpi), frameon=False)
                    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    ax.imshow(np.asarray(img), interpolation="nearest")
                    fig.savefig(out_pdf, dpi=float(self.screenshot_dpi), transparent=False, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.0)
                    _plt.close(fig)
                except Exception:
                    pass
            except Exception:
                self.plotter.screenshot(str(out_png), window_size=(tgt_w, tgt_h), transparent_background=False)
        finally:
            for actor_name in ("synapse_pre_capture", "synapse_post_capture"):
                try:
                    self.plotter.remove_actor(actor_name, render=False)
                except Exception:
                    pass
            try:
                self.plotter.camera_position = cam_before
            except Exception:
                pass
            try:
                if bg_before is not None:
                    self.plotter.set_background(bg_before)
            except Exception:
                pass
            try:
                self.plotter.show_axes()
            except Exception:
                pass
            self._update_overlay()
            self.plotter.render()

        self.last_action = f"saved synapse screenshot: {out_png.name}"
        self._update_overlay()
        self.plotter.render()
        print(f"[synapse screenshot] saved {out_png}")
        if out_pdf.exists():
            print(f"[synapse screenshot] saved {out_pdf}")

    def _toggle_synapse_overlay(self) -> None:
        if not self.synapse_overlay_enabled:
            try:
                ds = self._synapse_overlay_dataset(quiet=False)
                pairs = list(ds.get("pair_order", []) or [])
                if not pairs:
                    raise ValueError("No synapse pairs available in the current run")
                if self.flow_focus_pair is not None and tuple(int(x) for x in self.flow_focus_pair) in pairs:
                    self.synapse_pair_index = pairs.index(tuple(int(x) for x in self.flow_focus_pair))
                else:
                    self.synapse_pair_index = int(np.clip(self.synapse_pair_index, 0, len(pairs) - 1))
                self.synapse_overlay_enabled = True
            except Exception as e:
                self.last_action = f"synapse overlay unavailable: {e}"
                self._update_overlay()
                self.plotter.render()
                print(f"[synapse overlay] unavailable: {e}")
                return
        else:
            self.synapse_overlay_enabled = False
        pair = self._current_synapse_pair()
        self.last_action = (
            f"synapse_overlay={'on' if self.synapse_overlay_enabled else 'off'}"
            + (f" pair={pair[0]}->{pair[1]}" if self.synapse_overlay_enabled and pair is not None else "")
        )
        self._update_synapse_overlay_actor()
        self._update_overlay()
        self.plotter.render()
        if self.synapse_overlay_enabled and pair is not None:
            print(f"[synapse overlay] showing pair {pair[0]} -> {pair[1]}")
        else:
            print("[synapse overlay] hidden")

    def _cycle_synapse_pair(self, step: int) -> None:
        try:
            ds = self._synapse_overlay_dataset(quiet=True)
        except Exception as e:
            print(f"[synapse overlay] unavailable: {e}")
            return
        pairs = list(ds.get("pair_order", []) or [])
        if not pairs:
            print("[synapse overlay] no synapse pairs to cycle")
            return
        if not self.synapse_overlay_enabled:
            self.synapse_overlay_enabled = True
        self.synapse_pair_index = (int(self.synapse_pair_index) + int(step)) % len(pairs)
        pair = self._current_synapse_pair()
        self.last_action = f"synapse_pair={pair[0]}->{pair[1]}" if pair is not None else "synapse_pair=none"
        self._update_synapse_overlay_actor()
        self._update_overlay()
        self.plotter.render()
        if pair is not None:
            print(f"[synapse overlay] pair -> {pair[0]} -> {pair[1]}")

    def _refocus_camera(self) -> None:
        self._focus_camera_on_neurons()
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        self.last_action = "camera_refocus"
        self._update_overlay()
        self.plotter.render()
        print("[view] camera refocused on loaded neurons")

    def _toggle_solo_mode(self) -> None:
        ids = [int(x) for x in self.project.neuron_ids]
        if not ids:
            print("[view] no loaded neurons to isolate")
            return
        if not self.solo_mode:
            self.solo_neuron_index = self._preferred_neuron_index()
            self.solo_mode = True
        else:
            self.solo_mode = False
        label = self._visible_neuron_ids()
        self.last_action = (
            f"view_mode={'solo' if self.solo_mode else 'all'}"
            + (f" neuron={label[0]}" if self.solo_mode and label else "")
        )
        self._rebuild_scene()
        self._focus_camera_on_neurons()
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        self._update_overlay()
        self.plotter.render()
        if self.solo_mode and label:
            print(f"[view] solo neuron mode -> {label[0]}")
        else:
            print("[view] showing all neurons")

    def _cycle_visible_neuron(self, step: int) -> None:
        ids = [int(x) for x in self.project.neuron_ids]
        if not ids:
            print("[view] no loaded neurons to cycle")
            return
        if not self.solo_mode:
            self.solo_mode = True
            self.solo_neuron_index = self._preferred_neuron_index()
        self.solo_neuron_index = (int(self.solo_neuron_index) + int(step)) % len(ids)
        visible = self._visible_neuron_ids()
        self.last_action = f"cycle_visible_neuron neuron={visible[0]}"
        self._rebuild_scene()
        self._focus_camera_on_neurons()
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        self._update_overlay()
        self.plotter.render()
        print(f"[view] active solo neuron -> {visible[0]}")

    def _toggle_visual_style(self) -> None:
        self.visual_style = "classic"
        self.last_action = "visual_style=classic"
        self._update_overlay()
        self.plotter.render()
        print("[view] classic visual style is always active")

    def _on_pick_cells(self, picked: Any) -> None:
        seg_idxs = _extract_selected_segment_indices(picked)
        if not seg_idxs:
            return

        sub = self.segment_df[self.segment_df["segment_idx"].isin(sorted(seg_idxs))]
        if sub.empty:
            return

        for _, r in sub.iterrows():
            key: SelectionKey = (int(r["neuron_id"]), int(r["child_node_id"]))
            if key not in self.selected_keys:
                self.selected_keys.add(key)
                self.selection_order.append(key)

        self._update_selected_actor()
        self._update_overlay()
        self.plotter.render()

    def _on_pick_point(self, point: Any, *args: Any) -> None:
        if self.pending_draw_anchor is None:
            return
        xyz = _as_xyz(point)
        if xyz is None:
            return

        anchor = self.pending_draw_anchor
        self.pending_draw_anchor = None

        self._push_undo()
        try:
            new_ids = self.project.apply_grow_branch_to_point(
                anchor[0],
                parent_node_id=anchor[1],
                target_xyz_um=[float(x) for x in xyz],
                segments=self.grow_segments,
                radius_scale=self.grow_radius_scale,
            )
        except Exception as e:
            print(f"[draw] failed: {e}")
            return

        if new_ids:
            new_key = (int(anchor[0]), int(new_ids[-1]))
            self.selected_keys.add(new_key)
            self.selection_order.append(new_key)

        self.last_action = (
            f"grow_branch_to_point anchor={anchor} target=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"
        )
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[draw] added {len(new_ids)} node(s) from {anchor} to clicked point")

    def _clear_selection(self) -> None:
        self.selected_keys.clear()
        self.selection_order.clear()
        self.pending_draw_anchor = None
        self._update_selected_actor()
        self._update_overlay()
        self.plotter.render()

    def _print_selection(self) -> None:
        if not self.selected_keys:
            print("[select] no segments selected")
            return
        rows = self.segment_df[self.segment_df["key"].isin(list(self.selected_keys))].copy()
        cols = ["neuron_id", "child_node_id", "parent_node_id", "swc_type", "radius_um"]
        print(rows[cols].sort_values(["neuron_id", "child_node_id"]).to_string(index=False))

    def _thin_selected(self) -> None:
        self._scale_selected(self.thin_factor, "thin")

    def _thicken_selected(self) -> None:
        self._scale_selected(self.thick_factor, "thicken")

    def _scale_selected(self, factor: float, label: str) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print(f"[{label}] no selected segments")
            return

        self._push_undo()
        changed = 0
        for nid, node_ids in by_n.items():
            out = self.project.apply_scale_radii(
                nid,
                node_ids,
                factor=float(factor),
                include_subtree=self.include_subtree_radius,
            )
            changed += len(out)

        self.last_action = f"{label} factor={factor:g} changed_nodes={changed}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[{label}] changed {changed} node radii")

    def _grow_selected(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[grow] no selected segments")
            return

        self._push_undo()
        new_keys: List[SelectionKey] = []
        total = 0
        for nid, node_ids in by_n.items():
            for node_id in node_ids:
                new_ids = self.project.apply_grow_branch_along_tangent(
                    nid,
                    parent_node_id=int(node_id),
                    length_um=self.grow_length_um,
                    segments=self.grow_segments,
                    radius_scale=self.grow_radius_scale,
                )
                total += len(new_ids)
                if new_ids:
                    new_keys.append((int(nid), int(new_ids[-1])))

        for k in new_keys:
            self.selected_keys.add(k)
            self.selection_order.append(k)

        self.last_action = f"grow_tangent added_nodes={total}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[grow] added {total} new node(s)")

    def _split_selected_edges(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[split] no selected segments")
            return

        self._push_undo()
        inserted_keys: List[SelectionKey] = []
        total = 0
        for nid, child_ids in by_n.items():
            inserted = self.project.apply_split_edges(nid, child_ids, frac=0.5)
            total += len(inserted)
            for iid in inserted:
                inserted_keys.append((int(nid), int(iid)))

        for k in inserted_keys:
            self.selected_keys.add(k)
            self.selection_order.append(k)

        self.last_action = f"split_edges inserted_nodes={total}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[split] inserted {total} node(s)")

    def _translate_selected(self, *, dx: float, dy: float, dz: float) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[move] no selected segments")
            return

        self._push_undo()
        changed = 0
        for nid, node_ids in by_n.items():
            out = self.project.apply_translate(
                nid,
                node_ids,
                dx_um=float(dx),
                dy_um=float(dy),
                dz_um=float(dz),
                include_subtree=self.translate_subtree,
            )
            changed += len(out)

        self.last_action = (
            f"translate dx={float(dx):g} dy={float(dy):g} dz={float(dz):g} changed_nodes={changed}"
        )
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[move] translated {changed} node(s)")

    def _detach_selected(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[detach] no selected segments")
            return

        self._push_undo()
        changed = 0
        for nid, node_ids in by_n.items():
            out = self.project.apply_detach(nid, node_ids)
            changed += len(out)

        self.last_action = f"detach changed_nodes={changed}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[detach] detached {changed} node(s) (new roots)")

    def _reparent_last_pair(self) -> None:
        pair = self._last_two_selected()
        if pair is None:
            print("[reparent] need at least two selected segments")
            return

        src, dst = pair
        if int(src[0]) != int(dst[0]):
            print(
                "[reparent] selected pair spans different SWCs. "
                "Use key 'j' to create an inter-neuron connection spec instead."
            )
            return

        self._push_undo()
        nid = int(src[0])
        child_node = int(dst[1])
        new_parent = int(src[1])

        try:
            self.project.apply_reparent(nid, [(child_node, new_parent)], allow_cycles=False)
        except Exception as e:
            print(f"[reparent] failed: {e}")
            return

        self.last_action = f"reparent neuron={nid} child={child_node} -> parent={new_parent}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[reparent] neuron {nid}: child {child_node} now parented to {new_parent}")

    def _arm_draw_to_click(self) -> None:
        if not self.selection_order:
            print("[draw] select a segment first; anchor will be the most recently selected node")
            return
        anchor = self.selection_order[-1]
        self.pending_draw_anchor = anchor
        self._update_overlay()
        self.plotter.render()
        print(
            f"[draw] armed on anchor={anchor}. Left-click a 3D point in the window to create a new branch."
        )

    def _add_connection_from_last_pair(self) -> None:
        pair = self._last_two_selected()
        if pair is None:
            print("[connect] need at least two selected segments")
            return

        pre, post = pair
        try:
            conn = self.project.add_connection(
                pre_neuron_id=int(pre[0]),
                pre_node_id=int(pre[1]),
                post_neuron_id=int(post[0]),
                post_node_id=int(post[1]),
                chemical_synapses=self.connection_chemical,
                gap_junctions=self.connection_gap,
                gap_mode=self.connection_gap_mode,
                gap_direction=self.connection_gap_direction,
                note="added_from_selector",
            )
        except Exception as e:
            print(f"[connect] failed: {e}")
            return

        self.last_action = (
            f"connect pre=({conn.pre_neuron_id},{conn.pre_node_id}) "
            f"post=({conn.post_neuron_id},{conn.post_node_id})"
        )
        self._update_overlay()
        self.plotter.render()
        print(
            "[connect] added",
            {
                "pre": (conn.pre_neuron_id, conn.pre_node_id),
                "post": (conn.post_neuron_id, conn.post_node_id),
                "chemical_synapses": conn.chemical_synapses,
                "gap_junctions": conn.gap_junctions,
                "gap_mode": conn.gap_mode,
                "gap_direction": conn.gap_direction,
            },
        )

    def _mark_selected_passive(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[biophys] no selected segments")
            return

        self._push_undo()
        changed = 0
        for nid, node_ids in by_n.items():
            out = self.project.set_nodes_passive(
                nid,
                node_ids,
                include_subtree=False,
                replace=False,
                note="set_from_selector",
            )
            changed += len(out)

        self.last_action = f"set_passive changed_nodes={changed}"
        self._update_overlay()
        self.plotter.render()
        print(f"[biophys] marked passive nodes={changed}")

    def _mark_selected_active(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[biophys] no selected segments")
            return

        self._push_undo()
        changed = 0
        for nid, node_ids in by_n.items():
            out = self.project.set_nodes_active(
                nid,
                node_ids,
                include_subtree=False,
                replace=False,
                note="set_from_selector",
            )
            changed += len(out)

        self.last_action = f"set_active changed_nodes={changed}"
        self._update_overlay()
        self.plotter.render()
        print(f"[biophys] marked active nodes={changed}")

    def _mark_selected_recording(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[recording] no selected segments")
            return

        self._push_undo()
        changed = 0
        summaries: List[Dict[str, Any]] = []
        for nid, node_ids in by_n.items():
            out = self.project.set_nodes_recording(
                int(nid),
                node_ids,
                include_subtree=False,
                replace=False,
                note="set_from_selector",
            )
            changed += len(node_ids)
            summaries.append(
                {
                    "neuron_id": int(nid),
                    "probe_node_ids": [int(x) for x in out],
                }
            )

        self.last_action = f"set_recording_probes changed_nodes={changed}"
        self._update_overlay()
        self.plotter.render()
        print("[recording] assigned probe nodes")
        print(json.dumps(summaries, indent=2))

    def _mark_selected_ais(self) -> None:
        by_n = self._selected_by_neuron()
        if not by_n:
            print("[ais] no selected segments")
            return

        ordered = [k for k in self.selection_order if k in self.selected_keys]
        last_by_neuron: Dict[int, int] = {}
        for nid, node_id in ordered:
            last_by_neuron[int(nid)] = int(node_id)

        self._push_undo()
        changed = 0
        summaries: List[Dict[str, Any]] = []
        for nid, node_ids in by_n.items():
            primary_node_id = int(last_by_neuron.get(int(nid), int(node_ids[-1])))
            out = self.project.set_ais_nodes(
                int(nid),
                node_ids,
                primary_node_id=primary_node_id,
                primary_xloc=0.5,
                replace=False,
                note="set_from_selector",
            )
            self.project.set_nodes_active(
                int(nid),
                list({int(primary_node_id)}.union(int(x) for x in out["extra_node_ids"])),
                include_subtree=False,
                replace=False,
                note="ais_region_auto_active",
            )
            changed += 1 + len(out["extra_node_ids"])
            summaries.append(
                {
                    "neuron_id": int(nid),
                    "primary_node_id": int(out["primary_node_id"]),
                    "extra_node_ids": [int(x) for x in out["extra_node_ids"]],
                }
            )

        self.last_action = f"set_ais_regions changed_nodes={changed}"
        # Persist AIS assignment immediately so the notebook can recover even
        # if the interactive window exits after this callback.
        autosave_info = None
        try:
            autosave_info = self.project.export_bundle(
                output_root=self.output_root,
                tag=self.tag,
                require_single_component=self.require_single_component,
                write_phase2_overlay=True,
            )
        except Exception as e:
            print(f"[ais] autosave failed: {e}")

        # Avoid extra VTK scene churn here; that has been the unstable part.
        print("[ais] assigned selected AIS regions")
        print(json.dumps(summaries, indent=2))
        if autosave_info is not None:
            print("[ais] autosaved bundle")
            print(json.dumps(autosave_info, indent=2))

    def _print_validation(self) -> None:
        rep = self.project.validate_all(require_single_component=self.require_single_component)
        print("[validate] require_single_component=", self.require_single_component)
        for nid in sorted(rep.keys()):
            r = rep[nid]
            print(
                f"  neuron={nid} valid={r['valid']} nodes={r['n_nodes']} "
                f"components={r['n_components']} roots={r['n_roots']} "
                f"cycles={len(r['cycle_node_ids'])} missing_parents={len(r['missing_parent_node_ids'])}"
            )

    def _serialize_camera_position(self) -> Optional[Dict[str, Any]]:
        try:
            cam = self.plotter.camera_position
        except Exception:
            return None
        try:
            pos = [float(x) for x in cam[0]]
            focal = [float(x) for x in cam[1]]
            up = [float(x) for x in cam[2]]
        except Exception:
            return None
        if len(pos) != 3 or len(focal) != 3 or len(up) != 3:
            return None
        return {
            "position": pos,
            "focal_point": focal,
            "view_up": up,
        }

    def _apply_camera_preset_data(self, data: Dict[str, Any]) -> bool:
        cam = dict(data or {})
        try:
            pos = [float(x) for x in cam.get("position", [])]
            focal = [float(x) for x in cam.get("focal_point", [])]
            up = [float(x) for x in cam.get("view_up", [])]
        except Exception:
            return False
        if len(pos) != 3 or len(focal) != 3 or len(up) != 3:
            return False
        try:
            self.plotter.camera_position = [tuple(pos), tuple(focal), tuple(up)]
            try:
                self.plotter.reset_camera_clipping_range()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _save_camera_preset(self, *, quiet: bool = False) -> Optional[Path]:
        cam = self._serialize_camera_position()
        if cam is None:
            if not quiet:
                print("[camera] unable to capture current camera pose")
            return None
        payload = {
            "tag": self.tag,
            "neuron_ids": [int(x) for x in self.project.neuron_ids],
            "saved_at": datetime.now().isoformat(),
            "camera": cam,
        }
        path = self.camera_preset_path
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not quiet:
            print(f"[camera] preset written: {path}")
        return path

    def _load_camera_preset(self, *, quiet: bool = False) -> Optional[Path]:
        path = self.camera_preset_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            if not quiet:
                print(f"[camera] failed to load preset {path}: {e}")
            return None
        cam = payload.get("camera") if isinstance(payload, dict) else None
        if not isinstance(cam, dict):
            cam = payload if isinstance(payload, dict) else {}
        if not self._apply_camera_preset_data(cam):
            if not quiet:
                print(f"[camera] preset invalid or incompatible: {path}")
            return None
        if not quiet:
            print(f"[camera] preset loaded: {path}")
        return path

    def _save_bundle(self) -> None:
        try:
            out = self.project.export_bundle(
                output_root=self.output_root,
                tag=self.tag,
                require_single_component=self.require_single_component,
                write_phase2_overlay=True,
            )
        except Exception as e:
            print(f"[save] failed: {e}")
            return

        cam_path = None
        try:
            cam_path = self._save_camera_preset(quiet=False)
        except Exception as e:
            print(f"[camera] failed to save preset: {e}")

        if isinstance(out, dict) and cam_path is not None:
            out = dict(out)
            out["camera_preset_json"] = str(cam_path)

        self.last_action = "save_bundle+camera" if cam_path is not None else "save_bundle"
        self._update_overlay()
        self.plotter.render()
        print("[save] bundle written")
        if cam_path is not None:
            print(f"[save] camera preset written: {cam_path}")
        print(json.dumps(out, indent=2))

    @staticmethod
    def _find_time_column(df: pd.DataFrame) -> str:
        for col in ("t_ms", "time_ms", "t"):
            if col in df.columns:
                return str(col)
        raise ValueError(f"records table missing time column. Columns={list(df.columns)[:20]}")

    @staticmethod
    def _parse_trace_neuron_id(col_name: str) -> Optional[int]:
        m = re.match(r"^(\d+)", str(col_name))
        if m is None:
            return None
        return int(m.group(1))

    @staticmethod
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
        frac = (float(thresh_mV) - v0) / denom
        return t0 + frac * (t1 - t0)

    def _load_flow_dataset(self) -> Dict[str, Any]:
        if self.flow_run_dir is None:
            raise RuntimeError("No flow run directory configured. Relaunch with --flow-run-dir.")
        if not self.flow_run_dir.exists():
            raise FileNotFoundError(f"Flow run directory not found: {self.flow_run_dir}")

        cfg_path = self.flow_run_dir / "config.json"
        rec_path = self.flow_run_dir / "records.csv"
        if not rec_path.exists():
            raise FileNotFoundError(f"records.csv missing in flow run dir: {rec_path}")

        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        df = pd.read_csv(rec_path)
        tcol = self._find_time_column(df)
        t_ms = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(t_ms)):
            raise RuntimeError("Flow run time vector contains NaNs/non-finite values.")

        trace_map: Dict[int, np.ndarray] = {}
        for col in df.columns:
            if "_soma_v" not in str(col):
                continue
            nid = self._parse_trace_neuron_id(str(col))
            if nid is None or int(nid) not in self.project.tables:
                continue
            trace_map[int(nid)] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if self.flow_focus_pair is not None:
            neuron_ids = [int(x) for x in self.flow_focus_pair if int(x) in trace_map]
        else:
            neuron_ids = sorted(int(x) for x in trace_map.keys())
        if not neuron_ids:
            raise RuntimeError("No overlapping recorded soma traces found for loaded mutation-app neurons.")

        spike_map: Dict[int, np.ndarray] = {}
        for spike_name in ("spike_times.csv", "spikes.csv"):
            p = self.flow_run_dir / spike_name
            if not p.exists():
                continue
            sdf = pd.read_csv(p)
            nid_col = "neuron_id" if "neuron_id" in sdf.columns else ("nid" if "nid" in sdf.columns else None)
            t_col = "spike_time_ms" if "spike_time_ms" in sdf.columns else ("t_ms" if "t_ms" in sdf.columns else None)
            if nid_col is None or t_col is None:
                continue
            sdf = sdf[[nid_col, t_col]].copy()
            sdf[nid_col] = pd.to_numeric(sdf[nid_col], errors="coerce")
            sdf[t_col] = pd.to_numeric(sdf[t_col], errors="coerce")
            sdf = sdf.dropna()
            for nid, grp in sdf.groupby(nid_col):
                ni = int(nid)
                if ni in self.project.tables:
                    spike_map[ni] = np.sort(grp[t_col].to_numpy(dtype=float))
            if spike_map:
                break

        for nid in neuron_ids:
            if int(nid) not in spike_map:
                spike_map[int(nid)] = self._spike_times_from_trace(
                    t_ms,
                    trace_map[int(nid)],
                    thresh_mV=self.flow_threshold_mV,
                )

        voltage_norm: Dict[int, np.ndarray] = {}
        for nid in neuron_ids:
            vv = np.asarray(trace_map[int(nid)], dtype=float)
            voltage_norm[int(nid)] = np.clip((vv + 70.0) / 100.0, 0.0, 1.0)

        default_delay = self.flow_syn_delay_ms
        if default_delay is None:
            default_delay = float(cfg.get("default_delay_ms", 0.5) or 0.5)

        conn_recs = self._connection_records()
        if self.flow_focus_pair is not None:
            pair = tuple(int(x) for x in self.flow_focus_pair)
            conn_recs = [
                r for r in conn_recs
                if int(r["pre_neuron_id"]) == int(pair[0]) and int(r["post_neuron_id"]) == int(pair[1])
            ]
            if not conn_recs:
                print(
                    f"[flow] no explicit mutation-connection spec for focus pair {pair}; "
                    "exporting per-neuron activity only"
                )
        conn_recs = [
            r for r in conn_recs
            if int(r["pre_neuron_id"]) in neuron_ids and int(r["post_neuron_id"]) in neuron_ids
        ]

        events_by_neuron: Dict[int, List[Dict[str, Any]]] = {int(nid): [] for nid in neuron_ids}
        for nid in neuron_ids:
            src = self._primary_source_node_id(int(nid))
            times = np.asarray(spike_map.get(int(nid), np.array([], dtype=float)), dtype=float)
            if times.size:
                events_by_neuron[int(nid)].append(
                    {
                        "source_node_id": int(src),
                        "times_ms": times,
                        "kind": "self",
                    }
                )

        conn_event_records: List[Dict[str, Any]] = []
        for r in conn_recs:
            pre_n = int(r["pre_neuron_id"])
            post_n = int(r["post_neuron_id"])
            pre_times = np.asarray(spike_map.get(pre_n, np.array([], dtype=float)), dtype=float)
            if pre_times.size == 0:
                continue
            arrivals = pre_times + float(default_delay)
            events_by_neuron[post_n].append(
                {
                    "source_node_id": int(r["post_node_id"]),
                    "times_ms": arrivals,
                    "kind": "incoming",
                    "from_neuron_id": int(pre_n),
                }
            )
            conn_event_records.append(
                {
                    **dict(r),
                    "event_times_ms": arrivals,
                }
            )

        return {
            "cfg": cfg,
            "t_ms": t_ms,
            "trace_map": trace_map,
            "neuron_ids": neuron_ids,
            "spike_map": spike_map,
            "voltage_norm": voltage_norm,
            "events_by_neuron": events_by_neuron,
            "connection_events": conn_event_records,
            "default_syn_delay_ms": float(default_delay),
        }

    def _distance_map(self, neuron_id: int, source_node_id: int) -> Dict[int, float]:
        key = (int(neuron_id), int(source_node_id))
        if key in self._distance_cache:
            return self._distance_cache[key]

        df = self.project.table(int(neuron_id))
        pos = {
            int(r["id"]): np.asarray([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
            for _, r in df.iterrows()
        }
        adj: Dict[int, List[Tuple[int, float]]] = {int(n): [] for n in pos.keys()}
        for _, r in df.iterrows():
            nid = int(r["id"])
            parent = int(r["parent"])
            if parent == -1 or parent not in pos:
                continue
            dist = float(np.linalg.norm(pos[nid] - pos[parent]))
            adj[nid].append((int(parent), dist))
            adj[parent].append((int(nid), dist))

        src = int(source_node_id)
        if src not in adj:
            raise KeyError(f"source_node_id {src} missing from neuron {int(neuron_id)}")

        dist_map: Dict[int, float] = {src: 0.0}
        heap: List[Tuple[float, int]] = [(0.0, src)]
        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur_d > dist_map.get(cur, float("inf")):
                continue
            for nxt, edge_d in adj.get(cur, []):
                cand = float(cur_d + edge_d)
                if cand < dist_map.get(int(nxt), float("inf")):
                    dist_map[int(nxt)] = cand
                    heapq.heappush(heap, (cand, int(nxt)))

        self._distance_cache[key] = dist_map
        return dist_map

    def _flow_signal_values(self, frame_t_ms: float, flow: Dict[str, Any]) -> np.ndarray:
        if self.segment_df is None:
            raise RuntimeError("segment_df unavailable")
        seg_neuron = self.segment_df["neuron_id"].to_numpy(dtype=np.int64)
        seg_child = self.segment_df["child_node_id"].to_numpy(dtype=np.int64)
        out = np.zeros(len(self.segment_df), dtype=float)

        for nid in flow["neuron_ids"]:
            mask = seg_neuron == int(nid)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            base_trace = np.asarray(flow["voltage_norm"].get(int(nid), np.zeros_like(flow["t_ms"])), dtype=float)
            base_val = float(np.interp(float(frame_t_ms), flow["t_ms"], base_trace))
            vals = np.full(len(idx), 0.42 * base_val, dtype=float)

            for ev in flow["events_by_neuron"].get(int(nid), []):
                dist_map = self._distance_map(int(nid), int(ev["source_node_id"]))
                dists = np.asarray([dist_map.get(int(seg_child[j]), np.nan) for j in idx], dtype=float)
                valid = np.isfinite(dists)
                if not np.any(valid):
                    continue
                for t0 in np.asarray(ev["times_ms"], dtype=float):
                    arrival = float(t0) + dists[valid] / self.flow_speed_um_per_ms
                    dt = float(frame_t_ms) - arrival
                    sigma = max(1e-6, float(self.flow_pulse_sigma_ms))
                    rise = max(1e-6, sigma * 0.75)
                    decay = max(1e-6, sigma * 1.8)
                    pulse = _asymmetric_flow_pulse(dt, rise_ms=rise, decay_ms=decay)
                    vals[valid] = np.maximum(vals[valid], np.clip(0.12 + 0.88 * pulse, 0.0, 1.0))

            out[idx] = np.clip(vals, 0.0, 1.0)
        return out

    def _build_flow_connection_mesh(self, flow: Dict[str, Any]) -> Tuple[Optional[pv.PolyData], List[Dict[str, Any]]]:
        recs = flow.get("connection_events", []) or []
        if not recs:
            return None, []
        pts: List[np.ndarray] = []
        lines: List[int] = []
        p_off = 0
        kept: List[Dict[str, Any]] = []
        for r in recs:
            try:
                p0 = self._node_xyz(int(r["pre_neuron_id"]), int(r["pre_node_id"]))
                p1 = self._node_xyz(int(r["post_neuron_id"]), int(r["post_node_id"]))
            except Exception:
                continue
            pts.extend([p0, p1])
            lines.extend([2, p_off, p_off + 1])
            p_off += 2
            kept.append(dict(r))
        if not kept:
            return None, []
        mesh = pv.PolyData()
        mesh.points = np.vstack(pts)
        mesh.lines = np.asarray(lines, dtype=np.int64)
        mesh.cell_data["flow_signal"] = np.zeros(len(kept), dtype=float)
        return mesh, kept

    def _flow_connection_values(self, frame_t_ms: float, conn_records: Sequence[Dict[str, Any]]) -> np.ndarray:
        vals = np.zeros(len(conn_records), dtype=float)
        for i, r in enumerate(conn_records):
            tms = np.asarray(r.get("event_times_ms", np.array([], dtype=float)), dtype=float)
            if tms.size == 0:
                continue
            dt = float(frame_t_ms) - tms
            sigma = max(0.5, float(self.flow_pulse_sigma_ms) * 1.5)
            rise = max(0.5, sigma * 0.75)
            decay = max(0.5, sigma * 1.8)
            pulse = _asymmetric_flow_pulse(dt, rise_ms=rise, decay_ms=decay)
            vals[i] = float(np.clip(np.max(pulse), 0.0, 1.0))
        return vals

    def _export_flow_movie(self) -> None:
        try:
            flow = self._load_flow_dataset()
        except Exception as e:
            self.last_action = f"flow export unavailable: {e}"
            self._update_overlay()
            self.plotter.render()
            print(f"[flow] export unavailable: {e}")
            return

        frame_times = _flow_movie_frame_times(
            flow["t_ms"],
            flow_max_ms=self.flow_max_ms,
            frame_stride=self.flow_frame_stride,
            fps=self.flow_fps,
            duration_sec=self.flow_duration_sec,
        )
        if frame_times.size < 2:
            print("[flow] not enough time samples to export movie")
            return

        movie_duration = float(frame_times.size) / float(max(1, self.flow_fps))
        print(
            f"[flow] exporting {frame_times.size} frames at {self.flow_fps} fps "
            f"({float(frame_times[0]):.3f}-{float(frame_times[-1]):.3f} ms "
            f"compressed to {movie_duration:.2f} s)"
        )

        pair_label = (
            f"{int(self.flow_focus_pair[0])}_to_{int(self.flow_focus_pair[1])}"
            if self.flow_focus_pair is not None
            else "all_loaded"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_mp4 = self.output_root / f"morphology_mutation_flow_{pair_label}_{stamp}.mp4"
        out_gif = self.output_root / f"morphology_mutation_flow_{pair_label}_{stamp}.gif"

        flow_poly = self.mesh.copy(deep=True)
        flow_poly.cell_data["flow_signal"] = np.zeros(flow_poly.n_cells, dtype=float)
        flow_signal_arr = flow_poly.cell_data["flow_signal"]
        conn_mesh, conn_records = self._build_flow_connection_mesh(flow)
        conn_signal_arr = None
        if conn_mesh is not None:
            conn_signal_arr = conn_mesh.cell_data["flow_signal"]

        flow_actor = None
        conn_actor = None
        time_actor_added = False
        cam_before = self.plotter.camera_position
        mwriter = None
        try:
            try:
                self.plotter.remove_actor("status", render=False)
            except Exception:
                pass
            try:
                self.plotter.remove_actor("keys", render=False)
            except Exception:
                pass
            if not self.flow_preserve_camera:
                self._focus_camera_on_neurons()
            volume_mode = self.render_mode == "neuroglancer"
            flow_actor = self.plotter.add_mesh(
                flow_poly,
                name="flow_overlay",
                scalars="flow_signal",
                cmap=_resolve_flow_cmap(self.flow_overlay_style),
                clim=[0.0, 1.0],
                show_scalar_bar=False,
                render_lines_as_tubes=True,
                line_width=(
                    max(8.0, float(self.skeleton_line_width) + 5.0)
                    if volume_mode else max(4.0, float(self.skeleton_line_width) + 2.0)
                ),
                opacity=0.98 if volume_mode else 1.0,
                pickable=False,
                copy_mesh=False,
                lighting=False,
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
            )
            if conn_mesh is not None:
                conn_actor = self.plotter.add_mesh(
                    conn_mesh,
                    name="flow_connections",
                    scalars="flow_signal",
                    cmap=_resolve_connection_cmap(self.flow_overlay_style),
                    clim=[0.0, 1.0],
                    show_scalar_bar=False,
                    render_lines_as_tubes=True,
                    line_width=(
                        max(6.0, float(self.skeleton_line_width) + 2.0)
                        if volume_mode else max(3.0, float(self.skeleton_line_width))
                    ),
                    opacity=0.98 if volume_mode else 0.95,
                    pickable=False,
                    copy_mesh=False,
                    lighting=False,
                    ambient=1.0,
                    diffuse=0.0,
                    specular=0.0,
                )

            self.plotter.open_movie(str(out_mp4), framerate=self.flow_fps)
            mwriter = getattr(self.plotter, "mwriter", None)
            wrote_movie = True
        except Exception as e:
            print(f"[flow][warn] mp4 export unavailable ({e}); falling back to GIF")
            wrote_movie = False
            self.plotter.open_gif(str(out_gif), fps=self.flow_fps)
            mwriter = getattr(self.plotter, "mwriter", None)

        try:
            for frame_idx, frame_t in enumerate(frame_times):
                flow_signal_arr[:] = self._flow_signal_values(float(frame_t), flow)
                if conn_signal_arr is not None:
                    conn_signal_arr[:] = self._flow_connection_values(float(frame_t), conn_records)
                try:
                    flow_poly.modified()
                except Exception:
                    pass
                if conn_mesh is not None:
                    try:
                        conn_mesh.modified()
                    except Exception:
                        pass
                try:
                    self.plotter.remove_actor("flow_status", render=False)
                except Exception:
                    pass
                self.plotter.add_text(
                    f"Flow preview  sim t={float(frame_t):.2f} ms  video={frame_idx / max(1, self.flow_fps):.2f}s",
                    position=(0.02, 0.96),
                    viewport=True,
                    name="flow_status",
                    font_size=11,
                    color=self._overlay_text_color(),
                )
                time_actor_added = True
                self.plotter.render()
                self.plotter.write_frame()
        finally:
            try:
                if mwriter is not None:
                    mwriter.close()
            except Exception:
                pass

        try:
            if flow_actor is not None:
                self.plotter.remove_actor("flow_overlay", render=False)
        except Exception:
            pass
        try:
            if conn_actor is not None:
                self.plotter.remove_actor("flow_connections", render=False)
        except Exception:
            pass
        if time_actor_added:
            try:
                self.plotter.remove_actor("flow_status", render=False)
            except Exception:
                pass
        try:
            self.plotter.camera_position = cam_before
        except Exception:
            pass
        self._update_overlay()
        self.plotter.render()
        if wrote_movie:
            self.last_action = f"saved flow movie: {out_mp4.name}"
            self._update_overlay()
            self.plotter.render()
            print(f"[flow] saved movie: {out_mp4}")
        else:
            self.last_action = f"saved flow gif: {out_gif.name}"
            self._update_overlay()
            self.plotter.render()
            print(f"[flow] saved gif: {out_gif}")

    def _crop_background(self, img_arr: np.ndarray, *, tol: int = 10, pad_px: int = 12) -> np.ndarray:
        arr = _rgba_from_any(img_arr)

        # Prefer alpha-mask cropping when transparent background capture is used.
        if arr.shape[2] >= 4:
            alpha = arr[..., 3]
            mask = alpha > 3
        else:
            arr_rgb = arr[..., :3].astype(np.int16)
            corners = np.array(
                [
                    arr_rgb[0, 0],
                    arr_rgb[0, -1],
                    arr_rgb[-1, 0],
                    arr_rgb[-1, -1],
                ],
                dtype=np.int16,
            )
            bg = np.median(corners, axis=0)
            diff = np.max(np.abs(arr_rgb - bg[None, None, :]), axis=-1)
            mask = diff > int(tol)

        if not bool(np.any(mask)):
            return arr

        ys, xs = np.where(mask)
        y0 = max(0, int(np.min(ys)) - int(pad_px))
        y1 = min(arr.shape[0], int(np.max(ys)) + int(pad_px) + 1)
        x0 = max(0, int(np.min(xs)) - int(pad_px))
        x1 = min(arr.shape[1], int(np.max(xs)) + int(pad_px) + 1)
        return arr[y0:y1, x0:x1]

    def _focus_camera_on_neurons(self) -> None:
        if self.mesh is None or int(self.mesh.n_points) <= 0:
            return
        visible = self._visible_neuron_set()
        seg_idxs = self.segment_df.loc[
            self.segment_df["neuron_id"].isin(list(visible)), "segment_idx"
        ].to_numpy(dtype=np.int64)
        if seg_idxs.size > 0 and len(visible) < len(self.project.neuron_ids):
            try:
                focus_mesh = self.mesh.extract_cells(seg_idxs)
                b = np.asarray(focus_mesh.bounds, dtype=float)
            except Exception:
                b = np.asarray(self.mesh.bounds, dtype=float)
        else:
            b = np.asarray(self.mesh.bounds, dtype=float)
        if b.size != 6 or (not np.all(np.isfinite(b))):
            return

        cx = 0.5 * (b[0] + b[1])
        cy = 0.5 * (b[2] + b[3])
        cz = 0.5 * (b[4] + b[5])
        # Small margin around neuron geometry.
        margin = 1.04
        hx = max(1e-3, 0.5 * (b[1] - b[0]) * margin)
        hy = max(1e-3, 0.5 * (b[3] - b[2]) * margin)
        hz = max(1e-3, 0.5 * (b[5] - b[4]) * margin)
        fit_bounds = (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)

        try:
            self.plotter.renderer.ResetCamera(*fit_bounds)
        except Exception:
            try:
                self.plotter.reset_camera()
            except Exception:
                pass

    def _strip_white_background_from_svg(self, svg_path: Path) -> None:
        try:
            txt = svg_path.read_text(encoding="utf-8")
        except Exception:
            return
        # Remove plain white full-canvas rect backgrounds when present.
        patterns = [
            r'<rect[^>]*fill="white"[^>]*/>\s*',
            r"<rect[^>]*fill='#ffffff'[^>]*/>\s*",
            r'<rect[^>]*fill="#ffffff"[^>]*/>\s*',
            r'<rect[^>]*fill="rgb\(100%,100%,100%\)"[^>]*/>\s*',
        ]
        out = txt
        for pat in patterns:
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
        if out != txt:
            try:
                svg_path.write_text(out, encoding="utf-8")
            except Exception:
                pass

    def _export_skeleton_stroke_svg(self, out_svg: Path, *, pad_px: float = 8.0) -> int:
        """Export current skeleton view as SVG with one <line> per segment.

        This is intended for downstream editing in tools like Inkscape where each
        segment should be an independent stroke object with explicit width.
        """
        if self.mesh is None or self.segment_df is None:
            raise RuntimeError("No mesh available for skeleton SVG export.")

        self.plotter.render()
        ren = self.plotter.renderer

        line_arr = np.asarray(self.mesh.lines, dtype=np.int64)
        if line_arr.size < 3:
            raise RuntimeError("Mesh has no line segments to export.")
        if (line_arr.size % 3) != 0:
            raise RuntimeError("Unexpected line connectivity format for skeleton export.")

        cells = line_arr.reshape(-1, 3)
        if not np.all(cells[:, 0] == 2):
            # Keep only 2-point segments.
            cells = cells[cells[:, 0] == 2]
        if cells.size == 0:
            raise RuntimeError("No 2-point line cells found for skeleton export.")
        point_ids = cells[:, 1:3]

        pts = np.asarray(self.mesh.points, dtype=float)
        cell_neuron = np.asarray(self.mesh.cell_data.get("neuron_id", np.zeros(len(point_ids))), dtype=int)
        cell_seg_idx = np.asarray(self.mesh.cell_data.get("segment_idx", np.arange(len(point_ids))), dtype=int)
        if len(cell_neuron) != len(point_ids) or len(cell_seg_idx) != len(point_ids):
            n = min(len(cell_neuron), len(cell_seg_idx), len(point_ids))
            point_ids = point_ids[:n]
            cell_neuron = cell_neuron[:n]
            cell_seg_idx = cell_seg_idx[:n]

        sel_idxs = set(
            int(x)
            for x in self.segment_df.loc[self.segment_df["key"].isin(list(self.selected_keys)), "segment_idx"].tolist()
        )
        base_w = float(self.skeleton_line_width)
        sel_w = max(3.0, base_w + 3.0)

        recs: List[Tuple[float, float, float, float, float, str, float, int, int]] = []

        def _project_xyz(xyz: np.ndarray) -> Tuple[float, float, float]:
            ren.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
            ren.WorldToDisplay()
            dx, dy, dz = ren.GetDisplayPoint()
            return float(dx), float(dy), float(dz)

        for i, (p0, p1) in enumerate(point_ids):
            xyz0 = pts[int(p0)]
            xyz1 = pts[int(p1)]
            x0, y0, z0 = _project_xyz(xyz0)
            x1, y1, z1 = _project_xyz(xyz1)
            if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
                continue

            seg_idx = int(cell_seg_idx[i]) if i < len(cell_seg_idx) else int(i)
            is_sel = int(seg_idx) in sel_idxs
            nid = int(cell_neuron[i]) if i < len(cell_neuron) else -1
            color = str(self.selected_color if is_sel else self.neuron_colors.get(nid, "#4c78a8"))
            width = float(sel_w if is_sel else base_w)
            zavg = 0.5 * (float(z0) + float(z1))
            recs.append((x0, y0, x1, y1, zavg, color, width, nid, seg_idx))

        if not recs:
            raise RuntimeError("No projectable skeleton segments for SVG export.")

        xs = [r[0] for r in recs] + [r[2] for r in recs]
        ys = [r[1] for r in recs] + [r[3] for r in recs]
        x0 = float(min(xs) - float(pad_px))
        x1 = float(max(xs) + float(pad_px))
        y0 = float(min(ys) - float(pad_px))
        y1 = float(max(ys) + float(pad_px))
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)

        # Draw far-to-near for cleaner overlap.
        recs.sort(key=lambda r: r[4], reverse=True)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
                f'width="{w:.3f}" height="{h:.3f}" viewBox="0 0 {w:.3f} {h:.3f}">'
            ),
        ]

        for i, (x_a, y_a, x_b, y_b, _z, color, stroke_w, nid, seg_idx) in enumerate(recs, start=1):
            # VTK display y grows upward; SVG y grows downward.
            xa = x_a - x0
            xb = x_b - x0
            ya = y1 - y_a
            yb = y1 - y_b
            lines.append(
                f'<line x1="{xa:.3f}" y1="{ya:.3f}" x2="{xb:.3f}" y2="{yb:.3f}" '
                f'stroke="{color}" stroke-width="{stroke_w:.3f}" '
                f'id="seg_n{int(nid)}_{int(seg_idx)}_{int(i)}" data-neuron-id="{int(nid)}" '
                f'data-segment-idx="{int(seg_idx)}" '
                'stroke-linecap="round" stroke-linejoin="round" fill="none" '
                'vector-effect="non-scaling-stroke" />'
            )
        lines.append("</svg>")

        out_svg.write_text("\n".join(lines), encoding="utf-8")
        return int(len(recs))

    def _save_jpeg_screenshot(self) -> None:
        out_dir = self.output_root
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_jpg = out_dir / f"morphology_mutation_selector_{stamp}.jpg"
        out_png = out_dir / f"morphology_mutation_selector_{stamp}.png"
        out_pdf = out_dir / f"morphology_mutation_selector_{stamp}.pdf"
        out_svg = out_dir / f"morphology_mutation_selector_{stamp}.svg"
        out_svg_raster = out_dir / f"morphology_mutation_selector_{stamp}_raster.svg"
        out_pdf_vec = out_dir / f"morphology_mutation_selector_{stamp}_vector.pdf"

        win_w, win_h = [int(v) for v in self.plotter.window_size]
        tgt_w = max(2200, int(round(win_w * self.screenshot_scale)))
        tgt_h = max(1400, int(round(win_h * self.screenshot_scale)))

        n_strokes = None
        try:
            cam_before = self.plotter.camera_position
            bg_before = None
            try:
                try:
                    bg_before = tuple(float(x) for x in self.plotter.renderer.GetBackground())
                except Exception:
                    bg_before = None
                # Capture only neuron geometry footprint: hide text overlays + axes widget.
                try:
                    self.plotter.remove_actor("status", render=False)
                except Exception:
                    pass
                try:
                    self.plotter.remove_actor("keys", render=False)
                except Exception:
                    pass
                try:
                    self.plotter.hide_axes()
                except Exception:
                    pass

                self._focus_camera_on_neurons()
                self.plotter.render()
                # Always export editable stroke SVG (one line object per segment).
                try:
                    n_strokes = self._export_skeleton_stroke_svg(out_svg, pad_px=8.0)
                    self._strip_white_background_from_svg(out_svg)
                except Exception as _e:
                    print(f"[screenshot][warn] custom stroke SVG export failed ({_e})")
                try:
                    self.plotter.save_graphic(str(out_pdf_vec), raster=False, painter=True)
                except Exception as _e:
                    print(f"[screenshot][warn] vector PDF export failed ({_e})")
                img = self.plotter.screenshot(
                    return_img=True,
                    window_size=(tgt_w, tgt_h),
                    transparent_background=False,
                )
            finally:
                try:
                    self.plotter.camera_position = cam_before
                except Exception:
                    pass
                try:
                    if bg_before is not None:
                        self.plotter.set_background(bg_before)
                except Exception:
                    pass
                try:
                    self.plotter.show_axes()
                except Exception:
                    pass
                self._update_overlay()
                self.plotter.render()

            img = self._crop_background(img, tol=10, pad_px=10)
            rgba = _rgba_from_any(img)
            if Image is not None:
                pil_rgba = Image.fromarray(rgba, mode="RGBA")
                pil_rgb = Image.new("RGB", pil_rgba.size, (255, 255, 255))
                pil_rgb.paste(pil_rgba, mask=pil_rgba.getchannel("A"))
                pil_rgba.save(
                    out_png,
                    format="PNG",
                    optimize=True,
                    dpi=(self.screenshot_dpi, self.screenshot_dpi),
                )

                # Photo-style exports keep the current viewer background.
                pil_rgb.save(
                    out_jpg,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                    subsampling=0,
                    dpi=(self.screenshot_dpi, self.screenshot_dpi),
                )
                try:
                    import matplotlib.pyplot as _plt

                    h, w = rgba.shape[:2]
                    fig = _plt.figure(
                        figsize=(float(w) / float(self.screenshot_dpi), float(h) / float(self.screenshot_dpi)),
                        dpi=float(self.screenshot_dpi),
                        frameon=False,
                    )
                    fig.patch.set_alpha(1.0)
                    fig.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))
                    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    ax.patch.set_alpha(1.0)
                    ax.patch.set_facecolor((1.0, 1.0, 1.0, 1.0))
                    ax.imshow(np.asarray(pil_rgb), interpolation="nearest")
                    save_kwargs = dict(
                        dpi=float(self.screenshot_dpi),
                        transparent=False,
                        facecolor="white",
                        edgecolor="white",
                        bbox_inches="tight",
                        pad_inches=0.0,
                    )
                    fig.savefig(out_pdf, **save_kwargs)
                    fig.savefig(out_svg_raster, **save_kwargs)
                    _plt.close(fig)
                except Exception as _e:
                    print(f"[screenshot][warn] transparent pdf/svg export failed ({_e})")
            else:
                self.plotter.screenshot(str(out_png), window_size=(tgt_w, tgt_h), transparent_background=True)
                print("[screenshot] PIL missing: saved PNG only; JPG/PDF/SVG export skipped")
            print(
                f"[screenshot] saved {out_png} + {out_jpg} + {out_pdf} + {out_svg} "
                f"(cropped to neuron bounds, current viewer background, dpi={self.screenshot_dpi}, quality={self.jpeg_quality})"
            )
            print(
                f"[screenshot] stroke SVG: {out_svg} "
                f"(one editable line per segment; stroke width preserved). "
                f"Raster fallback SVG: {out_svg_raster}. Vector PDF: {out_pdf_vec}"
            )
            if self.render_mode == "neuroglancer":
                print(
                    "[screenshot] neuroglancer-like mode is raster-first: the PNG/JPG/PDF capture "
                    "the textured volume look, while SVG/vector PDF remain line-based guides."
                )
            if n_strokes is not None:
                print(f"[screenshot] custom stroke count = {n_strokes}")
            return
        except Exception as e:
            print(f"[screenshot] jpeg export failed ({e}); saving png fallback")

        self.plotter.screenshot(str(out_png), window_size=(tgt_w, tgt_h), transparent_background=False)
        print(f"[screenshot] saved {out_png}")

    def _print_help(self) -> None:
        print(
            "Morphology Mutation keys:\n"
            "  r drag-box select (additive)\n"
            "  c clear selection | p print selection\n"
            "  t thin | y thicken | g grow tangent | d draw branch to clicked point (right click)\n"
            "  b split selected edges | a reparent last pair | x detach selected\n"
            "  i/k +X/-X, l/o +Y/-Y, u/n +Z/-Z translate\n"
            "  j add pre->post connection spec from last two selected\n"
            "  f mark selected passive | q mark selected active | e mark recording probes | 5 assign AIS regions\n"
            "  3 save static all-synapse screenshot | 7/8 previous/next synapse pair\n"
            "  w toggle skeleton<->tube | 1 toggle neuroglancer-like volume\n"
            "  [ previous neuron | ] next neuron | 4/6 previous/next neuron | - toggle solo/all neurons\n"
            "  z undo | v validate | s save bundle + camera | m photo export (current bg) | 0 export flow movie\n"
            "  [/]/4/6: previous/next neuron | - or 2: solo/all toggle\n"
            "Flow export uses the current camera by default. Key 3 saves a static all-synapse screenshot from the current run's edge table/synmap coordinates."
        )

    def show(self) -> None:
        self.plotter.show(title="Morphology Mutation - Multi SWC Selector")



def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Morphology Mutation desktop app: load multiple SWCs, select compartments, "
            "mutate morphology, define connection specs, validate, and export bundles."
        )
    )
    p.add_argument("--swc-dir", type=str, required=True, help="Root directory containing SWCs.")
    p.add_argument("--neuron-ids", type=str, required=True, help="Comma-separated neuron IDs.")
    p.add_argument(
        "--phase2-root",
        type=str,
        default=None,
        help="Optional path to Phase 2 repo root so digifly imports resolve.",
    )

    default_output = (_PROJECT_ROOT / "notebooks" / "debug" / "outputs").resolve()
    p.add_argument("--output-root", type=str, default=str(default_output), help="Bundle output root dir.")
    p.add_argument("--tag", type=str, default="mutant", help="Short mutation tag for saved files.")

    p.add_argument("--selected-color", type=str, default="#ff0000")
    p.add_argument("--thin-factor", type=float, default=0.8)
    p.add_argument("--thick-factor", type=float, default=1.2)

    p.add_argument("--grow-length-um", type=float, default=18.0)
    p.add_argument("--grow-segments", type=int, default=4)
    p.add_argument("--grow-radius-scale", type=float, default=0.85)

    p.add_argument("--include-subtree-radius", action="store_true", help="Apply thin/thick to full subtree.")
    p.add_argument("--move-step-um", type=float, default=3.0)
    p.add_argument("--translate-subtree", action="store_true", help="Translate full subtree, not just selected node.")

    p.add_argument("--connection-chemical", type=int, default=1)
    p.add_argument("--connection-gap", type=int, default=0)
    p.add_argument(
        "--connection-gap-mode",
        type=str,
        default="none",
        choices=["none", "non_rectifying", "rectifying"],
    )
    p.add_argument("--connection-gap-direction", type=str, default=None)
    p.add_argument("--render-mode", type=str, default="tube", choices=["tube", "skeleton", "neuroglancer"])
    p.add_argument("--skeleton-line-width", type=float, default=4.0)
    p.add_argument(
        "--visual-style",
        type=str,
        default="classic",
        choices=["classic", "vaa3d"],
        help="Legacy option retained for notebook compatibility; the app now always uses classic styling.",
    )

    p.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="Allow multiple connected components during validation.",
    )

    p.add_argument("--screenshot-scale", type=float, default=3.0)
    p.add_argument("--jpeg-quality", type=int, default=97)
    p.add_argument("--screenshot-dpi", type=int, default=300)
    p.add_argument(
        "--neuroglancer-quality",
        type=str,
        default="auto",
        choices=["auto", "balanced", "high", "ultra"],
        help="Synthetic Neuroglancer-like volume quality preset.",
    )
    p.add_argument(
        "--neuroglancer-voxel-um",
        type=float,
        default=None,
        help="Optional explicit synthetic voxel size in microns for neuroglancer mode.",
    )
    p.add_argument(
        "--neuroglancer-max-dim",
        type=int,
        default=None,
        help="Optional explicit per-axis cap for the synthetic neuroglancer volume grid.",
    )
    p.add_argument(
        "--neuroglancer-max-voxels",
        type=int,
        default=None,
        help="Optional explicit total voxel budget for the synthetic neuroglancer volume.",
    )
    p.add_argument("--flow-run-dir", type=str, default=None, help="Simulation run dir with config.json/records.csv.")
    p.add_argument(
        "--neuron-color-overrides",
        type=str,
        default=None,
        help="Comma-separated neuron_id:color overrides, e.g. 10000:#1d4ed8,10002:#93c5fd,10068:#b91c1c,10110:#fca5a5",
    )
    p.add_argument(
        "--flow-focus-pair",
        type=str,
        default=None,
        help="Optional pre,post neuron ID pair to focus the flow movie on, e.g. 10000,10068.",
    )
    p.add_argument("--flow-fps", type=int, default=30)
    p.add_argument("--flow-frame-stride", type=int, default=4)
    p.add_argument("--flow-speed-um-per-ms", type=float, default=25.0)
    p.add_argument("--flow-pulse-sigma-ms", type=float, default=18.0)
    p.add_argument("--flow-syn-delay-ms", type=float, default=None)
    p.add_argument("--flow-threshold-mv", type=float, default=0.0)
    p.add_argument(
        "--flow-overlay-style",
        type=str,
        default="electric_cyan",
        choices=["neuron_yellow", "viridis", "electric_cyan"],
        help="Flow overlay palette style.",
    )
    p.add_argument(
        "--flow-preserve-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the current camera during flow export instead of refocusing before rendering.",
    )
    p.add_argument(
        "--flow-max-ms",
        type=float,
        default=0.0,
        help="Upper time limit for movie export. Use <=0 for full run.",
    )
    p.add_argument(
        "--flow-duration-sec",
        type=float,
        default=20.0,
        help="Movie duration in seconds. Positive values resample the selected simulation span to fps*duration frames.",
    )
    p.add_argument(
        "--start-solo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start the app in single-neuron solo mode.",
    )
    p.add_argument(
        "--start-neuron-id",
        type=int,
        default=None,
        help="Optional neuron ID to show first when starting in solo mode.",
    )
    return p



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arg_parser().parse_args(argv)

    if args.phase2_root:
        phase2_root = Path(args.phase2_root).expanduser().resolve()
        if str(phase2_root) not in sys.path:
            sys.path.insert(0, str(phase2_root))

    neuron_ids = _parse_neuron_ids(args.neuron_ids)
    project = MorphologyMutationProject.from_neuron_ids(args.swc_dir, neuron_ids)

    app = MorphologyMutationApp(
        project=project,
        output_root=Path(args.output_root),
        tag=str(args.tag),
        selected_color=str(args.selected_color),
        thin_factor=float(args.thin_factor),
        thick_factor=float(args.thick_factor),
        grow_length_um=float(args.grow_length_um),
        grow_segments=int(args.grow_segments),
        grow_radius_scale=float(args.grow_radius_scale),
        include_subtree_radius=bool(args.include_subtree_radius),
        move_step_um=float(args.move_step_um),
        translate_subtree=bool(args.translate_subtree),
        connection_chemical=int(args.connection_chemical),
        connection_gap=int(args.connection_gap),
        connection_gap_mode=str(args.connection_gap_mode),
        connection_gap_direction=args.connection_gap_direction,
        require_single_component=not bool(args.allow_disconnected),
        render_mode=str(args.render_mode),
        skeleton_line_width=float(args.skeleton_line_width),
        screenshot_scale=float(args.screenshot_scale),
        jpeg_quality=int(args.jpeg_quality),
        screenshot_dpi=int(args.screenshot_dpi),
        visual_style=str(args.visual_style),
        flow_run_dir=(None if args.flow_run_dir is None else Path(args.flow_run_dir)),
        flow_focus_pair=_parse_focus_pair(args.flow_focus_pair),
        flow_fps=int(args.flow_fps),
        flow_frame_stride=int(args.flow_frame_stride),
        flow_speed_um_per_ms=float(args.flow_speed_um_per_ms),
        flow_pulse_sigma_ms=float(args.flow_pulse_sigma_ms),
        flow_syn_delay_ms=args.flow_syn_delay_ms,
        flow_threshold_mV=float(args.flow_threshold_mv),
        flow_max_ms=args.flow_max_ms,
        flow_duration_sec=args.flow_duration_sec,
        neuron_color_overrides=_parse_neuron_color_overrides(args.neuron_color_overrides),
        flow_preserve_camera=bool(args.flow_preserve_camera),
        flow_overlay_style=str(args.flow_overlay_style),
        start_solo_mode=bool(args.start_solo),
        start_neuron_id=args.start_neuron_id,
        neuroglancer_quality=str(args.neuroglancer_quality),
        neuroglancer_voxel_um=args.neuroglancer_voxel_um,
        neuroglancer_max_dim=args.neuroglancer_max_dim,
        neuroglancer_max_voxels=args.neuroglancer_max_voxels,
    )
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
