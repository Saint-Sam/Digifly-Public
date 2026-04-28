from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyvista as pv

try:
    from PIL import Image
except Exception:
    Image = None

_PHASE2_ROOT = Path(__file__).resolve().parents[5]

from .morphology_mutation import MorphologyMutationProject


SelectionKey = Tuple[int, int]  # (neuron_id, child_node_id)


def _hex_from_rgb01(rgb: Sequence[float]) -> str:
    r = int(np.clip(round(float(rgb[0]) * 255.0), 0, 255))
    g = int(np.clip(round(float(rgb[1]) * 255.0), 0, 255))
    b = int(np.clip(round(float(rgb[2]) * 255.0), 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


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
        if self.render_mode not in {"tube", "skeleton"}:
            self.render_mode = "tube"
        self.skeleton_line_width = max(1.0, float(skeleton_line_width))

        self.screenshot_scale = max(1.0, float(screenshot_scale))
        self.jpeg_quality = int(np.clip(int(jpeg_quality), 50, 100))
        self.screenshot_dpi = int(np.clip(int(screenshot_dpi), 72, 1200))

        self.mesh: Optional[pv.PolyData] = None
        self.segment_df = None
        self.root_points: Optional[pv.PolyData] = None
        self.neuron_colors: Dict[int, str] = _color_map_for_ids(self.project.neuron_ids)

        self.selected_keys: Set[SelectionKey] = set()
        self.selection_order: List[SelectionKey] = []
        self.pending_draw_anchor: Optional[SelectionKey] = None

        self.undo_stack: List[Dict[str, Any]] = []
        self.last_action: str = "none"

        self.plotter = pv.Plotter(window_size=(1600, 980))
        self._setup_scene()

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

    def _setup_scene(self) -> None:
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes(viewport=(0.88, 0.02, 0.995, 0.16))
        self.plotter.add_slider_widget(
            callback=self._on_skeleton_width_slider,
            rng=(1.0, 14.0),
            value=float(self.skeleton_line_width),
            title="Skeleton Width",
            pointa=(0.68, 0.03),
            pointb=(0.96, 0.03),
            style="modern",
        )

        self._rebuild_scene(first_time=True)

        self.plotter.enable_cell_picking(
            callback=self._on_pick_cells,
            through=True,
            # Show the picked region immediately so rectangle selection is
            # visually obvious while editing.
            show=True,
            show_message=False,
            style="wireframe",
            color="magenta",
            line_width=3,
            start=False,
            show_frustum=True,
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

        self._update_overlay()

    def _rebuild_scene(self, *, first_time: bool = False) -> None:
        cam = None if first_time else self.plotter.camera_position
        self.mesh, self.segment_df, self.root_points = self._build_mesh()

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

        # Hidden-ish pick mesh (line segments) keeps robust cell-picking behavior.
        self.plotter.add_mesh(
            self.mesh,
            color="#ffffff",
            line_width=1,
            render_lines_as_tubes=False,
            name="morphology",
            pickable=True,
            show_scalar_bar=False,
            opacity=0.03,
        )

        # Render morphology with either 3D tubes or colored skeleton lines.
        for nid in self.project.neuron_ids:
            idxs = self.segment_df.loc[self.segment_df["neuron_id"] == int(nid), "segment_idx"].to_numpy(dtype=np.int64)
            if idxs.size == 0:
                continue
            sub_raw = self.mesh.extract_cells(idxs)
            # extract_cells returns UnstructuredGrid; convert to PolyData for tube filter.
            sub = sub_raw.extract_surface().clean()
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
            else:
                radii = np.asarray(sub.cell_data.get("radius_um", np.array([], dtype=float)), dtype=float)
                med_r = float(np.nanmedian(radii)) if radii.size else 0.2
                tube_r = max(0.18, min(3.5, med_r * 2.0))
                try:
                    geom = sub.tube(radius=tube_r, n_sides=18, capping=True)
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
        self._update_selected_actor()

        if cam is not None:
            self.plotter.camera_position = cam

    def _sanitize_selection(self) -> None:
        valid = set((int(r["neuron_id"]), int(r["child_node_id"])) for _, r in self.segment_df.iterrows())
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
        picked = picked_raw.extract_surface().clean()
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
        passive_total = 0
        active_total = 0
        for pol in self.project.biophys_policies.values():
            passive_total += len(pol.get("passive", set()))
            active_total += len(pol.get("active", set()))
        status = (
            "Morphology Mutation (multi-SWC)\n\n"
            f"Loaded neuron IDs: {self.project.neuron_ids}\n\n"
            f"Selected segments: {nsel}\n\n"
            f"Connection specs: {len(self.project.connections)}\n\n"
            f"Biophys policy nodes: passive={passive_total}, active={active_total}\n\n"
            f"Render mode: {self.render_mode} (line width {self.skeleton_line_width:.1f})\n\n"
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
            "w: toggle skeleton <-> 3D\n"
            "z: undo\n"
            "v: validate\n"
            "s: save bundle\n"
            "m: save cropped transparent screenshot (png+jpg+pdf)\n"
            "h: help"
        )

        self.plotter.add_text(
            status,
            position=(0.02, 0.93),
            viewport=True,
            name="status",
            font_size=11,
        )
        self.plotter.add_text(
            keys,
            position=(0.02, 0.03),
            viewport=True,
            name="keys",
            font_size=10,
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
        self.render_mode = "skeleton" if self.render_mode == "tube" else "tube"
        self.last_action = f"render_mode={self.render_mode}"
        self._rebuild_scene()
        self._update_overlay()
        self.plotter.render()
        print(f"[view] render mode -> {self.render_mode}")

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

        self.last_action = "save_bundle"
        self._update_overlay()
        self.plotter.render()
        print("[save] bundle written")
        print(json.dumps(out, indent=2))

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
            try:
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
                    transparent_background=True,
                )
            finally:
                try:
                    self.plotter.camera_position = cam_before
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
                pil_rgba.save(
                    out_png,
                    format="PNG",
                    optimize=True,
                    dpi=(self.screenshot_dpi, self.screenshot_dpi),
                )

                # JPEG has no alpha; composite transparent pixels to black instead of white.
                bg = Image.new("RGB", pil_rgba.size, (0, 0, 0))
                bg.paste(pil_rgba, mask=pil_rgba.getchannel("A"))
                bg.save(
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
                    fig.patch.set_alpha(0.0)
                    fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
                    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    ax.patch.set_alpha(0.0)
                    ax.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
                    ax.imshow(rgba, interpolation="nearest")
                    save_kwargs = dict(
                        dpi=float(self.screenshot_dpi),
                        transparent=True,
                        facecolor="none",
                        edgecolor="none",
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
                f"(cropped to neuron bounds, transparent bg, dpi={self.screenshot_dpi}, quality={self.jpeg_quality})"
            )
            print(
                f"[screenshot] stroke SVG: {out_svg} "
                f"(one editable line per segment; stroke width preserved). "
                f"Raster fallback SVG: {out_svg_raster}. Vector PDF: {out_pdf_vec}"
            )
            if n_strokes is not None:
                print(f"[screenshot] custom stroke count = {n_strokes}")
            return
        except Exception as e:
            print(f"[screenshot] jpeg export failed ({e}); saving png fallback")

        self.plotter.screenshot(str(out_png), window_size=(tgt_w, tgt_h), transparent_background=True)
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
            "  f mark selected passive | q mark selected active\n"
            "  w toggle skeleton<->3D (use slider for skeleton width)\n"
            "  z undo | v validate | s save bundle | m screenshot (cropped transparent png+jpg+pdf)"
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

    default_output = (_PHASE2_ROOT / "outputs" / "glia_editing" / "mutation").resolve()
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
    p.add_argument("--render-mode", type=str, default="tube", choices=["tube", "skeleton"])
    p.add_argument("--skeleton-line-width", type=float, default=4.0)

    p.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="Allow multiple connected components during validation.",
    )

    p.add_argument("--screenshot-scale", type=float, default=3.0)
    p.add_argument("--jpeg-quality", type=int, default=97)
    p.add_argument("--screenshot-dpi", type=int, default=300)
    return p



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arg_parser().parse_args(argv)

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
    )
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
