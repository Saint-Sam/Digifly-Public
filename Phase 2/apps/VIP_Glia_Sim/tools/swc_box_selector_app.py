from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyvista as pv
try:
    from PIL import Image
except Exception:
    Image = None

# Allow running this file directly from any cwd.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.swc_interactive_selector import (
    apply_glia_loss_to_selected_sections,
    load_swc_cell_for_selection,
    section_metadata_table,
)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _decimate_xyz(xyz: np.ndarray, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    if stride <= 1 or xyz.shape[0] <= 3:
        return xyz
    keep = np.r_[0, np.arange(1, xyz.shape[0] - 1, stride), xyz.shape[0] - 1]
    keep = np.unique(keep.astype(int))
    return xyz[keep, :]


def _build_polyline_mesh(cell: Any, section_df, point_stride: int = 3) -> Tuple[pv.PolyData, List[int]]:
    points: List[np.ndarray] = []
    lines: List[int] = []
    section_df_indices: List[int] = []
    point_section_ids: List[np.ndarray] = []
    point_swc_types: List[np.ndarray] = []
    offset = 0

    for df_idx, row in section_df.iterrows():
        sec = row["_sec_obj"]
        pts = getattr(cell, "_cache_sec_pts", {}).get(sec)
        if pts is None or pts.shape[0] < 2:
            continue
        xyz = _decimate_xyz(np.asarray(pts[:, :3], dtype=float), point_stride)
        if xyz.shape[0] < 2:
            continue

        n = int(xyz.shape[0])
        points.append(xyz)
        lines.extend([n, *list(range(offset, offset + n))])
        offset += n

        sec_idx = int(df_idx)
        swc_type = _safe_int(row.get("swc_type", 0), default=0)
        section_df_indices.append(sec_idx)
        point_section_ids.append(np.full(n, sec_idx, dtype=np.int64))
        point_swc_types.append(np.full(n, swc_type, dtype=np.int32))

    if not points:
        raise RuntimeError("No drawable SWC sections found for the selected neuron.")

    mesh = pv.PolyData(np.vstack(points))
    mesh.lines = np.asarray(lines, dtype=np.int64)
    mesh.point_data["section_df_idx"] = np.concatenate(point_section_ids, axis=0)
    mesh.point_data["swc_type"] = np.concatenate(point_swc_types, axis=0)
    return mesh, section_df_indices


def _extract_selected_df_indices(picked: Any) -> Set[int]:
    out: Set[int] = set()
    if picked is None:
        return out

    blocks: Sequence[Any]
    if isinstance(picked, pv.MultiBlock):
        blocks = [b for b in picked if b is not None]
    else:
        blocks = [picked]

    for block in blocks:
        try:
            if int(getattr(block, "n_cells", 0)) < 1 and int(getattr(block, "n_points", 0)) < 1:
                continue
            arr = block.point_data.get("section_df_idx")
            if arr is None:
                arr = block.cell_data.get("section_df_idx")
            if arr is None:
                continue
            out.update(int(x) for x in np.asarray(arr).astype(int).tolist())
        except Exception:
            continue
    return out


def _capture_section_ion_state(sec: Any) -> List[Dict[str, Any]]:
    state: List[Dict[str, Any]] = []
    for seg in sec:
        has_ko = hasattr(seg, "ko")
        has_ek = hasattr(seg, "ek")
        ko_val = float(seg.ko) if has_ko else float("nan")
        ek_val = float(seg.ek) if has_ek else float("nan")
        state.append(
            {
                "has_ko": bool(has_ko),
                "has_ek": bool(has_ek),
                "ko": float(ko_val),
                "ek": float(ek_val),
            }
        )
    return state


def _restore_section_ion_state(sec: Any, state: Sequence[Dict[str, Any]]) -> int:
    restored = 0
    segs = list(sec)
    n = min(len(segs), len(state))
    for i in range(n):
        seg = segs[i]
        s = state[i]
        changed = False
        try:
            if bool(s.get("has_ko", False)) and hasattr(seg, "ko"):
                seg.ko = float(s.get("ko", seg.ko))
                changed = True
        except Exception:
            pass
        try:
            if bool(s.get("has_ek", False)) and hasattr(seg, "ek"):
                seg.ek = float(s.get("ek", seg.ek))
                changed = True
        except Exception:
            pass
        if changed:
            restored += 1
    return int(restored)


class SWCBoxSelectorApp:
    def __init__(
        self,
        cell: Any,
        *,
        ko_mM: float,
        ki_mM: float,
        update_ek: bool,
        point_stride: int,
        output_spec_json: Optional[Path],
        selected_color: str,
        append_output: bool,
        screenshot_scale: float,
        jpeg_quality: int,
        screenshot_dpi: int,
    ):
        self.cell = cell
        self.neuron_id = int(getattr(cell, "gid", -1))
        self.ko_mM = float(ko_mM)
        self.ki_mM = float(ki_mM)
        self.update_ek = bool(update_ek)
        self.output_spec_json = output_spec_json
        self.selected_color = str(selected_color)
        self.append_output = bool(append_output)
        self.screenshot_scale = max(1.0, float(screenshot_scale))
        self.jpeg_quality = int(np.clip(int(jpeg_quality), 50, 100))
        self.screenshot_dpi = int(np.clip(int(screenshot_dpi), 72, 1200))

        self.section_df = section_metadata_table(cell)
        self.selected_df_indices: Set[int] = set()
        self.last_applied_rows: Optional[int] = None
        self._baseline_by_section_name: Dict[str, List[Dict[str, Any]]] = {}
        self._touched_section_names: Set[str] = set()

        self.mesh, _ = _build_polyline_mesh(cell, self.section_df, point_stride=point_stride)
        self.plotter = pv.Plotter(window_size=(1450, 900))
        self._setup_scene()

    def _setup_scene(self) -> None:
        self.plotter.add_mesh(
            self.mesh,
            scalars="swc_type",
            cmap="tab10",
            line_width=3,
            render_lines_as_tubes=True,
            name="morphology",
            pickable=True,
            show_scalar_bar=False,
            opacity=0.7,
        )
        # Keep orientation axes away from the lower-left key legend.
        self.plotter.add_axes(viewport=(0.88, 0.02, 0.995, 0.16))
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.enable_cell_picking(
            callback=self._on_pick_cells,
            through=True,
            show=False,
            # We render our own top-left instruction block with consistent spacing.
            show_message=False,
            style="wireframe",
            color="magenta",
            line_width=3,
            start=False,
            show_frustum=False,
        )

        self.plotter.add_key_event("c", self._clear_selection)
        self.plotter.add_key_event("g", self._apply_glia_loss)
        self.plotter.add_key_event("u", self._revert_selected)
        self.plotter.add_key_event("x", self._revert_all_touched)
        self.plotter.add_key_event("s", self._save_spec)
        self.plotter.add_key_event("k", self._save_jpeg_screenshot)
        self.plotter.add_key_event("p", self._print_selection)
        self.plotter.add_key_event("h", self._print_help)
        self._update_overlay()

    def _selected_rows(self):
        if not self.selected_df_indices:
            return self.section_df.iloc[0:0].copy()
        return self.section_df.loc[sorted(self.selected_df_indices)].copy()

    def _selected_sections(self) -> List[Any]:
        rows = self._selected_rows()
        return [r["_sec_obj"] for _, r in rows.iterrows() if "_sec_obj" in r]

    def _update_overlay(self) -> None:
        n = len(self.selected_df_indices)
        applied = "none" if self.last_applied_rows is None else str(self.last_applied_rows)
        status_txt = (
            'Press "r" then drag a box to select compartments (additive)\n\n'
            f"Neuron: {self.neuron_id}\n\n"
            f"Selected sections: {n}\n\n"
            f"Last glia apply rows: {applied}\n\n"
            f"Touched sections: {len(self._touched_section_names)}"
        )
        help_txt = (
            "Keys\n\n"
            "r: drag-box select (additive)\n"
            "c: clear selection\n"
            "g: apply glia-loss\n"
            "u: undo selected\n"
            "x: undo all touched\n"
            "s: save JSON spec\n"
            "k: save HD JPEG screenshot\n"
            "p: print selection\n"
            "h: help"
        )
        # Use normalized viewport anchors to preserve spacing on resize.
        self.plotter.add_text(
            status_txt,
            position=(0.02, 0.93),
            viewport=True,
            name="status",
            font_size=11,
        )
        self.plotter.add_text(
            help_txt,
            position=(0.02, 0.03),
            viewport=True,
            name="help",
            font_size=10,
        )

        if n == 0:
            try:
                self.plotter.remove_actor("selected", render=False)
            except Exception:
                pass
            return

        point_ids = np.where(
            np.isin(np.asarray(self.mesh.point_data["section_df_idx"]), list(self.selected_df_indices))
        )[0]
        if point_ids.size == 0:
            return
        picked_mesh = self.mesh.extract_points(point_ids, adjacent_cells=True, include_cells=True)
        if int(getattr(picked_mesh, "n_cells", 0)) == 0 and int(getattr(picked_mesh, "n_points", 0)) == 0:
            return
        self.plotter.add_mesh(
            picked_mesh,
            color=self.selected_color,
            line_width=7,
            render_lines_as_tubes=True,
            pickable=False,
            name="selected",
            opacity=1.0,
        )

    def _on_pick_cells(self, picked: Any) -> None:
        picked_ids = _extract_selected_df_indices(picked)
        if not picked_ids:
            return
        self.selected_df_indices.update(picked_ids)
        self._update_overlay()
        self.plotter.render()

    def _clear_selection(self) -> None:
        self.selected_df_indices.clear()
        self._update_overlay()
        self.plotter.render()

    def _apply_glia_loss(self) -> None:
        rows = self._selected_rows()
        sections = [r["_sec_obj"] for _, r in rows.iterrows() if "_sec_obj" in r]
        if not sections:
            print("[glia] No compartments selected. Press 'r' and drag-select first.")
            return
        for _, row in rows.iterrows():
            sec_name = str(row["section_name"])
            sec = row["_sec_obj"]
            if sec_name not in self._baseline_by_section_name:
                self._baseline_by_section_name[sec_name] = _capture_section_ion_state(sec)
        self._touched_section_names.update(str(x) for x in rows["section_name"].tolist())
        df = apply_glia_loss_to_selected_sections(
            self.cell,
            sections,
            ko_mM=self.ko_mM,
            ki_mM=self.ki_mM,
            update_ek=self.update_ek,
        )
        self.last_applied_rows = int(len(df))
        print(
            f"[glia] Applied ko={self.ko_mM:.4g} mM, ki={self.ki_mM:.4g} mM, update_ek={self.update_ek} "
            f"to {len(df)} section(s)."
        )
        show_cols = [c for c in ["section_name", "node_id", "segment_updates", "ko_mM", "ek_mV"] if c in df.columns]
        if show_cols:
            print(df[show_cols].head(15).to_string(index=False))
        self._update_overlay()
        self.plotter.render()

    def _revert_selected(self) -> None:
        rows = self._selected_rows()
        if rows.empty:
            print("[undo] No selected sections to restore.")
            return
        restored_sections = 0
        restored_segments = 0
        for _, row in rows.iterrows():
            sec_name = str(row["section_name"])
            sec = row["_sec_obj"]
            baseline = self._baseline_by_section_name.get(sec_name)
            if baseline is None:
                continue
            restored_segments += _restore_section_ion_state(sec, baseline)
            restored_sections += 1
            self._touched_section_names.discard(sec_name)
        self.last_applied_rows = None
        print(
            f"[undo] Restored {restored_sections} selected section(s) "
            f"({restored_segments} segment value updates) to baseline."
        )
        self._update_overlay()
        self.plotter.render()

    def _revert_all_touched(self) -> None:
        if not self._touched_section_names:
            print("[undo] No touched sections recorded.")
            return
        rows = self.section_df[self.section_df["section_name"].isin(sorted(self._touched_section_names))]
        restored_sections = 0
        restored_segments = 0
        for _, row in rows.iterrows():
            sec_name = str(row["section_name"])
            sec = row["_sec_obj"]
            baseline = self._baseline_by_section_name.get(sec_name)
            if baseline is None:
                continue
            restored_segments += _restore_section_ion_state(sec, baseline)
            restored_sections += 1
        self._touched_section_names.clear()
        self.last_applied_rows = None
        print(
            f"[undo] Restored all touched sections: {restored_sections} section(s), "
            f"{restored_segments} segment value updates."
        )
        self._update_overlay()
        self.plotter.render()

    def _build_glia_loss_spec(self) -> List[Dict[str, Any]]:
        rows = self._selected_rows()
        out: List[Dict[str, Any]] = []
        for _, row in rows.iterrows():
            sec_name = str(row["section_name"])
            out.append(
                {
                    "neuron_id": int(self.neuron_id),
                    "compartment": f"sec:{sec_name}",
                    "ko_mM": float(self.ko_mM),
                    "ki_mM": float(self.ki_mM),
                }
            )
        return out

    def _save_spec(self) -> None:
        if self.output_spec_json is None:
            print("[save] No --output-spec-json path provided.")
            return
        spec = self._build_glia_loss_spec()
        if self.append_output and self.output_spec_json.exists():
            try:
                existing = json.loads(self.output_spec_json.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
            merged = list(existing) + list(spec)
            dedup: Dict[Tuple[int, str], Dict[str, Any]] = {}
            for item in merged:
                try:
                    key = (int(item["neuron_id"]), str(item["compartment"]))
                except Exception:
                    continue
                dedup[key] = item
            spec = list(dedup.values())
        self.output_spec_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_spec_json.write_text(json.dumps(spec, indent=2))
        mode = "append+dedup" if self.append_output else "overwrite"
        print(f"[save] Wrote {len(spec)} entries to: {self.output_spec_json} ({mode})")

    def _save_jpeg_screenshot(self) -> None:
        out_dir = self.output_spec_json.parent if self.output_spec_json is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_jpg = out_dir / f"swc_selector_neuron_{self.neuron_id}_{stamp}.jpg"
        win_w, win_h = [int(v) for v in self.plotter.window_size]
        tgt_w = max(1920, int(round(win_w * self.screenshot_scale)))
        tgt_h = max(1080, int(round(win_h * self.screenshot_scale)))

        try:
            img = self.plotter.screenshot(return_img=True, window_size=(tgt_w, tgt_h))
            if Image is not None:
                Image.fromarray(img).save(
                    out_jpg,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                    subsampling=0,
                    dpi=(self.screenshot_dpi, self.screenshot_dpi),
                )
            else:
                self.plotter.screenshot(str(out_jpg), window_size=(tgt_w, tgt_h))
            print(
                f"[screenshot] Saved HD JPEG: {out_jpg} "
                f"({tgt_w}x{tgt_h}, quality={self.jpeg_quality}, dpi={self.screenshot_dpi})"
            )
            if Image is None:
                print("[screenshot] PIL not available; JPEG DPI metadata may be unavailable.")
            return
        except Exception as e:
            print(f"[screenshot] JPEG export failed ({e}); saving PNG fallback.")

        out_png = out_jpg.with_suffix(".png")
        self.plotter.screenshot(str(out_png), window_size=(tgt_w, tgt_h))
        print(f"[screenshot] Saved PNG fallback: {out_png}")

    def _print_selection(self) -> None:
        rows = self._selected_rows().drop(columns=["_sec_obj"], errors="ignore")
        if rows.empty:
            print("[select] No sections selected.")
            return
        show_cols = [c for c in ["section_name", "node_id", "swc_label", "length_um"] if c in rows.columns]
        print(rows[show_cols].head(30).to_string(index=False))
        if len(rows) > 30:
            print(f"... ({len(rows) - 30} more rows)")

    def _print_help(self) -> None:
        print(
            'Press "r" then drag box to select, "c" clear, "g" apply glia-loss, '
            '"u" undo selected, "x" undo all touched, "s" save JSON, '
            '"k" save HD JPEG screenshot, "p" print.'
        )

    def show(self) -> None:
        self.plotter.show(title=f"SWC Box Selector - Neuron {self.neuron_id}")


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone SWC box selector with drag-select and glia-loss apply.")
    p.add_argument("--neuron-id", type=int, required=True, help="Neuron gid / SWC id.")
    p.add_argument("--swc-dir", type=str, required=True, help="Directory containing SWC files.")
    p.add_argument(
        "--phase2-root",
        type=str,
        default=None,
        help="Path to Phase 2 repo root so `digifly.phase2` imports resolve.",
    )
    p.add_argument("--ko-mM", type=float, default=6.5, help="Target extracellular K+ (mM).")
    p.add_argument("--ki-mM", type=float, default=140.0, help="Intracellular K+ (mM) for EK update.")
    p.add_argument("--update-ek", action="store_true", default=True, help="Also update EK on selected sections.")
    p.add_argument("--no-update-ek", action="store_false", dest="update_ek", help="Do not update EK.")
    p.add_argument("--point-stride", type=int, default=3, help="Point decimation stride for faster rendering.")
    p.add_argument(
        "--selected-color",
        type=str,
        default="#ff0000",
        help="Color used to highlight selected compartments.",
    )
    p.add_argument(
        "--output-spec-json",
        type=str,
        default=None,
        help="Optional path to write GLIA_LOSS_SPEC JSON on key 's'.",
    )
    p.add_argument(
        "--append-output",
        action="store_true",
        help="Append to --output-spec-json and deduplicate by (neuron_id, compartment).",
    )
    p.add_argument(
        "--screenshot-scale",
        type=float,
        default=3.0,
        help="Scale factor for HD screenshot resolution when pressing key 'k'.",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=97,
        help="JPEG quality (50-100) used for key 'k' screenshots.",
    )
    p.add_argument(
        "--screenshot-dpi",
        type=int,
        default=300,
        help="DPI metadata for JPEG screenshots (default 300).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arg_parser().parse_args(argv)

    work_root = Path(__file__).resolve().parents[1]
    if str(work_root) not in sys.path:
        sys.path.insert(0, str(work_root))
    if args.phase2_root:
        phase2_root = Path(args.phase2_root).expanduser().resolve()
        if str(phase2_root) not in sys.path:
            sys.path.insert(0, str(phase2_root))

    cell = load_swc_cell_for_selection(
        neuron_id=int(args.neuron_id),
        swc_dir=Path(args.swc_dir).expanduser().resolve(),
        cfg_overrides={"celsius_C": 22.0, "ais_min_xloc": 0.05},
    )

    app = SWCBoxSelectorApp(
        cell,
        ko_mM=float(args.ko_mM),
        ki_mM=float(args.ki_mM),
        update_ek=bool(args.update_ek),
        point_stride=max(1, int(args.point_stride)),
        output_spec_json=Path(args.output_spec_json).expanduser().resolve() if args.output_spec_json else None,
        selected_color=str(args.selected_color),
        append_output=bool(args.append_output),
        screenshot_scale=float(args.screenshot_scale),
        jpeg_quality=int(args.jpeg_quality),
        screenshot_dpi=int(args.screenshot_dpi),
    )
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
