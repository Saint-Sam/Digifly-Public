from __future__ import annotations
import argparse, json, yaml, os
from typing import Dict, Any, Optional

from neuro_adapter.adapter import NeuroMuJoCoAdapter
from neuron_bridge.bridge import NeuronBridge  # Replace with your real bridge when ready

def run_episode(
    mode: str,
    mapping_yaml: str,
    stim_plan: Dict[str, Any] | str,
    duration_s: float,
    video_path: Optional[str] = None,
    render_fps: int = 60,
    render_size=(720, 720),
) -> None:
    with open(mapping_yaml, "r") as f:
        mapping = yaml.safe_load(f)

    if isinstance(stim_plan, str) and os.path.exists(stim_plan):
        with open(stim_plan, "r") as f:
            stim_plan = yaml.safe_load(f)
    elif isinstance(stim_plan, str):
        try:
            stim_plan = json.loads(stim_plan)
        except Exception:
            raise ValueError("stim_plan must be a dict, path to YAML, or JSON string.")

    bridge = NeuronBridge(config={})
    adapter = NeuroMuJoCoAdapter(
        mode=mode,
        mapping=mapping,
        bridge=bridge,
        render_fps=render_fps,
        video_path=video_path,
        render_size=render_size,
    )
    adapter.run(stim_plan=stim_plan, duration_s=duration_s)

def main():
    ap = argparse.ArgumentParser(description="Run NEURON→MuJoCo episode (no muscles, no feedback)")
    ap.add_argument("--mode", type=str, default="walk", help="walk | walk_on_ball | flight")
    ap.add_argument("--mapping", type=str, required=True, help="Path to mapping.yaml")
    ap.add_argument("--stim", type=str, required=True, help="Path to stim YAML (or JSON string)")
    ap.add_argument("--duration", type=float, default=5.0, help="Seconds")
    ap.add_argument("--video", type=str, default=None, help="Output MP4 path")
    ap.add_argument("--fps", type=int, default=60, help="Render FPS")
    ap.add_argument("--width", type=int, default=720, help="Render width")
    ap.add_argument("--height", type=int, default=720, help="Render height")
    args = ap.parse_args()

    run_episode(
        mode=args.mode,
        mapping_yaml=args.mapping,
        stim_plan=args.stim,
        duration_s=args.duration,
        video_path=args.video,
        render_fps=args.fps,
        render_size=(args.height, args.width),
    )

if __name__ == "__main__":
    main()
