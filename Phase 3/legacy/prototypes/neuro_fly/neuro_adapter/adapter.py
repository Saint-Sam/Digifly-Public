from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

from .env_loader import load_env, actuator_index_map, control_dt_seconds
from .filters import ActivationFilter
from .video import VideoWriter

DEFAULT_R_MAX = 200.0  # Hz for rate normalization

class NeuroMuJoCoAdapter:
    def __init__(self, mode: str, mapping: Dict[str, Any], bridge, render_fps: int = 60, video_path: str | None = None, render_size=(720, 720)):
        self.mode = mode
        self.mapping = mapping
        self.bridge = bridge
        self.render_fps = int(render_fps)
        self.video_path = video_path
        self.render_size = render_size

        self.env = load_env(mode=mode)
        self.actu_index = actuator_index_map(self.env)
        self.action_size = self.env.action_spec().shape[0]

        self.control_dt = control_dt_seconds(self.env)  # seconds
        self.neural_dt = float(mapping.get("global", {}).get("neural_dt_ms", 0.025)) * 1e-3  # seconds
        self.steps_per_action = max(1, int(round(self.control_dt / self.neural_dt)))
        self.clip_actions = bool(mapping.get("global", {}).get("clip_actions", True))

        self._filters: Dict[str, ActivationFilter] = {}
        for spec in mapping.get("actuators", []):
            pool = spec["pool"]
            if pool not in self._filters:
                tau_d = float(spec.get("tau_decay_ms", 10.0))
                tau_r = float(spec.get("tau_rise_ms", 2.0))
                kappa = float(spec.get("kappa", 0.05))
                self._filters[pool] = ActivationFilter(tau_d, tau_r, kappa)

        self.vw = VideoWriter(video_path, fps=self.render_fps) if self.video_path else None

        self._validate_mapping()

    def _validate_mapping(self):
        for spec in self.mapping.get("actuators", []):
            name = spec["actuator"]
            if name not in self.actu_index:
                raise ValueError(f"Actuator '{name}' not found in environment. Check names in your mapping file.")
        avail = set(self.bridge.pools())
        missing = [spec["pool"] for spec in self.mapping.get("actuators", []) if spec["pool"] not in avail]
        if missing:
            print(f"[WARN] Pools missing in bridge (will be zero unless provided): {missing[:10]}{' ...' if len(missing)>10 else ''}")

    def run(self, stim_plan: Dict[str, Any], duration_s: float = 5.0, render_cam: int | str | None = None):
        ts = self.env.reset()
        self.bridge.set_stim_plan(stim_plan)

        t_ms = 0.0
        end_ms = float(duration_s) * 1000.0
        control_dt_ms = self.control_dt * 1000.0
        neural_dt_ms = self.neural_dt * 1000.0

        last_render_t = 0.0
        render_period = (1000.0 / self.render_fps) if self.vw else None

        while t_ms < end_ms:
            self.bridge.window_reset()

            for _ in range(self.steps_per_action):
                self.bridge.step(neural_dt_ms)
                t_ms += neural_dt_ms

            pool_names = [spec["pool"] for spec in self.mapping.get("actuators", [])]
            readout = self.bridge.readout(pool_names)

            activ = {}
            for spec in self.mapping.get("actuators", []):
                pool = spec["pool"]
                flt = self._filters[pool]
                ro = readout.get(pool, {"mode": "rate", "value": 0.0})
                if ro.get("mode") == "spikes":
                    times = ro.get("times_ms", [])
                    a = flt.update_from_spikes(times, t_ms, t_ms - control_dt_ms)
                else:
                    rate = float(ro.get("value", 0.0))
                    a = flt.update_from_rate(rate, control_dt_ms, r_max=spec.get("r_max", DEFAULT_R_MAX))
                activ[pool] = a

            u = np.zeros(self.action_size, dtype=np.float64)
            for spec in self.mapping.get("actuators", []):
                name = spec["actuator"]
                idx = self.actu_index[name]
                typ = spec.get("type", "torque").lower()
                if typ == "torque":
                    gain = float(spec.get("gain", 1.0))
                    sign = float(spec.get("sign", 1.0))
                    u[idx] += gain * sign * activ.get(spec["pool"], 0.0)
                elif typ == "position":
                    angle0 = float(spec.get("angle0_deg", 0.0))
                    alpha = float(spec.get("alpha_deg", 30.0))
                    a = activ.get(spec["pool"], 0.0)
                    u[idx] = angle0 + alpha * a
                elif typ == "tendon":
                    gain = float(spec.get("gain", 1.0))
                    u[idx] += gain * activ.get(spec["pool"], 0.0)
                elif typ == "adhesion":
                    a = activ.get(spec["pool"], 0.0)
                    k = float(spec.get("k", 10.0))
                    x0 = float(spec.get("x0", 0.5))
                    fmax = float(spec.get("fmax", 6.0))
                    u[idx] = fmax / (1.0 + np.exp(-k * (a - x0)))
                else:
                    pass

            if self.clip_actions:
                spec_act = self.env.action_spec()
                u = np.clip(u, spec_act.minimum, spec_act.maximum)

            ts = self.env.step(u)

            if self.vw is not None:
                if (render_period is None) or ((t_ms - last_render_t) >= render_period):
                    try:
                        H, W = self.render_size
                        frame = self.env.physics.render(height=H, width=W, camera_id=render_cam)
                        self.vw.add(frame)
                        last_render_t = t_ms
                    except Exception as e:
                        print(f"[WARN] Render failed: {e}")

        if self.vw is not None:
            self.vw.close()
