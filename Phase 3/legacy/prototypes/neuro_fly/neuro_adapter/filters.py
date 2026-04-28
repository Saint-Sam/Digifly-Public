from __future__ import annotations
import math
from typing import List

class ActivationFilter:
    def __init__(self, tau_decay_ms: float = 10.0, tau_rise_ms: float = 2.0, kappa: float = 0.05, a0: float = 0.0):
        self.tau_decay = max(1e-6, tau_decay_ms)
        self.tau_rise = max(1e-6, tau_rise_ms)
        self.kappa = kappa
        self.a = max(0.0, min(1.0, a0))

    def reset(self, a0: float = 0.0):
        self.a = max(0.0, min(1.0, a0))

    def update_from_rate(self, rate_hz: float, dt_ms: float, r_max: float = 200.0):
        rate_term = (rate_hz / max(1e-6, r_max))
        da = (-self.a / self.tau_decay + self.kappa * rate_term) * dt_ms
        self.a += da
        self.a = max(0.0, min(1.0, self.a))
        return self.a

    def update_from_spikes(self, spike_times_ms: List[float], window_end_ms: float, window_start_ms: float):
        dt_ms = (window_end_ms - window_start_ms)
        decay_factor = math.exp(-dt_ms / self.tau_decay) if dt_ms > 0 else 1.0
        a_end = self.a * decay_factor
        for t in spike_times_ms:
            if window_start_ms <= t <= window_end_ms:
                age = window_end_ms - t
                a_end += self.kappa * math.exp(-age / self.tau_rise)
        self.a = max(0.0, min(1.0, a_end))
        return self.a
