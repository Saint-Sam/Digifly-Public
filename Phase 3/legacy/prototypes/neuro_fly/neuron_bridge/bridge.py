"""
DummyBridge + interface you can replace with your NEURON/NetPyNE hookup.
For now, it synthesizes smooth activations for any pool names the adapter asks for.
"""
from __future__ import annotations
import math
from typing import Dict, Any, List

class DummyBridge:
    def __init__(self, config: Dict[str, Any] | None = None):
        self._stim_plan: Dict[str, Any] | None = None
        self._t_ms = 0.0
        self._window_start_ms = 0.0
        self._pools_known: set[str] = set()

    def pools(self) -> List[str]:
        return list(self._pools_known)

    def set_stim_plan(self, stim_plan: Dict[str, Any]):
        self._stim_plan = stim_plan

    def window_reset(self):
        self._window_start_ms = self._t_ms

    def step(self, dt_ms: float):
        self._t_ms += float(dt_ms)

    def readout(self, pool_names: List[str]) -> Dict[str, Dict[str, Any]]:
        out = {}
        self._pools_known.update(pool_names)
        for i, name in enumerate(pool_names):
            phase = (i * 0.37) % 1.0
            rate = 75.0 * (1.0 + math.sin(2*math.pi*0.5*(self._t_ms/1000.0) + 2*math.pi*phase))
            out[name] = {"mode": "rate", "value": max(0.0, rate)}
        return out

NeuronBridge = DummyBridge
