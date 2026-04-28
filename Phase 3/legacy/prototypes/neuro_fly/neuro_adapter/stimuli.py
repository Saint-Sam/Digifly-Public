from __future__ import annotations
from typing import Dict, Any, List

def parse_stim_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(plan)
    out.setdefault("stimuli", [])
    return out

def eval_waveform(wf: str, params: Dict[str, Any], t_ms: float) -> float:
    wf = wf.lower()
    if wf == "tonic":
        return float(params.get("amp_nA", 0.0))
    elif wf == "square":
        amp = float(params.get("amp_nA", 0.0))
        freq = float(params.get("freq_Hz", 10.0))
        duty = float(params.get("duty", 0.5))
        if freq <= 0.0: return 0.0
        period_ms = 1000.0 / freq
        phase = (t_ms % period_ms) / period_ms
        return amp if phase < duty else 0.0
    elif wf == "ramp":
        a0 = float(params.get("a0_nA", 0.0))
        a1 = float(params.get("a1_nA", 1.0))
        t0 = float(params.get("t0_ms", 0.0))
        t1 = float(params.get("t1_ms", 1000.0))
        if t_ms <= t0: return a0
        if t_ms >= t1: return a1
        return a0 + (a1 - a0) * ((t_ms - t0) / max(1e-6, (t1 - t0)))
    elif wf == "pulses":
        amp = float(params.get("amp_nA", 0.5))
        times = list(params.get("times_ms", []))
        width = float(params.get("width_ms", 5.0))
        for t0 in times:
            if t0 <= t_ms <= t0 + width: return amp
        return 0.0
    else:
        return 0.0

def instantaneous_injection_for_targets(plan: Dict[str, Any], t_ms: float) -> Dict[str, float]:
    currents = {}
    for stim in plan.get("stimuli", []):
        target = stim.get("target")
        wf = stim.get("waveform", "tonic")
        params = stim.get("params", {})
        window = stim.get("window", {})
        t_on = float(window.get("t_on_ms", -1e9))
        t_off = float(window.get("t_off_ms", 1e9))
        if t_on <= t_ms <= t_off:
            currents[target] = currents.get(target, 0.0) + eval_waveform(wf, params, t_ms)
    return currents
