# neuro_fly — NEURON → MuJoCo adapter (no muscles, no feedback)

This scaffold lets you run a **ventral nerve cord (VNC)** neural model and directly drive the
**flybody** MuJoCo model using torque/position/tendon/adhesion actuators — **no muscles yet**.
You provide which cells to stimulate; the adapter handles timing, mapping motor pools to actuators,
and writes an **MP4** of the behavior.

> Status: Minimal, testable scaffold. Drop in your real NEURON/NetPyNE bridge in `neuron_bridge/bridge.py`
> (see the `DummyBridge` for the interface).

## Quick start

1. Install deps (Python ≥3.10):
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure your `flybody` package is importable (e.g., `export PYTHONPATH=/path/to/flybody-main/src:$PYTHONPATH`
   or `pip install -e /path/to/flybody-main`).

3. Edit `neuro_adapter/mapping.example.yaml` to match your motor pools and your env's actuator names.
   Copy it as `mapping.yaml` once ready.

4. Run a demo episode (uses a DummyBridge that synthesizes activations so you can test the mapping/video):
   ```bash
   python run_neuro_fly.py --mode walk        --mapping neuro_adapter/mapping.example.yaml        --stim examples/stim_walk.yaml        --duration 5.0        --video /tmp/walk_demo.mp4        --fps 60
   ```

## File layout

```
neuro_fly/
  run_neuro_fly.py              # one-block entry (CLI + importable run_episode())
  neuro_adapter/
    adapter.py                  # stepper loop (NEURON ↔ actions ↔ env)
    env_loader.py               # loads flybody envs + actuator introspection
    filters.py                  # spike/rate → activation smoothing
    stimuli.py                  # simple stimulus waveform utilities
    video.py                    # mp4 writer (imageio)
    mapping.example.yaml        # mapping template: pools → actuators
  neuron_bridge/
    bridge.py                   # DummyBridge + interface; replace with your NEURON/NetPyNE
  examples/
    stim_walk.yaml
    stim_flight.yaml
  requirements.txt
  README.md
```

## Interface you must implement (later)

Replace `DummyBridge` with your own class exposing:

- `pools(self) -> list[str]`
- `set_stim_plan(self, stim_plan: dict) -> None`
- `step(self, dt_ms: float) -> None`  # advance neural sim by dt_ms
- `window_reset(self) -> None`        # mark start of a control window
- `readout(self, pool_names: list[str]) -> dict[str, dict]`

`readout` returns for each pool either:
```python
{ "mode": "rate", "value": float }  # Hz normalized or raw 0..1
# or
{ "mode": "spikes", "times_ms": [t1, t2, ...] }  # spikes in the last control window
```

The adapter converts this to **activation a∈[0,1]** using `filters.py`.
