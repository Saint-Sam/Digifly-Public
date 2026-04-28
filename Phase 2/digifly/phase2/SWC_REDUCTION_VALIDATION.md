# SWC Reduction + Validation Guide

This document defines how reduced morphologies are generated and how to validate that reduction preserves simulation behavior.

## 1) Non-destructive policy

- Original SWCs in `export_swc/` are never modified.
- Reduced SWCs are written to a separate root (for example `export_swc_reduced/v1/`).
- Relative folder structure is preserved.
- Each reduction run writes:
  - `_swc_reduction_summary.csv`
  - `_swc_reduction_manifest.json`
  - optional per-neuron `.map.csv` (old node id -> reduced representative node id)

## 2) Reduction method (current implementation)

Tool:
- `python -m digifly.tools.reduce_swc_dataset`

Core idea:
- Preserve topology and biologically important points.
- Merge only linear chains between protected points.

Protected by default:
- root nodes
- branch points
- terminal nodes
- soma-type nodes (`type == 1`)
- nodes with strong parent/child radius jump (`max_diam_rel`)
- nearest nodes to synapse coordinates (default enabled)

Chain merge controls:
- `max_path_um`: keep a node if path distance since last kept node exceeds threshold
- `max_turn_deg`: keep a node at strong geometric bends
- `max_diam_rel`: keep nodes with significant relative radius change

This is a topology-preserving geometric simplification pass, not a full cable-theory optimization.

## 3) Why this is reasonable for large datasets

For large-scale network simulation (20k to 160k+ neurons), this approach gives:
- predictable memory/runtime reduction
- deterministic output
- robust preservation of branch topology and key wiring anchors

It is intentionally conservative around wiring and branch structure.

## 4) Using reduced SWCs in simulation

`swc_dir` remains the original dataset root (for edges/synapse catalogs).

Use:
- `morph_swc_dir` = reduced SWC root

In `run_simulation.ipynb`:
- `SWC_DIR` -> original `export_swc`
- `MORPH_SWC_DIR` -> reduced root (or `None` to use originals)

## 5) Validation protocol (required)

Run paired simulations with identical settings except morphology source:

1. Baseline run:
- original SWCs (`MORPH_SWC_DIR=None`)
- save run id (for example `run_baseline_orig`)

2. Reduced run:
- reduced SWCs (`MORPH_SWC_DIR=/.../export_swc_reduced/v1`)
- save run id (for example `run_reduced_v1`)

3. Compare outputs using the notebook validation cell:
- first spike time per tracked neuron
- pathway latency for key pairs (for example `10000->10110`, `10002->10068`)
- peak soma voltage per tracked neuron
- trace RMSE (baseline time grid, reduced interpolated)

Suggested acceptance thresholds (edit per use case):
- `first_spike_abs_diff_ms <= 0.20`
- `latency_abs_diff_ms <= 0.20`
- `peak_abs_diff_mV <= 8.0`
- `trace_rmse_mV <= 5.0`

If any threshold fails, review morphology reduction aggressiveness and rerun.

## 6) Recommended rollout

1. Pilot on a subset of neurons/runs.
2. Validate behavior against baseline.
3. Freeze reducer settings in manifest (`v1`).
4. Batch-generate full reduced dataset.
5. Revalidate on multiple circuits before broad use.

## 7) Reproducibility checklist

- Record:
  - reducer version (git commit)
  - exact reducer args/thresholds
  - input root checksum/version
  - output manifest path
  - validation run ids and metrics
- Keep reduced datasets versioned (`v1`, `v2`, ...), never overwrite without explicit version bump.

## 8) Limitations

- This method does not yet optimize electrotonic equivalence explicitly.
- It is geometric/topological and synapse-aware, but not a full multi-objective cable reduction.
- For highly sensitive analyses, tighten thresholds or disable reduction in critical neurons.
