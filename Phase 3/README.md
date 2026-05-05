# Phase 3 Working Copy

This is the cleaned Phase 3 workspace to use going forward.

## Recommended entry point

Open:

- `notebooks/user/Phase 3 - NEURON to MuJoCo.ipynb`

That notebook is the best current baseline for:

1. Loading a Phase 2 run
2. Converting motor spikes into actuator controls
3. Applying a named signal/render profile
4. Rendering a MuJoCo video
5. Running an interactive hemilineage flow that can prompt for `Hemi_XXX` and a run folder

## What changed in this working copy

- The current folder was backed up to `../Phase 3_OLD`.
- Legacy root-level clutter was moved under `legacy/`.
- The main notebook resolves paths from this workspace instead of older local workspace folders.
- Profile `control_map` settings now affect MuJoCo scaling during render.
- Hemilineage mode now filters `spike_times.csv` through `added_motor_neuron_ids.csv` before building controls.
- Outputs now default to `data/.../<save_tag>/<run_name>/`, with `save_tag` defaulting to the `Hemi_XXX` folder name.
- A rebuilt rule-based mapping now lives at `data/inputs/mappings/mn_to_actuator_mapping_rebuilt.csv`.
- Mapping rebuild audit outputs now land under `data/derived/mapping_rebuild/`, including the unresolved list and the `Hemi_09A` baseline coverage report.

## Mapping rebuild

Rebuild the Phase 3 mapping with:

```bash
python3 scripts/rebuild_phase3_mapping.py
```

That script regenerates the main mapping CSV plus:

- `data/derived/mapping_rebuild/mn_to_actuator_mapping_rebuilt_unresolved.csv`
- `data/derived/mapping_rebuild/mapping_rebuild_type_coverage.csv`
- `data/derived/mapping_rebuild/hemi_09a_baseline_audit.csv`

## Which legacy notebook is the better reference?

If you need to inspect the old notebooks:

- `Phase 3 v2.0.ipynb` is the better legacy reference.
- `Phase 3 - NEURON to MuJoCo.ipynb` is much more bloated and includes many abandoned experiments, duplicate cells, and old mapping attempts.

Both are archived for reference only. The working notebook in `notebooks/user` should be the one we improve from here.

## Keep vs archive

Keep using:

- `notebooks/user`
- `notebooks/debug`
- `src/phase3_bridge`
- `configs`
- `data/inputs`
- `data/derived`
- `data/outputs`

Treat as legacy/reference:

- `legacy/root_notebooks`
- `legacy/root_inputs`
- `legacy/reference`
- `legacy/prototypes`

## Dependencies

Install the notebook/runtime dependencies with:

```bash
pip install -r requirements.txt
```

MuJoCo rendering also needs a working local graphics setup or an appropriate `MUJOCO_GL` backend.
