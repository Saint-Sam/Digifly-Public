# Morphology Mutation

New additive workflow for multi-SWC morphology editing, built to avoid breaking existing glia selector/simulation paths.

## Entry points

- Desktop app: `../tools/morphology_mutation_app.py`
- Core API: `../tools/morphology_mutation.py`
- Public smoke-test launcher: `test_launch_morphology_mutation_standalone.ipynb`
- Direct launcher notebook: `launch_morphology_mutation.ipynb`

## What it supports

1. Load multiple SWCs at once.
2. Interactive box-selection on 3D rendered morphology segments.
3. Mutation operations:
- thickening/thinning
- branch growth (tangent-based)
- branch growth to clicked 3D point
- edge split
- detach (creates new roots/components)
- reparent/rejoin (same SWC)
- translation in XYZ
- compartment biophys policy tagging (selected nodes as passive or active)
 - AIS region tagging:
 - one primary AIS node per neuron for Phase 2 `ais_overrides.csv`
 - any number of extra AIS-like locations per neuron
4. Connection-spec authoring between selected nodes:
- chemical synapse count
- gap-junction count
- gap mode (`none`, `non_rectifying`, `rectifying`)
- optional direction metadata
5. Flow visualization:
- load an existing simulation run directory (`config.json` + `records.csv`)
- export an activity-flow movie over the skeleton or neuroglancer-like volume view
- optional pair focus, e.g. `10000 -> 10068`
6. Validation reports:
- connectivity components
- root count
- cycle detection
- missing parent references
- non-positive radius checks
7. Non-destructive saving:
- mutated SWCs saved under a new run folder
- original SWCs are never overwritten

## Save bundle outputs

When pressing `s` in the app, outputs are written under:

`notebooks/debug/outputs/morphology_mutation_<tag>_<timestamp>/`

Includes:

- `mutated_swc/*.swc`
- `phase2_morph_overlay/*.swc` (canonical filenames for Phase 2 loading)
- `morphology_mutation_manifest.json`
- `mutation_connections.json`
- `mutation_connections.csv`
- `mutation_biophys_policies.json`
- `mutation_ais_policies.json`
- `ais_overrides.csv`
- `mutation_validation.json`

## Runtime controls (desktop app)

- `r`: drag-box select (additive)
- `c`: clear selection
- `p`: print selected segments
- `t` / `y`: thin / thicken
- `g`: grow branch along tangent
- `d`: arm draw, then click 3D point to grow branch to that point
- `b`: split selected edges
- `a`: reparent last selected pair (same SWC)
- `x`: detach selected nodes
- `i/k`, `l/o`, `u/n`: translate selected by fixed step in +X/-X, +Y/-Y, +Z/-Z
- `j`: add connection spec from last selected pair
- `f` / `q`: mark selected nodes passive / active for biophys policy
- `e`: assign selected nodes as AIS regions
- `z`: undo
- `v`: validate
- `s`: save bundle
- `w`: toggle skeleton / 3D tubes
- `3`: toggle classic / Vaa3D-like style
- `m`: save white-background photo export
- `0`: export flow movie from the configured run dir; by default the full run is compressed into a smooth 20-second movie with widened rise/decay pulses

## Using mutated SWCs in simulations

Use `phase2_morph_overlay` as `morph_swc_dir` in simulation configs when the loaded neuron set matches the mutation-app neuron set.

This keeps baseline source SWCs untouched while letting Phase 2 resolve canonical per-neuron SWC names.
If an AIS primary node was assigned in the mutation app, the saved bundle also writes `ais_overrides.csv` into the overlay so Phase 2 can pick it up automatically.

Notebook helper module:

- `../tools/morphology_mutation_notebook_helpers.py`

Main helper:

- `build_sim_overrides_from_mutation_manifest(manifest_path, base_overrides=...)`

Direct connectivity overlay helper:

- `build_forced_chem_edges_from_mutation_connections(...)`
- Produces a `_forced_chem_only_edges`-style CSV for `edges_path` or `edges_csv`.
