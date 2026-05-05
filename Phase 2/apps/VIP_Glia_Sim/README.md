# VIP Glia Morphology Mutation App

Standalone Phase 2 morphology mutation app.

This copy preserves the current morphology mutation app behavior while keeping it separate from the older `digifly.phase2.extensions.glia_editing` package.

## Entry Points

- `notebooks/test_launch_morphology_mutation_standalone.ipynb`: public smoke-test launcher for the copied standalone app.
- `notebooks/launch_morphology_mutation.ipynb`: copied direct launcher, patched to resolve this public app root.
- `tools/morphology_mutation_app.py`: PyVista desktop app.
- `tools/morphology_mutation.py`: mutation backend, SWC mutation bundle writing, Phase 2 overlay writing.
- `tools/morphology_mutation_notebook_helpers.py`: helpers for using saved mutation bundles in simulations.

Selector support is included too:

- `notebooks/launch_swc_box_selector.ipynb`
- `notebooks/interactive_swc_compartment_selector.ipynb`
- `tools/swc_box_selector_app.py`
- `tools/swc_interactive_selector.py`

## Runtime Paths

The launch notebooks resolve paths in this order:

- `DIGIFLY_VIP_GLIA_ROOT`, otherwise this copied app folder
- `DIGIFLY_PHASE2_ROOT`, otherwise the surrounding `Phase 2` folder
- `DIGIFLY_SWC_DIR` or `notebooks/local_config.py`
- `Phase 1/manc_v1.2.1/export_swc`
- other public `Phase 1/*/export_swc` folders when they contain the requested neuron IDs

If the requested SWCs are missing from Digifly Public, the launcher seeds the local MANC cache by copying matching neuron folders from `DIGIFLY_SOURCE_SWC_DIR` or `FALLBACK_SWC_DIR`.

For flow visualization, the launcher auto-searches finished Phase 2 runs in this order:

- `DIGIFLY_FLOW_RUNS_ROOTS` / `FLOW_RUNS_ROOTS`
- `DIGIFLY_FLOW_RUNS_ROOT` / `FLOW_RUNS_ROOT`
- `SWC_DIR/hemi_runs`
- `Phase 1/manc_v1.2.1/export_swc/hemi_runs`
- `Phase 2/data/export_swc/hemi_runs`
- `notebooks/debug/runs`

Set `DIGIFLY_FLOW_RUN_DIR` or `FLOW_RUN_DIR` to force a specific run directory.

Press `0` in the app to export the flow movie. The default export uses the full simulation time span and compresses it into a 20-second, 30-fps movie with a widened rise/decay pulse; set `--flow-duration-sec`, `--flow-fps`, `--flow-pulse-sigma-ms`, `--flow-speed-um-per-ms`, or `--flow-max-ms` to override that behavior. The same flow overlay works in skeleton and neuroglancer-like volume render modes.

`Phase 2/data/export_swc` is intentionally ignored by git. For real testing, point `DIGIFLY_SWC_DIR` at a local Phase 1 SWC export folder, or copy `notebooks/local_config.example.py` to the ignored `notebooks/local_config.py` and set `SWC_DIR`.

The launch notebooks validate requested neuron IDs before starting the desktop process, so a missing SWC folder now fails with an actionable path error instead of silently launching and exiting.

## Generated Outputs

Runtime outputs are ignored by git and should stay local:

- `notebooks/debug/outputs/`
- `notebooks/debug/runs/`
- mutation bundle folders such as `morphology_mutation_<tag>_<timestamp>/`

Saved mutation bundles contain the useful handoff artifacts:

- `mutated_swc/*.swc`
- `phase2_morph_overlay/*.swc`
- `morphology_mutation_manifest.json`
- `mutation_connections.json`
- `mutation_connections.csv`
- `mutation_biophys_policies.json`
- `mutation_ais_policies.json`
- `ais_overrides.csv`
- `mutation_validation.json`

Use `phase2_morph_overlay` as `morph_swc_dir` in Phase 2 simulation configs.
