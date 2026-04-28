# VIP Glia Morphology Mutation App

Standalone Phase 2 app imported from `Digifly_NEW/VIP_Glia_Sim`.

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
- `DIGIFLY_SWC_DIR`, otherwise `Phase 2/data/export_swc`

`Phase 2/data/export_swc` is intentionally ignored by git. For real testing, point `DIGIFLY_SWC_DIR` at a local Phase 1 SWC export folder or copy data into the ignored local location.

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
