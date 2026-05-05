# Mutation App Copy Manifest

Date: 2026-04-27

Public destination:

- `Phase 2/apps/VIP_Glia_Sim`

This manifest documents the standalone morphology mutation app bundle and the files required for the launcher notebooks.

## App Bundle

Required runtime files:

- `tools/morphology_mutation_app.py`
  Main desktop app entry point.

- `tools/morphology_mutation.py`
  Backend mutation project, SWC loading, save bundle logic, manifest writing, and Phase 2 overlay writing.

- `tools/morphology_mutation_notebook_helpers.py`
  Notebook helper functions for reading mutation manifests, connection specs, AIS/biophysics sidecars, and simulation overrides.

- `tools/__init__.py`
  Package marker for clean imports.

Required launcher/testing notebooks:

- `notebooks/launch_morphology_mutation.ipynb`
  Standard standalone launcher.

- `notebooks/test_launch_morphology_mutation_standalone.ipynb`
  Minimal standalone launch test.

- `notebooks/README_MORPHOLOGY_MUTATION.md`
  App usage notes.

## Optional Selector Workflow

Selector-side files are optional and can be included when the SWC box selector workflow is needed:

- `tools/swc_box_selector_app.py`
- `tools/swc_interactive_selector.py`
- `notebooks/launch_swc_box_selector.ipynb`
- `notebooks/README_INTERACTIVE_SELECTOR.md`

## Runtime Paths

The launcher resolves paths relative to the public repository whenever possible:

- App root: `Phase 2/apps/VIP_Glia_Sim`
- App script: `Phase 2/apps/VIP_Glia_Sim/tools/morphology_mutation_app.py`
- Public MANC SWC cache: `Phase 1/manc_v1.2.1/export_swc`
- Public MANC run cache: `Phase 1/manc_v1.2.1/export_swc/hemi_runs`
- Phase 2 root: `Phase 2`
- Default output root: `Phase 2/apps/VIP_Glia_Sim/notebooks/debug/outputs`
- Default flow runs root: `Phase 2/apps/VIP_Glia_Sim/notebooks/debug/runs`

Additional SWC source roots may be configured through `local_config.py` or environment variables. Private source locations are not required by the public bundle.

The launcher searches `SWC_DIR/hemi_runs` and the public MANC run cache before the older debug run folder, so Docker/Jupyter simulation results remain visible to the mutation app from the mounted repository.

Flow movie export defaults to the full simulation span compressed into 20 seconds at 30 fps, with a widened rise/decay pulse profile for demo visibility. The overlay is available in both skeleton and neuroglancer-like volume render modes.

## External Dependencies

Required Python packages:

- `numpy`
- `pandas`
- `pyvista`

Recommended Python packages:

- `pillow`
- `matplotlib`

Optional packages for helper paths that read parquet/feather:

- `pyarrow`

## Flow Overlay Input

When launching with `--flow-run-dir`, the app expects the run directory to contain:

- `config.json`
- `records.csv`

Spike CSVs may also be read from the same run directory when present.

## Generated Outputs

Runtime outputs are not source files and should remain untracked:

- `notebooks/debug/outputs`
- `notebooks/debug/runs`
- `morphology_mutation_*` bundle folders

## Refresh Notes

When refreshing the standalone app bundle, update the copied files in place and keep launcher paths repository-relative. Any intentionally external SWC source should be configured through `local_config.py` or environment variables, not hardcoded in notebooks.
