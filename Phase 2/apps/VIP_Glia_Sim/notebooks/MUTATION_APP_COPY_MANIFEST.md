# Mutation App Copy Manifest

Date: 2026-04-27

Imported public destination:

- `Phase 2/apps/VIP_Glia_Sim`

This note is the safe copy map for moving the current `VIP_Glia_Sim` mutation app into `Digifly Public` as a new standalone source tree.

## Source of truth

If the goal is to preserve the current app behavior, including the newer `neuroglancer` render mode and the exact launcher behavior in `launch_morphology_mutation.ipynb`, the source of truth is:

- `<source VIP_Glia_Sim>/tools/morphology_mutation_app.py`
- `<source VIP_Glia_Sim>/tools/morphology_mutation.py`
- `<source VIP_Glia_Sim>/tools/morphology_mutation_notebook_helpers.py`
- `<source VIP_Glia_Sim>/notebooks/launch_morphology_mutation.ipynb`

Do not use the older `Phase 2/digifly/phase2/extensions/...` copies as the source for the new standalone version if you want the latest testing changes.

## Minimum files to copy

These are the files required to preserve the direct launcher workflow in `launch_morphology_mutation.ipynb`.

- `<source VIP_Glia_Sim>/tools/morphology_mutation_app.py`
  Why: main desktop app entry point. It imports `tools.morphology_mutation` directly near the top of the file.

- `<source VIP_Glia_Sim>/tools/morphology_mutation.py`
  Why: backend mutation project, SWC loading, save bundle logic, manifest writing, Phase 2 overlay writing.

- `<source VIP_Glia_Sim>/tools/__init__.py`
  Why: not strictly required for the app launch itself, but safe to copy with the package so notebook imports remain clean.

- `<source VIP_Glia_Sim>/notebooks/launch_morphology_mutation.ipynb`
  Why: this is the direct launcher notebook the user asked to preserve as-is.

- `<source VIP_Glia_Sim>/notebooks/README_MORPHOLOGY_MUTATION.md`
  Why: usage notes and app-level documentation.

## Backend/support files worth copying with it

These are not strictly required to open the app window from `launch_morphology_mutation.ipynb`, but they are part of the mutation-app ecosystem and are worth copying if the new location should be fully reusable.

- `<source VIP_Glia_Sim>/tools/morphology_mutation_notebook_helpers.py`
  Why: lets notebooks read mutation manifests, mutation connection specs, AIS/biophysics sidecars, and build simulation overrides from saved mutation bundles.

- `<source VIP_Glia_Sim>/notebooks/glia_simulation.ipynb`
  Why: contains the helper launch path that calls `tools/morphology_mutation_app.py`, passes `--flow-run-dir`, and later imports `tools.morphology_mutation_notebook_helpers`.

- `<source VIP_Glia_Sim>/notebooks/glia_circuit_growth_stages_v1.ipynb`
  Why: this notebook reuses helper definitions from `glia_simulation.ipynb` and can launch the mutation app through that path.

## Optional selector-side files

Copy these too if the new standalone source tree should keep the selector workflow, not just the mutation app launcher.

- `<source VIP_Glia_Sim>/tools/swc_box_selector_app.py`
- `<source VIP_Glia_Sim>/tools/swc_interactive_selector.py`
- `<source VIP_Glia_Sim>/notebooks/launch_swc_box_selector.ipynb`
- `<source VIP_Glia_Sim>/notebooks/README_INTERACTIVE_SELECTOR.md`

## What the launcher depends on

The launcher notebook currently points at these runtime locations:

- App root:
  `<source VIP_Glia_Sim>`

- App script:
  `<source VIP_Glia_Sim>/tools/morphology_mutation_app.py`

- SWC data root:
  `<source Phase 2>/data/export_swc`

- Phase 2 repo root:
  `<source Phase 2>`

- Default output root:
  `<source VIP_Glia_Sim>/notebooks/debug/outputs`

- Default flow runs root:
  `<source VIP_Glia_Sim>/notebooks/debug/runs`

When the code is copied to `Digifly Public`, the notebook paths should be updated to the new app root. The SWC and Phase 2 paths may stay external if that is intentional.

## External dependencies not inside VIP_Glia_Sim

The current mutation app is mostly self-contained, but it still expects some external data/runtime context.

- SWC files under a real SWC root
  Current expectation:
  `<source Phase 2>/data/export_swc`

- Optional Phase 2 import fallback
  `tools/morphology_mutation.py` can fall back to `digifly.phase2.neuron_build.swc_cell.find_swc` if the local SWC glob lookup does not find a match.

- Optional flow overlay input
  If launching with `--flow-run-dir`, the app expects that directory to contain at least:
  - `config.json`
  - `records.csv`
  It may also read spike CSVs from the same run directory.

- Python packages
  Required:
  - `numpy`
  - `pandas`
  - `pyvista`
  Recommended:
  - `pillow`
  - `matplotlib`
  Optional for helper paths that read parquet/feather:
  - `pyarrow`

## Generated outputs that do not need to be copied as source

These are runtime outputs, not source files:

- `<source VIP_Glia_Sim>/notebooks/debug/outputs`
- `<source VIP_Glia_Sim>/notebooks/debug/runs`
- mutation bundle folders such as `morphology_mutation_*`

Copy them only if you want to preserve prior saved runs or manifests.

## Existing mutation-app copies found on Desktop

### Current working standalone/testing source

- `<source VIP_Glia_Sim>/tools/morphology_mutation_app.py`
- `<source VIP_Glia_Sim>/tools/morphology_mutation.py`
- `<source VIP_Glia_Sim>/tools/morphology_mutation_notebook_helpers.py`
- `<source VIP_Glia_Sim>/notebooks/launch_morphology_mutation.ipynb`

### Older Phase 2 extension-style copies

- `<source Phase 2>/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<source Phase 2>/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<source Phase 2>/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

- `Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

- `<source Digifly_MASTER>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<source Digifly_MASTER>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<source Digifly_MASTER>/Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

- `<source Digifly-MASTER_MULTIPROCESS>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

### Archive/backups

- `<archive Digifly-Master_2026-03-13>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<archive Digifly-Master_2026-03-13>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<archive Digifly-Master_2026-03-13>/Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

- `<archive Digifly_MASTER_2026-03-14_Pre-refactor>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<archive Digifly_MASTER_2026-03-14_Pre-refactor>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<archive Digifly_MASTER_2026-03-14_Pre-refactor>/Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

- `<archive Digifly-MASTER_MULTIPROCESS_backup_2026-03-15>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation_app.py`
- `<archive Digifly-MASTER_MULTIPROCESS_backup_2026-03-15>/Phase 2/digifly/phase2/extensions/glia_editing/mutation/morphology_mutation.py`
- `<archive Digifly-MASTER_MULTIPROCESS_backup_2026-03-15>/Phase 2/digifly/phase2/extensions/glia_editing/selectors/swc_box_selector_app.py`

### Hemilineage simulation-local copies

- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/MANC_121_SIZ_activeK_pas/tools/morphology_mutation_app.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/MANC_121_SIZ_activeK_pas/tools/morphology_mutation.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/MANC_121_SIZ_only/tools/morphology_mutation_app.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/MANC_121_SIZ_only/tools/morphology_mutation.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/Escape-SIZ/tools/morphology_mutation_app.py`
- `<source Digifly-MASTER_MULTIPROCESS>/Hemilineage Simulations/Escape-SIZ/tools/morphology_mutation.py`

## Safest copy unit

If the new home under `Digifly Public` should work without hunting for sibling imports, the safest source unit to copy is:

- the entire folder:
  `<source VIP_Glia_Sim>/tools`

plus:

- `<source VIP_Glia_Sim>/notebooks/launch_morphology_mutation.ipynb`
- `<source VIP_Glia_Sim>/notebooks/README_MORPHOLOGY_MUTATION.md`

If you also want the simulation-side integration to remain intact, add:

- `<source VIP_Glia_Sim>/notebooks/glia_simulation.ipynb`
- `<source VIP_Glia_Sim>/notebooks/glia_circuit_growth_stages_v1.ipynb`

## Recommended destination layout

To avoid colliding with the older `Digifly Public/Phase 2/digifly/phase2/extensions/...` copy, prefer a new standalone source tree such as:

- `Phase 2/apps/VIP_Glia_Sim/tools/...`
- `Phase 2/apps/VIP_Glia_Sim/notebooks/...`

That keeps the new standalone mutation source clearly separate from the legacy Phase 2 extension copy.

## Practical next step

If the next job is the actual migration, copy from `VIP_Glia_Sim`, not from `Digifly Public/Phase 2`, then patch the copied notebook root paths so:

- `WORK_ROOT` points to the new `Digifly Public/.../VIP_Glia_Sim`
- `APP_PATH` points to the copied `tools/morphology_mutation_app.py`
- any intentionally external `SWC_DIR` and `PHASE2_ROOT` paths remain correct
