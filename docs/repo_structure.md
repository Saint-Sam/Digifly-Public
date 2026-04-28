# Digifly Public Repo Structure

## Source Of Truth

Use `Digifly-MASTER_MULTIPROCESS` as the ground truth for current structure, filenames, and notebook/helper formats.

## Chosen Public Layout

The public project intentionally mirrors the master repo's active workflow roots:

- `Phase 1/`
- `Phase 2/` later
- `Phase 3/`
- `docs/`

This keeps migration low-risk because the active Digifly notebooks and helper files already assume phase-oriented roots.

## Phase 1 Public Import

Included from `Digifly-MASTER_MULTIPROCESS/Phase 1/`:

- `Phase 1.ipynb`
- `phase1_bridge.py`
- `filter_ids_by_size_and_export_swc.py`
- `digifly_phase1/`

Public hygiene applied to the copied notebook:

- cleared code-cell outputs
- cleared execution counts
- replaced hardcoded neuPrint JWTs with `NEUPRINT_TOKEN` environment lookups

Public hygiene applied to the copied bridge module:

- removed fallback token lookup into older Desktop Digifly sibling folders
- kept environment-variable, `~/.neuprint_token`, and local ignored `Neuprint Token.txt` token paths

The notebook also imports `filter_ids_by_size_and_export_swc.py` for its Option 26 Excel-ID filtering workflow, so that helper is included to keep the public notebook runnable without hidden local files.

## Modular Phase 1 Menu

`Phase 1.ipynb` is now a launcher. It sets up a token and calls `digifly_phase1.menu.main_menu()`.

The default public menu shows the core workflow choices:

- Choice 1: `digifly_phase1/choice_1_build_exports.py`
- Choice 2: `digifly_phase1/choice_2_batch_filter_export.py`
- Choice 3: `digifly_phase1/choice_3_metadata_template.py`
- Choice 4: `digifly_phase1/choice_4_pathfinding.py`

The utility choices are kept importable but hidden from the default menu. They
can be shown with `main_menu(show_utilities=True)` or by typing `U` in the menu:

- Choice 5: `digifly_phase1/choice_5_glia_volume.py`
- Choice 6: `digifly_phase1/choice_6_label_coverage.py`
- Choice 7: `digifly_phase1/choice_7_proximity_scan.py`

Shared implementation currently lives in `digifly_phase1/workflow_core.py` to preserve the tested notebook behavior while the public codebase is cleaned up. Future cleanup can move helpers from `workflow_core.py` into more focused modules without changing the notebook menu contract.

Choice 1 prompts for the neuPrint connectome/dataset before exporting. Generated
Choice 1 outputs are grouped by dataset slug, for example:

- `Phase 1/manc_v1.2.1/export_swc/`
- `Phase 1/manc_v1.2.1/Glia IDs/`
- `Phase 1/male-cns_v0.9/export_swc/`

## Phase 3 Working Import

`Phase 3/` was imported directly from
`Digifly-MASTER_MULTIPROCESS/Phase 3_WORKING/` as a cleanup staging copy.
Only obvious local noise was excluded during import:

- `.DS_Store`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `*.pyc`

The imported folder currently preserves the working layout:

- `configs/`
- `data/`
- `legacy/`
- `notebooks/`
- `scripts/`
- `src/phase3_bridge/`

Most local size comes from generated artifacts under `Phase 3/data/derived/`
and `Phase 3/data/outputs/`; both are ignored by git.

## Phase 2 Framework Staging

`Phase 2/` now contains a lean public staging copy of the reusable NEURON
simulation framework from `Digifly-MASTER_MULTIPROCESS/Phase 2/`.

Included:

- `digifly/`
- `tests/`
- `config/structure_manifest.yaml`
- `data/*.mod`
- `notebooks/run_simulation.ipynb`
- `apps/VIP_Glia_Sim/` standalone morphology mutation app and launcher notebooks

Excluded:

- compiled mechanism folders such as `data/arm64/`
- exported SWC/synapse datasets
- large generated metadata CSVs
- debug notebooks
- Hemilineage Simulations run/output/cache folders

See `docs/phase2_simulation_framework_audit.md` for the detailed Phase 2 and
Hemilineage triage.

See `docs/hemilineage_experiment_feature_inventory.md` for the notebook-level
inventory of unique Hemilineage/DNg100 experiment features, simulation knobs,
error-log requirements, and the proposed Phase 2 workbench direction.

## GitHub Curation Rules

Track:

- reusable code
- notebooks that are meant to be read or rerun
- lightweight configs/manifests
- documentation

Ignore:

- neuPrint token files
- exported SWC trees
- large local datasets
- run outputs and debug artifacts
- notebook checkpoints
- Python/Jupyter caches
- compiled or generated mechanism/build outputs
