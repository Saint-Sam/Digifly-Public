# Phase 3 Migration Inventory

This inventory tracks what was copied into the new structure.

## Copied into `notebooks/archive`

- `Phase 3 - NEURON to MuJoCo.ipynb`
- `Phase 3 - NEURON to MuJoCo-Copy1.ipynb`
- `Phase 3 v2.0.ipynb`
- `MuJoCo Video tutorials.ipynb`

## Copied into `notebooks/debug`

- `Phase 3 - Legacy Workspace.ipynb` (copy of current main notebook)
- `Phase 3 - Mapping and Coverage Diagnostics.ipynb` (new)
- `Phase 3 - Mapping Enrichment from Phase1.ipynb` (new)

## Added in `notebooks/user`

- `Phase 3 - NEURON to MuJoCo.ipynb` (clean user workflow)
- `Phase 3 - NEURON to MuJoCo (Clean).ipynb` (same content with explicit suffix)

## Copied into `data/inputs/mappings`

- `mn_to_actuator_mapping.csv`
- `NEURON_to_MuJoCo_bridge.csv`
- `NEURON_to_MuJoCo_bridge__TODO_low_confidence.csv`
- `MN to Muscle Mapping FANC IDs.docx`

## Copied into `data/outputs/videos`

- all existing mp4 files from `neuro_fly/outputs`

## New source module

- `src/phase3_bridge/__init__.py`
- `src/phase3_bridge/pipeline.py`
- `src/phase3_bridge/video_pipeline.py`
- `src/phase3_bridge/mapping_enrichment.py`

## New configs

- `configs/phase3_video_profiles.yaml`

## Note

No original Phase 3 files were deleted or overwritten.
