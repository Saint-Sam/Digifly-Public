# Phase 1

Phase 1 imports neuPrint data and exports neuron morphology/synapse artifacts for downstream Digifly phases.

## Files

- `Phase 1.ipynb`: public copy of the current master Phase 1 notebook.
- `phase1_bridge.py`: helper functions for Phase 1 to Phase 2 export handoff.
- `filter_ids_by_size_and_export_swc.py`: Option 26 helper for Excel body-ID filtering and SWC export.
- `digifly_phase1/`: modular Phase 1 package used by the notebook menu.
- `Neuprint Token.example.txt`: tracked instructions for the local token file.

## Token

Before running notebook cells that call neuPrint, either set `NEUPRINT_TOKEN` in the shell/notebook environment or save the token to the local file `Neuprint Token.txt`.

```bash
export NEUPRINT_TOKEN="<neuprint-token>"
```

The notebook can also create `Neuprint Token.txt` through `ensure_neuprint_token()`. Local token files named `Neuprint Token.txt` are intentionally ignored by git. A 401 response usually means the local token file is incomplete, expired, or still contains placeholder text.

## Menu Structure

The notebook launches `digifly_phase1.menu.main_menu()`. The default menu shows the core Phase 1 -> Phase 2/3 workflow choices:

- `choice_1_build_exports.py`: core SWC export workflow. First asks which neuPrint connectome/dataset to use, then takes body IDs, family prefixes, `ALL`, `UNLABELED`, or `UNLABELED_STRICT`; writes healed SWCs, synapse CSVs, mapped SWCs, and metadata.
- `choice_2_batch_filter_export.py`: reads an Excel body-ID list, filters IDs by a selected size metric, and exports kept/dropped/failed reports plus a filtered SWC bundle.
- `choice_3_metadata_template.py`: exports a NeuronCriteria-style metadata CSV for downstream Phase 2/3 planning.
- `choice_4_pathfinding.py`: runs male-CNS pathfinding between selected upstream/downstream neurons and can save the combined annotated path table.

Additional one-off utility choices are kept in code but hidden from the default menu. Type `U` in the menu, or call `main_menu(show_utilities=True)`, to reveal:

- `choice_5_glia_volume.py`: queries MANC glia, computes skeleton bounding-box size metrics, and writes CSV/parquet volume tables.
- `choice_6_label_coverage.py`: reports labeled vs unlabeled neuron counts in the active dataset.
- `choice_7_proximity_scan.py`: scans exported SWCs from a master metadata CSV for proximity to reference skeleton IDs.

Shared legacy workflow code lives in `workflow_core.py`, and token/client handling lives in `token_store.py` and `clients.py`.

To add a future choice, add a new `choice_*.py` module with a `run(client=None)` function, then register it in `menu.py`.

## Generated Outputs

Relative output paths are resolved under this `Phase 1` folder. Choice 1 writes dataset-scoped outputs such as `manc_v1.2.1/export_swc/`, `manc_v1.2.1/Glia IDs/`, and `male-cns_v0.9/export_swc/`. General generated reports use `outputs/`. These are ignored because they can become large and machine-specific.
