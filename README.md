# Digifly Public

Digifly acts as a framework for simulating single or networked Hodgkin-Huxley neurons with varying degrees of biophysical accuracy. 
The project follows a three-phase pipeline:
- `Phase 1`: Import connectomic datasets from NeuPrint+
- `Phase 2`: Run NEURON-based simulations on pulled neurons
- `Phase 3`: Use motor-neuron outputs from Phase 2 to control NeuroMechFly, a MuJoCo model of Drosophila. 

This repo keeps the master workspace's phase-based layout so notebooks and helper files can migrate without a second path-refactor project.

## Current Contents

- `Phase 1/Phase 1.ipynb`: sanitized public copy of the current Phase 1 notebook.
- `Phase 1/phase1_bridge.py`: current Phase 1 bridge/helper module.
- `Phase 1/filter_ids_by_size_and_export_swc.py`: helper used by the Phase 1 notebook for Excel ID filtering and SWC export.
- `Phase 1/digifly_phase1/`: modular Phase 1 package with the menu, token handling, and one module per current Choice.
- `Phase 2/`: lean public staging copy of the reusable NEURON simulation framework.
- `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb`: notebook-first interactive workbench for shared runs and hemilineage project runs.
- `Phase 2/apps/VIP_Glia_Sim/`: standalone morphology mutation app imported from the current `VIP_Glia_Sim` source.
- `Phase 3/`: direct import of `Digifly-MASTER_MULTIPROCESS/Phase 3_WORKING`, with cache/checkpoint files excluded.
- `docs/repo_structure.md`: layout and curation notes for future Phase 2/Phase 3 imports.
- `docs/phase2_simulation_framework_audit.md`: audit of Phase 2 and Hemilineage Simulations, with framework-vs-experiment triage.
- `docs/hemilineage_experiment_feature_inventory.md`: inventory of Hemilineage Simulation experiment features, knobs, error-log requirements, and the proposed Phase 2 workbench shape.

## neuPrint Token Setup

The public notebook does not include a hardcoded neuPrint token. Before running Phase 1, set:

```bash
export NEUPRINT_TOKEN="your-token-here"
```

Alternatively, place a local token in `Phase 1/Neuprint Token.txt`. The local project token file is gitignored, and the notebook can create it through `ensure_neuprint_token()`.

## Install

```bash
python -m pip install -e ".[notebooks]"
```

## To-do

- Phase 3 needs original NeuroMechFly v2 files to work, add information on download and citations
- More use notes on Phase 1
- Update Visualizer
- Phase 2 app: make app

No license has been selected yet.
