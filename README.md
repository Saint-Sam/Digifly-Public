# Digifly Public

Digifly acts as a framework for simulating single or networked Hodgkin-Huxley neurons with varying degrees of biophysical accuracy. 
The project follows a three-phase pipeline:
- `Phase 1`: Import connectomic datasets from NeuPrint+
- `Phase 2`: Run NEURON-based simulations on pulled neurons
- `Phase 3`: Use motor-neuron outputs from Phase 2 to control NeuroMechFly, a MuJoCo model of Drosophila. 

This repo keeps the active Digifly phase-based layout so notebooks, helper files, and simulation assets remain easy to locate.

## Current Contents

- `Phase 1/Phase 1.ipynb`: sanitized public copy of the current Phase 1 notebook.
- `Phase 1/phase1_bridge.py`: current Phase 1 bridge/helper module.
- `Phase 1/filter_ids_by_size_and_export_swc.py`: helper used by the Phase 1 notebook for Excel ID filtering and SWC export.
- `Phase 1/digifly_phase1/`: modular Phase 1 package with the menu, token handling, and one module per current Choice.
- `Phase 2/`: lean public staging copy of the reusable NEURON simulation framework.
- `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb`: notebook-first interactive workbench for shared runs and hemilineage project runs.
- `Phase 2/notebooks/launch_browser_flow_visualizer.ipynb`: browser-native Plotly flow viewer for Docker/JupyterLab sessions.
- `Phase 2/apps/VIP_Glia_Sim/`: standalone morphology mutation app imported from the current `VIP_Glia_Sim` source.
- `Phase 3/`: staged Phase 3 working tree, with cache/checkpoint files excluded.
- `docs/repo_structure.md`: layout and curation notes for future Phase 2/Phase 3 imports.
- `docs/phase2_simulation_framework_audit.md`: audit of Phase 2 and Hemilineage Simulations, with framework-vs-experiment triage.
- `docs/hemilineage_experiment_feature_inventory.md`: inventory of Hemilineage Simulation experiment features, knobs, error-log requirements, and the proposed Phase 2 workbench shape.

## neuPrint Token Setup

The public notebook does not include a hardcoded neuPrint token. Before running Phase 1, set:

```bash
export NEUPRINT_TOKEN="<neuprint-token>"
```

Alternatively, place a local token in `Phase 1/Neuprint Token.txt`. The local project token file is gitignored, and the notebook can create it through `ensure_neuprint_token()`.

## Install

```bash
python -m pip install -e ".[notebooks]"
```

For Phase 2 on Windows, prefer Docker instead of native NEURON setup:

```bash
docker compose up --build phase2-jupyter
```

Then open `http://localhost:8888` and use `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb`.
After a run finishes, use **Open Browser Visualizer** in the workbench to view the activity animation inside JupyterLab.
See `docs/phase2_docker_setup.md` for the full Windows-first path, including the optional prebuilt GitHub Container Registry image.

## To-do

- Phase 3 needs original NeuroMechFly v2 files to work, add information on download and citations
- More use notes on Phase 1
- Update Visualizer
- Continue polishing the Phase 2 workbench and browser-based visualization path

No license has been selected yet.
