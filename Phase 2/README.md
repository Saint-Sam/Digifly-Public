# Digifly Phase 2

Phase 2 is the reusable NEURON simulation layer for Digifly.

This public copy is intentionally lean. It includes the framework code, tests, mechanism source files, and one canonical runner notebook. It does not include large SWC exports, compiled NEURON build artifacts, local run outputs, or Hemilineage/DNg100 experiment folders.

## Included

- `digifly/phase2/`: shared simulation package
- `digifly/phase2/workbench/`: notebook-first control surface, presets, validation, manifests, and execution helpers
- `digifly/tools/`: repo validation and small helper launchers
- `apps/VIP_Glia_Sim/`: standalone PyVista morphology mutation app and testing launch notebooks
- `data/*.mod`: NEURON gap-junction mechanism sources
- `notebooks/run_simulation.ipynb`: unified Phase 2 runner notebook
- `notebooks/Digifly_Phase2_Workbench.ipynb`: interactive workbench notebook built on the public preset/manifest layer
- `tests/`: focused framework tests
- `config/structure_manifest.yaml`: lightweight structure validation manifest

## Not Included

- exported SWC datasets
- `*_synapses_new.csv` files
- compiled mechanism folders such as `data/arm64/`
- hemilineage project outputs
- benchmark outputs
- paper-specific DNg100 run folders

Generate or copy local datasets outside git, then point Phase 2 configs at them.

## Quick Verification

Run from this folder:

```bash
env NEURON_MODULE_OPTIONS=-nogui python -m pytest -q tests
```

The `NEURON_MODULE_OPTIONS=-nogui` setting avoids display-window startup issues on headless or notebook-driven runs.

## Docker Runtime

For Windows users, the recommended Phase 2 runtime is Docker instead of native Windows NEURON:

```bash
docker compose up --build phase2-jupyter
```

Open `http://localhost:8888`, then open `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb`.

To test the container:

```bash
docker compose --profile test run --rm phase2-test
```

The Docker image installs NEURON, compiles the Phase 2 `.mod` mechanisms, and mounts the repo at `/workspace`. See `../docs/phase2_docker_setup.md`.

## Main Entry Points

- `digifly.phase2.api.run_walking_simulation`
- `digifly.phase2.api.build_config`
- `digifly.phase2.hemi.sim_project.run_full_hemilineage_project`
- `digifly.phase2.hemi.sim_project.run_hemilineage_benchmark`
- `digifly.phase2.workbench.launch_workbench`

The notebook `notebooks/Digifly_Phase2_Workbench.ipynb` is now the notebook-first public control surface. `notebooks/run_simulation.ipynb` remains the lower-level runner notebook.

## Workbench Planning

The Hemilineage Simulations notebooks contain many experimental modes that should become presets and controls rather than copied notebook code. See `../docs/hemilineage_experiment_feature_inventory.md` for the current inventory of:

- reusable simulation features and knobs
- DNg100, gap, HH/SBI, instability, morphology, and Phase 3 bridge workflows
- error-log and run-artifact requirements
- the proposed future Phase 2 Simulation Workbench layout

The first scaffold of that workbench now exists in `digifly/phase2/workbench/` plus `notebooks/Digifly_Phase2_Workbench.ipynb`.

## Mutation App

The standalone VIP glia morphology mutation app is staged under `apps/VIP_Glia_Sim/`.

Use `apps/VIP_Glia_Sim/notebooks/test_launch_morphology_mutation_standalone.ipynb` for a clean public smoke-test launch. The notebook resolves the copied app root, checks dependencies, prints the launch command, runs an app `--help` probe, and launches the PyVista desktop app from a final explicit cell.

## Mechanism Notes

Gap mechanisms are stored as source `.mod` files:

- `data/Gap.mod`
- `data/RectGap.mod`
- `data/HeteroRectGap.mod`

Compiled outputs are local machine artifacts and should stay untracked. The loader in `digifly.phase2.neuron_build.gaps` can search common local mechanism locations and compile when possible.
