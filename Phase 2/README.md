# Digifly Phase 2

Phase 2 is the reusable NEURON simulation layer for Digifly.

This public copy is intentionally lean. It includes the framework code, tests, mechanism source files, and one canonical runner notebook. It does not include large SWC exports, compiled NEURON build artifacts, local run outputs, or Hemilineage/DNg100 experiment folders.

## Included

- `digifly/phase2/`: shared simulation package
- `digifly/phase2/workbench/`: notebook-first control surface, presets, validation, manifests, and execution helpers
- `digifly/tools/`: repo validation and small helper launchers
- `Projects/`: canonical project notebooks and project-local output folders
- `apps/VIP_Glia_Sim/`: standalone PyVista morphology mutation app and testing launch notebooks
- `data/*.mod`: NEURON gap-junction mechanism sources
- `Projects/baseline_run_simulation/run_simulation.ipynb`: unified Phase 2 runner project notebook
- `Projects/phase2_workbench/Digifly_Phase2_Workbench.ipynb`: interactive workbench project notebook built on the public preset/manifest layer
- `Projects/elliott_sparrow_2012_beam/launch_elliott_sparrow_beam.ipynb`: Elliott/Sparrow beam project notebook
- `notebooks/launch_browser_flow_visualizer.ipynb`: browser-native Plotly flow visualizer for completed runs
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

## Windows Runtime Options

For Windows users, Docker and WSL are both supported.

Use Docker when you want the simplest browser-only runtime:

```text
Windows-Docker.bat
```

That launcher starts Docker and opens the standard Phase 2 Workbench directly.

Manual startup:

```bash
docker compose up --build phase2-jupyter
```

Open `http://localhost:8888`, then open `Phase 2/Projects/phase2_workbench/Digifly_Phase2_Workbench.ipynb`.
After a run completes, click `Open Browser Visualizer` to view the morphology/activity animation directly in JupyterLab.

To test the container:

```bash
docker compose --profile test run --rm phase2-test
```

The Docker image installs NEURON, compiles the Phase 2 `.mod` mechanisms, and mounts the repo at `/workspace`. See `../docs/phase2_docker_setup.md`.

Use WSL when you want NEURON installed in Linux and the real PyVista mutation app. On Windows, double-click `Windows-WSL.bat`; from an Ubuntu/WSL terminal, run:

```bash
bash scripts/wsl/start_phase2_workbench_wsl.sh
```

The WSL path creates `.venv-wsl`, compiles the local NEURON mechanisms, starts JupyterLab, and enables the workbench `Open Mutation App` button when WSLg/X display support is available. Set `DIGIFLY_WSL_DISTRO=<distro-name>` before running `Windows-WSL.bat` when you want a non-default WSL distro. See `../docs/phase2_wsl_setup.md`.

## Public Workbench Presets

The Phase 2 Workbench exposes four public defaults:

- `Single Giant Fiber`: one giant fiber neuron, `10000`.
- `Simple Escape`: giant fibers `10000/10002` plus TTMns `10068/10110`.
- `Escape With PSI`: simple escape plus PSI neurons `11446/11654`.
- `Full Escape`: escape with PSI plus DLMns `10074, 10361, 18309, 169914, 10014, 10088, 10589, 10592, 10892`.

## Main Entry Points

- `digifly.phase2.api.run_walking_simulation`
- `digifly.phase2.api.run_cached_simulation`
- `digifly.phase2.api.ensure_phase2_environment`
- `digifly.phase2.api.build_config`
- `digifly.phase2.hemi.sim_project.run_full_hemilineage_project`
- `digifly.phase2.hemi.sim_project.run_hemilineage_benchmark`
- `digifly.phase2.workbench.launch_workbench`
- `digifly.phase2.workbench.launch_browser_flow_visualizer`

The notebook `Projects/phase2_workbench/Digifly_Phase2_Workbench.ipynb` is now the notebook-first public control surface. `Projects/baseline_run_simulation/run_simulation.ipynb` is the lower-level runner project notebook.

## Workbench Planning

The Hemilineage Simulations notebooks contain many experimental modes that should become presets and controls rather than copied notebook code. See `../docs/hemilineage_experiment_feature_inventory.md` for the current inventory of:

- reusable simulation features and knobs
- DNg100, gap, HH/SBI, instability, morphology, and Phase 3 bridge workflows
- error-log and run-artifact requirements
- the proposed future Phase 2 Simulation Workbench layout

The first scaffold of that workbench now exists in `digifly/phase2/workbench/` plus `Projects/phase2_workbench/Digifly_Phase2_Workbench.ipynb`.

## Live Cache, MPI, and CoreNEURON

Phase 2 now exposes a reusable live simulation cache through `digifly.phase2.api.run_cached_simulation`. It builds a NEURON network once, keeps it alive in a background Python process, and accepts repeated runtime-safe reruns for changing duration, dt, current clamps, recording, selected stimulation targets, and supported synapse/gap/cell overrides. Use `nproc > 1` to launch the cache under `mpiexec`; the helper sets the distributed `ParallelContext` backend and reuses the same request/response protocol from notebooks.

One-shot runs still use the existing `enable_coreneuron`, `coreneuron_gpu`, and `coreneuron_nthread` config fields. Live cached reruns intentionally execute on regular NEURON after the warm build because CoreNEURON is not reliable for mutating and reusing an already-built network inside a long-lived notebook session.

Notebook apps should call `ensure_phase2_environment(auto_install_python=True)` before launching simulations. That helper detects macOS, WSL, Docker, MPI, `nrnivmodl`, and gap mechanism paths; it can install missing Python packages into the active notebook kernel. System runtimes such as OpenMPI, Docker, WSL packages, and the NEURON desktop app are detected and reported rather than silently installed.

## Mutation App

The standalone VIP glia morphology mutation app is staged under `apps/VIP_Glia_Sim/`.

Use `apps/VIP_Glia_Sim/notebooks/test_launch_morphology_mutation_standalone.ipynb` for a clean public smoke-test launch. The notebook resolves the copied app root, checks dependencies, prints the launch command, runs an app `--help` probe, and launches the PyVista desktop app from a final explicit cell.

The PyVista mutation app is a desktop GUI path. In Docker/JupyterLab, use the workbench `Open Browser Visualizer` button after a run completes; it renders the latest run directly in the browser with Plotly. In WSL with WSLg/X display support, use the workbench `Open Mutation App` button to launch the full mutation app from the latest run.

SWC exports are intentionally not tracked. Set `DIGIFLY_SWC_DIR`, or copy `apps/VIP_Glia_Sim/notebooks/local_config.example.py` to the ignored `local_config.py` and point `SWC_DIR` at a real `export_swc` folder. Project notebooks route shared-run outputs under their own `Projects/<project>/outputs/runs` folders. For local testing, the launcher can seed `Phase 1/manc_v1.2.1/export_swc` by copying requested neuron folders from a configured source export.

## Mechanism Notes

Gap mechanisms are stored as source `.mod` files:

- `data/Gap.mod`
- `data/RectGap.mod`
- `data/HeteroRectGap.mod`

Compiled outputs are local machine artifacts and should stay untracked. The loader in `digifly.phase2.neuron_build.gaps` can search common local mechanism locations and compile when possible.
