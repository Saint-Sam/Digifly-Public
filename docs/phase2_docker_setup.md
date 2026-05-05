# Phase 2 Docker Setup

This is the recommended Phase 2 path for Windows users. It avoids native Windows NEURON setup by running Digifly Phase 2 inside a Linux container.

## Quick Start

1. Install Docker Desktop.
2. Clone or download the Digifly Public repo.
3. Open this notebook from the repo folder and run its single code cell:

```text
START_HERE_Digifly_Phase2.ipynb
```

The notebook starts Docker when needed and opens the Docker-hosted Phase 2 Workbench.

If Windows does not know how to open `.ipynb` files yet, double-click this Command Prompt launcher from the repo folder:

```text
Start_Digifly_Phase2_Windows.bat
```

The launcher uses Command Prompt, not PowerShell. It starts Docker Compose in the background and opens the same start notebook:

```text
START_HERE_Digifly_Phase2.ipynb
```

4. In the Workbench, use the `Single Neuron Debug` preset, then click `Run`. After the run finishes, click `Open Browser Visualizer`.

Manual terminal startup is also available:

```bash
docker compose up --build phase2-jupyter
```

Then open:

```text
http://localhost:8888
```

If port `8888` is already in use, choose another local port:

```bash
DIGIFLY_JUPYTER_PORT=8889 docker compose up --build phase2-jupyter
```

Then open:

```text
http://localhost:8889
```

In JupyterLab, open:

```text
START_HERE_Digifly_Phase2.ipynb
```

After a workbench run completes, click `Open Browser Visualizer` to load the latest run into the Plotly browser-native flow viewer. This path works inside Docker on Windows because it renders in JupyterLab and does not require PyVista, VTK desktop windows, or a Windows NEURON install.

The container sets `NEURON_MODULE_OPTIONS=-nogui`, installs NEURON, compiles the Phase 2 gap-junction mechanisms, and exposes the repo at `/workspace`.

## Test The Container

Run:

```bash
docker compose --profile test run --rm phase2-test
```

This checks that NEURON imports, the compiled mechanisms load, Digifly Phase 2 imports, and the Phase 2 tests pass.

## Optional Prebuilt Image

The GitHub Actions workflow at `.github/workflows/phase2-docker.yml` builds the Phase 2 image on pull requests and publishes it to GitHub Container Registry on pushes to the default branch or manual workflow runs.

After the first successful publish, the image tag will be:

```text
ghcr.io/saint-sam/digifly-public-phase2:latest
```

If GitHub marks the package private after the first run, open the package settings in GitHub and make it public.

To use the prebuilt image with Compose on macOS/Linux:

```bash
DIGIFLY_PHASE2_IMAGE=ghcr.io/saint-sam/digifly-public-phase2:latest docker compose pull phase2-jupyter
DIGIFLY_PHASE2_IMAGE=ghcr.io/saint-sam/digifly-public-phase2:latest docker compose up --no-build phase2-jupyter
```

On Windows PowerShell:

```powershell
$env:DIGIFLY_PHASE2_IMAGE = "ghcr.io/saint-sam/digifly-public-phase2:latest"
docker compose pull phase2-jupyter
docker compose up --no-build phase2-jupyter
```

## Data Paths

The repo is mounted into the container at:

```text
/workspace
```

By default, Phase 2 looks for SWC data at:

```text
/workspace/Phase 1/manc_v1.2.1/export_swc
```

This maps to the same folder in the downloaded repo:

```text
Phase 1/manc_v1.2.1/export_swc
```

Generated outputs stay in ignored repo folders such as `Phase 2/outputs/` and notebook debug folders.

Completed shared-run outputs also persist under:

```text
Phase 1/manc_v1.2.1/export_swc/hemi_runs
```

The standalone browser viewer can be opened from:

```text
Phase 2/notebooks/launch_browser_flow_visualizer.ipynb
```

## VS Code Dev Container

If using VS Code:

1. Install Docker Desktop.
2. Install the Dev Containers extension.
3. Open the repo folder.
4. Choose **Reopen in Container**.

VS Code will use `.devcontainer/devcontainer.json`, build the same Phase 2 image, and run the smoke test after the container is created.

## Why Docker

Phase 2 needs Python, NEURON, MPI/build tools, and compiled `.mod` mechanisms. That combination is fragile on native Windows. Docker keeps the simulation runtime Linux-based and reproducible while still letting Windows users work from a normal browser and repo folder.
