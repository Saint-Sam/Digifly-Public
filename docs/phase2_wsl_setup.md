# Phase 2 WSL Setup

WSL is the Windows path for running native Linux NEURON and launching the full PyVista morphology mutation app from the Phase 2 Workbench.

Docker remains the easiest browser-only runtime. Use WSL when you want the real mutation app window and your Windows machine has Ubuntu/WSL available.

## Quick Start

From Windows, you can run the `.sh` from Git Bash or another Bash-compatible
shell. The launcher detects that it is not inside WSL and relaunches itself in
your default WSL distro:

```bash
cd /c/Users/<you>/path/to/Digifly-Public
bash Start_Digifly_Phase2_WSL.sh
```

For the most reliable Windows double-click launch, use the companion batch file:

```text
Start_Digifly_Phase2_WSL.bat
```

Or open the Ubuntu/WSL CLI, go to the downloaded repo, and run:

```bash
cd /mnt/c/Users/<you>/path/to/Digifly-Public
bash Start_Digifly_Phase2_WSL.sh
```

On first launch, the script installs Ubuntu packages, creates `.venv-wsl`, installs the Python dependencies, compiles the NEURON `.mod` mechanisms, and starts JupyterLab.

If a previous launch was interrupted and left `.venv-wsl` incomplete, the launcher moves the broken environment aside and rebuilds it automatically.

After JupyterLab opens:

1. Open `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb` if it is not already open.
2. Select a preset.
3. Click `Run`.
4. After the run completes, click `Open Mutation App`.

The PyVista app launches the staged VIP glia morphology mutation app and connects it to the latest workbench run.

## Requirements

- Windows with WSL and Ubuntu, Debian, or another apt-based WSL distro installed.
- WSLg for desktop GUI windows, or a separately configured X server.
- Internet access on first setup so Ubuntu and Python packages can install.

The startup script warns when `DISPLAY` and `WAYLAND_DISPLAY` are missing. In that case, simulations can still run, but the PyVista mutation app will not open until WSL desktop display support is fixed.

## What The Script Sets

The WSL launcher keeps everything repo-local:

```text
.venv-wsl/                         Python environment, ignored by git
Phase 2/data/x86_64/               compiled NEURON mechanisms, ignored by git
Phase 1/manc_v1.2.1/export_swc/    default SWC root
```

It also exports:

```bash
DIGIFLY_PHASE2_ROOT="<repo>/Phase 2"
DIGIFLY_SWC_DIR="<repo>/Phase 1/manc_v1.2.1/export_swc"
DIGIFLY_GAP_MECH_DIR="<repo>/Phase 2/data"
PYTHONPATH="<repo>/Phase 2:<repo>/Phase 1:<repo>"
NEURON_MODULE_OPTIONS="-nogui"
```

Override the Jupyter port when needed:

```bash
DIGIFLY_JUPYTER_PORT=8890 bash Start_Digifly_Phase2_WSL.sh
```

Skip automatically opening the Windows browser:

```bash
DIGIFLY_OPEN_BROWSER=0 bash Start_Digifly_Phase2_WSL.sh
```

Use a non-default WSL distro when needed:

```bash
DIGIFLY_WSL_DISTRO=Ubuntu-24.04 bash Start_Digifly_Phase2_WSL.sh
```

## Widget Troubleshooting

If JupyterLab shows a JavaScript error such as:

```text
Failed to load model class 'VBoxModel' from module '@jupyter-widgets/controls'
```

the kernel and JupyterLab frontend are using incompatible or stale widget state. The WSL launcher checks for JupyterLab 4, Notebook 7, ipykernel 6, and the ipywidgets 8 stack, then repairs the Python packages automatically when needed. After that upgrade, close the old notebook tab, restart JupyterLab from `Start_Digifly_Phase2_WSL.bat`, reopen `Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb`, and rerun the workbench cell.

For a manual repair from the repo root inside WSL:

```bash
bash scripts/wsl/setup_phase2_wsl.sh --python-only
```

## Docker Versus WSL

Use Docker when you want the simplest reproducible browser runtime and the Plotly browser visualizer.

Use WSL when you want NEURON installed in Linux and the workbench button to launch the full PyVista morphology mutation app.

Both paths use the same repo folders and the same public workbench presets.
