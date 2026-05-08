# Phase 2 Colab Notes

Colab is a good browser-notebook target for Phase 2 because it provides a fresh Linux runtime and avoids native Windows NEURON setup. It should be treated as a notebook/runtime path, not as a full replacement for the WSL PyVista desktop app.

## Recommended Scope

- Use Colab for NEURON simulation runs, preset testing, generated artifacts, and browser-native Plotly visualization.
- Keep the full interactive PyVista mutation desktop app on WSL or local Linux with desktop display support.
- Add a Colab start notebook such as `START_HERE_Digifly_Phase2_Colab.ipynb` that bootstraps the runtime from a clean VM.

## Bootstrap Shape

A Colab bootstrap notebook should:

1. Clone this branch or pull the current checkout.
2. Install Linux build/runtime packages with `apt-get`.
3. Install Python dependencies from the Phase 2/WSL requirements file.
4. Compile the NEURON `.mod` files with `nrnivmodl`.
5. Export the same environment variables used by Docker and WSL:

```bash
export DIGIFLY_PHASE2_ROOT="/content/Digifly-Public/Phase 2"
export DIGIFLY_SWC_DIR="/content/Digifly-Public/Phase 1/manc_v1.2.1/export_swc"
export DIGIFLY_GAP_MECH_DIR="/content/Digifly-Public/Phase 2/data"
export PYTHONPATH="/content/Digifly-Public/Phase 2:/content/Digifly-Public/Phase 1:/content/Digifly-Public"
export NEURON_MODULE_OPTIONS="-nogui"
```

6. Launch the Phase 2 workbench inside the notebook, or provide a simpler Colab-specific runner cell for selected presets.

## Visualization Guidance

Colab does not naturally expose a local desktop window for the PyVista mutation app. Prefer the Plotly browser visualizer for Colab. If PyVista output is needed later, add a separate headless rendering path that writes screenshots or videos rather than opening a desktop GUI.

## Data Persistence

Colab runtimes are temporary. Treat `/content` as disposable and save important outputs to Google Drive, GitHub artifacts, or manual downloads.

## Widget Compatibility

Use the JupyterLab 4 / ipywidgets 8 stack:

```text
ipywidgets>=8.1,<9
jupyterlab_widgets>=3,<4
jupyterlab>=4,<5
notebook>=7,<8
ipykernel>=6,<7
```

If a notebook shows a browser error like `Failed to load model class 'VBoxModel'` with `@jupyter-widgets/controls`, restart the runtime/kernel and clear stale widget outputs before re-running the workbench cell.
