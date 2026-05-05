# Interactive SWC Selector

Notebook: `interactive_swc_compartment_selector.ipynb`

This notebook provides:

1. Interactive 3D SWC compartment picking for one neuron.
2. Local glia-loss-like potassium perturbation on selected sections.
3. AIS assignment from selected section(s), with optional persistent override.

Module used:

- `../tools/swc_interactive_selector.py`

Expected dependencies in the notebook kernel:

- `numpy`
- `pandas`
- `plotly`
- `ipywidgets`
- `neuron`

If `plotly` or `ipywidgets` is missing, the selector constructor raises an import error with guidance.

Backend notes:

- `backend="widget"` uses Plotly `FigureWidget` with click-to-select.
- `backend="static"` avoids `FigureWidget/anywidget` frontend issues and uses a list picker plus 3D hover.

Standalone desktop option (outside Jupyter):

- Script: `../tools/swc_box_selector_app.py`
- Purpose: drag-box select in a desktop 3D window, apply glia-loss to selected compartments, export JSON spec.
- Notebook launcher: `launch_swc_box_selector.ipynb`
  - Compare flow support in `glia_simulation.ipynb`:
  - baseline selector on `SWC_DIR`
  - reduced selector on `GLIA_COMPARE_REDUCED_MORPH_SWC_DIR`
  - reuse policy prompt (`ask`/`reuse`/`new`) for fast parameter sweeps
  - back-to-back baseline vs reduced/coalesced run summary in Cell 13
- Controls:
  - `r` then drag: select sections inside the box (additive)
  - `c`: clear selection
  - `g`: apply glia-loss to selected sections
  - `u`: undo glia-loss on currently selected sections (restore baseline `ko/ek`)
  - `x`: undo glia-loss on all touched sections in this app session
  - `s`: save `GLIA_LOSS_SPEC` JSON (when `--output-spec-json` is provided)
  - with `--append-output`, each save appends and deduplicates by `(neuron_id, compartment)`

## Morphology Mutation (multi-SWC)

For multi-SWC structural editing (grow/split/rejoin/thin/thicken/translate + connection specs), use:

- `tools/morphology_mutation_app.py`
- `tools/morphology_mutation.py`
- `notebooks/launch_morphology_mutation.ipynb`
- `notebooks/README_MORPHOLOGY_MUTATION.md`
