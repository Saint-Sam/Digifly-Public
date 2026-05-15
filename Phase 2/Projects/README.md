# Phase 2 Projects

Project folders are the notebook-facing layer for Phase 2 work.

Each project can have its own notebook, configs, notes, and local outputs while importing reusable code from `Phase 2/digifly/phase2`. Project notebooks should change freely for the needs of a specific experiment. Shared simulation code should stay in the package unless the change is useful across projects.

## Layout

```text
Phase 2/Projects/
  project_paths.py
  baseline_run_simulation/
    run_simulation.ipynb
    outputs/
  phase2_workbench/
    Digifly_Phase2_Workbench.ipynb
    outputs/
  elliott_sparrow_2012_beam/
    launch_elliott_sparrow_beam.ipynb
    outputs/
```

The `outputs/` folders are local runtime folders and are ignored by git except for their `.gitkeep` placeholders.

## Project Rule

A project notebook should:

1. call `activate_project("<project_slug>")` near the top
2. import the reusable Digifly code it needs
3. write run outputs under its own `outputs/` folder
4. avoid editing `digifly/phase2` code for project-specific behavior

If a notebook discovers reusable behavior that belongs in the framework, promote that behavior into `digifly/phase2` deliberately and keep the notebook as a thin control surface.
