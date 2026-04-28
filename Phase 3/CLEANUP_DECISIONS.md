# Phase 3 Cleanup Decisions

## Best notebook baseline

- Use `notebooks/user/Phase 3 - NEURON to MuJoCo.ipynb` as the main Phase 3 notebook.
- Keep `Phase 3 v2.0.ipynb` only as a legacy reference.
- Do not build new work on the old 115-cell notebook. It contains duplicated render attempts, outdated path assumptions, and exploratory dead ends.

## Keep

- `src/phase3_bridge`: reusable bridge and render helpers
- `notebooks/user`: main workflow
- `notebooks/debug`: diagnostics and mapping enrichment
- `configs`: named signal/render profiles
- `data/inputs/mappings`: current mapping tables
- `data/derived` and `data/outputs`: generated artifacts

## Archive

- Root-level legacy notebooks
- Root-level duplicate mapping CSVs and docs
- `neuro_fly/` prototype scaffold
- Old logs and one-off reference artifacts

## Next technical priorities

1. Improve MN-to-actuator coverage for the motor neurons that actually spike in your selected Phase 2 run.
2. Tune the `control_map` and `signal` profiles against visible MuJoCo behavior.
3. Add a small validation notebook or script that reports which mapped actuators are active before rendering.
4. Prioritize hemilineage coverage gaps first, since many added motor neurons can spike before they have Phase 3 actuator mappings.
