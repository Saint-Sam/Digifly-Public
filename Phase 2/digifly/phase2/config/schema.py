"""
digifly/phase2/config/schema.py

Phase 2 scaffold file.
Logic will be added during refactor.
"""
from __future__ import annotations

"""
Schema notes for CONFIG keys used by the shareable Phase 2 runner.

This is intentionally lightweight (dict-based) to preserve the notebook workflow
from Phase 2 Master-Copy1.

Key groups:

- Paths:
    swc_dir (required): root directory containing SWCs and local synapses_new.csv files
    morph_swc_dir (optional): alternate SWC root for morphology loading (for reduced SWCs)
    edges_root (derived): <swc_dir>/edges
    runs_root  (derived): <swc_dir>/hemi_runs
    master_csv (derived): <swc_dir>/../all_neurons_neuroncriteria_template.csv

- Selection:
    selection.mode:
        'single'      : selection.neuron_id
        'custom'      : selection.neuron_ids (list[int])
        'hemilineage' : selection.label (e.g. '09A') and optional seeds override

- Edges:
    edges_path (optional): explicit path to edges CSV/Parquet
    edges_csv  (used by neuron_build builders): resolved edges path stored here before wiring
    edge_cache (optional): local SQLite edge cache for custom mode
        - enabled: bool
        - db_path: optional explicit SQLite file
        - build_mode: 'from_edges_files' | 'from_synapses_csv'
        - source_paths: optional list of edge files for cache build
        - query.mode: 'loaded_subgraph' | 'seed_io_1hop'

  Expected edges columns (your canonical format):
    pre_id, post_id, weight_uS, delay_ms, tau1_ms, tau2_ms, syn_e_rev_mV, post_x, post_y, post_z, syn_count

  predictedNT is ignored when tau/erev columns exist.

- Stimulation:
    iclamp_* globals and/or per-driver specs through drivers dict.
    Driver specs support:
        amp, delay, dur, site ('soma'|'ais'), optional neg_pulse, and optional pulse_train

- Biophysics:
    pre_soma_hh / pre_branch_hh / post_soma_hh / post_branch_hh etc.
    Nested dicts are deep-merged by loader.build_config.

- Parallel:
    threads (optional): if set, configures `neuron.h.ParallelContext().nthread(...)`
    for threaded simulation setup in the shared NEURON runner.
    parallel.build_backend (optional): currently defaults to 'single_host'; the
    new ownership/build-planning layer uses this namespace so we can stage into
    distributed GID ownership without changing the notebook/run API.
    parallel.maxstep_ms (optional): default `pc.set_maxstep(...)` ceiling used by
    the distributed_gid backend so cross-rank spike delivery is valid in NEURON.
    parallel.ownership_strategy (optional): stable gid->owner assignment policy
    such as 'round_robin' or 'contiguous'.
    coreneuron_nthread (optional): best-effort CoreNEURON thread count when supported.
    io_workers (optional): worker count used for the output-writing stage.

This module exists so you can open it to quickly see what CONFIG keys are expected.
"""
