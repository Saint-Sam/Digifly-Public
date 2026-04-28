# Hemilineage Experiment Feature Inventory

Date: 2026-04-22

Ground truth reviewed:

- `Digifly-MASTER_MULTIPROCESS/Hemilineage Simulations`
- top-level Hemilineage notebooks
- `Escape-SIZ/Escape-SIZ.ipynb`
- Hemilineage support modules
- method notes, benchmark notes, cache notes, and error logs

## Purpose

This document gathers the experimental features scattered across the Hemilineage Simulations notebooks so Phase 2 can become a reusable public tool instead of a collection of one-off research notebooks.

The long-term public target should be a Phase 2 Simulation Workbench: one executable notebook first, and later a small application, that exposes the important knobs from the notebooks while routing all runs through stable reusable code.

The error logs and checklist files are part of the method. They explain why certain validation, cache, logging, and output rules need to exist in the public version.

## Highest Level Takeaway

Most notebooks share the same foundation:

- choose a circuit or neuron set
- resolve SWCs and metadata from Phase 1
- materialize a chemical edge set
- optionally add gap junctions
- choose passive or HH biophysics
- choose stimulation and recording
- launch NEURON, MPI NEURON, or CoreNEURON
- write a reproducible run folder
- analyze voltage, spikes, timing, instability, rhythmicity, or gait outputs

The reusable public framework should make those shared steps explicit. The notebooks should become presets, recipes, or saved workbench configurations.

## Feature Families

### 1. Environment And Backend Verification

Origin:

- `core_neuron_install_and_verify_v1.ipynb`
- `core_neuron_setup_v2.ipynb`

Unique features:

- NEURON import verification
- CoreNEURON availability checks
- MPI path checks
- launch environment construction
- report JSON generation
- notebook-visible environment diagnostics

Important knobs:

- Python executable
- OpenMPI binary path
- OpenMPI library path
- whether to run an MPI probe
- MPI probe rank count
- CoreNEURON candidate locations
- preferred wheel/site-package location

Public role:

These should become a `Verify Environment` page or notebook section. A public user needs an obvious way to know whether they are running plain NEURON, MPI NEURON, or CoreNEURON.

### 2. Hemilineage Project Launch

Origin:

- `master_hemi_sims.ipynb`
- `hemi_sims.ipynb`
- `hemi_sims_v2.ipynb`
- `hemi_sim_benchmarking.ipynb`
- `hemi_sim_benchmarking_v2.ipynb`
- `hemi_setup_validator.py`
- `hemi_v2_parameter_spec.json`

Unique features:

- full hemilineage project generation
- quick launch mode
- cached build reuse
- serial preflight before MPI
- benchmark sweeps over core size and repeats
- reduction and coalescing pipelines
- distributed gid ownership options
- per-rank timing and build imbalance summaries
- healed morphology override support
- setup validation against a parameter spec

Important knobs:

- project root
- SWC root
- master metadata CSV
- hemilineage label
- neuron/core ID list
- edge-set name
- neuPrint dataset
- edge registry root
- Phase 1 fallback paths
- force rebuild and force rerun flags
- MPI worker count
- distributed gid mode
- max step
- runtime, timing, biophysics, stimulation, recording, and gap profiles
- reduction profile
- coalescing profile
- benchmark core sizes and repeats
- ownership profile
- setup validation toggle
- healed SWC override root and neuron IDs

Public role:

This is the main public Phase 2 workflow. It should become the default `Hemilineage Simulation` preset in the workbench.

### 3. DNg100 Network Construction

Origin:

- `hemi_DNg100_sims_v1.ipynb`
- `dng100_downstream_sims_v1.ipynb`
- `dng100_rhythm_probe_v1.ipynb`
- `dng100_network_support.py`

Unique features:

- DNg100-to-hemilineage bridge network construction
- downstream/motor partner expansion
- minimum synapse thresholds for bridge and partner discovery
- DNg100 focus network materialization
- recurrent shell expansion around a focus circuit
- optional DNg internal edges
- optional DNg-to-motor edges
- silenced motor lists
- DNg100-specific gap presets
- pulse-train stimulation
- paper-rate cache construction
- motor rhythmicity scoring

Important knobs:

- source hemilineage project version
- hemilineage label
- DNg100 IDs
- target motor IDs
- direct partner types
- extra partner IDs
- recurrent shell enabled/disabled
- recurrent shell min synapse threshold
- recurrent shell direction
- allowed partner types/classes
- min DNg100-to-target synapses
- min partner-to-motor synapses
- edge cache rebuild/workers/chunk size
- pulse train frequency, amplitude, width, start, stop, and max pulses
- gap preset, mechanism, direction, conductance, max synapses, and extra pairs
- paper-rate profile
- tracked IDs and latency IDs

Public role:

This should become a `DNg100 Circuit` preset family. It is too specific to be the only Phase 2 workflow, but it contains important reusable machinery for explicit circuit construction and parameterized edge materialization.

### 4. HH, SBI, And Paper-Rate Search

Origin:

- `dng100_hh_sbi_to_paper_rate_v1.ipynb`
- `dng100_hh_sbi_gap_hypothesis_v1.ipynb`
- `dng100_hh_sbi_intrinsic_build_family_v1.ipynb`
- `dng100_hh_gap_feedback_diagnostic_v1.ipynb`
- `dng100_hh_sbi_support.py`
- `run_dng100_hh_stage_shard_worker.py`
- `run_dng100_multileg_walking_sbi.py`
- `run_dng100_multileg_tripod_gait_sbi.py`
- `DNg100_SBI_strategy_and_stage1_record.md`
- `DNg100_local_paper_hh_cached_runtime_results_and_stage3_launch.md`

Unique features:

- staged approximate Bayesian computation style search
- Latin hypercube sampling
- paper-rate target scoring
- walking/gait target scoring
- local circuit persistence diagnostics
- runtime-only search stages
- build-time intrinsic family search
- gap build family search
- sharded stage runs
- resume and merge support
- spike-only recording mode for speed
- failure reports and status polling
- parameter posterior plots
- distance landscapes
- voltage rhythmicity scoring
- multi-leg walking and tripod gait objective functions

Important knobs:

- target actuator
- target leg set
- target period
- stimulation target IDs
- sample count
- random seed
- sample mode
- prior ranges
- stage name
- shard count and shard tag
- sample timeout
- risk thresholds
- tonic current overrides
- intrinsic group template
- synapse group template
- gap pair template
- manual build parameters
- selected build sample
- analysis start time
- bin width
- persistence window
- record profile
- spike-only record toggle
- target frequency and rhythmicity metrics
- accepted period limits
- burst and activity windows

Public role:

This should become an `Optimization / Search` tab in the workbench. It should not live as copied notebook code. The public version needs a schema that declares each searchable parameter, whether it is runtime-safe or build-time, and what cache invalidation it requires.

### 5. Gap Junction Hypotheses And Conductance Design

Origin:

- `dng100_rhythm_probe_v1.ipynb`
- `dng100_hh_sbi_gap_hypothesis_v1.ipynb`
- `dng100_hh_gap_feedback_diagnostic_v1.ipynb`
- `custom_circuit_gap_diagnostics_v1.ipynb`
- `GAP_G_US_per_pair_design_note.md`
- `Gap.mod`
- `RectGap.mod`
- `HeteroRectGap.mod`

Unique features:

- ohmic gap junctions
- rectifying gap junctions
- hetero-rectifying gap junctions
- per-pair gap overrides
- contact-count scaled conductance design
- glia-reference conductance ranges
- all-synapse gap expansion
- max-synapse caps
- component on/off gap motifs
- global gap conductance sweeps
- delayed rectifier feedback hypotheses

Important knobs:

- gap enabled/disabled
- gap mechanism
- global conductance
- pair-specific conductance
- directionality
- contact-count normalization
- maximum synapses per pair
- allowed gap pair lists
- automatic gap pair discovery
- glia reference contact count
- effective conductance target
- target period and recovery constraints

Public role:

Gap logic should be a first-class workbench module, not a hidden dictionary. The public UI should make it clear whether the user is changing a global conductance, a pair-specific conductance, or a contact-normalized rule.

### 6. Custom Circuits, Glia Parity, And Instability Ladders

Origin:

- `custom_circuit_gap_diagnostics_v1.ipynb`
- `custom_circuit_instability_ladder_v1.ipynb`
- `glia_15n_master_parity_v1.ipynb`
- `hemi09a_motor_morph_safety_v1.ipynb`
- `hemi_instability_diagnostics.ipynb`
- `custom_circuit_v2_support.py`
- `custom_instability_ladder_support.py`
- `glia_parity_debug_support.py`
- `hemi09a_motor_morph_safety_support.py`
- `instability_cache.py`
- `Hemi_09A_v2_run_and_instability_findings.md`

Unique features:

- manually specified custom circuit IDs
- DLM/glia seed sets
- exact 15-neuron glia parity circuit
- chemistry-aware edge subsetting
- VNC glutamate policy
- passive non-seed posts
- edge cache coverage checks
- gap on/off parity comparisons
- instability ladder cases
- deep diagnostic recording
- morphology safety cohorts
- severe and mild offender tracking
- rerun variants from prior unstable outputs
- cache session compatibility checks

Important knobs:

- custom neuron ID list
- seed IDs
- target IDs
- DLM IDs
- glia IDs
- gap pairs
- selected case names
- passive-post mode
- stimulation profile
- no-stim profile
- deep trace variables
- deep diagnostic sites
- voltage clipping threshold
- pathological voltage threshold
- offender thresholds
- rerun backend
- cache policy
- variant timeout and polling

Public role:

These should become `Custom Circuit` and `Diagnostics` workbench modes. They are important because they reveal failure modes that normal successful examples hide.

### 7. Morphology Overrides And Mutation Workflows

Origin:

- `hemi_sims_v2.ipynb`
- `hemi09a_motor_morph_safety_v1.ipynb`
- `Escape-SIZ/Escape-SIZ.ipynb`
- `Escape-SIZ/escape_siz_support.py`
- `Escape-SIZ/tools/morphology_mutation.py`
- `Escape-SIZ/tools/morphology_mutation_app.py`

Unique features:

- healed SWC override roots
- sparse healed morphology override sets
- offender-specific morphology safety checks
- SIZ morphology mutation app launch
- mutation manifest discovery
- all-compartment recording attachments
- SWC validation after mutation
- node radius scaling
- node translation
- branch growth
- edge splitting
- detach/reparent mutations

Important knobs:

- healed override enabled/disabled
- override source root
- override profile tag
- override neuron IDs
- mutation project root
- SIZ neuron IDs
- mutation app launch toggle
- max compartments per neuron
- all-compartment recording toggle

Public role:

Morphology override handling belongs in core Phase 2. The full visual mutation app can remain a later optional tool, but the public simulator should understand override manifests and record which morphology source was used for every cell.

### 8. Phase 3 And Rendering Bridges

Origin:

- `dng100_multileg_phase3_render_only.ipynb`
- `render_dng100_generic_video.py`
- DNg100 paper/render scripts
- plot comparison scripts

Unique features:

- render-only Phase 3 bridge from Phase 2 outputs
- video profile selection
- source signal selection
- signal scaling
- crop windows
- expected gait rendering
- paper-rate versus best-fit comparison plots
- topology, spiking-biophysics, locked, and sparse motor comparison plots

Important knobs:

- run directory
- save tag
- video profile
- source signal kind
- global scale
- target control fraction
- crop time window
- floor friction override
- neuron IDs for mutation flow visualization

Public role:

This should be a downstream output/export tab, not part of Phase 2 core simulation. Phase 2 should write standardized outputs so Phase 3 can consume them reliably.

## Notebook And File Crosswalk

This crosswalk is meant to preserve provenance. The public workbench should absorb the method, not copy these notebooks cell-for-cell.

| Source | Distinct method or feature | Public role |
| --- | --- | --- |
| `core_neuron_install_and_verify_v1.ipynb` | NEURON/CoreNEURON/MPI install verification and report writing | environment check |
| `core_neuron_setup_v2.ipynb` | wheel/path cleanup and Digifly-visible CoreNEURON validation | environment check |
| `master_hemi_sims.ipynb` | full-control hemilineage project generation | hemilineage preset |
| `hemi_sims.ipynb` | quick hemilineage launcher with build cache reuse | hemilineage preset |
| `hemi_sims_v2.ipynb` | healed SWC overrides, setup validation, edge-set versioning | hemilineage preset plus morphology controls |
| `hemi_sim_benchmarking.ipynb` | serial benchmark sweeps over core size/repeats | benchmark preset |
| `hemi_sim_benchmarking_v2.ipynb` | MPI benchmark sweeps, ownership profiles, imbalance metrics | benchmark preset |
| `hemi_DNg100_sims_v1.ipynb` | DNg100-to-hemilineage bridge network | DNg100 preset |
| `dng100_downstream_sims_v1.ipynb` | DNg100 downstream and motor partner expansion | DNg100 preset |
| `dng100_rhythm_probe_v1.ipynb` | reduced DNg100 focus circuit, pulse trains, paper-rate cache, gap presets | DNg100 rhythm preset |
| `dng100_hh_sbi_to_paper_rate_v1.ipynb` | staged HH search toward paper-rate target | optimization preset |
| `dng100_hh_sbi_gap_hypothesis_v1.ipynb` | gap hypothesis search, sharded stage runs, resume/merge | optimization preset |
| `dng100_hh_sbi_intrinsic_build_family_v1.ipynb` | intrinsic build family sampling and selected-build runtime stages | optimization preset |
| `dng100_hh_gap_feedback_diagnostic_v1.ipynb` | gap feedback diagnostics, voltage rhythmicity, delayed rectifier hypotheses | diagnostics preset |
| `custom_circuit_gap_diagnostics_v1.ipynb` | custom circuit gap diagnostics and chemistry-aware edge materialization | custom circuit preset |
| `custom_circuit_instability_ladder_v1.ipynb` | instability ladder with deep diagnostic recording | diagnostics preset |
| `glia_15n_master_parity_v1.ipynb` | exact 15-neuron glia parity comparison | validation example |
| `hemi09a_motor_morph_safety_v1.ipynb` | offender morphology safety tests | morphology diagnostics preset |
| `hemi_instability_diagnostics.ipynb` | rerun unstable outputs with variants and cached sessions | diagnostics preset |
| `dng100_multileg_phase3_render_only.ipynb` | Phase 3 render handoff from Phase 2 outputs | Phase 3 bridge |
| `Escape-SIZ/Escape-SIZ.ipynb` | morphology mutation app flow and all-compartment recording | optional morphology tool |
| `dng100_network_support.py` | reusable DNg100 network materialization, gap presets, paper-rate cache helpers | promote to public code |
| `dng100_hh_sbi_support.py` | HH/SBI materialization, sampling, cache, scoring, plotting helpers | promote to public code |
| `custom_circuit_v2_support.py` | custom circuit edge subset, VNC glutamate policy, cache coverage checks | promote to public code |
| `custom_instability_ladder_support.py` | instability ladder profiles, deep trace defaults, cohort summaries | promote to public code |
| `glia_parity_debug_support.py` | compact parity-case setup and comparison helpers | example helper |
| `hemi09a_motor_morph_safety_support.py` | motor morphology safety cohorts, case configs, diagnostics | diagnostics helper |
| `instability_cache.py` | cached variant-session start/reuse/submit/status logic | promote to public code |
| `hemi_setup_validator.py` | case/run/edge schema validation | promote to public code |
| `hemi_v2_parameter_spec.json` | required keys, paths, and critical edge columns | public schema seed |
| `run_hemi_quick_project.py` | command-line hemilineage project launcher | future public script |
| `run_hemi_cached_session.py` | long-lived hemilineage cache server | future public script |
| `run_explicit_cached_session.py` | long-lived explicit-config cache server | future public script |
| `run_parity_case.py` | small parity run launcher | future public script |
| `run_dng100_hh_stage_shard_worker.py` | sharded HH/SBI stage worker | future public script |
| `run_dng100_multileg_walking_sbi.py` | multi-leg walking target stage | optimization example |
| `run_dng100_multileg_tripod_gait_sbi.py` | tripod gait target stage | optimization example |
| `GAP_G_US_per_pair_design_note.md` | contact-normalized and per-pair gap conductance design | public method note |
| `CACHE_FUTURE_WORK.md` | structural cache plus runtime overlay design | public cache design |
| `NOTEBOOK_ERROR_LOG_AND_CHECKLIST.md` | failure history and guardrail checklist | public logging design |

## Error Logs As Design Requirements

Origin:

- `NOTEBOOK_ERROR_LOG_AND_CHECKLIST.md`
- `CACHE_FUTURE_WORK.md`
- `DNg100_SBI_strategy_and_stage1_record.md`
- `DNg100_local_paper_hh_cached_runtime_results_and_stage3_launch.md`
- `Hemi_09A_v2_run_and_instability_findings.md`
- `hemi_sim_benchmarking_v2.md`

The public workbench should preserve the lessons from the error logs. These are not just historical notes; they are rules for reproducible simulation.

### Required Run Artifacts

Every public Phase 2 run should write:

- `run_manifest.json`
- `case.json`
- `resolved_config.json`
- `_phase_timings.json`
- `stdout.txt`
- `stderr.txt`
- `environment_report.json`
- `cache_fingerprint.json`
- `edge_schema_report.json`
- `run_status.json`
- `failure_report.json` if the run fails
- `notes.md` or `fix_log.md` for user-visible diagnosis

### Required Log Fields

Each run manifest should include:

- Digifly public version or source digest
- notebook/app preset name
- simulation mode
- input SWC root
- morphology override root
- neuPrint dataset
- edge-set name
- neuron IDs
- gap configuration digest
- biophysics configuration digest
- stimulation configuration digest
- record configuration digest
- cache identity
- backend requested
- backend actually used
- MPI rank count requested
- MPI rank count actually used
- CoreNEURON requested
- CoreNEURON actually used
- start time
- stop time
- exit status
- output files written

### Known Failure Modes To Guard Against

The public tool should detect or warn on:

- run-name reuse hiding stale config
- missing neurotransmitter, Erev, or synapse timing enrichment
- glutamate mapping mismatches
- stale helper modules after notebook edits
- CoreNEURON or GPU fallback when the user thinks it is active
- MPI mismatch where `nhost` stays at 1
- stale in-memory imports inside notebooks
- broken healed SWC symlinks
- stale morphology override directories
- cache fingerprint mismatch
- stale custom edge-cache coverage
- edge-set name collisions
- JSON key type mismatches in owner maps
- DNg100 downstream materialization hanging from loose thresholds
- edge-cache subcircuits losing chemistry text enrichment
- wrong spike threshold defaults
- Phase 3 bridges assuming unavailable spike files
- pulse-train runs accidentally behaving like sustained clamps
- distributed cached seed-clamp rebuilds on non-owner ranks
- missing standard-library imports in helper modules

### Cache Identity Rules

The public workbench needs to classify every control as one of:

- build-time: changing it requires rebuilding cells, edges, mechanisms, or wiring
- runtime-safe: changing it can reuse the structural cache
- analysis-only: changing it only reprocesses existing output
- unsafe/unknown: changing it requires a warning until tested

Build-time examples:

- SWC root
- morphology override root
- neuron ID set
- edge-set name
- edge CSV contents
- chemistry enrichment policy
- gap pair topology
- intrinsic biophysics family
- synaptic mechanism family
- passive versus active post policy

Runtime-safe candidates:

- tstop
- dt, with caution
- IClamp amplitude, delay, and duration
- pulse-train settings
- record subset
- analysis windows
- some tonic stimulation overlays

Unsafe or build-time until proven otherwise:

- arbitrary synaptic conductance family changes
- arbitrary intrinsic conductance family changes
- gap mechanism type changes
- topology changes
- cache reuse after code edits without a source digest check

The cache redesign note points toward a structural cache plus runtime overlays. That is the right public direction.

## Proposed Phase 2 Simulation Workbench

The first public version can be a notebook named:

```text
Phase 2/notebooks/Digifly_Phase2_Workbench.ipynb
```

Later it can become a small app built on the same schema and runner code.

### Recommended Workbench Tabs

1. `Project`
   - choose project root, output root, dataset, and run name
   - show environment/backend status

2. `Circuit`
   - choose hemilineage, DNg100, custom circuit, glia parity, or saved preset
   - choose neuron IDs and expansion rules

3. `Inputs`
   - choose SWC root, master CSV, edge registry, and Phase 1 fallback paths
   - validate required files before launch

4. `Edges`
   - choose edge-set name and chemistry policy
   - materialize, inspect, or rebuild edge cache
   - show schema completeness

5. `Biophysics`
   - passive, HH, intrinsic family, active posts, and spike threshold policy
   - make cache impact visible

6. `Stimulation`
   - IClamp, pulse train, tonic current, silencing, and target IDs
   - make runtime-safe controls visible

7. `Gap Junctions`
   - select mechanism, pair rules, conductance mode, and presets
   - show per-pair/contact-normalized values

8. `Recording`
   - choose voltage, spike-only, deep diagnostics, all-compartment, or Phase 3 export records

9. `Backend`
   - plain NEURON, MPI NEURON, CoreNEURON, worker count, ownership profile, cache mode
   - show requested versus actual backend after launch

10. `Search`
    - configure SBI/ABC stages, parameter priors, samples, seeds, shards, and resume/merge behavior

11. `Diagnostics`
    - instability ladder, offender detection, morphology safety, voltage rhythmicity, gait scoring

12. `Outputs`
    - list written artifacts, plots, benchmark tables, Phase 3 handoff files, and error reports

13. `Error Log`
    - append user notes and machine failures into the run folder
    - surface known checklist warnings

### Required Registries

The workbench should be driven by registries instead of copied notebook cells.

Parameter registry:

- display name
- config path
- type
- default value
- allowed range or options
- source notebooks
- cache impact class
- validation rule
- public help text

Preset registry:

- hemilineage quick
- hemilineage benchmark
- DNg100 bridge
- DNg100 downstream
- DNg100 rhythm probe
- DNg100 HH paper-rate search
- DNg100 gap hypothesis
- custom circuit gap diagnostic
- glia parity
- Hemi09A instability diagnostic
- morphology safety
- Escape-SIZ morphology mutation

Run history registry:

- run ID
- preset
- start/stop times
- status
- case file
- resolved config
- output directory
- cache fingerprint
- backend actuals
- error report
- fix notes

## What Should Move Into Public Code

Promote:

- environment verification helpers
- setup validator and parameter spec
- edge materialization helpers
- DNg100 circuit construction helpers
- custom circuit edge subset helpers
- gap preset and per-pair conductance helpers
- cache start/reuse/submit/status helpers
- HH/SBI sampling and analysis helpers
- instability diagnostic helpers
- morphology override resolution
- run artifact and error-log writers

Keep as examples or presets:

- exact DNg100 paper workflows
- glia parity case
- Hemi09A instability notebooks
- motor morphology safety notebooks
- Phase 3 render-only notebooks
- paper comparison plots

Do not import as public framework:

- generated project folders
- local cache folders
- videos and render outputs
- checkpoint notebooks
- local result tables
- Desktop-specific absolute paths

## Recommended Next Build Step

Before building the workbench notebook, create a small public module layer:

```text
Phase 2/digifly/phase2/workbench/
  __init__.py
  controls.py
  presets.py
  validation.py
  run_artifacts.py
  cache_identity.py
  error_log.py
```

Then the notebook can stay clean:

- load a preset
- render controls
- validate inputs
- launch a run
- summarize outputs
- append error/fix notes

That shape keeps future Choice/preset additions simple. A new experiment becomes a new preset plus a few registered controls, not another copied notebook.
