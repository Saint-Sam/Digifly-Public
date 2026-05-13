# Elliott and Sparrow 2012 Beam Replication

This standalone project targets:

Elliott CJH, Sparrow JC. 2012. In vivo measurement of muscle output in intact Drosophila. Methods 56:78-86. DOI: 10.1016/j.ymeth.2011.10.005.

The lost dataset was not a voltage dataset. It was a two-axis flexible optical beam readout from intact flies or larvae. This project keeps the paper-specific replication work separate from the main Digifly checkout, while still providing a Digifly Phase 2 template for the neural side of the experiment.

The model should be organized as:

1. A Phase 2 neural activation experiment for the escape circuit.
2. A muscle/behavior transform from neural events to force.
3. A flexible-beam sensor transform from force to vertical/horizontal traces.

## What The Paper Constrains

- The adult jump is driven by the giant fiber pathway: GF -> TTM motor neuron -> TTM/TDT.
- The stimulus was electrical eye stimulation of the GF pathway: 28 V, 1 ms, about 10x threshold for a 3-day wild type fly.
- The optical fiber was 160 mm long, card platform 5 x 5 mm, card placed 7.5 mm from the movable end, quadrant photodiode 0.5-1 mm from the light-pipe end.
- Calibration was linear up to 500 um of beam displacement, with about 0.5 uV/um sensitivity.
- For fast jump events, the paper treats the output as an energy/output index because the force duration is shorter than the beam resonant period.
- For slow larval peristalsis, the output is a direct force measure because contraction period is longer than beam resonance.

## Figure Targets

- Fig. 1: shak-B2 affects GF coupling. Wild type one-leg responses are consistent. shak-B2 one-leg responses are all-or-none. With six legs, only about 50% of shak-B2 flies jump; successful responses are less than half wild type and often begin with the opposite vertical polarity.
- Fig. 3: amph26 null nearly abolishes jump output while leaving walking force and vertical climbing speed not significantly different from wild type.
- Fig. 4: adhesion/grip force rises as the fly is lifted, then drops in at least two release steps with beam ringing.
- Fig. 5: flight downdraft appears as about 5 uN mean downforce plus beam-dominated oscillations near 14.95 Hz vertical and 10.88 Hz horizontal, followed by upward leg-grip motion at landing.
- Fig. 6: parkin25 larvae show similar contraction force amplitude to wild type but lower contraction frequency and lower crawling velocity.

The numeric values in `configs/paper_targets.json` are paper-text values where available and visual estimates from the figures where the PDF does not provide tabular data. The first serious calibration pass should digitize the PDF figure traces and bar plots.

## Digifly Integration

Use this repository as the paper-specific experiment layer beside a local `Digifly Public` checkout. The waveform generator runs standalone; the Phase 2 config is a template to run from the Digifly repo root or to copy into `Digifly Public/Phase 2/apps/Elliott_Sparrow_2012_Beam/`.

Use the existing Phase 2 public escape IDs as the neural side of the experiment:

- GF: `10000`, `10002`
- TTMn: `10068`, `10110`
- Optional PSI: `11446`, `11654`
- Optional DLMn set from the `Full Escape` preset when modeling flight initiation rather than only jump output.

Add three layers on top of the existing Phase 2 run:

1. **Escape activation preset**
   - Stimulate both GF IDs with a single short event near `stim_time_ms`.
   - Record spikes and soma voltages from GF and TTMn IDs.
   - Use gap configuration to represent GF/GCI coupling hypotheses.
   - Use synaptic/activation attenuation presets for shak-B2 rather than treating it as a morphology change.

2. **Muscle output transform**
   - Convert TTMn spikes or thresholded TTMn voltages into left/right TTM twitch impulses.
   - Wild type: bilateral TTM activation, strong downward vertical trace, smaller variable horizontal component.
   - shak-B2: probabilistic unilateral or failed activation.
   - `phase2_gated_jump`: jump only when the Phase 2 run contains a TTMn spike.
   - `phase2_gated_shakB2`: use the same Phase 2 spike gate, but with the reduced/opposite-polarity shak-B2 beam response.
   - `standing_still`: flat no-jump baseline for a fly that does not move.
   - amph26: normal upstream neural activation but greatly reduced TTM muscle gain.
   - parkin25 larva: unchanged contraction amplitude, reduced oscillator frequency.

3. **Beam readout transform**
   - For jumps: second-order underdamped beam response to short impulse-like muscle output.
   - For walking/grip/larva: slower direct force traces with optional beam ringing at sudden release.
   - Store outputs as `beam_waveforms.csv` with `t_ms`, `vertical`, `horizontal`, and `vector` columns.

## Files

- `configs/paper_targets.json`: paper-derived target table for figures and genotypes.
- `configs/digifly_escape_run_template.json`: Phase 2 template for a single escape-circuit run.
- `notebooks/launch_elliott_sparrow_beam.ipynb`: notebook launcher for beam traces, Phase 2 voltage plots, and optional Digifly escape runs.
- `tools/beam_waveform_model.py`: standalone waveform generator and Phase 2 spike-to-beam adapter.

## Quick Start

Preferred local workflow: open `notebooks/launch_elliott_sparrow_beam.ipynb` from Jupyter. It provides dropdown/button controls for paper beam waveforms and plots soma voltages from Digifly Phase 2 `records.csv` files.

For causal neural-to-beam runs, choose `phase2_gated_jump` or `phase2_gated_shakB2` in the notebook and paste/select a Phase 2 run folder. If `spike_times.csv` contains a TTMn spike (`10068` or `10110`), the beam jumps at that spike time; if not, the output stays flat. The optional Digifly launch cell includes notebook tunables for GF current clamp, synaptic weight/timing, GF-to-TTMn gap conductance, and GF-to-TTMn summed-strength equalization, with `wildtype_gap`, `shakB2_no_gap`, and `fly_stands_still` presets.

Generate paper proxy waveforms without running NEURON:

```bash
python tools/beam_waveform_model.py \
  --condition wildtype_jump \
  --out-dir /tmp/elliott_sparrow_demo
```

Adapt a completed Phase 2 run:

```bash
python tools/beam_waveform_model.py \
  --condition wildtype_jump \
  --phase2-run "/path/to/Phase 2 run folder" \
  --out-dir /tmp/elliott_sparrow_from_phase2
```

The adapter currently uses a deliberately simple twitch-to-beam model. That is intentional: the first fork should make assumptions explicit, then tune them against digitized figure traces.
