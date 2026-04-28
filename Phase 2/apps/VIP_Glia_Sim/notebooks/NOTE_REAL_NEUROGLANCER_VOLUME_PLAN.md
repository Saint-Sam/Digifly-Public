# Real Neuroglancer Volume Plan

Date: 2026-04-27

## Current state

The current `neuroglancer` render mode in the mutation app is **synthetic**, not a true streamed Neuroglancer volume.

Relevant code:

- App entry: `../tools/morphology_mutation_app.py`
- Synthetic SWC-to-volume build: `../tools/morphology_mutation_app.py`
- Synthetic render branch: `../tools/morphology_mutation_app.py`
- Screenshot note that this mode is raster-first: `../tools/morphology_mutation_app.py`

This explains the current limitations:

- close zoom can become blurrier because the app is rendering a fixed synthetic voxel grid
- interactive speed drops because PyVista/VTK is resampling one in-memory volume rather than streaming multiscale chunks
- the surface can look slightly "built from shapes" because it is inferred from SWC radii, not from real EM/segmentation voxels

## What we learned about the real MANC data

The source seen in Neuroglancer through neuPrint points to the MANC bucket:

- Segmentation clue from neuPrint scene: `precomputed://gs://manc-seg-v1p2/manc-seg-v1.2`
- Base Neuroglancer scene referenced publicly: `gs://manc-seg-v1p2/manc-v1.2.3-neuprint-layers.json`

Public references:

- Janelia MANC page: <https://www.janelia.org/project-team/flyem/manc-connectome>
- neuPrint dataset metadata example: <https://natverse.org/neuprintr/reference/neuprint_info.html>
- TensorStore Neuroglancer precomputed driver: <https://google.github.io/tensorstore/driver/neuroglancer_precomputed/index.html>
- cloud-volume: <https://github.com/seung-lab/cloud-volume>

Important distinction:

- `flyem-manc-exports` is for flat connectome tables
- the real EM and segmentation layers are hosted separately as Neuroglancer `precomputed` volumes
- the MANC page says subvolume download is possible via TensorStore or cloud-volume

## Likely requirement for the "real" look

For the nice Neuroglancer-style textured render, segmentation alone may not be enough.

Likely needed:

1. the segmentation layer for neuron/body labels
2. the EM image layer from the published scene
3. a way to crop aligned subvolumes around the neuron of interest

Using the EM layer is what should give the real surface texture instead of the current synthetic shell.

## Suggested future plan

### Phase 1: verify the public layer paths

1. Open the published MANC Neuroglancer scene and inspect the exact image-layer and segmentation-layer URLs.
2. Confirm which layer path corresponds to EM and which corresponds to segmentation.
3. Record the exact scale/resolution names used by the bucket.

### Phase 2: test local subvolume download

1. Make a small standalone script or notebook, separate from the mutation app.
2. Use TensorStore or cloud-volume to pull a tiny crop from the MANC bucket.
3. Save the crop locally and verify orientation, voxel spacing, and axis order.
4. Confirm whether access is truly public or if any credentials are needed.

### Phase 3: align to morphology coordinates

1. Pick one known neuron already loaded as SWC in the mutation app.
2. Determine whether the SWC coordinates already match the Neuroglancer voxel/world coordinates directly.
3. If not, compute and document the transform between SWC space and volume space.
4. Validate alignment by overlaying the SWC on a downloaded crop.

### Phase 4: integrate without breaking the current workflow

1. Keep existing `tube`, `skeleton`, and synthetic `neuroglancer` modes intact.
2. Add a new opt-in mode such as `neuroglancer_real` or `manc_volume`.
3. Continue using the hidden line mesh for picking/editing so mutation tools remain stable.
4. Render the real cropped volume only for display, not for interaction logic.

### Phase 5: improve performance

1. Start in solo mode by default for real-volume rendering.
2. Download/view only bounded local crops around the visible neuron.
3. Prefer multiscale chunk reads over full-volume downloads.
4. Cache recent crops on disk so repeated launches are fast.

## Proposed minimal deliverable

The smallest useful next step is:

1. identify the exact public EM + segmentation layer URLs from the MANC scene
2. download one small crop around one DNp01 neuron
3. prove that the crop and SWC can be aligned locally

If that works, then integrating it into the mutation app becomes much safer and more predictable.

## Recommendation

Keep the current synthetic renderer for now.

When revisiting this, do **not** replace it first. Add the real-volume path as a separate experimental mode until:

- local crop download works reliably
- SWC/volume alignment is confirmed
- performance is acceptable in solo mode
