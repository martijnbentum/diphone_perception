# Representation decomposition

## Step 1: random time-point selection

Implemented in `sampling.py`. `make_manifest` selects exact `comp-k` or `comp-o`
and `nl` path components from a Phraser CGN audio inventory by default. Recordings
must be at least **1,000 ms** long. Within eligible recordings, all audio is
available for sampling, including silence and unannotated intervals. Phone and
speaker annotations do not influence selection.

Frame counts come from `frame.make_frames_from_duration`; sampled timestamps
come from indexed `frame.Frames` objects. Both use the library's default window,
step, and origin: 25 ms windows starting at 0, 20, 40, ... ms. Phraser durations
and the manifest's `duration_ms` are in milliseconds; expanded sample timestamps
and context collars are in **seconds**. Frame timing settings are not stored in
the manifest, so reuse the same frame-library timing defaults when expanding it.

The default selection is 418,500 unique frame slots uniformly sampled without
replacement, matching the size of the existing balanced phone inventory. Longer
recordings contribute more eligible frames. Windows can overlap by 5 ms, but
with a unique recording inventory the same recording/frame pair cannot be
selected twice. Trailing audio that cannot form another complete window adds no
frame. No waveforms are physically concatenated.

### Recording splits and reproducibility

`sample_frames` sorts the inventory by `audio_key`, then calls
`assign_fit_and_eval_splits`. This function sorts and shuffles recordings within
each component and assigns half to `fitting` and half to `evaluation`. For odd
counts, the extra recording goes to evaluation; a component containing one
recording contributes only to evaluation. There is no minimum-recording check.
The function updates rows in place and returns the same list.

Sampling then draws globally across cumulative frame counts. The requested
sample count is the total across both splits. Recording counts are approximately
balanced within each component; duration and sampled-frame counts need not be.
Splits isolate recordings, not speakers. Group related files separately if they
represent the same recording/session; this module does not detect that relation.

Frame selection uses a fixed seed of 42. Split assignment has its own random
generator, also defaulting to 42. For the same inventory and sample count,
selection is independent of input audio order. Split assignments do not depend
on sample count. Neither `sample_frames` nor `make_manifest` accepts a `seed`
argument. `assign_fit_and_eval_splits(audio_infos, seed=7)` can explicitly
reassign splits without changing selected frame indices. Adding/removing
recordings can change both selection and assignments; save and reuse the
manifest for an exact selection.

### Create and reload a manifest

Run Python from the repository root in an environment with Phraser, frame, and
progressbar2 installed. The configured CGN store must be available. There is no
command-line entry point in `sampling.py`.

```python
from decomposition.sampling import load_cgn_store, make_manifest
from decomposition.sampling import load_manifest, iter_samples

store = load_cgn_store()
manifest = make_manifest(store, n_samples=418_500)

# In a later session, reuse the saved selection:
manifest = load_manifest()
sample = next(iter_samples(manifest))
```

The default source is `locations.cgn_lmdb`; you can instead pass an already open
Phraser store to `make_manifest`. The default output is
`locations.decomposition_random_frames_base / 'region-nl_comps-k_o.json'`,
currently `../data/decomposition/random_frames/region-nl_comps-k_o.json` relative
to the repository root. An existing output is never overwritten. The output
filename depends on region and component order, not sample count.

Use matching arguments when creating and reloading another selection:

```python
manifest = make_manifest(store, region='vl', components=('comp-o',),
    n_samples=10_000)
manifest = load_manifest(region='vl', components=('comp-o',))
```

### Lower-level sampling and custom output

`sample_frames` returns a **list of audio-info dictionaries**, not a manifest.
It does not filter region, component, or duration. Supply unique recordings with
valid durations sufficient for at least one complete frame. The normal filter
avoids short recordings; directly passing recordings shorter than one frame can
raise an error in `audio_to_info`.

```python
from decomposition.sampling import filter_audios_on_component, sample_frames
from decomposition.sampling import save_manifest, iter_samples

selected = filter_audios_on_component(store.audios)
audio_infos = sample_frames(selected, n_samples=10_000)
manifest = {'audio_infos': audio_infos}
save_manifest(manifest, 'custom_frames.json')
sample = next(iter_samples(manifest, collar_seconds=2))
```

This minimal wrapper is sufficient for `iter_samples`; use `make_manifest` for
the full metadata described below. `save_manifest` accepts a custom path and
creates parent directories, refusing to overwrite an existing file. Read custom
paths with `json.load`; `load_manifest` only uses the configured directory and
region/component filename. Requests exceeding the available frame population,
or negative sample counts, raise `ValueError`. A zero sample count retains the
recording inventory and split assignments with empty frame selections.

### Stored and expanded fields

`make_manifest` stores `n_samples`, `region`, `components`, `sampling_unit`
(`uniform_unique_frame`), `include_all_audio`, `phraser_store`, and `audio_infos`.
`include_all_audio=True` refers to retaining silence and unannotated intervals
inside eligible recordings, not bypassing the recording filters. `phraser_store`
is the string representation of the source store.

Each audio-info row contains `audio_key` (a hexadecimal Phraser key), `filename`,
`component`, `duration_ms`, `n_frames`, `split`, and sorted `frame_indices`.
Recordings with no selected frames remain in the inventory. Seeds, frame timing
parameters, and aggregate split counts are not stored as separate fields.

`iter_samples` takes a manifest dictionary containing `audio_infos` and yields:

- `sample_id`, formatted as `<audio_key>:<frame_index>`;
- `audio_key`, `filename`, `component`, `split`, and `frame_index`;
- `start_second`, the selected frame's recording-relative start;
- `collar_start_second` and `collar_end_second`, the frame start minus the collar
  and frame end plus the collar, clipped to the recording boundaries.

The default collar is 2 seconds on each side; pass a nonnegative
`collar_seconds` value to customize it. Frame-end and frame-center timestamps
are not separate output fields. The module does not print a split summary;
inspect inventory and selected-frame counts before extraction, for example:

```python
for split in ('fitting', 'evaluation'):
    rows = [row for row in manifest['audio_infos'] if row['split'] == split]
    n_frames = sum(row['n_frames'] for row in rows)
    n_selected = sum(len(row['frame_indices']) for row in rows)
    print(split, len(rows), n_frames, n_selected)
```

## Subsequent steps

1. Attach Phraser phone, speaker, and context annotations at sampled positions.
   Retain unannotated points and represent overlapping annotations explicitly.
2. Extract and store embeddings through Echoframe for the final checkpoint and
   selected layer. Use 2 seconds of context on each side, clipped at recording
   boundaries, and retain each selected frame's position. Accept Echoframe's
   frame-selection behavior; do not snap random samples to phone centers.
   The integration must preserve the recording-relative frame grid when windows
   are cropped. Persist actual extraction bounds and available context.
3. Build the sample-by-dimension matrix, fit centered SVD, and check held-out
   strengths and direction/subspace stability.
4. Interpret modes using phone, sonority, voicing, speaker, and recording
   metadata; compare with the existing balanced sample later.

Step 1 uses recording metadata only and does not run model inference.

## Tests

From the repository root, using an environment with the above dependencies and
pytest installed:

```sh
python -m pytest tests/test_decomposition_sampling.py
```

Tests use lightweight recording metadata and temporary manifest directories;
they do not require the CGN store or waveforms.

## Extract saved marker embeddings

Use a Phraser version with marker support and an open store containing the
saved sample. Marker timestamps must be recording-relative milliseconds;
extraction uses them unchanged, with a 2,000 ms collar on each side by default.

```python
from decomposition.extract_embeddings import load_markers
from decomposition.extract_embeddings import extract_marker_embeddings
from decomposition.extract_embeddings import extract_marker_embeddings_for_models

markers = load_markers(cgn_store, label='decomp_random_frames')

# One model in its dedicated decomposition store; caller closes the result.
store = extract_marker_embeddings(markers, gpu=True)
try:
    pass  # Inspect or use the open output store here.
finally:
    store.remove_cached_model()
    store.close()

# Dedicated store per model; model and store cleanup is automatic.
store_paths = extract_marker_embeddings_for_models(markers,
    ['wav2vec2_nl1_checkpoint-200000'], layers=[9], gpu=True)
```

Both workflows default to model-specific stores below
`locations.decomposition_random_frames_echoframe_model_stores`
(`data/decomposition/random_frames/echoframe_model_stores/<model_name>`),
with path separators in model names escaped. Pass an open Echoframe store
as `store=` to override this for single-model extraction. Both workflows include CNN
features, default to batches of 120 markers, and skip existing outputs through
Echoframe. The multi-model function returns model names mapped to store paths
and leaves the markers' Phraser store open. Both extraction functions assume
a non-empty marker iterable whose markers all belong to the same open store. Load markers separately to inspect
or subset them before starting extraction; fitting/evaluation assignments
remain in the sampling manifest.
