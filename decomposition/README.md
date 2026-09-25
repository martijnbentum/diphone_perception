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

Use `load_splits` to retrieve the saved recording assignments. Its `region`
and `components` arguments have the same defaults as `load_manifest`:

```python
from decomposition.sampling import load_splits

filenames = load_splits()  # {'fit': [filenames...], 'eval': [filenames...]}
marker_splits = load_splits(per_marker=True)  # ['fit', 'fit', 'eval', ...]
```

Filename lists include all manifest recordings, even those with no selected
frames. With `per_marker=True`, each recording's label is repeated once per
selected frame. The result follows manifest order, matching `iter_marker_info`;
do not assume it matches the order returned by the Phraser marker store or a
collection with missing embeddings skipped.

Check that loaded markers match this order before using per-marker splits:

```python
from decomposition.load_embeddings import check_marker_alignment

check_marker_alignment(markers)  # True, or ValueError describing the mismatch
```

The check compares marker count, `marker.label`, and `marker.audio.filename`
against every selected frame in manifest order. It loads the default manifest,
accepts the same `region` and `components` options, or takes an already loaded
dictionary as `manifest=manifest`. It does not reorder markers. Run it on the
full marker list; it does not check whether embedding loading skipped entries.

After loading embeddings, use `check_embedding_alignment` to check the actual
loaded collection, including whether entries were skipped:

```python
from decomposition.load_embeddings import check_embedding_alignment

embeddings = load_embeddings(markers, phraser_store=cgn)
check_embedding_alignment(embeddings)
```

It accepts both `Embeddings` and `CNNFeatures`, resolves each item's
`.phraser_object`, and delegates to `check_marker_alignment` with the same
manifest options. Keep the attached Phraser store open and check before
converting to a NumPy matrix. It returns `True` or raises on a mismatch.

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

## Extract marker-start MFCCs

`extract_mfcc.py` stores one MFCC row per saved random-frame marker in a
separate Echoframe store at
`locations.decomposition_random_frames_echoframe_mfcc_store`. The store is
distinct from the phone-probing MFCC store and the decomposition model stores.

```python
from decomposition.extract_mfcc import extract_marker_mfcc
from decomposition.extract_mfcc import find_unaligned_markers
from decomposition.load_embeddings import load_cgn, load_markers

cgn = load_cgn()
try:
    markers = load_markers(cgn)
    unaligned = find_unaligned_markers(markers)
    mfcc_store = extract_marker_mfcc(markers, workers=8)
    try:
        keys = [mfcc_store.make_echoframe_key('acoustic_feature',
            feature_name='mfcc', phraser_key=marker.key)
            for marker in markers]
        vectors = mfcc_store.load_many_frames(keys, frame='center',
            keep_missing=True)
    finally:
        mfcc_store.close()
finally:
    cgn.close()
```

Each stored payload has shape `(1, 39)`: 13 static MFCCs, 13 deltas, and 13
delta-deltas. The 25 ms MFCC window starts exactly at `marker.start`. The
extractor uses Phraser's efficient recording-aligned batch output when that
grid matches the marker. Otherwise it reads a short audio interval and anchors
the MFCC grid at the marker start. Both paths use neighboring recording audio
for delta calculations, clipped at recording boundaries. Markers need enough
remaining audio for one complete 25 ms window; they need not follow Phraser's
recording-aligned grid. Existing payloads are skipped after a shape check;
`vectors` follows marker order and contains `None` for missing payloads. The
caller closes the returned Echoframe store and the Phraser store.

`find_unaligned_markers(markers)` prints the number of marker starts outside
Phraser's recording grid and returns those marker objects in input order. It
uses each recording's sample rate and the same sample rounding as extraction.
It checks only start alignment, without loading audio or checking whether a
complete 25 ms window fits.

## Load saved marker embeddings

`load_embeddings.py` opens the decomposition store, loads saved markers, and
retrieves their stored features without running inference. `load_cgn()` reuses
the default CGN loader in `sampling.py`.

```python
from decomposition.load_embeddings import load_cgn, load_store, load_markers
from decomposition.load_embeddings import load_embedding, load_embeddings
from decomposition.load_embeddings import embeddings_to_matrix

cgn = load_cgn()
try:
    store = load_store(phraser_store=cgn)
    try:
        markers = load_markers(cgn)
        embedding = load_embedding(markers[0], store)
        array = embedding.data
    finally:
        store.close()
    embeddings = load_embeddings(markers[:100], phraser_store=cgn)
    try:
        matrix = embeddings_to_matrix(embeddings)
    finally:
        embeddings.store.close()
    cnn_matrix = load_embeddings(markers[:100], phraser_store=cgn,
        layer='cnn', to_matrix=True)
finally:
    cgn.close()
```

The example assumes at least one saved marker. Defaults match extraction:
model `wav2vec2_nl1_checkpoint-200000`, layer 9, and a 2,000 ms collar.
For `load_embedding`, pass matching `model_name` arguments to the store and
single-embedding loader. `load_embeddings` opens the appropriate Echoframe store
itself using `model_name`; pass a Phraser store through `phraser_store`, or omit
it to open default CGN. Feature loaders return full stored objects, with no frame
selection or pooling. `load_embedding()` returns one `Embedding` or, for
`layer='cnn'`, one `CNNFeature`. Missing metadata or payload raises `ValueError`.

`load_embeddings()` uses Echoframe's batched reads and returns an `Embeddings`
collection or, for `layer='cnn'`, a `CNNFeatures` collection. Access the individual
objects through `.embeddings` or `.cnn_features`, respectively; both attributes
contain tuples. Echoframe warns and skips missing or invalid features, preserving
the order of retained markers. Use each object's `.phraser_key` to match it to
its marker when entries are skipped. Empty input or no valid features raises
`ValueError`. Supply unique marker keys; duplicate keys among loaded features
also raise `ValueError`.

The bulk collection keeps its Echoframe store open as `.store`. Close it after
use. If you omitted `phraser_store`, also call
`embeddings.store.close_phraser_stores()` to close the CGN store opened for you.
With `to_matrix=True`, the loader closes its internally opened stores before
returning. It leaves a supplied Phraser store open.

Use `embeddings_to_matrix(embeddings)` to convert either collection to a NumPy
matrix, or pass `to_matrix=True` to `load_embeddings()` to return the matrix
directly. Each row corresponds to a retained marker and averages its stored
frames, giving shape `(n_markers, embedding_dimension)`. Matrix output does not
include marker identities; retain the collection when you need its Phraser keys.

To obtain separate fitting and evaluation matrices:

```python
from decomposition.load_embeddings import split_matrix
from decomposition.sampling import load_splits

matrix = embeddings_to_matrix(embeddings)
split_list = load_splits(per_marker=True)
matrices = split_matrix(matrix, split_list)
X_fit, X_eval = matrices['fit'], matrices['eval']
```

`split_matrix` only checks that the matrix and split list have equal length,
raising `ValueError` otherwise. It preserves row order within each split and
omits rows with labels other than `'fit'` or `'eval'`. A split with no matching
rows retains the matrix's column count. Ensure the split labels correspond to
matrix row order; alignment is not checked by this function.

`load_store()` can open default CGN automatically. Retrieve that attached store
with `store.load_phraser_store('cgn-awd')`; the caller must close both stores.
The existing `extract_embeddings.load_markers` import remains available.

## Fit centered SVD

`svd.py` operates on NumPy matrices with one sample per row and one embedding
dimension per column. Use the manifest's recording assignments to separate
fitting and evaluation markers before loading their matrices. Keep model,
layer, collar, and pooling settings consistent across both splits.

```python
from decomposition.svd import fit_svd, transform, summarize_spectrum
from decomposition.svd import evaluate_svd, save_svd, load_svd

# X_fit and X_eval are matrices from the respective recording splits.
decomposition = fit_svd(X_fit)
scores = transform(X_eval, decomposition)
summary = summarize_spectrum(decomposition)
evaluation = evaluate_svd(X_eval, decomposition)
save_svd(decomposition, 'final_layer9_svd.npz')
decomposition = load_svd('final_layer9_svd.npz')
```

`fit_svd` gives rows equal weight, subtracts the fitting column mean, and runs
NumPy's reduced SVD in float64 without normalization or coordinate scaling.
It returns a dictionary containing `mean`, `directions`, `singular_values`,
`eigenvalues`, and `n_rows`. Directions are orthonormal **columns**, ordered by
decreasing singular value. There are `min(n_rows - 1, n_dimensions)` components,
including any zero modes; eigenvalues are `singular_values ** 2 / (n_rows - 1)`.
Fitting requires at least two rows and one dimension, with finite real values.
This implementation loads the full matrix and computes all components in memory.

`transform` uses the saved fitting mean and returns sample-by-component scores.
Pass `n_components=k` to retain the first `k` directions. `summarize_spectrum`
returns total variance, variance fractions, cumulative variance fractions,
component counts for 90% and 95% variance, participation ratio, row count, and
the centered rank ceiling. For zero total variance, fractions, component counts,
and participation ratio are zero.

`evaluate_svd` requires at least two evaluation rows. It reports sample variance
along fitted directions, total evaluation variance, and their ratios. It also
reports the evaluation-minus-fitting mean in embedding coordinates (`mean_shift`)
and fitted coordinates (`score_mean_shift`). Variances are measured around
evaluation means with `ddof=1`, separately from those shifts. Fractions can sum
below one when the fitted basis does not span the evaluation variation.

`save_svd` creates parent directories and refuses to overwrite an existing file.
It saves the fitted dictionary only; keep model and sample provenance separately.
`load_svd` reads it without enabling pickle.

Run the SVD tests from the repository root:

```sh
../diphone_env/bin/python -m pytest tests/test_decomposition_svd.py
```

These tests use synthetic matrices and temporary files; no corpus or model
stores are needed. They cover reconstruction, covariance, spectrum summaries,
held-out variance and mean shifts, input validation, and persistence.
