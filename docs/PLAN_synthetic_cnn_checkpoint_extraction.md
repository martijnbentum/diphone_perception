# Plan: CNN embeddings for synthetic acoustic probes

Status: the experiment-specific Phraser storage feature is implemented and
tested. CNN-only Echoframe exposure and checkpoint extraction remain planned.

Scope: store the complete CNN frame matrix for each of the 799 F0 pure-tone
stimuli and each of 122 models: 121 trained `wav2vec2_nl1` checkpoints plus
the random-initialized `wav2vec2_checkpoint-0`. Use one Phraser store per
stimulus experiment and one shared Echoframe store. Extraction is sequential
and unbatched.

## Feature 1: expose CNN-only extraction in Echoframe

### Requirements

- Add a public `compute_cnn` function accepting a Phraser segment, registered
  model name, Echoframe store, collar, device choice, and optional tags.
- Reuse Echoframe's existing segment-time, missing-feature, validation, and
  `CNNFeature` construction helpers.
- Call `to_vector.filename_to_cnn` directly. Do not invoke
  `filename_to_vector`, request transformer layers, or run the transformer.
- Store the complete two-dimensional `(frames, channels)` output under the
  existing model-qualified CNN key. Skip an already complete feature unless
  overwrite is requested.
- Support wav2vec2-family checkpoints first; reject unsupported SpidR models
  clearly. Export and document the function as part of Echoframe's public API.

### Tests

- Mock the model and `filename_to_cnn`; verify segment boundaries and collar
  handling and verify that the full frame matrix round-trips as a `CNNFeature`.
- Assert the full-model extraction path is never called.
- Verify existing output is skipped, overwrite recomputes it, and unsupported
  model types fail clearly.

## Feature 2: add stored stimuli to Phraser

### Requirements

- Add `synthetic_acoustic_probes/phraser_store.py` with two simple public
  functions: `add_stimuli(stimulus_package, store)`, which writes Phraser
  objects and returns `None`, and `load_stimuli(store)`, which returns the
  store's native Phraser Phrase objects.
- Accept any package written by `write_stimuli`, rather than coupling the
  Phraser integration to pure tones or CNN extraction.
- Represent every WAV as one Audio and one full-duration Phrase under a shared
  synthetic speaker. Use the manifest `stimulus_id` as the Phrase label.
- Keep `stimulus_id` and manifest fields as the authoritative identity and
  parameters; use the Phrase label to join the corresponding manifest row.
- Keep the original stimulus manifest authoritative for identity, parameters,
  files, and ordering. Do not maintain a sidecar catalog or wrapper class; the
  experiment-specific Phraser store holds all Phrase keys.
- Require an empty experiment store and fail clearly on a second addition.
  Report missing WAVs, duplicate IDs, and duration mismatches.

### Tests

- Build a temporary store from a small manifest and verify Audio/Phrase
  timing, labels, and keys.
- Verify `add_stimuli` returns `None` and `load_stimuli` returns every native
  Phrase object from the experiment store.
- Run `add_stimuli` twice and verify that the second call fails without
  creating duplicate objects.
- Reject missing files, duplicate IDs, and inconsistent durations.

## Feature 3: extract all 122 checkpoints into one Echoframe store

### Requirements

- Discover exactly `wav2vec2_checkpoint-0` and model names matching
  `wav2vec2_nl1_checkpoint-<integer>` from `model_paths.json`.
- Require an inventory of one random model plus 121 trained NL1 checkpoints.
  Order the random model first and trained checkpoints by numeric step.
- Register all models and the dedicated Phraser source in one Echoframe store.
  Process one loaded checkpoint at a time and release it before advancing.
- For every Phrase, call Echoframe `compute_cnn` with collar zero and retain
  the complete frame matrix. Do not batch inference and do not store a pooled
  duplicate.
- Resume at the model/stimulus level by skipping valid existing CNN features.
  Report computed, skipped, and failed counts per checkpoint and overall.
- Default the stores beneath `data/synthetic_acoustic_probes/`, alongside but
  separate from the generated `f0_pure_tones` WAV directory.

### Tests

- Test discovery, exact inventory validation, and numeric ordering with a
  fixture model catalog.
- With fake models and a small experiment store, verify all models share one
  Echoframe store,
  full frame matrices are retained, checkpoint loading is sequential, and
  collar zero is used.
- Prepopulate selected outputs and verify an interrupted sweep resumes without
  recomputing them.
- Keep real-checkpoint extraction as an explicit smoke test, not a default
  unit test.

## Feature 4: load complete or time-averaged probe matrices

### Requirements

- Load stored CNN features in stimulus-manifest order for a named checkpoint.
- Return full per-stimulus frame matrices by default and expose a helper that
  averages over frames on read, reproducing the article's representation for
  downstream frequency analysis.
- Return stimulus IDs, frequencies, and model metadata with the matrix, and
  fail clearly when a checkpoint is incomplete or channel shapes disagree.

### Tests

- Verify manifest ordering and exact recovery of full matrices.
- Verify on-read frame averaging against a hand-computed result.
- Reject incomplete model inventories and incompatible channel dimensions.

## Non-goals

- Batched inference, transformer hidden-state extraction, NL2/HuBERT/SpidR
  sweeps, frequency metrics, and plots.
- Persisting time-averaged copies of embeddings; pooling remains a read-time
  operation.

## Implementation order

1. Add and release the CNN-only Echoframe API.
2. Update this repository's Echoframe dependency.
3. Build and verify the experiment-specific Phraser store.
4. Implement the resumable 122-checkpoint sweep.
5. Add the read API and run one real-model smoke test before the full sweep.
