# Handoff: F0 CNN normalization and phase diagnostics

Date: 2026-08-12

## Current objective

Use the same CNN representation boundary as full Wav2Vec2 inference and the
paper *Opening the Black Box of wav2vec Feature Encoder*: apply the model's
learned feature-projection `LayerNorm` independently to every final CNN frame,
then perform temporal aggregation (`mean` or `middle`). Re-run the complete F0
experiment with those normalized frames and determine which previously observed
discontinuities remain.

This is **not** L2 normalization and it is **not** the learned projection to the
Transformer hidden size. The relevant representation remains 512-dimensional:

```text
waveform
  -> convolutional feature extractor             raw (frames, 512)
  -> feature_projection.layer_norm                normalized (frames, 512)
  -> feature_projection.projection                Transformer width (e.g. 768)
  -> positional/pre-encoder processing            Transformer layer 0 input
```

The paper retained the second item: normalized `(frames, 512)` features.

## Repository and environment state

- Main project: `/Users/martijn.bentum/repos/diphone/repo`
  - These changes build on `b771411`
    (`Add F0 checkpoint result persistence and plots`).
- Data: `/Users/martijn.bentum/repos/diphone/data`.
- `to-vector`: `/Users/martijn.bentum/repos/to-vector`
  - `main` is at pushed commit `68d1445`.
  - Version `0.1.51` makes `audio_to_cnn(..., normalize=True)` and
    `filename_to_cnn(..., normalize=True)` return frame-wise LayerNorm output by
    default; `normalize=False` retains access to raw convolutional output.
  - Tests compare the default result directly with full-inference
    `outputs.extract_features` for Wav2Vec2 and WavLM and cover HuBERT and a
    nested `Wav2Vec2ForPreTraining` model.
  - Do not modify or delete its unrelated untracked files
    `docs/NOTE_normalized_cnn_features_2026-08-12.md` and
    `docs/gpu_pipeline_snippets_2026-05-04.py`.
- Echoframe: `/Users/martijn.bentum/repos/echoframe`
  - `main` is clean.
  - Its CNN extraction already calls `to_vector.filename_to_cnn` and stores
    `outputs.extract_features`, so no special aggregation code is needed after
    updating `to-vector`.
- The Diphone virtual environment was updated on 2026-08-12 from
  `to-vector==0.1.50` to `to-vector==0.1.51` at commit `68d1445`. The installed
  `audio_to_cnn` and `filename_to_cnn` signatures now default to
  `normalize=True`, and the installed source applies
  `feature_projection.layer_norm` before returning the frame matrix. All 107
  installed packages pass `uv pip check`.

Existing Echoframe `cnn` keys do not encode whether the stored matrix is raw or
normalized. Never mix old and new payloads in one store. Use a greenfield store
or deliberately overwrite every CNN payload. The user prefers greenfield
conditions.

## Why normalization belongs before aggregation

LayerNorm acts independently on all 512 channels of each frame. Therefore the
correct mean representation is:

```python
normalized_frames.mean(axis=0)
```

not:

```python
layer_norm(raw_frames.mean(axis=0))
```

Normalizing the complete matrix during CNN-only extraction preserves the
existing Echoframe API: both
`aggregate_segment(..., method='mean')` and `method='middle'` continue to work
without branches. The operation is computationally negligible compared with
CNN extraction; loading the checkpoint dominates its cost.

## Artifacts and provenance

Raw results were moved to:

- `data/synthetic_acoustic_probes/experiment_f0/_raw_cnn_features/f0_data.npz`
- `data/synthetic_acoustic_probes/experiment_f0/_raw_cnn_features/f0_umap.pdf`
- `data/synthetic_acoustic_probes/experiment_f0/_raw_cnn_features/phase_diagnostics/cnn_phase_diagnostics.npz`

A normalized phase-offset result now exists at:

- `data/synthetic_acoustic_probes/experiment_f0/phase_diagnostics/cnn_phase_diagnostics.npz`

Its timestamp follows the `to-vector` 0.1.51 commit and its norms differ from
the archived raw artifact as expected. The exact runtime environment used to
generate it was not stored in the NPZ, but its values have the expected
normalized-feature signature. The active Diphone environment is now verified
at `to-vector==0.1.51` for subsequent extraction.

The normalized full rerun produced model-specific result bundles and plots:

- `data/synthetic_acoustic_probes/experiment_f0/output_data/wav2vec2_checkpoint-0.npz`
- `data/synthetic_acoustic_probes/experiment_f0/output_data/wav2vec2_nl1_checkpoint-200000.npz`
- `data/synthetic_acoustic_probes/experiment_f0/plots/wav2vec2_checkpoint-0.pdf`
- `data/synthetic_acoustic_probes/experiment_f0/plots/wav2vec2_nl1_checkpoint-200000.pdf`

Each NPZ contains `mean_cnn_features`, `coordinates`, `frequencies`,
`random_state`, `metric`, `model_name`, and `aggregation`. The 799 feature and
coordinate rows follow the ascending 10-through-7,990 Hz frequency array.

## Findings established from the raw complete F0 array

These statements apply specifically to archived raw CNN means in
`_raw_cnn_features/f0_data.npz`, not to normalized CNN features:

- The 799 one-second tones from 10 through 7,990 Hz, their IDs, and their
  10-Hz ordering were correct. There was no stimulus-order or path-order bug.
- The gray path in the archived F0 figure connects consecutive frequencies.
  Each apparent long bridge is two almost overlapping segments around an
  isolated point:
  - 4,790 -> 4,800 and 4,800 -> 4,810 Hz;
  - 6,390 -> 6,400 and 6,400 -> 6,410 Hz.
- Those breaks already existed in 512-D cosine geometry; UMAP enlarged them.
- Raw 4,800 Hz was closest to about 3,600 Hz in CNN space and about 3,560 Hz
  in the two-dimensional UMAP. Those are two different nearest-neighbor
  calculations, and the gray path does not explicitly join those frequencies.
- Raw 3,200 and 6,400 Hz means were almost directionally identical (cosine
  distance about `0.00000334`) and had the two lowest raw norms, about
  `0.07265`.
- The raw-norm sweep showed narrow notches at approximately 3,200, 4,000,
  4,800, 5,600, 6,000, 6,400, and 7,600 Hz; strong responses around
  6,200-6,300 and 7,200 Hz; and rising norms near Nyquist.

The existing raw-feature report
[`REPORT_f0_cnn_phase_diagnostics.md`](REPORT_f0_cnn_phase_diagnostics.md)
describes these archived raw findings. The normalized checkpoint bundles and
plots above are the authoritative full-rerun artifacts.

## Temporal and phase findings

The original full raw diagnostic sweep found:

- Every multiple of 50 Hz was temporally stationary to displayed precision.
  At 16 kHz, the final CNN stride is 320 samples = 20 ms, so such a tone makes
  an integer number of cycles between adjacent output frames.
- The raw 3,200, 4,000, 4,800, and 6,400 Hz vectors were already unusual in
  individual frames; temporal averaging did not create their main behavior.
- Even/odd frame means were almost identical. This was not a decisive
  stride-five test because nonmultiples of 50 Hz cycle through five phase
  alignments and even/odd grouping mixes them.

The implemented decisive control generates sample offsets 0-4 while retaining
frequency, amplitude, duration, and FFT magnitude. See
[`cnn_phase_diagnostics.py`](../synthetic_acoustic_probes/cnn_phase_diagnostics.py).
The current panel contains 34 frequencies x 5 offsets = 170 stimuli, including
the special frequencies, neighbors, 3,590-3,610 Hz, and denser ranges around
4,000 and 4,800 Hz.

From the current normalized phase NPZ:

- 3,200 and 6,400 Hz remain essentially invariant across input offsets; their
  maximum cosine distances from offset zero are approximately `0.00000775` and
  `0.0000206`.
- 4,000 and 4,800 Hz remain strongly phase-sensitive; their corresponding
  maxima are approximately `0.605` and `0.672`.
- 3,600 Hz is also meaningfully phase-sensitive (`0.225`), whereas nearby
  3,590 and 3,610 Hz are about `0.001`.
- Across the targeted panel, excluding 3,200 and 6,400, the median of each
  frequency's maximum distance from offset zero is about `0.216` at multiples
  of 50 Hz versus `0.0056` at other frequencies. If maximum **all-pairs**
  offset distance is used instead, the corresponding medians are about `0.243`
  and `0.0065`. Always state which statistic is meant. An earlier conversational
  value near `0.222` used a slightly different calculation/panel and should not
  be quoted without recomputing it from the chosen artifact.

Interpretation: at a multiple of 50 Hz, every output frame sees the same phase.
Changing the input offset selects a different constant phase, so the temporal
mean can change substantially. At other frequencies, consecutive output frames
already traverse a five-phase cycle, so averaging largely marginalizes phase
and a small input shift has less effect. This does **not** mean different
frequencies spaced by 50 Hz are generally more similar.

The raw norm peaks and notches can therefore partly reflect fixed-phase/
stride-locking. After LayerNorm, raw activation magnitude is removed, but the
phase-dependent directional differences remain relevant to cosine UMAP. The
3,200/6,400 behavior is different: it remains stable across offsets and is not
explained by selecting an unlucky zero phase.

## What the paper actually did

Paper: Kwanghee Choi and Eun Jung Yeo, *Opening the Black Box of wav2vec
Feature Encoder*, arXiv:2210.15386. A local copy is available as
[`Opening_the_Black_Box.pdf`](Opening_the_Black_Box.pdf); official code:
<https://github.com/juice500ml/unbox-w2v-convnet>.

For its F0 experiment, the authors:

1. Generated one-second, 16-kHz, amplitude-one, zero-phase sine waves from 10
   through 8,000 Hz in 10-Hz increments.
2. Ran the convolutional extractor, transposed to `(frames, 512)`, called
   `feature_projection`, discarded its projected first return value, and kept
   its normalized second return value.
3. Averaged the 49 normalized frames and ran cosine UMAP.
4. Plotted an unconnected scatter for the Base model, so their figure did not
   expose long consecutive-frequency path segments.

Important limitations found during inspection:

- Their temporal-stationarity control used only 100, 200, 300, 400, and 500 Hz:
  all are stride-locked multiples of 50 Hz. It does not establish general
  stationarity.
- A zero-phase 8,000-Hz sine sampled at 16 kHz is zero at every ideal sample,
  so that endpoint is effectively silence/numerical residue. Our complete grid
  correctly stops at 7,990 Hz.
- Their frequency-scale notebook calculates neighboring distance as
  `2.0 - cosine_similarity`, despite defining cosine distance as
  `1.0 - cosine_similarity`. Cumulative summation therefore injects a constant
  linear term, weakening their claim that Wav2Vec2 produces a uniformly linear
  frequency scale.
- They did not inspect raw CNN norms, phase offsets, or layer-by-layer CNN
  responses.

## Recommended next actions

1. Run the existing `to-vector` normalization tests and the relevant Echoframe
   CNN tests. Confirm directly on the NL1 checkpoint that CNN-only output equals
   full-inference `outputs.extract_features` frame for frame.
2. Audit remaining `to-vector` raw-CNN callers:
   - the single-item codebook path now inherits normalized `audio_to_cnn` and
     should match full inference;
   - `_audio_batch_to_cnn_outputs` in `wav2vec2_codebook.py` still appears to
     feed raw convolutional output to the quantizer;
   - `hf_batch_helper.inputs_to_cnn` also returns raw frames if its fallback is
     reached.
   Fix and test those paths only if confirmed; do not silently normalize an
   API that explicitly promises raw output.
3. Preserve the completed normalized checkpoint bundles and compare raw and
   normalized results using cosine neighbor distances and UMAP:
   - 3,190/3,200/3,210;
   - 3,500-3,620;
   - 3,990/4,000/4,010;
   - 4,790/4,800/4,810;
   - 6,390/6,400/6,410;
   - the complete adjacent-frequency distance ranking.
4. Extend phase-offset diagnostics across the 400 Hz locking grid and inspect
   layer-by-layer CNN representations.

The completed full normalized sweep establishes that the original UMAP bridges
survive at the exact representation boundary used by the paper and by full
Wav2Vec2 inference. Layer-by-layer CNN inspection remains a later experiment.
