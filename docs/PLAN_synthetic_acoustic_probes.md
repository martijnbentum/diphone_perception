# Plan: synthetic acoustic structure probes

Status: model-independent features 1-3 are implemented. Praat via Parselmouth
is the source-filter backend. Dutch monophthong anchors are drawn from
source-separated literature tables and can be estimated independently from the
selected phone corpus. Model extraction remains deferred.

Paper: Kwanghee Choi and Eun Jung Yeo, *Opening the Black Box of
wav2vec Feature Encoder* (https://arxiv.org/abs/2210.15386).

## Goal

The current milestone implements the model-independent foundation for
scientific probes of representation structure with respect to:

- pure-tone frequency;
- synthetic F1/F2 component frequencies;
- component amplitude/energy.

It will generate controlled audio, verify that audio acoustically, and compute
structure scores from caller-supplied representation matrices. Model loading,
CNN extraction, single-model runners, and checkpoint sweeps are deferred until
these three pieces have stable APIs and tests.

These are scientific probes, not pass/fail software tests. Their scores should
remain continuous until empirical baselines justify scientific thresholds.
Ordinary unit tests will verify that the implementation itself is correct.

## Proposed directory

Use `synthetic_acoustic_probes/`.

This is more precise than `tests/`, which Python tooling normally interprets
as software tests, and it keeps these controlled, model-level experiments
separate from the existing supervised phoneme probes in `probing/`.

Proposed layout:

```text
synthetic_acoustic_probes/
    __init__.py
    stimuli.py
    formants.py
    acoustics.py
    metrics.py
vowel_formant_reference/
    __init__.py
    literature.md
    formant_tables.py
    measurement.py
    selected_phones.py
    aggregation.py
data/
    formants/
        manifest.json
        literature/
        selected_phones/
tests/
    test_synthetic_acoustic_stimuli.py
    test_synthetic_acoustic_formants.py
    test_synthetic_acoustic_acoustics.py
    test_synthetic_acoustic_metrics.py
    test_vowel_formant_tables.py
    test_vowel_formant_measurement.py
    test_vowel_formant_selected_phones.py
    test_vowel_formant_aggregation.py
```

Do not add plotting to the core milestone. Seeded UMAP/PCA plots can be a
separate reporting feature after the full-space metrics are agreed.

`vowel_formant_reference` is an upstream analysis package, not part of audio
synthesis. `formant_tables.py` is the public read API for literature data,
local token measurements, and derived summaries. It writes nothing during
import. It keeps every source as a separate table; it does not concatenate or
pool sources. Stimulus generators consume one explicitly selected standardized
summary table without
importing corpus-selection or formant-estimation logic.

`literature.md` documents every publication and implementation reference used
by this work; it contains prose and citations, not executable table-loading
logic.

Store the formant artifacts in
`vowel_formant_reference/formants/`. Preserve small literature tables,
summarized anchors, and local token measurements as CSV, and keep large local
files out of version control if their size becomes material. Store structured
metadata and manifests as JSON. Each table has citation metadata
containing its full reference, URL, page range, page and table number where
available, record level, schema version, and notes. `manifest.json`
additionally records the generating command, configuration, software versions,
source checksums, and artifact paths.

## Paper method to reproduce

The paper feeds 16 kHz, one-second synthetic signals to the convolutional
frontend and averages the resulting time frames. It uses cosine distance
between representations and UMAP with a cosine metric.

The relevant stimulus families are:

- Temporal consistency: pure sines at 100, 200, ..., 500 Hz.
- Temporal detail: replace the middle of a 200 Hz signal with 800 Hz for
  320, 160, 80, 40, 20, or 10 ms.
- Frequency: unit-amplitude pure sines from 10 Hz toward 8 kHz in 10 Hz
  increments.
- Bias invariance: amplitude 0.5, frequencies 100, ..., 500 Hz, and additive
  biases -0.5, -0.25, 0, 0.25, and 0.5.
- Formants: sum three sines with amplitudes 0.5, 0.35, and 0.15. Use
  F0 = 120 Hz, 30 F1 values from 235 to 850 Hz, and 30 F2 values from 595
  to 2400 Hz. The extended experiment uses F0 values 100, 125, ..., 225 Hz.
- Amplitude: sum 100 Hz and 700 Hz components while varying both component
  amplitudes from 0.1 to 0.5.

The paper's “formants” are independently added sinusoidal components, not
formants produced by filtering a harmonic source. Its “amplitude” experiment
changes spectral balance as well as total energy. The implementation and
reports should use those precise names and avoid claiming that these probes
fully measure pitch perception, speech formants, or perceptual loudness.

## Feature 1: deterministic signal synthesis

### Requirements

- Introduce an immutable `Stimulus` record containing:
  `waveform`, `sample_rate`, `parameters`, and `stimulus_id`.
- Implement a general sum-of-sinusoids function with explicit frequency,
  amplitude, phase, duration, sample rate, and DC bias.
- Implement the centered frequency-replacement signal used by the temporal
  detail experiment.
- Implement named generators for the paper grids:
  `pure_tone_stimuli`, `bias_stimuli`, `formant_stimuli`,
  `amplitude_stimuli`, and `temporal_burst_stimuli`.
- Keep the paper's three-sine “formant” stimuli as an explicitly named
  `sinusoidal_component_formant_stimuli` reproduction baseline.
- Return one-dimensional `float32` NumPy arrays with exactly
  `round(duration * sample_rate)` samples.
- Reject non-finite parameters, negative durations/amplitudes, frequencies at
  or above Nyquist, and accidental clipping unless explicitly allowed.
- Preserve all controlled values in metadata; generation must not depend on
  estimating F0 or formants back from the signal.

### Tests

- Exact sample count, dtype, repeatability, and metadata.
- A single integer-Hz sine has the expected RMS and dominant FFT bin.
- A sum of integer-Hz sines contains all requested spectral peaks.
- The centered replacement occupies the requested number of samples.
- Invalid Nyquist, duration, shape, and clipping cases fail clearly.
- Every default paper grid has the expected size, ordering, and endpoints.

## Feature 2a: Dutch vowel formant reference tables

### Requirements

- Keep every source in a separate physical and logical table.
  `formant_tables.py` exposes each source's complete monophthong table, a
  standardized observation view of that same source, and a standardized
  speaker-balanced summary view where its observation level permits one.
- Do not provide a concatenating or cross-source pooling helper. Any comparison
  must name the source tables explicitly.
- Expose a catalog API such as `available_formant_tables()` and
  `load_formant_table(name, view="native")`.
- Return a `FormantTable` record containing the data plus:
  `name`, `reference`, `url`, `page_range`, `source_page`, `table_number`,
  `record_level`, `schema_version`, and `notes`.
- Preserve every available formant field rather than truncating data to F1/F2:
  `f0_hz`, `f1_hz`, `f2_hz`, `f3_hz`, literature formant levels
  `l1_db`, `l2_db`, `l3_db`, and locally measured bandwidths or confidence
  fields when present.
- The standardized schema contains at least:
  `source`, `dataset`, `record_level`, `population`, `speaker_id`,
  `speaker_type`, `gender`, `age`, `vowel_label`, `ipa`, `f0_hz`, `f1_hz`,
  `f2_hz`, `f3_hz`, `l1_db`, `l2_db`, `l3_db`,
  `duration_seconds`, `stress`, `n_speakers`, `n_tokens`, aggregation method,
  uncertainty fields, measurement settings, and provenance. Fields not
  supplied by a source remain explicitly missing.
- Load the literature tables that provide reproducible numeric data:
  - Pols, Tromp & Plomp (1973): 12 Dutch monophthongs from 50 male speakers
    in `/hVt/` context, including F1-F3 and formant levels;
  - Van Nierop, Pols & Plomp (1973): 12 Dutch monophthongs from 25 female
    speakers in `/hVt/` context, including F1-F3 and formant levels;
  - Weenink (1985): 12 Dutch monophthongs from 10 men, 10 women, and
    10 children, including F0 and F1-F3;
  - Adank, Van Hout & Smits (2004): published male/female group means for
    Northern and Southern Standard Dutch, retaining only the monophthong rows
    from Table I: duration and F0/F1/F2/F3 at 50%.
- Praat supplies the two 1973 datasets through one combined built-in table.
  Split it by source population into separately named and stored Pols and Van
  Nierop tables immediately after loading; do not expose the combined table as
  a scientific source.
- Label the Adank data as published group summaries. The article describes
  1,200 underlying tokens but does not publish their individual measurements
  in its tables, so `formant_tables.py` must not present them as token- or
  speaker-level observations.
- Provide literature-derived speaker-balanced male and female anchors rather
  than approximating all female formants with one multiplier.
- Estimate a second anchor set from the repository's selected phone corpus:
  - select IPA-labelled monophthongs only;
  - consume Phraser Phone objects directly rather than copying the metadata
    phone table;
  - obtain gender from `phraser_phone.speaker.gender()`;
  - use `phraser_phone.start_seconds` and `end_seconds` for audio selection;
  - store only the Phraser phone key, measurement gender, and acoustic result
    in the token artifact;
  - do not inspect stress for full vowels or schwa;
  - estimate F0 and F1-F3 with Praat;
  - summarize a stable central portion of each phone;
  - use sex/gender-appropriate formant ceilings as configurable defaults;
  - record failed and rejected measurements with explicit reasons.
- Treat the selected-phone inventory as observed data, not as the Dutch
  inventory. Missing monophthong categories must remain absent from the local
  table and must not be synthesized by interpolation.
- Aggregate tokens in two stages:
  - first compute a robust center for each speaker and vowel;
  - then compute the gender/population center across speakers.
  This prevents speakers with many selected tokens from dominating the result.
- Report medians, dispersion, speaker-bootstrap confidence intervals, and
  per-vowel speaker/token counts.
- Preserve token-level measurements as an optional detailed artifact, but feed
  only the summarized anchor table to stimulus generation.
- Compare corpus estimates against each literature table separately. Large
  discrepancies should be reported, not silently reconciled or pooled.
- Exclude all phonemic diphthongs from literature tables, local measurements,
  summaries, and stimulus anchors in this milestone.
- Include `/eː, øː, oː/` as phonological monophthongs. Use only their stable
  central measurements in this milestone and document that some Netherlandic
  speakers realize them dynamically; do not add trajectory measurements.
- Include every non-overlapping observed monophthong regardless of stress.
- Use the median of per-speaker medians as the primary group center.
- Measure, aggregate, and persist local formants in one public function. Print
  the selected-vowel count dictionary and every output path.
- Write `phone_formants.csv`, `phone_formants_metadata.json`, and
  `gender_formants.csv`; do not persist an intermediate speaker table.
- Make writing artifacts explicit. Write versioned data only under
  `vowel_formant_reference/formants/`, never during import.

### Tests

- The table catalog lists every documented source and rejects unknown tables
  or views.
- The Pols male and Van Nierop female observations load as separate tables
  despite originating from one combined Praat command.
- Every loaded table exposes its reference, URL, page information, table
  number where available, record level, and schema version.
- Native literature views retain the expected vowels, populations, columns,
  speaker counts, F3 values, and formant levels where supplied.
- The Adank native view is marked as a group summary and contains only the
  monophthong rows from published Table I without inventing speaker-level rows.
- Standardized views retain values and provenance while representing genuinely
  unavailable fields as missing.
- Selection includes monophthongs and rejects consonants and every diphthong,
  including `/ɛi/` present in the selected-phone metadata.
- Selection includes `/eː, øː, oː/` when present, and their standardized rows
  contain central measurements only.
- Central-window selection behaves correctly for short and long phones.
- Praat measurements on controlled vowels recover plausible ordered F1/F2
  values, retain F3 when measurable, and expose failures.
- Speaker-first aggregation gives equal speaker weight despite unequal token
  counts.
- Bootstrap summaries are deterministic for a fixed seed.
- CSV output round-trips through the explicit column schema without losing
  units or provenance.
- The manifest identifies every artifact and its exact measurement
  configuration.
- Fake Phraser phones test gender, second-based timing, overlap exclusion, and
  rejection handling without requiring the cluster audio store. Their stress
  property fails if inspected.

## Feature 2b: acoustic verification and improved formants

### Requirements

- Provide waveform measurements for peak magnitude, RMS, energy, dBFS,
  crest factor, DC mean, and dominant FFT peaks.
- Treat F0 and formant settings stored in `Stimulus.parameters` as ground
  truth. Verification confirms the generator; it does not re-label stimuli
  with estimates from the generated waveform.
- Add a Praat source-filter formant generator alongside the paper
  reproduction:
  - call Praat from Python through `praat-parselmouth`;
  - construct the voiced source through Praat's
    `PitchTier -> PointProcess -> Sound (phonation)` workflow;
  - construct the vocal-tract filter as a `FormantGrid`;
  - vary F1 and F2 only, with an explicit center frequency and bandwidth for
    each;
  - use `Sound & FormantGrid: Filter (no scale)` and handle normalization
    explicitly;
  - apply a short fade at both boundaries;
  - optionally normalize all stimuli to the same RMS so the formant probe
    isolates spectral-envelope shape from overall level.
- Record the Parselmouth and embedded Praat versions with generated stimuli.
- Convert the final Praat `Sound` directly to the package's NumPy-based
  `Stimulus` representation.
- Retain the paper's rectangular F1/F2 grid for comparison, but label
  combinations with F2 <= F1 as acoustically invalid rather than treating
  them as plausible vowels.
- Add a speech-plausible grid requiring ordered, sufficiently separated
  formants (`F1 < F2`) plus Dutch monophthong anchor points.
- Add language-independent cardinal-vowel anchors after the Dutch anchors and
  generator are verified.
- Support multiple F0 values for every F1/F2 setting. This permits later
  measurements of whether formant geometry generalizes across pitch.
- Validate the designed filter using its frequency response. Optionally
  provide an LPC-based waveform estimate as a diagnostic, but do not make an
  estimator's error part of stimulus ground truth.
- Keep F3, higher formants, breath noise, time-varying formants, consonant
  transitions, and a full Klatt synthesizer out of the first implementation.
  A later realism check may add fixed upper formants without making them grid
  dimensions.

### Tests

- Analytic RMS/energy checks for pure and summed sinusoids.
- Peak, dBFS, DC, and crest-factor checks on hand-built waveforms.
- The Praat filter's impulse/frequency response peaks near each configured
  center and has the expected approximate bandwidth.
- The source-filter generator is deterministic for a fixed source phase/seed.
- F0 changes harmonic spacing while fixed filter settings preserve the
  designed spectral envelope.
- Equal-RMS mode equalizes level without changing formant metadata.
- Speech-plausible grids reject or mark crossed and insufficiently separated
  formants.

## Feature 3: quantitative structure metrics

### Requirements

- Compute cosine distance as `1 - cosine_similarity` in the original
  representation space.
- Treat UMAP as an optional visualization only, never as the source of the
  scientific score.
- Report several complementary scores instead of collapsing the result into
  one premature “structuredness” number:
  - pairwise distance-geometry Spearman correlation;
  - local grid-neighbor preservation;
  - cross-validated ridge-regression R² for each controlled dimension;
  - conditional axis monotonicity while the other dimensions are fixed.
- Standardize multi-dimensional stimulus coordinates by their configured
  spans before computing target-space distances.
- For the F0 experiment, compare raw-Hz geometry with log-Hz, Mel, and Bark
  controls, and report linearity of the accumulated adjacent cosine-distance
  scale.
- For the formant experiment, report separate F1 and F2 results in addition
  to their joint grid score.
- For improved formants, evaluate cross-F0 generalization: fit a simple
  decoder or neighborhood mapping on some F0 values and evaluate it on held-out
  F0 values. This distinguishes formant structure from memorizing pitch.
- Support both rectangular grids and irregular speech-plausible grids. Define
  local target neighbors from distances in normalized acoustic-parameter
  space, not from array indices.
- For the amplitude experiment, report A0, A1, total RMS/energy, and relative
  spectral balance separately.
- Make undefined cases explicit: zero-norm representations, constant targets,
  too few samples for a fold, and NaNs must raise or return a documented
  missing result rather than silently becoming zero.

### Tests

- Hand-built ordered embeddings score better than a seeded shuffled control.
- An ideal two-axis grid has perfect neighbor preservation and target
  decoding.
- A representation encoding only one axis scores well only for that axis.
- Cosine distance is zero for identical vectors and handles zero vectors
  explicitly.
- Metrics are deterministic for a fixed seed and invariant to sample order.

## Deferred feature 4: CNN representation extraction

### Requirements

- Accept a loaded model rather than coupling experiment logic to one checkpoint
  path.
- Add a small adapter boundary returning a `(time, feature)` NumPy array for
  each waveform.
- Initially support the model family selected for the first end-to-end run.
  Keep the boundary able to support both Hugging Face wav2vec2 checkpoints and
  SpidR `.pt` checkpoints.
- Reuse `to_vector` model preparation where possible. Its model outputs expose
  `extract_features`, the natural cross-model boundary for the CNN frontend.
- Run in evaluation mode without gradients and preserve the model's device.
- Support bounded batches and preserve stimulus order.
- Compute both stepwise representations and time-averaged representations.
- Derive output frame counts/receptive fields from the model configuration;
  do not assume 49 frames, a 400-sample window, or a 320-sample stride for
  every model.
- Record the exact extraction point, model identifier, frontend shape,
  pooling method, and software versions in result metadata.

### Tests

- A fake adapter verifies shape validation, ordering, batching, and mean
  pooling without loading a large model.
- Evaluation mode and no-gradient behavior are enforced.
- Variable frontend shapes are accepted and reported.
- A separately marked integration test checks one real model when its
  checkpoint is available.

### Deferred question

Which first checkpoint family must work end to end: a Hugging Face wav2vec2
directory or one of the SpidR `.pt` models in `data/model_paths.json`?

## Deferred feature 5: single-model experiment API

### Requirements

- Expose separate `run_f0_probe`, `run_formant_probe`, and
  `run_amplitude_probe` functions.
- Compose them with `run_probe_suite(model, ...)`.
- Keep stimulus generation, extraction, and scoring independently callable.
- Return a structured result containing:
  configuration, stimulus table, pooled representations, optional stepwise
  representations, individual metrics, and provenance.
- Avoid implicit file writes in the scientific functions. Add serialization
  only as a thin explicit helper.
- Permit small smoke-test grids without changing the paper defaults.

### Tests

- A lightweight deterministic fake encoder exercises every suite end to end.
- Each runner forwards its exact stimulus metadata to its result.
- Re-running a suite with the same inputs produces identical scores.
- Small grids work for development; invalid grids fail before model execution.

## Deferred feature 6: checkpoint sweep

This feature and the single-model API are explicitly out of scope for the
current implementation.

### Requirements

- Enumerate checkpoint metadata and sort by actual training step.
- Generate each stimulus set once and reuse it for every checkpoint.
- Run the same extraction point, pooling, grids, and metrics for all models.
- Write a tidy table with one row per checkpoint, probe, score, and controlled
  dimension.
- Save configuration and model provenance beside the table.
- Include untrained and final/pretrained reference models where available.
- Add bootstrap confidence intervals over stimuli or stimulus pairs.
- Plot scores against training step only after the result schema is stable.

### Tests

- Checkpoint ordering parses steps correctly.
- Resume skips only complete results with matching configuration hashes.
- A two-model fake sweep produces the same per-model scores as direct
  single-model calls.

## Recommended implementation order

1. Add `praat-parselmouth` and implement the published Dutch table loaders.
2. Implement the agreed Phraser-native selection and median-of-speaker-medians
   policies without stress filtering.
3. Implement selected-phone measurement, quality control, speaker-balanced
   aggregation, and versioned anchor-table output.
4. Implement `Stimulus`, the paper-reproduction synthesis functions, and their
   parameter grids.
5. Implement the Praat source-filter generator using Dutch anchors and
   speech-plausible grids.
6. Implement waveform and filter-response verification.
7. Implement and unit-test structure metrics using ideal, partially
   structured, and shuffled representation matrices.
8. Review the APIs and metric behavior before introducing a model dependency.
9. Later, implement model extraction, single-model runners, optional plots,
   and finally the checkpoint sweep.

## Reproduction corrections and cautions

- At 16 kHz, a zero-phase 8 kHz sine samples to all zeros. The default grid
  must stop below Nyquist; an exact-reproduction mode may retain the paper/code
  endpoint only with a warning.
- The paper defines cosine distance as `1 - similarity`, while its public
  notebook accumulates `2 - similarity` for the F0 scale. The corrected
  definition should be primary.
- The paper says the examined wav2vec2 frontends have five convolutional
  layers, while the released Base and Large configurations list seven.
  Frontend geometry must be derived from each actual model.
- The prose says amplitude values are spaced evenly in energy, but the public
  code's transformation does not make squared amplitudes evenly spaced. The
  default should follow the stated energy criterion, with exact code
  reproduction available only as an explicit mode.
- A grid visible in UMAP is not sufficient evidence of structure because UMAP
  can distort global distances. All primary scores must use the original
  representations.
- Pure sinusoids isolate controlled frequency geometry, which is useful, but
  they do not establish invariance to harmonic content, phase, speaker, or
  realistic vowel production. A later robustness suite should vary phase and
  source characteristics before making broader speech claims.
