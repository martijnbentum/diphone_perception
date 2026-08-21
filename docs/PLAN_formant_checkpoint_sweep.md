# Plan: formant checkpoint sweep

Date: 2026-08-20

This records decisions made while planning a wav2vec2 checkpoint sweep for
the F1/F2 formant grid (paper §3.4, "vowel space construction"), mirroring
the sweep already implemented for F0 (§3.3) in `f0_experiment.py` and
`f0_checkpoint.py`. It is a design note, not a requirements spec like the
other `PLAN_*` docs — it exists so the reasoning behind these choices isn't
lost before the formant experiment module is actually written.

Naming: the codebase's established noun for this axis is "formant"
(`formants.py`, `formant_stimuli`, `praat_formant_stimuli`,
`locations.formant_grid_stimuli`). The module was nonetheless built as
`f1f2_experiment.py` (`f1f2_` prefix throughout, including `locations.py`)
per explicit instruction, kept distinct from the existing
`locations.formant_grid_stimuli`/`sinusoidal_component_formant_stimuli`
default location rather than reusing it — `locations.f1f2_stimuli` lives
nested under a new `f1f2_experiment/` root, mirroring `f0_experiment/`
exactly.

## Status: implemented

`f1f2_experiment.py`, `locations.f1f2_*`, and
`tests/test_synthetic_acoustic_f1f2_experiment.py` exist, mirroring
`f0_experiment.py` structurally: own Phraser store
(`F1F2_PHRASER_SOURCE_ID = 'f1f2-formant-grid'`), own Echoframe store
(`locations.f1f2_echoframe_store`), `make_f1f2_x_y` built on
`checkpoint_naming`/`storage.read_manifest`/
`phraser_store.align_phrases_to_manifest`, and
`save_f1f2_checkpoint_result(s)` using the `cnn`/`umap_*`/no-`aggregation`
npz schema decided below. Stimuli are the fixed-F0=120Hz, 30x30 F1xF2 grid
(`sinusoidal_component_formant_stimuli()` defaults) -- the paper's Fig. 7
setup, not the extended F0xF1xF2 Fig. 8 sweep. `aggregation` is hardcoded
to `'mean'` inside `make_f1f2_x_y` rather than exposed as a parameter, per
the decision that F0's `'center'`/`'mean'` flexibility won't be exercised
here. `_f1f2_targets(row)` returns `(f1_hz, f2_hz)` as a single tuple
"target" per manifest row -- `align_phrases_to_manifest`'s `extract_target`
callback was already generic enough to carry a multi-value target without
any change to that function.

Not yet done: actually running the CNN extraction and checkpoint sweep
(`create_f1f2_echoframe_store`, `extract_f1f2_cnn_features`,
`save_f1f2_checkpoint_results`) against real wav2vec2 checkpoints, and any
formant-side analysis/plotting (§3.4's grid-structure and vowel-space
questions) -- that's the `metrics.py` Feature-3 functions plus new plotting
code, not a port of `f0_distances.py`/`f0_plot.py`.

## Decision: keep the Phraser store and Echoframe store F0-specific

Formants get their own Phraser store, their own Echoframe store, and their
own `locations.py` directory tree, entirely separate from F0's. Nothing
about F0's stored data or paths changes because formants exist.

## Decision: reuse the already-extracted generic modules, duplicate the rest

`checkpoint_naming.py` (renamed `_checkpoint_naming.py`),
`storage.read_manifest`, and `phraser_store.align_phrases_to_manifest` were
extracted from `f0_checkpoint.py`/`f0_experiment.py`/`cnn_phase_diagnostics.py`
because they were exact, zero-design-cost duplicates. `cnn_extraction.py`,
`echoframe_store.make_x_y`/`create_store`/`select_wav2vec2_nl1_checkpoints`,
`umap_projection.project_umap`, and `metrics.py`'s Feature-3 scores were
already generic. `formant_experiment.py` should call all of these directly,
the same way `f0_experiment.py` does.

The orchestration layer around them
(`create_*_echoframe_store`, `make_*_x_y`, `save_*_checkpoint_result(s)`)
should **not** be pulled into a shared parameterized module. It's short,
experiment-specific glue with real per-experiment differences (target
shape, Phraser source ID, `locations` paths), and abstracting it now would
be designing for a hypothetical third experiment based on only two data
points — indirection without simplification. `formant_experiment.py` should
read as its own short file, structurally similar to `f0_experiment.py` but
independent, not parameterized by it.

`f0_distances.py`/`f0_metrics.py`'s scalar adjacency/Hz-Mel-Bark framing is
not reusable for a 2D F1/F2 grid and isn't meant to be reused — the formant
side should use `metrics.py`'s already-generic Feature-3 functions
(`local_neighbor_preservation`, `pairwise_geometry_spearman`,
`conditional_axis_monotonicity`) instead.

## Decision: `locations.synthetic_acoustic_probes_echoframe_store` → `f0_echoframe_store`

Done. It was nested under `f0_experiment/` and misleadingly named as if it
were shared package-wide infrastructure; renamed to match its `f0_`-prefixed
siblings. Pure attribute rename, no path value change, no data migration.

Formants need their own parallel constants when the module is built
(stimuli dir, Phraser store dir, Echoframe store dir, output/plots dirs),
following the same `f0_*` naming pattern but with a `formant_` prefix.

## Decision: no F0 storage-format migration

The checkpoint-result npz schema changes proposed below apply to new
formant code only. F0's existing on-disk results
(`mean_cnn_features`, `coordinates`, `frequencies`, `metric`, `model_name`,
`aggregation`, `random_state`) are left exactly as they are — regenerating
the full 122-checkpoint F0 sweep is expensive (see
`docs/HANDOFF_f0_normalized_cnn.md`), and nothing about the *storage
format* actually requires it: a formant bundle with different field names
can be read by its own loader without F0's loader or stored files changing
at all.

## Checkpoint-result npz schema for formants (not F0)

F0's saved bundle has two rough edges worth not repeating:

- `mean_cnn_features` bakes an aggregation choice into an array name, and
  a separate `aggregation` metadata field then repeats the same
  information. If formants only ever use mean aggregation, recording it
  redundantly adds no information.
- `coordinates` and `metric` are UMAP-specific but stored under generic
  names. `F0Checkpoint._set_info` already renames them to `umap_metric`/
  `umap_coordinates` immediately on load — the reader itself treats the
  on-disk names as opaque enough to need renaming, which is a sign they
  should just be named that way on disk. An unprefixed `metric` also risks
  being misread as a `metrics.py` structure-metric score.

Proposed formant bundle fields: `cnn` (the CNN representation matrix, no
aggregation baked into the name), `umap_coordinates`, `umap_metric`,
`umap_random_state`, `model_name`, plus one array per controlled target
(`f1_hz`, `f2_hz`). No `aggregation` field. If a future formant checkpoint
loader validates required fields the way `F0Checkpoint._validate` does, its
field set should be defined independently rather than shared with F0's,
since the two bundles are no longer field-compatible by design.

## Decision: checkpoint discovery is shared, npz field validation isn't

`_checkpoint_naming.discover_checkpoints(output_directory)` globs a
directory's `*.npz` files, converts each to a model name, and returns them
sorted by training step — this part genuinely is identical for any
checkpoint sweep regardless of what's inside the npz files, so it now lives
in `_checkpoint_naming.py` and `F0Checkpoints.__init__` calls it directly
(replacing its own glob/sort/`_validate` logic). A future
`FormantCheckpoints` should do the same. What stays per-experiment is the
npz field validation and distance/metric logic inside each `*Checkpoint`
class, per the schema decision above.
