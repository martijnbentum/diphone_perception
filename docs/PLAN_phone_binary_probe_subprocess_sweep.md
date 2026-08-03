# Plan: subprocess phone binary-probe sweep

Features 1 through 4 are implemented first. Tests and documentation follow
only after those interfaces have stabilized, to avoid churn while the worker,
cache, scheduler, and report contracts are still changing.

## Scope and defaults

- Support the Netherlandic phone inventory only. Flemish probe training is
  explicitly out of scope.
- Keep the existing low-level binary-probe training and five-fold cache
  behavior in `probing/train_binary_embedding_probe.py` and
  `probing/probe_utils.py`.
- Add `probing/phone_binary_probe.py` as the path-based worker, sweep, metadata
  preflight, report, and command-line module.
- Default all paths to the existing Netherlandic locations, including
  `data/echoframe_model_stores`, `data/phone_probes`, and
  `data/probe_results`, so `run_phone_binary_probe_sweep()` needs no
  arguments.
- Run independent operating-system subprocesses without Slurm and without an
  internal process pool. Default to 31 concurrent jobs, while accepting any
  positive user-supplied concurrency limit.

## Feature 1: single-phone worker API and CLI

### Requirements

- Add a path-based `train_phone_binary_probe(...)` API for exactly one phone
  label, model, and layer. It handles every cross-validation fold through the
  existing `train_binary_embedding_probe(...)` implementation.
- Add a CLI entry point callable as:

  ```text
  python -m probing.phone_binary_probe train \
      --phone <label> --model-name <name> --layer <number>
  ```

- Require phone, model, and layer task identity at the worker CLI. Default the
  metadata CSV, sentence TSV, Phraser key, duplicate-replacement key, model
  store, probe, and prediction paths to the current Netherlandic paths.
- Forward the existing collar, sample count, fold count, random seed,
  standardization, persistence, verbosity, and overwrite settings.
- Reuse the existing run manifest and per-fold completion markers. With
  overwrite disabled, skip a fully complete task, reuse valid completed folds
  in a partial task, and train only work that remains. With overwrite enabled,
  retrain every fold.
- When a sweep supplies a task-status path, atomically write `running`, then
  `completed`, `already_complete`, or `failed`, including task identity,
  timestamps, elapsed time, compact metrics, and a compact error description.
- Return exit status zero for completed or already-complete work and nonzero
  for failure. Close stores in all normal and error paths.
- Construct one `Phones` object per worker process, open its Phraser store
  once, and attach that same handle to the Echoframe store. Never open the
  same Phraser LMDB a second time in one process.
- After a completed or already-complete task, atomically write its selected
  `run_id` to a settings-specific pointer below that task's probe-artifact
  directory. Independent pointer files prevent contention between workers.

### Tests

- Parse required task identity and Netherlandic path defaults.
- Forward every training setting and custom path to the existing trainer.
- Report completed, already-complete, and failed outcomes atomically.
- Honor overwrite and preserve the existing partial-fold recovery behavior.
- Close owned stores after success and failure.

## Feature 2: persistent all-model metadata preflight

### Requirements

- Add a reusable metadata-preflight function that discovers all supported
  checkpoint stores and checks every required layer before any probe worker is
  launched.
- Lazily construct one shared `Phones` object for the complete parent-process
  preflight and close its single Phraser store when checking finishes.
- Keep the current checkpoint policy: layers 1 through 12 for
  `wav2vec2_checkpoint-0` and `wav2vec2_nl1_checkpoint-200000`, and layer 9
  for other `wav2vec2_nl1_checkpoint-<integer>` stores.
- Persist one atomic JSON cache at
  `data/echoframe_model_stores/phone_binary_probe_metadata_status.json`, or
  directly below a custom model-store root.
- Store schema/version information, the effective phone-inventory fingerprint,
  collar, timestamps, and separate total/available/missing/complete records
  for every model and layer.
- Reuse only a cached complete entry whose schema, model, layer, collar, and
  phone-inventory fingerprint match. Always recheck incomplete or failed
  entries. A force-metadata-check option ignores every cache entry and rewrites
  the results.
- Skip probe tasks for incomplete or failed model/layer inventories, warn, and
  retain their details for the final report. Continue checking later stores.
- Replace the current line per 1,000 metadata items with progress bars: an
  overall model-store bar and a current-store bar spanning all required
  `(layer, phone)` checks for that store. Show the active layer in the
  current-store prefix and mark reused stores as cached.

### Tests

- Write and atomically update the root-level JSON cache at the default and a
  custom store root.
- Reuse matching complete entries without opening/checking their store.
- Recheck incomplete, failed, stale-fingerprint, stale-collar, and forced
  entries.
- Preserve counts and failures for every model/layer while continuing.
- Update current-store and overall-store progress without emitting one line
  per metadata batch.

## Feature 3: bounded subprocess sweep and live progress

### Requirements

- Add `run_phone_binary_probe_sweep(...)`, callable with no arguments under
  the Netherlandic defaults, plus a `sweep` CLI subcommand with equivalent
  options.
- Build the complete task list only after the complete metadata preflight.
  Create one task for every phone label in every complete model/layer.
- Launch the same module's `train` CLI as an independent subprocess for every
  task. Pass the phone, model, layer, all paths, all training settings,
  overwrite, and a unique task-status file explicitly.
- Manage subprocesses directly and cap active children at the user-selected
  `jobs` value, which defaults to 31 and may be any positive integer. Do not use
  `multiprocessing`, `concurrent.futures`, a process pool, or Slurm.
- Create a unique `/tmp/diphone-phone-probes-<run-id>/` directory containing
  worker status and log files. Capture each worker's stdout/stderr there so 31
  workers cannot interleave terminal output.
- Poll workers/status files and display aggregate task progress with finished,
  active, trained, already-complete, and failed counts, elapsed time, and ETA.
- On worker failure, print the model/layer/phone identity and the end of its
  captured log, record the failure, and continue scheduling other tasks.
- On interrupt, stop active child processes and do not launch more work. On
  every handled/normal sweep exit, remove the temporary run directory after
  its information has been incorporated into the persistent report.

### Tests

- Build the expected phone/model/layer command lines and unique status paths.
- Never exceed the requested concurrency and default to 31 jobs.
- Distinguish trained, already-complete, failed, and preflight-skipped tasks.
- Continue after worker failure and include its log tail in the report.
- Update aggregate progress from worker status/process completion.
- Stop children on interruption and remove the temporary directory after the
  run.

## Feature 4: rebuildable persisted report

### Requirements

- Add `build_phone_binary_probe_report(...)` as a reusable, non-training
  function and expose it through a `report` CLI subcommand.
- Report only artifacts matching the requested settings; with no overrides,
  use the standard sweep settings.
- Combine the metadata-status cache with persisted probe `run.json` manifests
  and fold-completion markers. Do not load embedding arrays and do not train.
- Record run settings, paths, timestamps, elapsed time when known, metadata
  status, expected task counts, complete/partial/missing/failed tasks,
  per-fold accuracies, per-task mean/std accuracy, cache outcomes, error/log
  tails available from the current sweep, and aggregate totals.
- Atomically overwrite one report at
  `<probe_save_dir>/phone_binary_probe_report.json`. The default is therefore
  `data/phone_probes/phone_binary_probe_report.json`; a custom probe directory
  owns its own report.
- Resolve multiple matching historical `run_id` directories through the
  settings-specific pointer written by the worker. With no pointer, accept one
  matching run; report multiple matches as ambiguous rather than silently
  choosing an obsolete run. Treat an invalid or missing-target pointer as a
  report error.
- Have the sweep call this reusable function after worker completion, enrich
  it with current worker outcomes/timings, write it, print a concise summary,
  and return the same report structure.

### Tests

- Rebuild a report from complete, partial, missing, malformed, and mismatched
  persisted artifacts without invoking training.
- Filter artifacts by the requested/default training settings.
- Follow valid selected-run pointers, accept one pointerless match, and report
  multiple pointerless matches as ambiguous.
- Merge metadata-skipped and current worker-failure details.
- Write the fixed report path atomically under default and custom probe roots.
- Verify that the sweep's returned report equals the persisted report.

## Feature 5: integration and documentation

Implement this feature only after Features 1 through 4 have stabilized, so
tests and documentation describe the settled interfaces rather than changing
in parallel with them.

### Requirements

- Keep the single-label trainer as the source of truth for selection,
  manifests, fold fitting, caching, and output paths.
- Replace the documented sequential checkpoint sweep with the new path-based
  subprocess sweep. Preserve the lower-level single-label and in-process
  all-label APIs for callers that still need them.
- Document the no-argument Python sweep, `jobs`, overwrite,
  force-metadata-check, single-task CLI, sweep CLI, and report-only API/CLI.
- Document all persistent JSON paths, cache semantics, progress behavior, and
  the fact that the current sweep supports Netherlandic phones only.

### Verification

- After Features 1 through 4 are complete, add and run focused worker,
  preflight-cache, subprocess-scheduler, report-builder, and existing
  binary-probe tests.
- Then run the full test suite and Python compilation for the changed modules.
- Finally check the complete diff for unrelated files and malformed
  whitespace.
