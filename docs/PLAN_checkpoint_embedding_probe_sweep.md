# Plan: checkpoint embedding probe sweep

## Feature 1: checkpoint discovery and load-time inventory validation

### Requirements

- Discover model stores directly below `data/echoframe_model_stores`.
- Include only the exact random model `wav2vec2_checkpoint-0` and directories
  matching `wav2vec2_nl1_checkpoint-<integer>`.
- Sort checkpoints numerically and reject unrelated directory names by
  ignoring them.
- Use layers 1 through 12 for `wav2vec2_checkpoint-0` and
  `wav2vec2_nl1_checkpoint-200000`; use only layer 9 for other checkpoints.
- Load each checkpoint/layer directly through `probe_data.build_probe_matrix`.
- Validate per-label feature counts from the loaded `phone_labels`; do not run
  a separate Echoframe metadata preflight.

### Tests

- Discover only supported checkpoint directory names in numeric order.
- Apply the special and ordinary layer policies exactly.
- Reject incomplete per-label inventories while building the probe matrix.
- Verify the sweep does not request Echoframe metadata before feature loading.

## Feature 2: all-checkpoint, all-label probe sweep

### Requirements

- Add the sweep to `probing/train_binary_embedding_probe.py` and build it on
  the existing `train_binary_embedding_probes` all-label function.
- Open and close one model-specific Echoframe store at a time.
- Check persisted fold results before opening a checkpoint store and run only
  layers with incomplete results unless overwrite is requested.
- Load and validate every planned model/layer before training all labels.
- Preserve the existing layer-specific probe and prediction artifact layout,
  cache behavior, sampling, cross-validation, and overwrite options.
- Print progress and warnings while running.
- Record unexpected store or training failures, warn, and continue.
- Print and return a compact final report rather than retaining fitted probe
  objects across the complete sweep.
- Include per-label mean/std accuracy and sample counts for completed runs,
  plus a per-run mean label accuracy in the printed summary. Preserve load or
  training errors for failed runs.

### Tests

- Train complete checkpoint/layer inventories and forward probe options.
- Reject incomplete inventories during probe-matrix loading.
- Record training failures, continue, and close every opened store.
- Return compact summaries without fitted classifier objects.
- Print a final completed/skipped/failed report.
- Document a minimal invocation snippet in `README.md`.
