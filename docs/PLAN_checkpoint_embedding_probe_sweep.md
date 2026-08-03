# Plan: checkpoint embedding probe sweep

## Feature 1: checkpoint discovery and embedding preflight

### Requirements

- Discover model stores directly below `data/echoframe_model_stores`.
- Include only the exact random model `wav2vec2_checkpoint-0` and directories
  matching `wav2vec2_nl1_checkpoint-<integer>`.
- Sort checkpoints numerically and reject unrelated directory names by
  ignoring them.
- Use layers 1 through 12 for `wav2vec2_checkpoint-0` and
  `wav2vec2_nl1_checkpoint-200000`; use only layer 9 for other checkpoints.
- Before training a checkpoint/layer, check Echoframe metadata for every phone
  in `phones.phraser_phones` at the requested collar.
- Skip training if even one embedding is missing and retain the total,
  available, and missing counts for the final report.

### Tests

- Discover only supported checkpoint directory names in numeric order.
- Apply the special and ordinary layer policies exactly.
- Count complete and incomplete inventories without loading embedding arrays.
- Check metadata in bounded batches and validate the batch-size argument.

## Feature 2: all-checkpoint, all-label probe sweep

### Requirements

- Add the sweep to `probing/train_binary_embedding_probe.py` and build it on
  the existing `train_binary_embedding_probes` all-label function.
- Open and close one model-specific Echoframe store at a time.
- Preflight every planned model/layer before training all labels.
- Preserve the existing layer-specific probe and prediction artifact layout,
  cache behavior, sampling, cross-validation, and overwrite options.
- Print progress and warnings while running.
- Record unexpected store, preflight, or training failures, warn, and continue.
- Print and return a compact final report rather than retaining fitted probe
  objects across the complete sweep.
- Include per-label mean/std accuracy, sample counts, missing counts, skipped
  state, and cache status for completed runs, plus a per-run mean label
  accuracy in the printed summary.

### Tests

- Train complete checkpoint/layer inventories and forward probe options.
- Skip incomplete inventories without calling probe training.
- Record training failures, continue, and close every opened store.
- Return compact summaries without fitted classifier objects.
- Print a final completed/skipped/failed report.
- Document a minimal invocation snippet in `README.md`.
