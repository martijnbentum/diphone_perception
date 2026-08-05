# Plan: load Flemish phones and extract model embeddings

## Feature 1: `FlemishPhones`

### Requirements

- Add `FlemishPhones` to `probing.metadata` as a metadata-free owner of the
  selected Flemish Phraser inventory.
- Accept an optional Phraser store and lazily open the existing CGN store when
  omitted.
- Default the key path to `data/flemish_phraser_phone_keys.bin`.
- Bulk-load keys from the binary without loading `metadata.csv`.
- Expose the cached list through canonical `.phraser_phones` and the
  `.flemish_phraser_phones` alias.
- Strictly require 155,000 fixed-width, non-placeholder, globally unique keys
  and 155,000 successfully loaded Phraser phones.
- Validate the label-major contract: exactly 5,000 phones for each of the 31
  labels in `probing.select_flemish_phones.flemish_phone_labels` order.
- Raise `ValueError` for malformed records, missing objects, duplicates,
  incorrect labels/order, or invalid per-label counts.

### Tests

- Lazily load the CGN store and cache it.
- Load and cache a valid small label-major inventory without touching local
  metadata.
- Expose the same list through both property names.
- Reject malformed key files, placeholders, duplicate keys, wrong total
  counts, missing Phraser objects, incorrect labels/order, and unbalanced
  labels.

## Feature 2: Flemish multi-model embedding extraction

### Requirements

- Use `locations.echoframe_model_flemish_stores` at
  `data/echoframe_model_flemish_stores` as the default store root.
- Add `extract_flemish_phone_embeddings_for_models(...)`, mirroring
  `extract_phone_embeddings_for_models(...)` and accepting a `FlemishPhones`
  instance.
- Use one model-specific Echoframe store below the Flemish root, preserving
  the existing quoted model-directory naming, cleanup, return mapping, model
  registration, and `cgn-awd` Phraser source ID behavior.
- Extract from the validated `flemish_phones.phraser_phones` inventory.
- Change the default `batch_size` to 120 for `extract_phone_embeddings`,
  `extract_phone_embeddings_for_models`, and
  `extract_flemish_phone_embeddings_for_models`.

### Tests

- Verify the Flemish default root and per-model paths.
- Verify the Flemish wrapper opens, extracts to, closes, and reports one
  dedicated store per model while forwarding all extraction options.
- Verify cleanup after failures and reject a string passed as `model_names`.
- Verify all three public extraction entry points default to batch size 120.
- Preserve the existing Netherlandic multi-model behavior and tests.
