# Plan: replace duplicate Phraser phone keys

## Feature 1: generate replacement keys

### Requirements

- Add `probing/fix_phone_duplicate.py` with small helpers and a public
  `save_duplicate_replacement_phraser_keys(...)` function.
- Accept a `probing.metadata.Phones` object.
- Default to `data/duplicate_phone_counts.json` and
  `data/duplicate_replacement_phraser_phone_keys.bin`.
- Require `overwrite=True` when the output already exists.
- Read each requested count from the JSON file and select half that number,
  for 1,286 replacements in total.
- Consider only Phraser audios whose path contains the exact `comp-k` or
  `comp-o` path component.
- Consider only phones with the requested exact label and a duration inside
  that label's hard-coded inclusive range from
  `notes/phone_duration_stats.md`.
- Exclude every key already present in `data/phraser_phone_keys.bin` and keep
  selected replacement keys unique.
- Use `random.seed(42)` and `random.sample(...)` by default.
- Show progress while filtering component audios, collecting candidate phones,
  and sampling replacement labels; allow callers to disable it.
- Write replacements in duplicate-occurrence order so they can be applied
  without losing the alignment between metadata phones and Phraser phones.
- Validate that applying the replacements yields 418,500 unique keys and
  exactly 13,500 keys per label. Fail rather than write a partial or invalid
  result.

### Tests

- Filter only exact `comp-k` and `comp-o` path components.
- Enforce inclusive minimum and maximum duration bounds.
- Exclude existing and repeated candidate keys.
- Produce deterministic samples for a fixed seed.
- Report each collection/sampling phase through the progress helper.
- Reject odd, invalid, or mismatched counts and insufficient candidate pools.
- Require explicit overwrite permission.
- Verify replacement-file record count and order against duplicate metadata
  occurrences.

## Feature 2: load replacements through `Phones`

### Requirements

- Add a replacement-key path argument to `Phones`, defaulting to
  `data/duplicate_replacement_phraser_phone_keys.bin`.
- When the file exists, replace only repeated occurrences of an original key,
  preserving list length and positional alignment.
- Store the loaded replacement Phraser phones on
  `Phones.duplicate_replacement_phones` in replacement-file order.
- Load from the key files and Phraser store without parsing metadata by
  default; provide `validate_against_metadata=True` for strict validation.
- Warn explicitly when the optional replacement file is absent or disabled.
- Reject a present replacement file when its record count is wrong, it reuses
  an original key, it contains duplicate replacements, the final keys are not
  unique, or a loaded replacement phone has the wrong label.
- Warn when loaded Phraser phones still contain duplicate keys or when any
  label does not contain exactly 13,500 phones.

### Tests

- Apply replacements in place while preserving nonduplicate keys and order.
- Retain original duplicates and warn when no replacement file exists.
- Verify the default load path does not load metadata phones.
- Validate metadata alignment only when explicitly requested.
- Reject malformed replacement inventories.
- Warn for duplicate keys and unbalanced label counts.
- Accept an all-unique, balanced inventory without warnings.
