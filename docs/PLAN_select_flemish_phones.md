# Plan: select a balanced Flemish Phraser phone inventory

## Feature: generate Flemish Phraser phone keys

### Requirements

- Add `probing/select_flemish_phones.py` with a public
  `save_flemish_phraser_phone_keys(...)` function that accepts a Phraser
  store directly and does not depend on `probing.metadata.Phones`.
- Default the output path to `data/flemish_phraser_phone_keys.bin`.
- Require `overwrite=True` when the output already exists.
- Use the provided 31-label order as the binary layout contract, with 5,000
  consecutive fixed-width keys per label.
- Consider only audios whose paths contain both an exact `comp-k` or `comp-o`
  component and the exact Flemish Dutch `vl` component.
- Use exact Phraser labels and the hard-coded inclusive duration bounds from
  `notes/phone_duration_stats.md` for all 31 labels.
- Deduplicate candidates globally by their 22-byte Phraser keys. Do not load
  or compare against the Netherlandic `data/phraser_phone_keys.bin` file.
- Use Python's module-level `random.seed(42)` and `random.sample(...)` by
  default, sampling labels in the documented label order.
- Show progress while filtering audios, scanning phone candidates, and
  sampling labels; allow callers to disable progress.
- Print the available unique-token count for every label and return a
  structured result containing available counts, selected counts, output
  path, and whether a file was written.
- If any label has fewer than 5,000 candidates, report every count, do not
  raise, do not create or alter the output, and return `written=False`.
- On success, validate 155,000 globally unique keys before writing the
  label-major binary and return `written=True`.

### Tests

- Accept only exact `comp-k`/`comp-o` paths containing exact `vl`; reject
  `nl`, language variants, component near-matches, and incorrect casing.
- Enforce inclusive duration bounds and exact requested labels.
- Reject malformed Phraser keys and deduplicate repeated keys globally.
- Invoke all progress phases and allow progress suppression.
- Report all available counts and return without writing when any label is
  insufficient.
- Use `random.seed`/`random.sample` deterministically and write keys in the
  documented label-major order.
- Require explicit overwrite permission and validate the final record count
  and global uniqueness before writing.
