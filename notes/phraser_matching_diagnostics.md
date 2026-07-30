# Phraser phone matching — failure diagnostics (2026-07-28)

Findings from running `Phones.save_phraser_keys()` then
`Phones.analyze_phraser_failures()` (see `metadata.py`) on the full corpus,
to characterize why ~43% of phones fail to find a corresponding phone in
the phraser store.

## Setup

- `metadata.Phone` rows (from `metadata.csv`) are matched against phones in
  a `phraser` LMDB store by audio, IPA label (`phoneme_ipa`, mapped from
  SAMPA via `phone_mapper.cgn.cgn_to_ipa`), and a start/end tolerance
  window (default ±25ms), with neighbor-label disambiguation when more
  than one candidate falls in the window.
- `analyze_phraser_failures()` breaks failures down by error type, label,
  overlap, comp, sentence-edge position, and the label of the phraser
  phone closest in time (regardless of label match).

## Headline numbers

- Total phones: 418,500
- Failed to match: 181,747 (43.4%)
- `NoCandidateError`: 181,486 (99.86% of failures)
- `AmbiguousMatchError`: 261 (0.14% of failures)
- `closest_matches_expected`: 78,099 / 181,747 (43.0%) — the correct label
  *does* sit nearby in time for this fraction of failures

## Finding 1 — four labels fail 100% of the time: an IPA labeling mismatch, not a timing issue

`ʉ`, `iː`, `ʋ`, and `uː` each have **exactly 13,500 failures** — every
single occurrence of these four phonemes in the corpus (the corpus is
perfectly balanced at 13,500 occurrences per label) failed to match. That
alone is 54,000 / 181,747 (~30%) of all failures.

Their closest-in-time labels (`by_label_closest_label`) show the nearby
phraser phone is essentially *never* the same symbol:

| expected label | closest label (dominant) | count |
|---|---|---|
| `ʉ` | `ʏ` | 8477 |
| `ʉ` | `u` | 2960 |
| `iː` | `i` | 12712 |
| `ʋ` | `w` | 11334 |
| `uː` | `u` | 12838 |

**Interpretation:** phraser's store appears to use a different IPA
convention than `phone_mapper.cgn.cgn_to_ipa` for these phones —
specifically, it looks like phraser doesn't mark vowel length on `i`/`u`
(bare `i`/`u` regardless of length), and uses `ʏ`/`w` where our mapping
produces `ʉ`/`ʋ`. Since the label literally doesn't exist near the
timestamp, no amount of widening `tolerance_ms` would fix this — it needs
either a symbol-mapping reconciliation or a fallback/equivalence lookup at
match time.

## Finding 2 — two hypotheses ruled out by the data

- **Overlap**: only 2 / 181,747 failures had `overlap=True`. Not a factor.
- **Sentence-edge position**: 96.9% of failures are `interior` phones (not
  first/last-in-sentence). Not a factor.

## Finding 3 — ambiguity is a non-issue

99.86% of failures are `NoCandidateError` (nothing in the tolerance window
at all); only 0.14% are `AmbiguousMatchError`. Consistent with genuine
label mismatches rather than borderline/competing timing.

## Finding 4 — remaining confusions look like real Dutch phonological assimilation, not bugs

For labels other than the four 100%-failure ones, `by_label_closest_label`
shows patterns consistent with surface-level phonetic assimilation:

- `v` → `f` (devoicing)
- `x` ↔ `ɣ` (voicing alternation)
- `n` → `ŋ`, `n` → `m` (place assimilation before velars/labials)
- `s` ↔ `z` (voicing)

**Interpretation:** phraser's forced-alignment transcript likely reflects
the surface-realized phone, while our CGN-derived label is the
citation-form phoneme. This is a different, harder problem than a symbol
bug — not mechanically "fixable," more a property of comparing two
different transcription philosophies.

## Per-label failure rate (of that label's 13,500 occurrences)

| label | failures | rate |
|---|---|---|
| ʉ | 13500 | 100.0% |
| iː | 13500 | 100.0% |
| ʋ | 13500 | 100.0% |
| uː | 13500 | 100.0% |
| v | 8741 | 64.7% |
| aː | 6802 | 50.4% |
| z | 6374 | 47.2% |
| ŋ | 6351 | 47.0% |
| x | 6120 | 45.3% |
| s | 6087 | 45.1% |
| ɛi | 5750 | 42.6% |
| oː | 5508 | 40.8% |
| eː | 5477 | 40.6% |
| r | 5110 | 37.9% |
| ɣ | 5067 | 37.5% |
| ɛ | 5056 | 37.5% |
| k | 5033 | 37.3% |
| h | 4958 | 36.7% |
| j | 4767 | 35.3% |
| l | 4704 | 34.8% |
| ɑ | 4015 | 29.7% |
| ə | 3968 | 29.4% |
| ɔ | 3702 | 27.4% |
| b | 3652 | 27.1% |
| n | 3630 | 26.9% |
| m | 3403 | 25.2% |
| ɪ | 3218 | 23.8% |
| t | 3197 | 23.7% |
| d | 2733 | 20.2% |
| f | 2176 | 16.1% |
| p | 2148 | 15.9% |

Note `aː`, `eː`, `oː` are *not* affected the way `iː`/`uː` are — they show
high self-match rates in their own `by_label_closest_label` entries, so
phraser does use length-marked labels for those three; the length-marking
gap seems specific to `i`/`u`.

## Open questions / next steps

- Check whether phraser exposes its own canonical IPA label set (e.g.
  `phone_features.py` expects "IPA symbol" inputs) and diff it directly
  against `phone_mapper.cgn.cgn_to_ipa`'s value set, to confirm the
  `ʉ→ʏ`, `iː→i`, `ʋ→w`, `uː→u` equivalences and check for any others.
- Consider whether `get_phraser_phone` should try a small set of known
  label equivalences as a fallback when the primary label yields zero
  candidates.
- Decide whether the assimilation-driven confusions (`v`/`f`, `x`/`ɣ`,
  `n`/`ŋ`, `n`/`m`, `s`/`z`) should be tolerated as acceptable near-matches
  or left as failures, since they likely reflect real linguistic
  phenomena rather than data errors.

## Follow-up — duplicate Phraser keys in embedding extraction (2026-07-30)

An embedding run over 418,500 matched phones and five layers reported:

```text
n segments: 418500
missing segments: 417214
found segments: 0
missing layer items: 2092500
found layer items: 0
```

The Echoframe store was empty. The 1,286-segment difference is consistent
with repeated Phraser keys in `phones.phraser_phones`, not with existing
embeddings or audio-bound filtering:

- Echoframe constructs all 418,500 × 5 = 2,092,500 layer keys before
  deduplication, which explains `missing layer items`.
- `MissingSegments._find_missing()` adds a compute request only the first
  time it sees a `segment.key`, which explains 417,214 missing segments.
- Audio bounds do not silently exclude requests here. Embedding extraction
  clips the collar to the audio bounds and raises for a non-positive segment
  duration.

This means multiple metadata phones may refer to the same stored Phraser
segment. One embedding is sufficient for identical Phraser keys, but the
duplicates should be inspected to distinguish genuine duplicate metadata
rows from matching collisions where distinct metadata phones were assigned
to the same Phraser phone.

### Count duplicate Phraser keys

```python
from collections import Counter

segments = phones.phraser_phones
counts = Counter(segment.key for segment in segments)
duplicates = {key: count for key, count in counts.items() if count > 1}

print('input segments:', len(segments))
print('unique keys:', len(counts))
print('duplicate groups:', len(duplicates))
print(
    'duplicate occurrences:',
    sum(count - 1 for count in duplicates.values()),
)
print('largest group:', max(duplicates.values(), default=1))
```

Expected from the embedding-run counts:

```text
input segments: 418500
unique keys: 417214
duplicate occurrences: 1286
```

### Inspect metadata phones sharing a Phraser key

```python
from collections import defaultdict

matches = defaultdict(list)

for index, (phone, segment) in enumerate(
    zip(phones.phones, phones.phraser_phones, strict=True)
):
    matches[segment.key].append((index, phone, segment))

duplicate_matches = {
    key: rows for key, rows in matches.items()
    if len(rows) > 1
}

for key, rows in list(duplicate_matches.items())[:10]:
    print('\nPhraser key:', key)
    for index, phone, segment in rows:
        print(
            index,
            phone.audio_filename,
            phone.phoneme_ipa,
            phone.start_seconds,
            phone.end_seconds,
            '->',
            segment.label,
            segment.start_seconds,
            segment.end_seconds,
        )
```
