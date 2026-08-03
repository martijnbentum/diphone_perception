# Phone duration statistics (2026-08-03)

Min/max/median duration (seconds) per IPA phone label, computed from the
`duration` column of `data/metadata.csv`. Every label occurs 13,500 times
(the corpus is perfectly balanced).

| ipa_phoneme | min | max | median | n |
|---|---|---|---|---|
| aː | 0.046 | 0.675 | 0.111 | 13500 |
| b | 0.046 | 0.304 | 0.070 | 13500 |
| d | 0.046 | 0.273 | 0.061 | 13500 |
| eː | 0.047 | 0.445 | 0.111 | 13500 |
| f | 0.046 | 0.439 | 0.071 | 13500 |
| h | 0.046 | 0.414 | 0.061 | 13500 |
| iː | 0.050 | 0.965 | 0.081 | 13500 |
| j | 0.046 | 0.563 | 0.080 | 13500 |
| k | 0.046 | 0.457 | 0.071 | 13500 |
| l | 0.046 | 0.358 | 0.061 | 13500 |
| m | 0.046 | 0.592 | 0.071 | 13500 |
| n | 0.046 | 0.296 | 0.061 | 13500 |
| oː | 0.050 | 0.530 | 0.111 | 13500 |
| p | 0.046 | 0.324 | 0.071 | 13500 |
| r | 0.046 | 0.644 | 0.061 | 13500 |
| s | 0.050 | 0.699 | 0.081 | 13500 |
| t | 0.046 | 0.366 | 0.067 | 13500 |
| uː | 0.050 | 0.351 | 0.081 | 13500 |
| v | 0.046 | 0.253 | 0.071 | 13500 |
| x | 0.046 | 0.373 | 0.071 | 13500 |
| z | 0.046 | 0.354 | 0.081 | 13500 |
| ɑ | 0.047 | 0.355 | 0.071 | 13500 |
| ɔ | 0.046 | 0.283 | 0.076 | 13500 |
| ɛ | 0.046 | 0.273 | 0.071 | 13500 |
| ɛi | 0.050 | 0.429 | 0.111 | 13500 |
| ɪ | 0.046 | 0.466 | 0.071 | 13500 |
| ɣ | 0.050 | 0.344 | 0.071 | 13500 |
| ŋ | 0.050 | 0.654 | 0.071 | 13500 |
| ʉ | 0.050 | 0.304 | 0.070 | 13500 |
| ʋ | 0.046 | 0.274 | 0.061 | 13500 |

## Notes

- Diphthongs and long vowels (`ɛi`, `eː`, `aː`, `oː`) have the highest
  medians (0.111s), consistent with their phonologically longer duration.
- `iː` has the largest max (0.965s) by a wide margin — worth a spot check
  if outlier durations matter downstream (e.g. sanity-checking
  `phraser_matching_diagnostics.md`'s tolerance window assumptions).
- Computed with:
  ```python
  import csv, statistics as stats
  from collections import defaultdict

  with open('data/metadata.csv', newline='') as f:
      rows = list(csv.DictReader(f))

  durations = defaultdict(list)
  for row in rows:
      durations[row['ipa_phoneme']].append(float(row['duration']))

  table = sorted(
      (label, min(ds), max(ds), stats.median(ds), len(ds))
      for label, ds in durations.items()
  )
  ```
