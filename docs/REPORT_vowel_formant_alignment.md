# Local CGN vowel formants compared with Dutch literature

Date: 2026-07-31

## Conclusion

The local measurements align well enough with the Dutch literature to use
F0, F1, and F2 for the first synthetic-material pilot. Across the ten full
vowels shared by the local corpus and the literature tables, the shape of the
F1/F2 vowel space is highly consistent: the across-vowel correlations range
from 0.968 to 0.997. Median absolute F1 differences range from 14 to 59 Hz and
median absolute F2 differences from 34 to 88 Hz, depending on reference
population.

Three qualifications should remain visible:

1. Local F0 is systematically lower than the Weenink and Adank values. This
   is consistent across vowels and is plausibly a speech-style/context effect,
   rather than evidence of mislabelled vowel categories.
2. F3 agreement is appreciably weaker than F1/F2 agreement. F3 is reported
   here as a diagnostic, but is excluded from the first synthesis design.
3. The local corpus contains ten of the twelve full-vowel categories plus
   schwa. `/yː/` and `/øː/` are literature-only; schwa is local-only.

No cross-source average is recommended. The local CGN and literature anchors
should remain named experimental conditions.

## Data included

The current local run consists of:

- `phone_formants.csv`: one row per selected Phraser phone key;
- `phone_formants_metadata.json`: selection and measurement settings;
- `gender_formants.csv`: median of per-speaker medians with speaker-bootstrap
  confidence intervals;
- `manifest.json`: paths, provenance, versions, and checksums.

There are 148,465 unique selected phones. Of these, 143,784 (96.85%) have a
successful joint F0/F1/F2/F3 measurement. All registered manifest checksums
match their files.

Six obsolete files from an earlier stress-filtering implementation were
removed before this report was written. They were not registered in the
current manifest.

## Inventory and example words

Yes: the local corpus has 11 observed categories, and removing schwa leaves
10 full vowels. The standard literature inventory has 12 full vowels. It adds
the locally absent `/yː/` and `/øː/`; schwa is a separate reduced vowel and is
not one of the twelve.

The project normalizes the traditional Dutch `/ʏ/` category to `/ʉ/`, and it
uses length marks in its labels for the tense-vowel series. The words below
are category examples; the length mark does not claim identical surface
duration in every context.

| Project IPA | Example | Meaning | Local CGN | Literature |
|---|---|---|---:|---:|
| `/ɪ/` | *pit* | kernel/pit | yes | yes |
| `/ɛ/` | *pet* | cap | yes | yes |
| `/ɑ/` | *pat* | stalemate | yes | yes |
| `/ɔ/` | *pot* | pot | yes | yes |
| `/ʉ/` | *put* | well | yes | yes, commonly written `/ʏ/` |
| `/iː/` | *biet* | beetroot | yes | yes |
| `/yː/` | *fuut* | grebe | no | yes |
| `/eː/` | *beet* | bite/bit | yes | yes |
| `/øː/` | *neus* | nose | no | yes |
| `/aː/` | *maat* | measure/mate | yes | yes |
| `/oː/` | *boot* | boat | yes | yes |
| `/uː/` | *boek* | book | yes | yes |
| `/ə/` | *gemak* | ease/convenience | yes | no |

The Praat source table provides the same twelve categories as a controlled
`/p_t/` series: *pit, pet, pat, pot, put, Piet, puut, peet, peut, paat, poot,*
and *poet*. See the Praat manual's
[Pols and Van Nierop table documentation](https://www.fon.hum.uva.nl/praat/manual/Create_formant_table__Pols___Van_Nierop_1973_.html).

One correction to the manually copied counts is worth recording: the
committed metadata contains 13,499 `/ɑ/` tokens, not 13,500.

## Local measurement coverage

“Selected” comes from the run metadata. “Successful” is the sum of the female
and male token counts contributing to the local group anchors.

| IPA | Selected | Successful | Success | Female speakers/tokens | Male speakers/tokens |
|---|---:|---:|---:|---:|---:|
| `/ɪ/` | 13,499 | 12,896 | 95.53% | 30/4,354 | 39/8,542 |
| `/ɛ/` | 13,500 | 12,768 | 94.58% | 25/4,601 | 34/8,167 |
| `/ɑ/` | 13,499 | 12,862 | 95.28% | 19/4,398 | 28/8,464 |
| `/ɔ/` | 13,500 | 12,799 | 94.81% | 28/4,061 | 38/8,738 |
| `/ʉ/` | 13,500 | 13,330 | 98.74% | 177/5,827 | 158/7,503 |
| `/iː/` | 13,492 | 13,379 | 99.16% | 32/4,360 | 42/9,019 |
| `/eː/` | 13,488 | 13,422 | 99.51% | 25/4,415 | 34/9,007 |
| `/aː/` | 13,490 | 13,245 | 98.18% | 22/4,695 | 32/8,550 |
| `/oː/` | 13,497 | 13,384 | 99.16% | 28/4,199 | 38/9,185 |
| `/uː/` | 13,500 | 13,442 | 99.57% | 73/4,453 | 92/8,989 |
| `/ə/` | 13,500 | 12,257 | 90.79% | 11/5,005 | 16/7,252 |

The low end of the speaker counts, especially female schwa (11 speakers),
should be remembered when interpreting bootstrap intervals. It does not
invalidate the center estimate, but that estimate represents fewer independent
speakers than the token count suggests.

## Local group anchors

Values are rounded to the nearest Hz. F3 is shown for diagnostic completeness
but will not be used in the first synthetic materials.

| IPA | Gender | Speakers | Tokens | F0 | F1 | F2 | F3 diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|
| `/ɪ/` | female | 30 | 4,354 | 191 | 411 | 2,364 | 2,943 |
| `/ɪ/` | male | 39 | 8,542 | 119 | 367 | 1,914 | 2,539 |
| `/ɛ/` | female | 25 | 4,601 | 179 | 608 | 1,906 | 2,774 |
| `/ɛ/` | male | 34 | 8,167 | 113 | 516 | 1,623 | 2,430 |
| `/ɑ/` | female | 19 | 4,398 | 182 | 688 | 1,200 | 2,810 |
| `/ɑ/` | male | 28 | 8,464 | 115 | 587 | 1,082 | 2,450 |
| `/ɔ/` | female | 28 | 4,061 | 186 | 510 | 970 | 2,926 |
| `/ɔ/` | male | 38 | 8,738 | 119 | 468 | 847 | 2,576 |
| `/ʉ/` | female | 177 | 5,827 | 189 | 415 | 1,626 | 2,661 |
| `/ʉ/` | male | 158 | 7,503 | 123 | 384 | 1,439 | 2,353 |
| `/iː/` | female | 32 | 4,360 | 192 | 307 | 2,571 | 3,043 |
| `/iː/` | male | 42 | 9,019 | 119 | 281 | 2,145 | 2,641 |
| `/eː/` | female | 25 | 4,415 | 184 | 416 | 2,411 | 2,906 |
| `/eː/` | male | 34 | 9,007 | 116 | 395 | 1,979 | 2,515 |
| `/aː/` | female | 22 | 4,695 | 175 | 854 | 1,548 | 2,607 |
| `/aː/` | male | 32 | 8,550 | 108 | 676 | 1,337 | 2,337 |
| `/oː/` | female | 28 | 4,199 | 184 | 462 | 957 | 2,724 |
| `/oː/` | male | 38 | 9,185 | 115 | 444 | 896 | 2,378 |
| `/uː/` | female | 73 | 4,453 | 201 | 348 | 974 | 2,806 |
| `/uː/` | male | 92 | 8,989 | 130 | 323 | 896 | 2,425 |
| `/ə/` | female | 11 | 5,005 | 169 | 398 | 1,681 | 2,892 |
| `/ə/` | male | 16 | 7,252 | 101 | 371 | 1,455 | 2,514 |

## Comparison method

Every source was kept separate. Comparisons use the ten full vowels common to
the local data and each literature group. Schwa has no literature comparison,
and `/yː, øː/` have no local comparison.

- Local values are medians of per-speaker medians.
- Pols, Van Nierop, and Weenink values are medians across their published
  speaker observations, as produced by the repository's within-source summary
  loader.
- Adank values are the published group means at the vowel midpoint. They are
  not speaker-level observations.
- Every difference below is `local minus literature`, in Hz.
- Bias is the median signed difference across vowels.
- MAE is the median absolute difference across vowels.
- `r` is the Pearson correlation across vowel centers. It describes agreement
  in vowel-space geometry; it is not a measure of absolute equality.
- F3 columns marked with a dagger are diagnostic-only.

## Overall alignment

| Reference | Gender | F0 bias | F0 MAE | F1 bias | F1 MAE | F1 r | F2 bias | F2 MAE | F2 r | F3 MAE† | F3 r† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pols 1973 | male | — | — | -51 | 51 | 0.997 | -27 | 51 | 0.997 | 100 | 0.733 |
| Van Nierop 1973 | female | — | — | -57 | 57 | 0.986 | +67 | 67 | 0.997 | 74 | 0.738 |
| Weenink 1985 | male | -28 | 28 | -47 | 47 | 0.993 | -27 | 88 | 0.989 | 116 | 0.782 |
| Weenink 1985 | female | -56 | 56 | -59 | 59 | 0.976 | +8 | 59 | 0.984 | 147 | 0.674 |
| Adank 2004 NSD | female | -36 | 36 | +12 | 42 | 0.970 | +15 | 60 | 0.990 | 86 | 0.510 |
| Adank 2004 NSD | male | -33 | 33 | +14 | 14 | 0.979 | -25 | 60 | 0.990 | 65 | 0.758 |
| Adank 2004 SSD | female | -43 | 43 | -12 | 31 | 0.981 | -35 | 53 | 0.986 | 215 | 0.603 |
| Adank 2004 SSD | male | -15 | 15 | +32 | 36 | 0.968 | -8 | 34 | 0.989 | 144 | 0.598 |

### Interpretation

F1/F2 agreement is strong in every comparison. The smallest absolute errors
occur against Adank's modern Standard Dutch groups, especially male NSD F1
(14 Hz median absolute difference) and male SSD F2 (34 Hz). The 1973 and 1985
tables still preserve almost exactly the same across-vowel F1/F2 geometry,
despite differences in period, speaking context, and summary statistic.

Local F0 is lower for both genders in every source that reports F0. The shift
is highly consistent: -15 to -33 Hz for male groups and -36 to -56 Hz for
female groups. This should be represented as a corpus-versus-literature source
difference, not silently averaged away.

F3 is less consistent, particularly for Adank female NSD and both SSD groups.
This supports the decision to leave F3 out of the first synthesis design.

The largest recurring F1/F2 source differences are category-specific rather
than signs of a global failure. Examples include lower local `/ʉ/` F2 than
Adank NSD (-204 Hz female, -156 Hz male), and lower local `/ɛ/` F2 than Adank
NSD for men (-116 Hz). These differences are scientifically useful source
contrasts and should remain visible in the stimulus metadata.

## Per-vowel differences

The detailed tables below contain the synthesis-relevant F0/F1/F2 differences.
F3 is omitted here because it is not part of the initial materials.

### Pols 1973, male

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | — | -33 | -86 |
| `/ɛ/` | — | -64 | -77 |
| `/ɑ/` | — | -93 | +22 |
| `/ɔ/` | — | -62 | -13 |
| `/ʉ/` | — | -56 | -61 |
| `/iː/` | — | -19 | -75 |
| `/eː/` | — | -15 | -41 |
| `/aː/` | — | -124 | +37 |
| `/oː/` | — | -46 | -14 |
| `/uː/` | — | -17 | +96 |

### Van Nierop 1973, female

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | — | -59 | +64 |
| `/ɛ/` | — | -42 | +56 |
| `/ɑ/` | — | -62 | +100 |
| `/ɔ/` | — | -90 | +70 |
| `/ʉ/` | — | -65 | -24 |
| `/iː/` | — | +27 | +71 |
| `/eː/` | — | -54 | +11 |
| `/aː/` | — | -146 | +98 |
| `/oː/` | — | -38 | +7 |
| `/uː/` | — | +48 | +124 |

### Weenink 1985, male

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -28 | -11 | -214 |
| `/ɛ/` | -29 | -66 | -260 |
| `/ɑ/` | -27 | -92 | +26 |
| `/ɔ/` | -22 | -8 | +131 |
| `/ʉ/` | -21 | -50 | -62 |
| `/iː/` | -31 | -2 | -108 |
| `/eː/` | -28 | -53 | -68 |
| `/aː/` | -34 | -121 | +7 |
| `/oː/` | -27 | -44 | +46 |
| `/uː/` | -31 | +5 | +195 |

### Weenink 1985, female

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -63 | -54 | +24 |
| `/ɛ/` | -62 | -14 | -219 |
| `/ɑ/` | -55 | -132 | -51 |
| `/ɔ/` | -52 | -53 | +79 |
| `/ʉ/` | -52 | -68 | -111 |
| `/iː/` | -57 | +23 | +11 |
| `/eː/` | -56 | -71 | +68 |
| `/aː/` | -67 | -64 | -25 |
| `/oː/` | -50 | -96 | +6 |
| `/uː/` | -58 | +25 | +245 |

### Adank 2004 Northern Standard Dutch, female

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -30 | +12 | +88 |
| `/ɛ/` | -41 | +73 | -84 |
| `/ɑ/` | -44 | -70 | -80 |
| `/ɔ/` | -32 | +91 | +52 |
| `/ʉ/` | -57 | -2 | -204 |
| `/iː/` | -56 | +13 | +47 |
| `/eː/` | -23 | -26 | +68 |
| `/aː/` | -19 | -58 | -24 |
| `/oː/` | -17 | +17 | -7 |
| `/uː/` | -48 | +62 | +36 |

### Adank 2004 Northern Standard Dutch, male

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -35 | +6 | -5 |
| `/ɛ/` | -41 | +41 | -116 |
| `/ɑ/` | -34 | +9 | -90 |
| `/ɔ/` | -33 | +66 | +26 |
| `/ʉ/` | -31 | +18 | -156 |
| `/iː/` | -38 | +3 | -17 |
| `/eː/` | -15 | -5 | -16 |
| `/aː/` | -26 | +6 | -88 |
| `/oː/` | -24 | +32 | -33 |
| `/uː/` | -34 | +64 | +91 |

### Adank 2004 Southern Standard Dutch, female

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -65 | -44 | +249 |
| `/ɛ/` | -45 | +27 | -26 |
| `/ɑ/` | -43 | -37 | -62 |
| `/ɔ/` | -47 | +35 | -17 |
| `/ʉ/` | -60 | -42 | -159 |
| `/iː/` | -42 | -10 | -76 |
| `/eː/` | -35 | -20 | -9 |
| `/aː/` | -28 | -14 | -92 |
| `/oː/` | -31 | +44 | -11 |
| `/uː/` | -36 | +27 | -45 |

### Adank 2004 Southern Standard Dutch, male

| IPA | ΔF0 | ΔF1 | ΔF2 |
|---|---:|---:|---:|
| `/ɪ/` | -16 | +3 | +169 |
| `/ɛ/` | -15 | +41 | +7 |
| `/ɑ/` | -11 | +32 | +16 |
| `/ɔ/` | -17 | +70 | -3 |
| `/ʉ/` | -15 | +31 | -53 |
| `/iː/` | -29 | +3 | -34 |
| `/eː/` | -3 | +11 | -14 |
| `/aː/` | -8 | -41 | -92 |
| `/oː/` | -10 | +75 | +34 |
| `/uː/` | -19 | +57 | -82 |

## Decision for synthetic materials

The evidence supports the following restrained design:

1. Use F0, F1, and F2 only in the first material set.
2. Use local CGN centers as the target-domain condition.
3. Use Adank Northern Standard Dutch as the main literature condition because
   it is modern, complete for the twelve full vowels, and aligns well with the
   local F1/F2 space.
4. Retain Adank SSD, Weenink, Pols, and Van Nierop as named robustness
   conditions rather than pooling them.
5. Compare only the ten shared vowels in direct local-versus-literature tests.
6. Include local schwa as a clearly labelled local-only supplement.
7. Include `/yː/` and `/øː/` as clearly labelled literature-only supplements;
   do not present interpolated values as local CGN measurements.
8. Cross formant-source condition with common F0 values so that the lower CGN
   pitch does not become an accidental cue for “local” versus “literature.”
9. Keep duration, RMS, fade, and the two formant bandwidths constant in the
   first controlled set.
10. Validate the generated waveforms acoustically after synthesis; this is a
    later material-validation step, not a reason to delay choosing anchors.

This design preserves the empirical source differences while keeping the
first experiment interpretable.
