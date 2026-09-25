# Speech and non-speech in held-out SVD modes

Date: 2026-09-24

## Conclusion

Mode 0 is the clearest speech/non-speech separator in the reported evaluation
results. Its non-speech mean score is 5.1369 higher than the speech mean, a
standardized difference of 3.3478. Mode 2 is second at 0.9552; every other
reported mode has an absolute standardized difference below 0.28. This is a
descriptive result. The outputs do not yet establish that the separation is
consistent across recordings or caused by speech status itself.

## Results

The results below were supplied from `significant_modes(speech, d)`,
`significant_modes(no_speech, d)`, and
`compare_speech_non_speech(speech, no_speech, d)` in
[`mode_significance.py`](../decomposition/mode_significance.py). The complete
significant-mode lists used 100,000 random draws and Benjamini-Hochberg
`alpha=0.01`. A later run reported the leading ten modes in each group with
1,000,000 draws. The matrices, recording identities, and labels were not
available for an independent check when this report was written.

| Mode | Speech variance fraction | Non-speech variance fraction | Non-speech minus speech mean | Standardized difference |
|---:|---:|---:|---:|---:|
| 0 | 0.0250 | 0.0205 | 5.1369 | 3.3478 |
| 2 | 0.0489 | 0.0132 | 1.7524 | 0.9552 |
| 3 | 0.0447 | 0.0204 | 0.0308 | 0.0166 |
| 4 | 0.0283 | 0.0660 | 0.0912 | 0.0423 |

Every mode from 0 through 54 passed the random-direction comparison in both
groups. Modes 3 and 4 illustrate why this result differs from group separation:
they capture unusually large variation within the groups, but their group
means are close. Mode 0 has a large group mean shift despite accounting for
only about 2% to 2.5% of each group's within-group variance.

The leading ten modes in each group still reached the Monte Carlo floor in the
1,000,000-draw run: `p=1/1000001`, about `1e-6`. Their adjusted values were
about `1.634e-5` for speech and `1.707e-5` for non-speech. This confirms they
pass the random-direction comparison, but still cannot rank them by p-value.
The 1,000,000-draw output did not include the full list, so the count of
passing modes at that resolution is unknown.

Each variance fraction uses its own group's total variance as the denominator,
so fractions across groups do not compare absolute variance. The standardized
differences have no confidence intervals or group-difference p-values here.
SVD direction signs are arbitrary; the sign of a mean difference can reverse
if the direction is flipped.

## Recommended next steps

1. Verify that neither evaluation matrix contributed embeddings to the SVD fit.
   Record sample counts, label rules, and the recordings represented in each
   group. Check whether speech and non-speech come from the same held-out
   recordings.
2. Plot score distributions for modes 0 and 2 by group, with recording-level
   summaries. Check whether mode 0 separates groups across recordings or is
   driven by a few recordings or outliers.
3. Estimate the mode 0 mean difference across recordings, using paired
   recording differences where both groups occur and a recording-level
   bootstrap confidence interval. Treat frame rows from one recording as
   correlated rather than independent observations.
4. Inspect whether mode 0 tracks acoustic energy, silence, background noise,
   speaker, or available speech context. Repeat the comparison on another set
   of held-out recordings before interpreting it as a speech-status mode.
