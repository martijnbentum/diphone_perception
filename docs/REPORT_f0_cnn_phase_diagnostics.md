# F0 CNN representation and phase diagnostics

Date: 2026-08-12

## Conclusion

The sharp F0 UMAP breaks are not frequency-ordering or audio-grid mistakes.
They reflect discontinuities already present in the final 512-dimensional CNN
representations, which UMAP magnifies in two dimensions. Frame diagnostics
further show that temporal mean aggregation does not create the main
discontinuities at 3,200, 4,800, or 6,400 Hz.

## Data and attribution

The representation-space findings below were computed from
[`f0_data.npz`](../../data/synthetic_acoustic_probes/experiment_f0/f0_data.npz). It contains
799 aligned rows for one-second pure tones from 10 through 7,990 Hz:
`mean_cnn_features` `(799, 512)`, UMAP `coordinates` `(799, 2)`, frequencies,
stimulus IDs, model name, aggregation, metric, and seed.

[`make_f0_x_y`](../synthetic_acoustic_probes/experiment_f0.py) assembled the
CNN mean representations and frequencies in stimulus-manifest order.
[`project_umap`](../synthetic_acoustic_probes/umap_projection.py) generated the
coordinates with cosine distance and seed 42, and
[`plot_f0_umap`](../synthetic_acoustic_probes/f0_plot.py) generated
[`f0_umap.pdf`](../../data/synthetic_acoustic_probes/experiment_f0/f0_umap.pdf). No
repository function currently writes `f0_data.npz`; it is the persisted
analysis bundle built from those outputs.

The temporal findings were computed separately from the stored frame-level
Echoframe payloads with
[`diagnose_cnn_phase`](../synthetic_acoustic_probes/cnn_phase_diagnostics.py).

## Findings from `f0_data.npz`

- All 799 frequencies, stimulus IDs, mean features, and UMAP coordinates are
  finite, unique where expected, and aligned in 10 Hz steps.
- The two apparent major bridges are pairs of nearly overlapping path
  segments around isolated points:
  - 4,790→4,800 and 4,800→4,810 Hz are each about 14.2 UMAP units, while
    4,790 and 4,810 Hz are only 0.051 apart when 4,800 Hz is skipped;
  - 6,390→6,400 and 6,400→6,410 Hz are each about 14.0 units, while 6,390 and
    6,410 Hz are only 0.070 apart.
- In the original CNN space, 4,800 Hz is closest to 3,600 Hz and most of its
  nearest neighbors lie around 3,500–3,620 Hz. UMAP places it closest to
  3,560 Hz; that exact neighbor ordering is a two-dimensional layout effect.
- The 3,200 and 6,400 Hz mean vectors are effectively identical (cosine
  distance `0.00000334`) and have the two lowest norms, approximately
  `0.07265`.
- The discontinuities exist before UMAP: the special points have large cosine
  distances from their ±10 Hz neighbors. UMAP exaggerates their visible size;
  adjacent feature and UMAP gaps correlate moderately in magnitude
  (`r≈0.58`) but weakly in rank (`ρ≈0.12`).
- The 4,840→4,850 Hz separation is more projection-specific: cosine distance
  is only `0.079`, versus `2.924` units in UMAP.

In [`f0_umap.pdf`](../../data/synthetic_acoustic_probes/experiment_f0/f0_umap.pdf), the
gray path is always drawn between consecutive frequencies. It does not
explicitly connect 4,800 to 3,560 or 3,600 Hz.

## Findings from `cnn_phase_diagnostics`

The diagnostic columns mean:

- **Mean:** norm of `aggregate_segment(..., method='mean')`, the vector used
  in `f0_data.npz` and the UMAP.
- **Mean frame:** average of the individual frame-vector norms before
  aggregation.
- **Middle:** norm of `aggregate_segment(..., method='middle')`.
- **Cancellation ratio:** `mean / mean frame`; near one means stable vector
  direction across frames, independently of response magnitude.
- **Even/odd cosine:** cosine distance between averages of alternating output
  frames; zero means the two averages point in the same direction.

| Frequency | Mean | Mean frame | Middle | Cancellation ratio |
|---:|---:|---:|---:|---:|
| 3,190 Hz | 0.4103 | 0.6156 | 0.3778 | 0.6665 |
| **3,200 Hz** | **0.0726** | **0.0726** | **0.0726** | **1.0000** |
| 3,210 Hz | 0.4336 | 0.6777 | 1.2815 | 0.6398 |
| 3,990 Hz | 0.4727 | 0.6075 | 0.2678 | 0.7781 |
| **4,000 Hz** | **0.3262** | **0.3262** | **0.3262** | **1.0000** |
| 4,010 Hz | 0.5076 | 0.6344 | 0.8038 | 0.8002 |
| 4,790 Hz | 1.3949 | 1.7393 | 2.3790 | 0.8020 |
| **4,800 Hz** | **0.7097** | **0.7097** | **0.7097** | **1.0000** |
| 4,810 Hz | 1.2935 | 1.6053 | 2.0438 | 0.8058 |
| 6,390 Hz | 2.6054 | 3.6844 | 9.0455 | 0.7071 |
| **6,400 Hz** | **0.0727** | **0.0727** | **0.0727** | **1.0000** |
| 6,410 Hz | 2.9729 | 4.2814 | 4.1353 | 0.6944 |

The complete diagnostic sweep shows a structured, non-smooth frequency
response in the **Mean** column. It contains narrow notches including 3,200,
4,000, 4,800, 5,600, 6,000, 6,400, and 7,600 Hz, strong responses around
6,200–6,300 and 7,200 Hz, and rising responses near Nyquist. Because **Mean**
is the norm of the vector stored in `mean_cnn_features`, this pattern can also
be reproduced directly from `f0_data.npz`.

At all four central frequencies, the mean, mean-frame, and middle norms are
equal and the cancellation ratio is one. Their unusual final representations
therefore exist per frame rather than arising from averaging. At 3,200 and
6,400 Hz the per-frame response is consistently low-norm; 4,000 Hz has a
smaller notch; and at 4,800 Hz the stable vector points toward the high-3 kHz
representation neighborhood.

Across the complete sweep, every multiple of 50 Hz is temporally stationary
to displayed precision. This is consistent with wav2vec2's 20 ms final CNN
step: those tones complete an integer number of cycles per frame. Even/odd
cosine distances are approximately zero throughout, so there is no simple
period-two alternation.

## Limitation and next test

With a 50 Hz CNN frame rate, the 10 Hz stimulus spacing creates a five-frame
phase cycle away from 50 Hz multiples. Even/odd grouping mixes those phases.
The initial sample-offset test used the ±10 Hz triplets around 3,200, 4,000,
4,800, and 6,400 Hz, plus a 3,500 Hz control triplet. The default panel is now
expanded with 3,590–3,610 Hz and 10 Hz grids spanning 3,950–4,050 and
4,750–4,850 Hz. The
[`run_phase_diagnostic_experiment`](../synthetic_acoustic_probes/cnn_phase_diagnostics.py)
function creates its stimuli and stores, extracts the final CNN features, and
writes the diagnostics below `experiment_f0/phase_diagnostics`.
Layer-by-layer CNN inspection remains a later experiment.
