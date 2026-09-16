# Representation decomposition

## Step 1: random time-point selection

Implemented in `sampling.py`. Use Phraser's CGN audio inventory and select exact
`comp-k` or `comp-o` and `nl` path components. Include all audio, including silence
and unannotated intervals. Phone labels do not influence selection.

Frame counts come from `frame.make_frames_from_duration`; sampled timestamps
come from indexed `frame.Frames` objects. Both use the library's default window,
step, and origin. The sampler and manifest do not define separate timing
settings. The library uses seconds internally, while Phraser durations and
expanded sample timestamps use milliseconds.

Treat each recording as a grid of complete 25 ms windows starting at 0, 20, 40,
... ms. Sample 418,500 unique frame slots uniformly without replacement, matching
the size of the existing balanced phone inventory. This is the discrete version
of sampling random time points: longer recordings and longer phones contribute
more because they contain more eligible frames. Windows can overlap by 5 ms,
but the same recording/frame pair cannot be selected twice. Trailing audio that
cannot form another complete window and recordings shorter than 25 ms have no
additional eligible frames. No waveforms are physically concatenated.

Before sampling, shuffle recordings within each component and assign half to
fitting and half to evaluation (the extra recording goes to evaluation for odd
counts). Each present component needs at least two usable recordings. Then
sample globally across their cumulative frame counts. The total is 418,500
across both splits, not per split. Recording counts are approximately balanced;
durations and resulting sample counts need not be. Splits isolate recordings,
not speakers. Inspect the reported counts before extraction.

Run from the repository root in an environment with Phraser and frame installed:

```sh
python -m decomposition.sampling --n-samples 418500 --seed 42
```

The default source is `locations.cgn_lmdb`; output is
`../data/decomposition/random_frames.json`. Override either with `--store` and
`--output`. An existing output is never overwritten. The local checkout does
not contain the configured CGN store; run selection where that store is present.

For an already open store:

```python
from decomposition.sampling import iter_samples, sample_frames, save_manifest

manifest = sample_frames(store.audios, n_samples=418_500, seed=42)
save_manifest(manifest, 'random_frames.json')
sample = next(iter_samples(manifest))
```

The JSON stores the recording inventory, stable Phraser keys, split assignments,
sampling configuration, counts, and sorted selected frame indices per recording.
`iter_samples` expands it into individual rows with frame starts, ends, and
centers in milliseconds. Sorting makes storage and later extraction convenient;
it does not change the sample probabilities. Input audio order does not affect
the result. For exact reuse, load the saved manifest rather than resampling.

## Subsequent steps

1. Attach Phraser phone, speaker, and context annotations at sampled positions.
   Retain unannotated points and represent overlapping annotations explicitly.
2. Extract and store embeddings through Echoframe for the final checkpoint and
   selected layer. Use 2 seconds of context on each side, clipped at recording
   boundaries, and retain each selected frame's position. Accept Echoframe's
   frame-selection behavior; do not snap random samples to phone centers.
   The integration must preserve the recording-relative frame grid when windows
   are cropped. Persist actual extraction bounds and available context.
3. Build the sample-by-dimension matrix, fit centered SVD, and check held-out
   strengths and direction/subspace stability.
4. Interpret modes using phone, sonority, voicing, speaker, and recording
   metadata; compare with the existing balanced sample later.

Step 1 uses recording metadata only and does not run model inference.
