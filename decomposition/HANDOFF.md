# Handoff: final-checkpoint decomposition POC

## Goal and decisions

Discover reproducible dominant modes in a speech model's final-checkpoint
embeddings, then interpret them using sonority, voicing, speaker, and other
metadata. Follow development across checkpoints later. Implement step by step.

- Start with random **time points**, represented by distinct frame slots, rather
  than phone occurrences or speaker–phone means.
- Use all audio, including silence and unannotated intervals, from Netherlandic
  CGN components k and o. Longer recordings contribute proportionally more.
- Sample 418,500 frames total, matching the balanced phone inventory's size.
- Use Phraser for metadata, `frame` for timing, and Echoframe for embeddings.
  Trust their frame-selection behavior. Use a 2-second collar on each side,
  clipped at recording boundaries, for the upcoming extraction step.

## Implemented

[sampling.py](sampling.py) builds a recording inventory, splits recordings into
fitting/evaluation sets, samples unique global frame indices, and saves a JSON
manifest. `iter_samples` expands it into recording-relative timestamps.

`build_audio_inventory(audios, components=('comp-k', 'comp-o'))` assumes unique
audios. Timing uses `frame` defaults (25 ms windows, 20 ms steps); no separate
window/step configuration is maintained. `frame` was added to requirements.

Run where the CGN store is available:

```sh
python -m decomposition.sampling --n-samples 418500 --seed 42
```

Defaults: `locations.cgn_lmdb` and
`../data/decomposition/random_frames.json`. Existing manifests are protected.
Sampling has not run against the real corpus: the configured store is absent
locally. Nine targeted tests and repository style checks passed against
`../../frame`, using the neighboring Phraser environment:

```sh
PYTHONPATH=../../frame ../../phraser/.venv/bin/python -B -m unittest discover -s tests -p 'test_decomposition_sampling.py'
python3 -B scripts/check_style.py decomposition tests/test_decomposition_sampling.py
```

## Next steps

1. Run and inspect the sample manifest where CGN is available; check component
   and split counts. Splits separate recordings, not speakers, and frame counts
   need not be equal across splits.
2. Attach Phraser annotations at selected positions, retaining unannotated
   samples and explicitly handling overlapping annotations.
3. Integrate Echoframe extraction while preserving selected frame positions and
   recording-relative timing through cropped context windows.
4. Fit centered SVD, assess held-out stability, then interpret the modes.

[README.md](README.md) is the current action plan. The broader
[POC note](../note_poc_final_checkpoint_modes.md) still describes phone-token
sampling and needs updating to the agreed random-time-point approach; preserve
its `OLD_PLAN` text. Existing `probing` loaders assume phone-balanced data and
its cross-validation splits individual tokens, so neither should be reused
unchanged for this experiment. Work is uncommitted.
