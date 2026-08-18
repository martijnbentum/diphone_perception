# Code style — synthetic_acoustic_probes

Date: 2026-08-18

This documents the compact style applied across
[`synthetic_acoustic_probes/`](../synthetic_acoustic_probes/) during a
file-by-file style-alignment pass. It's descriptive of the current codebase,
not aspirational — new code in this package should follow it, and existing
code should be brought in line when it's next touched.

## Docstrings

- **No blank line between a docstring and the code that follows it.** The
  docstring's closing `'''` is immediately followed by the first statement.
- **Public functions get a full per-parameter docstring**: one-line summary,
  then each parameter as `name:  description`, with colons aligned to the
  longest parameter name (pad with spaces so every description starts in the
  same column). No blank line separates the summary, the parameter list, and
  any trailing prose — they all run together.
- **Private (`_`-prefixed) helpers get a short one-line docstring** stating
  what they validate or compute, not a full parameter list.

Example ([`stimuli.py`](../synthetic_acoustic_probes/stimuli.py)):

```python
def sum_of_sinusoids(frequencies, amplitudes=1.0, phases=0.0,
    duration=DURATION, sample_rate=SAMPLE_RATE, dc_bias=0.0,
    allow_clipping=False, stimulus_id=None, extra_parameters=None):
    '''Create an exact-length sum of sinusoids.
    frequencies:       Frequencies in Hz for each sinusoid component.
    amplitudes:        Amplitude per component, or one shared amplitude.
    ...
    '''
    frequencies = _as_1d_array(frequencies, 'frequencies')
```

## Blank lines

- **No blank lines inside a function body or inside a docstring.** Logical
  groups of statements run together without separating whitespace.
- **Keep** the standard two-blank-line separator between top-level
  definitions, blank lines that group import statements, and the blank line
  between a dataclass's field list and its first method (that's a
  class-level structural separator, not "inside a function").

## Line wrapping

- **79-character soft limit.** Pre-existing lines already over it were left
  alone rather than reflowed as a drive-by change; new or touched code stays
  under it.
- **Multi-line signatures, calls, and dict/set/list literals are packed**,
  not one-item-per-line. Fill each line greedily up to the limit; attach the
  closing paren/brace/bracket to the last item instead of giving it its own
  line. This applies just as much to dict literals (manifest rows,
  `extra_parameters`, provenance dicts) as it does to function signatures —
  don't leave a dict at one-key-per-line just because the surrounding call
  or signature already got packed.

```python
# not this
def f(
    a,
    b,
    c,
):

# this
def f(a, b,
    c):

# not this
row = {
    'a': 1,
    'b': 2,
    'c': 3,
}

# this
row = {'a': 1, 'b': 2,
    'c': 3}
```

- **Avoid dangling-parenthesis expressions** (`x = (\n    ...\n)`,
  ternaries split across lines). Prefer an explicit `if`/`else`, an `elif`
  chain, or pulling the sub-expression into a named variable.

```python
# not this
response *= 1 / np.sqrt(
    1 + np.square(2 * (frequencies - formant) / bandwidth)
)

# this
ratio = 2 * (frequencies - formant) / bandwidth
response *= 1 / np.sqrt(1 + np.square(ratio))
```

- **Single-line `if condition: statement`** for simple guard clauses
  (raises, one-line assignments) is preferred over a three-line `if` block.
- **Prefer single-line comprehensions** over multi-line ones. If elements
  need casting (e.g. to `float`), cast the whole source array/list once
  before the comprehension rather than per element inside it.

## Structure

- **Compute into a named variable, then return or append it** — avoid a bare
  `return <expression>` or `list.append(<expression>)` when the expression
  does real work (constructs an object, builds a dict, calls a function).
  Trivial one-liners (`return int(value.item())`) are fine as bare returns.
- **Hoist loop-invariant work above the loop**; cast or compute values that
  don't change per iteration once, before the loop, not inline on every
  pass. Values that do vary per iteration get cast once at the top of the
  loop body, not repeated inline at each use.
- **Don't re-cast the same value on both sides of a call.** If a caller casts
  a value before passing it to a helper, the helper should trust that cast
  and use the value as-is rather than casting it again — cast once, at
  whichever side owns the decision, not in both places. In
  [`formants.py`](../synthetic_acoustic_probes/formants.py),
  `praat_vowel_stimulus` casts `f0_hz`/`f1_hz`/`f2_hz` to `float` once
  before calling `_praat_vowel_parameters`, which then uses them directly
  instead of wrapping them in `float(...)` again.
- **Extract a private helper when a block does real, self-contained work**
  (building a manifest/provenance dict from several independent inputs is
  the clearest case — see `_manifest_row` in
  [`storage.py`](../synthetic_acoustic_probes/storage.py) and
  `_praat_vowel_parameters` in
  [`formants.py`](../synthetic_acoustic_probes/formants.py)). **Don't**
  extract a helper that only forwards its arguments to one other call with
  no added logic — that's indirection without simplification.
- **File order**: the module's central/experiment-specific public functions
  come first, shared generic engine functions and dataclasses come next,
  and private helpers are grouped at the bottom.
- **Don't cache external module state into a module-level constant at
  import time** (e.g. `_ROOT = locations.some_path`) — reference the
  attribute directly at the call site so it can still be monkeypatched and
  reflects the current value.
- **Drop dead code** — unused parameters, unused public functions — rather
  than keeping them for hypothetical future use.

## Comments

- Comments explain the **goal** of a step, not the parameter values already
  visible in the code. Reserve them for genuinely non-obvious API behavior.
- Verify factual/API claims before writing them into a comment — a wrong
  comment is worse than no comment.

## Naming

- Prefer names that describe behavior over vague ones
  (`_repeat_or_match_length` over `_broadcast_values`).
- Short local names (`sid`, `m`) are acceptable when they're the difference
  between a line fitting the 79-column limit and not, in a small enough
  scope that the abbreviation stays unambiguous. It's a fit-driven
  exception, not a general preference for terseness.
