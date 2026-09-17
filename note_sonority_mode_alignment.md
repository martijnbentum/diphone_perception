# Testing sonority alignment with an SVD mode

This worked example illustrates how to test whether one mode separates phones
according to sonority. All scores are invented for illustration; they are not
measurements from a speech model.

Related plan:
[Speech decomposition experiment](experiment_plan_decomposition.md).

## 1. Specify a sonority ordering

Assign a simple ordering before examining the mode:

| Phone | Description | Sonority rank |
|---|---|---|
| /p/ | Stop | 1 |
| /m/ | Nasal | 2 |
| /l/ | Liquid | 3 |
| /a/ | Vowel | 4 |

These ranks are an analysis choice, not measured acoustic quantities. Using
rank differences also assumes equal spacing between adjacent categories.
Alternative reasonable rankings and spacings should be checked in real data.

## 2. Obtain phone scores along a mode

Fit SVD on a matrix of centered embeddings. For a unit embedding-space mode
direction q, project each phone representation onto that direction:

`mode_score(phone) = (embedding(phone) - fitting_mean)^T q`

Use the centering mean from the matrix on which SVD was fitted. Interpret the
mode using held-out occurrences, with matched speaker and context weighting.
For a joint speaker-phone matrix, retain speaker-specific scores as well as
balanced phone averages to assess whether the pattern generalizes.

Suppose the resulting phone scores are:

| Phone | Sonority rank | Mode score |
|---|---|---|
| /p/ | 1 | -1.5 |
| /m/ | 2 | -0.5 |
| /l/ | 3 | +0.5 |
| /a/ | 4 | +1.5 |

The sonority ranks were supplied; the mode scores would be obtained from model
representations. They are separate quantities.

## 3. Compare differences between phone pairs

For each pair, compute:

`sonority_difference = abs(rank(phone_i) - rank(phone_j))`

`mode_difference = abs(score(phone_i) - score(phone_j))`

For the toy scores:

| Pair | Sonority difference | Mode-score difference |
|---|---|---|
| /p/-/m/ | 1 | 1 |
| /p/-/l/ | 2 | 2 |
| /p/-/a/ | 3 | 3 |
| /m/-/l/ | 1 | 1 |
| /m/-/a/ | 2 | 2 |
| /l/-/a/ | 1 | 1 |

Phones farther apart in the chosen sonority scale are also farther apart along
this mode. If /p/ and /a/ instead had almost identical scores while /p/ and /m/
were widely separated, the correspondence would be poorer.

## 4. Quantify the correspondence

Calculate Spearman rank correlation between the two difference columns. In
this example it is 1: perfect agreement, including the tied distances.

Reversing the mode's sign leaves all pairwise distances unchanged. Multiplying
all its scores by a nonzero constant also leaves their rank ordering unchanged.
This makes the measure about the pattern of separation rather than amplitude.
If all mode distances are equal, the correlation is undefined and should be
reported as such.

The six pairs are not six independent observations: pairs share phones. Use
phone-level label permutations, recomputing all distances together, for a null
comparison. Resample speakers or utterances to assess uncertainty in estimated
phone scores. Avoid significance tests that treat pairwise distances as
independent samples. Four phones provide only a toy demonstration.

## 5. Relate alignment to mode importance and training

Repeat the calculation for each mode and compare its alignment with its
singular value or singular-value rank. These measure different things:

- Alignment: whether the mode expresses the chosen sonority-distance pattern.
- Strength: how much the mode contributes to the decomposed matrix.

One strongly aligned mode supports a sonority interpretation for that mode.
The broader hypothesis is that sonority helps organize the spectrum: stronger
modes express broad sonority distinctions, while weaker modes express closer
or within-group distinctions. One aligned component alone does not establish
that hierarchy; inspect the contrasts expressed across the remaining modes.

Track the identified phone-score patterns across training checkpoints to ask
whether the dominant sonority-related distinctions emerge earlier. Account for
changing coordinates and component order, as described in the main plan.

Sonority can covary with intensity, periodicity, and other acoustic properties.
Compare those explanations using held-out data. Rank correlation does not
remove the spacing assumption introduced when defining sonority differences.
The result supports correspondence with a proposed organization, not proof
that the model explicitly represents sonority or uses it as a learning rule.
