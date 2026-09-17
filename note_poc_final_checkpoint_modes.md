# POC: final-checkpoint modes from individual phone tokens

## Aim and scope

Discover the dominant variation across individual speech occurrences at one
final checkpoint and one predefined layer. Fit SVD to center-frame token
embeddings, then use phone, speaker, sonority, voicing, and recording annotations
to interpret the modes. Group means are interpretation tools computed after
decomposition; they do not define its rows.

This POC asks whether leading modes are reproducible and interpretable. It does
not test learning order or require a sonority hierarchy as its outcome.

Related documents:
[experiment plan](experiment_plan_decomposition.md) and
[sonority alignment example](note_sonority_mode_alignment.md).

## 1. Define the token sample and independent splits

1. Select aligned phone occurrences from multiple speakers. Retain phone,
   speaker, utterance, boundaries, neighboring phones, and recording/session
   labels where available. Prespecify sonority groups/ranks and matched voicing
   contrasts; include several phones within each sonority group.
2. Split by utterance into fitting set A and evaluation set B within speakers.
   Separate recording sessions across splits where feasible. Evaluation then
   concerns new utterances from the same speakers; unseen-speaker transfer is
   an extension.
3. Start with random-token sampling: within each split, draw uniformly without
   replacement from all eligible phone tokens. Do not impose speaker or phone
   quotas. Fix sample sizes before inspecting embeddings and save the selection
   seed, token IDs, and counts by speaker, phone, and recording condition.
4. The primary spectrum describes the eligible corpus population, including its
   frequency differences. Rare groups may have insufficient observations for
   interpretation; report this rather than changing the random sample to fill
   cells. Use balanced speaker–phone sampling as a subsequent comparison, as
   specified below.

## 2. Extract one center-frame vector per token

1. Run complete utterances through the checkpoint in evaluation mode with a
   fixed masking policy. Save model and layer identifiers.
2. For each phone interval `[start, end)`, select the model frame whose center
   lies inside the interval and is nearest its midpoint. Break ties by choosing
   the earlier frame. Document model-specific frame timestamps and offsets.
3. Retain that frame's embedding without averaging frames. Exclude intervals
   containing no frame center and report exclusions by phone and speaker.
   Apply the sampling policy to the usable tokens after these exclusions.

A center-frame vector can still contain surrounding-speech information through
the model's receptive field and contextual attention.

## 3. Construct and decompose the token matrix

1. Build `Z_A` with shape `N_A × D`: one row per token and one column per
   embedding dimension. Two occurrences of the same speaker–phone combination
   remain two separate rows. Every selected token has equal weight.
2. Compute the fitting column mean `mu_A`, then fit centered SVD:
   `X_A = Z_A - mu_A = U_A S_A V_A^T`. Do not standardize coordinates or normalize
   token vectors in the primary analysis.
3. Save directions `q_k = V_A[:, k]`, token scores `X_A q_k`, singular values,
   covariance eigenvalues `s_k^2 / (N_A - 1)`, and variance fractions. Report total
   variance and the rank ceiling `min(N_A - 1, D)`.
4. Initially inspect five nonzero modes, or fewer if rank is lower. Include a
   whole cluster of nearly equal singular values when it crosses this boundary;
   interpret its subspace if its individual directions are unstable.

## 4. Check reproducibility on independent tokens

1. **Measure held-out strength.** Project B using A's mean and directions:
   `b_k = (Z_B - mu_A) q_k`. Compare fitting and evaluation score variances and
   mean scores. Divide the evaluation variance along each direction by total
   evaluation variance to obtain its held-out variance fraction. Compare
   covariance eigenvalues or variance fractions, rather than raw singular
   values, when sample sizes differ.
2. **Refit independently.** Fit another SVD to B centered by its own mean.
   Compare leading directions with absolute dot products, allowing mode order
   to change. Compare near-degenerate mode groups with principal angles.
   No coordinate alignment is needed for the same checkpoint and layer.
3. **Check interpretation transfer.** With A's directions fixed, average token
   scores by phone, speaker, or speaker–phone combination separately in A and B.
   Use matched weighting and compare corresponding group-score patterns and
   contrast sizes. Only these matched summaries can be compared row by row:
   A and B contain different tokens, so their raw score vectors are not paired.
4. **Estimate uncertainty.** Bootstrap whole utterances within speakers,
   separately in each split, starting with 200 replicates. Keep A's directions
   fixed for conditional uncertainty in held-out metrics; refit SVDs separately
   to assess direction/subspace stability. Report variation in token counts and
   missing interpretation groups. These intervals concern the sampled speakers.

Report pattern stability and strength separately. A large fitting singular value
alone is insufficient evidence of reproducibility. Retain A's modes for the
interpretation stage; use B's refit only as a stability diagnostic.

## 5. Interpret the modes after decomposition

1. **Inspect token-score distributions.** Plot B's scores by phone, speaker, and
   recording situation. Show within-group spread as well as means: a group
   contrast may explain only a small part of a token mode's total variation.
2. **Summarize phone and speaker patterns.** Average scores within speaker–phone
   cells for heatmaps, then compute phone summaries with equal speaker weighting
   and speaker summaries with matched phone weighting. Use common covered groups
   for comparisons and report omitted groups. These averages never replace the
   token matrix used to discover the modes.
3. **Test sonority correspondence.** From balanced phone-score means, calculate
   the pairwise-distance rank correlation in the sonority note. Check agreement
   across speakers and splits. Also report between-sonority-group versus
   within-group variation of the phone means. This fraction describes the
   phone summaries, not the fraction of all token variance explained by sonority.
   Permute phone labels together across speakers for a descriptive null; do not
   treat phone pairs as independent observations.
4. **Test voicing alignment.** In A, estimate a matched contrast such as mean /d/
   minus mean /t/, balancing speakers and relevant contexts. Normalize the
   nonzero contrast vector to `v`, then compute `abs(q_k^T v)`. Validate using
   B and independent contrasts such as /b/–/p/ and /g/–/k/, where available.
   Check the signed contrast pattern across pairs as well as absolute alignment.
5. **Test speaker alignment.** Compute phone-balanced speaker-mean vectors in A
   as a reference, center them, and fit speaker PCA. Use at most three nonzero
   PCs for this POC, reporting sensitivity to one and two. Measure
   `||Q_s^T q_k||^2`, compare with dimension-matched random subspaces, and check
   speaker-score patterns in B. Both reference and token modes use A, so their
   fitting alignment is descriptive; transfer provides independent evidence.
6. **Record alternative explanations.** Check whether apparent linguistic or
   speaker patterns also track recording, intensity, periodicity, or context
   where annotations permit. Write a short interpretation supported by held-out
   results; retain mixed and unresolved interpretations rather than forcing a
   single label.

Treat these as exploratory comparisons. Sonority alignment declining across
modes can partly follow from orthogonality and does not establish a hierarchy.
Geometric alignment alone does not show that the model uses the distinction.

## 6. Deliver the POC result

- Token manifest, sampling policy, split assignments, and extraction settings.
- Token-matrix spectrum and held-out variance fractions.
- Direction/subspace reproducibility estimates and uncertainty.
- Held-out token-score plots, grouped summaries, and sonority, voicing, and
  speaker interpretations for the leading reproducible modes.
- A compact mode table: strength, reliability, interpretation, and alternatives.

Success means identifying repeatable dominant variation in the sampled token
representations. Improve sampling if reliability is poor. Defer training
trajectories, additional layers, and discrimination experiments until this
result is established. SVD of speaker–phone means remains an optional comparison
for asking how the spectrum changes when within-cell variation is averaged out.

## Token sampling strategies

Sampling determines which population's variation the SVD describes. Each
selected token remains a separate matrix row under all strategies below.

| Strategy | How to sample | Main tradeoff |
|---|---|---|
| **Random tokens** | Give every eligible token equal selection probability. | Reflects corpus frequencies; common phones and prolific speakers contribute more. |
| **Equal tokens per speaker** | Draw the same number from each speaker. | Balances speakers but preserves differences in their phone distributions. |
| **Equal tokens per phone** | Draw the same number for each phone category. | Balances phones, but speaker contributions may remain uneven. |
| **Equal tokens per speaker–phone combination** | Draw a fixed number from every cell of a selected, well-covered grid. | Balances both factors; rare combinations can limit sample size or coverage. Tokens remain separate SVD rows. |
| **Equal tokens per sonority or manner group** | Balance broad groups, then sample phones within them using a specified rule. | Useful for targeted comparisons, but the chosen grouping influences the spectrum through sampling. |
| **Balance recording conditions or contexts** | Match token counts across conditions, ideally within speaker and phone. | Helps separate competing explanations; requires sufficient crossed coverage. |
| **Balance utterance contributions** | Cap tokens per utterance or sample a fixed number from each. | Reduces domination by long utterances; does not automatically balance phones or speakers. |

Strategies can be combined, but each additional constraint changes the sampled
population and should be documented.

### Two strategies for this POC

1. **Random tokens — starting point.** Sample uniformly from eligible tokens in
   each utterance-disjoint split. This asks what dominates the corpus as
   observed, without equalizing speaker or phone contributions.
2. **Balanced speaker–phone tokens — subsequent comparison.** Select a
   well-covered speaker-by-phone grid and sample equal numbers of tokens per
   cell. This asks what dominates when selected speakers and phones contribute
   equally. Fit token-level SVD here too; do not average the embeddings by cell.

For a comparison that isolates weighting, use the same speaker/phone inventory,
eligible token pool, split assignments, and total sample size per split for the
random and balanced samples. If balancing requires a restricted inventory or a
smaller sample, draw an additional random sample under those same restrictions;
keep the original random-sample result as the primary analysis.

Split utterances before sampling, sample without replacement, and repeat with a
few sampling seeds to assess sensitivity. Overlap between the two strategies'
samples is acceptable; no utterance may cross the fitting/evaluation boundary.

## Why we do not start with the old plan

Averaging embeddings by speaker–phone combination before SVD suppresses
within-cell variation and can make differences between those groups relatively
more prominent. It answers what dominates the group means. We first want to
discover what dominates individual token representations, then interpret it
using those groups. Token sampling still defines the population being studied,
but grouping is no longer built into the decomposition through averaging.

# OLD_PLAN

# POC: reproducible modes at the final checkpoint

## Aim and scope

Use one final checkpoint and one predefined layer to establish whether dominant
representation modes are reproducible and interpretable as phonetic, sonority,
speaker, or mixed patterns. A sonority hierarchy is a possible finding, not a
requirement for success. This POC does not test learning order.

Related documents:
[experiment plan](experiment_plan_decomposition.md) and
[sonority alignment example](note_sonority_mode_alignment.md).

## 1. Select the sample and split utterances

1. Select a manageable set of speakers and phones with good coverage. Include
   several phones within each proposed sonority group so within-group
   distinctions can actually be evaluated.
2. Specify sonority groups and ranks before inspecting embeddings. Record phone,
   speaker, utterance, boundaries, neighboring phones, and recording/session
   identifiers where available.
3. Split utterances into fitting set A and evaluation set B within each speaker.
   All tokens from an utterance stay together. Use separate recording sessions
   across sets where feasible; otherwise report that session generalization is
   untested.
4. Retain a complete speaker-by-phone grid: every selected speaker must have
   usable tokens of every selected phone in both sets. Choose and record a
   minimum token count per cell before decomposition. Reduce the speaker/phone
   selection if necessary; do not replace missing cells with zero vectors.
5. Save the token IDs, split assignment, cell counts, and selection seed. Exclude
   silence and non-speech. Record context imbalance that remains between cells
   or splits.

The primary reproducibility question concerns new utterances from the same
speakers and phones. Generalization to unseen speakers is a later extension.

## 2. Extract one center-frame embedding per phone token

1. Run the same complete utterances through the final checkpoint in evaluation
   mode with a fixed masking policy. Record the checkpoint and layer.
2. For a phone interval `[start, end)`, calculate its midpoint. Using the model's
   frame timestamps, select the frame whose center lies inside the interval and
   is closest to that midpoint. Break ties by choosing the earlier frame.
3. Use that single frame's hidden vector as the token embedding. Do not average
   frames. Exclude intervals containing no frame center and report exclusions
   by phone and speaker; apply the cell-coverage requirement after exclusions.
4. Save token embeddings and selected frame indices. Document the model-specific
   timestamp convention, including any encoder offset.

The selected frame can still represent surrounding speech because of the
encoder receptive field or contextual attention. Center-frame selection defines
where the representation is sampled, not the extent of its acoustic context.

## 3. Construct speaker–phone means precisely

For each split separately, group token embeddings by speaker identity `s` and
phone category `p`. If the cell contains `N_sp` tokens, compute:

`m_sp = (1 / N_sp) * sum_i z_spi`

Here, `z_spi` is the single center-frame embedding of token `i`. Thus:

- Each token contributes equally to its speaker–phone mean, regardless of phone
  duration. Frames are never averaged within a token.
- Different occurrences of the same phone from the same speaker are averaged.
  Different phones and speakers remain separate rows.
- The matrix has `S * P` rows and `D` columns for `S` speakers, `P` phones, and
  embedding dimension `D`. Use identical row identities and ordering in A and B.
- Every row receives equal weight in SVD. A cell with more tokens has a better
  estimated mean but does not receive more decomposition weight.

For example, the row for speaker 03 and /a/ averages that speaker's /a/ token
vectors in the relevant split. It contains no /a/ vectors from other speakers
and no other phones from speaker 03.

## 4. Fit the final-checkpoint decomposition

1. Compute the column mean `mu_A` across fitting rows and center matrix `M_A`:
   `X_A = M_A - mu_A`.
2. Fit `X_A = U_A S_A V_A^T`. Save the mean, singular values, directions, and row
   scores. Do not standardize individual embedding coordinates.
3. Plot singular values and variance fractions `s_k^2 / sum_j s_j^2`, and report
   total variance, row count, and the rank ceiling `min(S * P - 1, D)`.
4. Inspect the first five nonzero modes, or fewer if rank is lower. If this
   boundary splits a cluster of nearly equal singular values, include the
   cluster and evaluate its subspace together. Treat this as an exploratory POC;
   report the retained dimensions and any grouping decisions.

For a unit direction `q_k = V_A[:, k]`, its fitting score pattern is the vector
`a_k = X_A q_k`: one score for every speaker–phone cell.

## 5. Check reproducibility before assigning interpretations

Use two complementary checks: transfer of a fixed pattern and stability of the
decomposition itself. No coordinate alignment is needed between A and B because
both use the same checkpoint and layer.

1. **Project independent means onto the fitted directions.** Compute
   `b_k = (M_B - mu_A) q_k`. Plot `a_k` against `b_k` for corresponding cells.
   Report Pearson correlation for pattern agreement, the regression slope of
   `b_k` on `a_k` for amplitude transfer, and their root-mean-square difference
   for absolute disagreement. Report mean shifts too: correlation alone can
   conceal a shift or rescaling. These are descriptive agreement measures;
   fitting scores also contain sampling error.
2. **Independently refit SVD in B.** Center B using its own column mean for this
   separate fit. Compare leading directions using absolute dot products and
   display the full correspondence matrix, allowing modes to exchange order.
   For nearly equal singular values, compare the corresponding spans using
   principal angles instead of requiring stable individual axes.
3. **Quantify sampling uncertainty.** Bootstrap whole utterances within each
   speaker, separately in A and B, rebuilding the cell means. Use 200 replicates
   initially for POC uncertainty estimates. With A's directions fixed, estimate
   uncertainty in the transfer statistics; additionally refit decompositions to
   assess direction/subspace stability. Never bootstrap frames independently.
   Skip and count replicates with missing cells; frequent failures indicate
   insufficient coverage.
4. **Judge patterns and amplitudes separately.** Strong agreement of cell scores
   supports a repeatable pattern; similar amplitude supports a repeatable
   strength. Stable spans with rotating axes support subspace interpretation.
   Low agreement means the mode is unresolved with this sample, even if its
   fitting singular value is large.

Do not impose an arbitrary universal correlation cutoff. Report estimates and
uncertainty, identify the strongest reproducible findings, and leave weak or
unstable modes uninterpreted. Independent refitting is a diagnostic; retain
A's directions for the held-out interpretations below.

## 6. Interpret the reproducible modes in explicit steps

For each retained mode, use its evaluation scores `b_sp,k`. Apply any sign flip
consistently to fitting and evaluation plots; the sign has no intrinsic meaning.

1. **Display the complete pattern.** Make a speaker-by-phone heatmap with a
   common phone ordering and a diverging score scale. This shows whether a
   distinction repeats across speakers, reflects speaker offsets, or changes
   with speaker identity.
2. **Separate phone, speaker, and interaction patterns.** Compute the grand
   mean of the scores, each phone's mean across equally weighted speakers, and
   each speaker's mean across equally weighted phones. Subtract the grand mean
   from each marginal mean to define `phone_effect_p` and `speaker_effect_s`.
   The interaction is `b_sp,k - grand_mean - phone_effect_p - speaker_effect_s`.
   Plot the marginal effects and
   report their shares of centered score variation alongside the interaction
   share. These shares are additive for the complete, equally weighted grid.
   They describe score structure, not causal sources.
3. **Test sonority correspondence.** Using the balanced phone means, compute
   the pairwise-distance Spearman correlation described in the sonority note.
   Repeat the calculation within individual speakers to check consistency.
   Bootstrap utterances for uncertainty and permute phone labels jointly across
   speakers for a descriptive null comparison. Do not treat phone pairs as
   independent observations or use uncorrected per-mode tests as confirmation.
4. **Measure broad versus within-group contrasts directly.** For the balanced
   phone scores, split their sum of squared deviations into between-sonority-
   group and within-group components. Weight each group mean by its number of
   phones. Report the between-group fraction for each mode and inspect which
   phone pairs account for the within-group component. If phone-score variation
   is zero, report the fraction as undefined. Several phones per group are
   essential here.
5. **Summarize speaker-direction alignment.** From A, average the speaker–phone
   vectors over phones with equal weights, center the resulting speaker means,
   and fit speaker PCA. For this POC use at most three nonzero speaker PCs,
   capped by the available rank, and report sensitivity to one and two PCs.
   For their orthonormal basis `Q_s`, calculate `||Q_s^T q_k||^2`. Compare with
   random subspaces of the same dimension and validate the associated speaker
   score pattern in B. Treat the geometric alignment as descriptive: both
   decompositions were fitted from A.
6. **Write a short evidence-based interpretation.** For example: a repeatable
   vowel–obstruent contrast across speakers; a mostly speaker-associated offset;
   or a mixed pattern that varies by speaker. Note alternative explanations,
   particularly intensity, periodicity, context, and recording differences.
   A speaker-associated pattern alone does not isolate voice characteristics.

Lower sonority alignment in later modes does not establish a hierarchy: SVD
orthogonality already constrains what remains after a leading sonority-aligned
direction. Treat across-spectrum patterns as exploratory until tested against
a null model preserving those constraints. No discrimination or functional-use
claim follows from these plots alone.

## 7. Deliver the minimal POC report

- Sample and extraction manifest, including split assignments, cell counts,
  exclusions, and the center-frame timing rule.
- Singular-value spectrum and explained-variance plot.
- Reproducibility plots and estimates for the leading modes or subspaces.
- Held-out mode heatmaps, phone/speaker effects, sonority correspondence,
  between-group fractions, and speaker-subspace alignment.
- A compact table with one row per interpreted mode or subspace: strength,
  reliability, interpretation, and unresolved alternatives.

Success means identifying repeatable leading structure that can be described
across phones and speakers. If reliability is poor, improve coverage before
extending to training trajectories. Defer checkpoint tracking, extra layers,
probe/ABX evaluations, and subspace-removal experiments until this basic result
is established.
