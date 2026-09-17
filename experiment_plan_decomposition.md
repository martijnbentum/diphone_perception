# Decomposing speech representations across training

## Aim

Determine whether the development of phonetic representations reflects
strengthening of existing directions, acquisition of additional useful
directions, or reorganization of the embedding space. Relate these changes to
speaker information, acoustic properties, and the targets used during training.

The first experiment extends
[Liu, Tang, and Goldwater](liu_orthogonal_subspaces_summary.md) across training
checkpoints. A later experiment approaches the stronger question posed by
[Saxe et al.](saxe_semantic_development_summary.md) and
[Karkada et al.](karkada_training_dynamics_summary.md) : can statistical
structure in the training problem predict which patterns are learned first?

This is a proposed study, not a report of completed experiments. Corpus, model,
checkpoint availability, and sample sizes remain implementation choices.

## Primary experiment: do dominant modes emerge earlier?

Use the fully trained model to identify strong modes, interpret them through
phonetic and speaker reference directions, and track the same patterns across
training checkpoints. Test whether final mode strength predicts emergence time.
This tests a retrospective relationship; final-model strength is not the
data-defined mode strength used in Saxe's theoretical prediction.

### Discover modes from complete representations

At a fixed layer, construct a matrix with one complete mean embedding per
speaker-phone combination. Apply centered SVD to this joint matrix. Retain the
phone-mean and occurrence-level decompositions as complementary analyses.
Phone means alone average over speakers and can suppress the speaker variation
needed for the joint question. Keep the same rows and weighting across time.

Fit the final-model SVD on one analysis partition. Record singular values,
variance fractions, embedding-space directions, and scores across joint rows.
Use separate occurrences for estimating reference directions and validating
interpretations. Averaging reduces occurrence-level noise but does not remove
all effects of phonetic context, recording conditions, or unequal coverage.

### Interpret phonetic alignment

Estimate Choi-style feature vectors from matched phone contrasts, for example
mean([d]) minus mean([t]) for voicing. For a unit mode direction q and a unit
feature direction v, use abs(q^T v) as alignment. Check transfer to independent
contrasts such as [b]-[p] and [g]-[k], held-out speakers, and contexts.

A mode can align with several correlated properties. Report its alignment
profile rather than forcing a unique feature label. Feature directions need
not be orthogonal, and ordinary projection scores are not independent feature
coefficients. Inspect mode scores across phones to support the interpretation.

### Interpret speaker alignment

Speaker variation is a candidate source of prominent modes, not a guaranteed
leading mode or a single direction. Following Liu, average representations by
speaker with matched phone weighting, center the speaker-mean matrix, and
apply PCA to obtain an orthonormal speaker basis Q_speaker. Select its size on
fitting/development data and report sensitivity to that choice.

For a unit joint-mode direction q, measure:

`speaker_alignment = ||Q_speaker^T q||^2`

This ranges from 0 to 1 and measures the fraction of the direction's squared
length lying in the estimated speaker subspace. Compare against random
subspaces of the same dimension, since larger subspaces capture more by chance.
For groups of near-degenerate modes, compare subspaces using principal angles.

A Choi-style adaptation is also possible: estimate a speaker contrast
mean(z | speaker A, phone p) minus mean(z | speaker B, phone p), then test its
agreement and transfer across other phones and held-out utterances. Estimate
the speaker-pair offset from fitting phones only. Unlike voicing, speaker
identity has many categories and no universal positive-versus-negative axis;
different speaker pairs can define different directions. PCA of balanced
speaker means provides a primary summary of this family of contrasts.

Control recording/channel effects and phone/context imbalance. Speaker
directions may encode those differences as well as voice characteristics.
Validate on new recordings and, for subspace generalization, unseen speakers.
Compare phonetic and speaker evaluation before and after subspace removal.

### Candidate principles organizing mode importance

- Sonority: stronger modes distinguish groups farther apart in sonority;
  weaker modes distinguish closer groups or phones within those groups.
- Articulatory structure: manner, place, and voicing organize the dominant
  contrasts, potentially with different strengths across these properties.
- Speaker structure: shared differences between speaker groups dominate
  finer differences between individual speakers within groups.
- Temporal structure: current-phone information dominates neighboring-phone
  or longer-context information, or another temporal ordering emerges.
- Prosodic structure: pitch, intensity, stress, and duration account for
  prominent patterns and their relative strengths.
- Recording conditions: channel, noise, or reverberation differences account
  for modes that may be stronger than linguistic distinctions.

These are competing or overlapping hypotheses about organization across the
spectrum. They are not assumed hierarchies or requirements that every mode
receive a single interpretation. Mode importance is defined by the singular
value; candidate principles may explain the resulting ordering of strengths.

For sonority, ask whether stronger modes express contrasts between phones
farther apart on a prespecified sonority ordering, while weaker modes express
closer or within-group distinctions. This differs from testing whether one
mode's phone scores correlate with sonority. A single sonority-aligned mode
does not establish that sonority organizes the rest of the spectrum.

See the [sonority alignment worked example](note_sonority_mode_alignment.md)
for phone scores, pairwise differences, and a rank-correlation calculation,
including the distinction between individual-mode alignment and organization
across the spectrum.

Characterize the contrasts expressed by each mode using its scores across
phones and speakers. Separate pattern shape from amplitude when comparing
interpretations with singular values, so larger scores alone do not determine
both explanatory measurements. Validate on held-out occurrences and compare
candidate explanations jointly where they covary, such as sonority, intensity,
and periodicity. Sonority rankings and groupings should be specified in advance
and checked for sensitivity to reasonable alternatives.

For speaker groups, a gender-associated contrast can be estimated by averaging
phone-balanced speaker means within supplied gender labels, giving speakers
equal weight. Test alignment and transfer on held-out speakers and phones,
including preservation of phonetic distinctions. Interpret this as a
group-associated contrast, potentially combining acoustic, style, and recording
differences; do not assume a universal gender axis or a binary gender system.

Retain occurrence-level data and context/prosody/recording annotations for
interpretations that speaker-phone averaging could suppress. Once modes are
characterized, follow them across checkpoints to test whether stronger final
modes emerge earlier and whether a candidate organizing principle also helps
explain that temporal ordering.

### Follow the same patterns through training

Measure the final-mode score patterns on a fixed set of held-out joint rows.
At earlier checkpoints, test how well the representation expresses those row
patterns using aligned projections or a prespecified linear mapping fitted on
separate data. Fit alignments without using evaluation rows. Distinguish
pattern accessibility under a fitted mapping from geometric alignment itself.

Track both amplitude and pattern agreement. Do not identify a mode by its PCA
index or apply a final direction to unaligned earlier coordinates. Repeated
eigenvalues permit rotations, and components can exchange order. Evaluate
groups of such modes together where individual identification is unstable.

Compare final singular-value rankings with learning curves and prespecified,
baseline-adjusted emergence criteria. Report uncertainty and sensitivity to
thresholds: larger final amplitudes alone can cross an absolute threshold
earlier. Allow concurrent development, reorganization, and modes that become
weaker or disappear before the final checkpoint. A final-model analysis alone
cannot discover those transient modes, so retain checkpoint-wise spectra.

The primary result is whether dominant final patterns emerge earlier, and
whether phonetic, speaker-related, and mixed modes have different trajectories.
Assess which candidate principles explain the ordering of final mode strengths
and whether that organization also characterizes their development.
The phases below specify the shared data, extraction, and validation workflow.

## Questions and hypotheses

1. Which directions account for the most variation at different training stages,
   and what do they encode?
2. Does phonetic discrimination improve through stronger existing directions,
   additional useful dimensions, or changing geometry?
3. How do speaker and phonetic subspaces overlap, and does this overlap change
   during training?
4. Does target refinement change phonetic and speaker spectra differently from
   simply training longer?
5. Can a simple predictive model's learning order be explained by the strength
   of acoustic input–target modes?
6. Do sonority, articulatory, speaker, temporal, prosodic, or recording-related
   principles explain which distinctions dominate the mode spectrum?

Possible outcomes include a broad phonetic distinction emerging before finer
contrasts, or phonetic development occurring mainly in weaker components.
Neither broad-to-fine learning nor monotonically increasing rank should be
assumed. Acoustic variance, frequency, and nuisance variation can all produce
strong components.

## Phase 1: establish a fixed analysis set

Use aligned phone occurrences from multiple speakers, with phone labels, speaker
and utterance identities, boundaries, and neighboring phones. Add duration,
energy, and pitch where reliably measurable. Document missing or inapplicable
acoustic measurements, particularly pitch in unvoiced regions.

Split by utterance before fitting analysis methods. Reserve speakers for a
separate generalization test; ensure that the evaluated phone categories are
represented in both fitting and test sets. Speaker-identity classification
requires a different split with the same speaker classes but different
utterances. Do not confuse that task with generalization of a speaker-removal
transform to unseen speakers.

Use a fixed, reproducible sample across all checkpoints. Balance phone and
speaker coverage where feasible and record missing speaker–phone combinations.
Do not insert zero vectors for missing combinations. For the initial joint
analysis, prefer a well-covered subset of speakers and phones; use matched
weighting or an explicitly modeled incomplete design when expanding coverage.

Balance or stratify relevant comparisons by phonetic context and duration.
Exclude silence and non-speech from the primary phone analysis and analyze them
separately if useful. Retain occurrence-level data even when the main
decomposition uses category means.

## Phase 2: extract comparable representations

Begin with one model family and its existing checkpoints, including
initialization if available. Sample training more densely near rapid changes.
Record both optimizer steps and audio exposure when comparing runs, because
batch sizes can differ.

Feed the same complete utterances to each checkpoint in evaluation mode with a
fixed masking policy. Use model-specific frame timing to map phone boundaries to
representation frames. Cache the extraction configuration, model identifiers,
layer identifiers, and token IDs.

For the primary analysis, average frames within each phone occurrence, giving
one vector per occurrence. Compare middle-third pooling as a sensitivity
analysis, following Huo and Dunbar's phonetic analysis. Define handling of very
short phones before extraction. Retain frame sequences for a conventional
triphone ABX evaluation if that evaluation is included.

Analyze local encoder outputs separately from contextual layers. A phone
interval sliced out of a contextual layer can contain information from
surrounding speech. For an initial manageable run, select the encoder output and
a few predefined Transformer layers, then expand coverage if the results justify
it.

Liu averages frames directly when constructing category means. Averaging
occurrence vectors instead gives equal weight to phone occurrences rather than
to their duration. Report this deliberate difference and compare the two
weighting schemes on a subset.

## Phase 3: decompose complementary matrices

For each checkpoint and layer, construct the following matrices using the
fitting partition:

- Occurrence matrix
  Rows: One vector per phone occurrence
  Main interpretation: Total variation, including variation within phones and
  speakers

- Phone matrix
  Rows: Mean vector per phone category
  Main interpretation: Differences between average phone representations

- Speaker matrix
  Rows: Mean vector per speaker
  Main interpretation: Differences between average speaker representations

- Joint matrix
  Rows: Mean vector per observed speaker–phone combination
  Main interpretation: Combined speaker and phonetic organization

Compute phone means with matched speaker weighting where coverage permits, and
speaker means with matched phone weighting. Otherwise, differences in phone
inventories or speaker mixtures can masquerade as category effects. Use the same
weighting policy at every checkpoint.

Center each matrix using its fitting-set column mean and apply SVD:

`X_centered = U S Vᵀ`

Columns of `V` specify embedding-space directions; columns of `U S` give scores
for the rows. The covariance eigenvalues are `lambda_i = s_i² / (n - 1)` for an
ordinary equally weighted matrix with `n` rows. Use the corresponding weighted
covariance when rows have unequal weights.

Record absolute eigenvalues, their proportions of total variance, total
variance, the number of components needed for 90% and 95% of variance, and
participation ratio:

`participation_ratio = (sum(lambda_i))² / sum(lambda_i²)`

Handle zero total variance explicitly. Report embedding norms alongside these
quantities. Analyze unit-normalized vectors as a separate sensitivity condition;
normalization changes the geometry being measured. Do not silently standardize
every learned coordinate before the primary PCA.

A centered matrix with `n` rows has rank at most `n - 1` . In particular, a
phone-mean matrix cannot reveal more than `number_of_phones - 1` dimensions,
regardless of embedding size. Its spectrum measures between-phone structure, not
the dimensionality of all speech representations. Report row counts and rank
ceilings and avoid treating spectra of differently aggregated matrices as
directly equivalent quantities.

## Phase 4: interpret and follow patterns

Inspect component scores against phone identity and candidate features such as
manner, voicing, place, vowel height/backness, duration, energy, pitch, and
speaker identity. Treat interpretations such as a sonority gradient as
hypotheses to validate on separate data. Because phone labels define the
phone-mean matrix, phonetic organization there is not evidence of fully
label-free discovery.

Evaluate correlations or simple linear probes on held-out occurrences. Use
speaker-held-out evaluation for phonetic interpretations. Fit transformations
and select dimensions on fitting/development data only. Check whether acoustic
measurements explain an apparent phonetic association and whether it persists
within suitable matched subsets.

Within a checkpoint, compare phone and speaker subspaces using principal angles:
for orthonormal bases `Q_phone` and `Q_speaker` , singular values of
`Q_phoneᵀ Q_speaker` are the cosines of the principal angles. Compare against
random subspaces of the same dimensions in the same ambient space;
high-dimensional random directions can already be nearly orthogonal.

Across checkpoints, do not equate components by index. Signs can reverse,
components can exchange order, and similar eigenvalues permit unstable
rotations. Compare component-score patterns or representation similarity on the
same token IDs. If comparing embedding-space bases directly, first estimate an
orthogonal alignment on fitting tokens and evaluate its generalization on
held-out tokens. Use subspace comparisons for groups of near-degenerate
components.

## Phase 5: relate geometry to discrimination

Measure phonetic performance using all dimensions and progressively larger
principal-component subsets, for example 1, 2, 4, 8, and onward within the
available rank. Also examine weaker component groups and complements, so the
analysis can detect useful low-variance directions.

Use a held-out linear phone probe and a discrimination task. A token-level ABX
test asks whether occurrence X is closer to A, sharing its phone category, than
to B, from another category. Match surrounding context where possible and report
within- and across-speaker conditions separately. Label this as a token-level
adaptation. Liu's conventional triphone ABX uses speech sequences and should be
reproduced separately if direct comparability is required.

For a selected subspace with orthonormal columns `Q` , remove its projection
from centered vectors:

`z_removed = z_centered - Q Qᵀ z_centered`

Compare phone-subspace removal, speaker-subspace removal, and repeated
dimension-matched random removals. Account for removed variance when
interpreting differences, and retain a centering-only baseline. Fit a new probe
after removal to test information still accessible in the remaining
representation. Testing a frozen original probe answers a different question
about reliance on the removed directions.

Bootstrap at the speaker or utterance level rather than treating neighboring
frames as independent. Repeat key findings across training seeds where
available. Subspace removal tests the contribution of a representation to an
evaluation; it does not establish the training mechanism that created that
representation.

## Phase 6: compare targets and training iterations

Following [Huo and Dunbar](huo_iterative_refinement_summary.md) , compare first-
and later-iteration HuBERT models and, if available, wav2vec 2.0 or a controlled
contrastive-HuBERT variant. Existing checkpoints can support descriptive
comparisons; claims about the effect of objectives require controlled training
comparisons.

Record target source, clustering configuration, teacher layer, initialization,
and cumulative training exposure including teacher training. HuBERT iterations
start new models: display within-iteration trajectories separately and connect
them through the teacher-to-target relationship, not as one continuous parameter
trajectory.

Test whether refinement strengthens phone-related variance, reduces
speaker-related variance, changes overlap, or relocates these properties across
layers. Include a longer-training comparison where available. Changes in target
granularity, teacher layer, and fresh initialization are possible contributing
factors, not automatically separable consequences of refinement.

## Phase 7: predict learning order in a tractable model

Use fixed acoustic features and a specified context window to predict a future
acoustic frame or window with a small deep linear network trained by squared
error. Vary prediction horizon in separate experiments. Define centering,
scaling, and input whitening explicitly, using training data only.

Construct the input–target cross-covariance `C_yx = E[y xᵀ]` after the specified
preprocessing and decompose it by SVD. In the whitened-input,
small-initialization regime, this provides a setting closely connected to Saxe's
analysis. Compare data-mode strengths with the growth of the model's
input–output mapping along those fixed modes, as well as hidden-representation
geometry and held-out prediction error.

This moves beyond observing PCA structure after training: modes are specified
from the learning problem before their learning trajectories are measured. For
nonlinear speech encoders, a relationship between these acoustic modes and
learning order remains an empirical hypothesis. There is no established
equivalence here between a raw acoustic covariance matrix and the target matrix
of Karkada's word2vec approximation.

## Initial deliverables and scope

The first pass should use one model trajectory, a fixed multi-speaker phone
sample, selected layers, and roughly 10–20 checkpoints if available. Complete
occurrence/phone/speaker/joint spectra, held-out phonetic evaluation with
increasing component counts, subspace overlap, and speaker-removal controls
before expanding to new model training.

Deliver token and checkpoint manifests, extraction settings, fitted
decompositions, held-out metrics, and plots of spectral trajectories, useful
dimension counts, subspace overlap, and discrimination after projection/removal.
Report whether changes indicate scale growth, additional useful directions, or
reorganization, including uncertainty and failures to distinguish these
explanations.

## References

- [Liu, Tang, and Goldwater (2023)](liu_orthogonal_subspaces.pdf) :
  category-mean PCA, speaker/phone subspaces, and speaker removal.
- [Huo and Dunbar (2025)](huo_iterative_refinement.pdf) : controlled comparison
  of loss choice and target refinement.
- [Saxe et al.](saxe_semantic_development.pdf) : data modes and analytical
  learning dynamics in deep linear networks.
- [Karkada et al.](karkada_training_dynamics.pdf) : target-matrix modes and
  effective-rank development in a word2vec approximation.
- [Jing et al.](jing_collapse.pdf) : dimensional collapse and the distinction
  between encoder and projected representations.
- [Valentini et al.](valentini_word_frequency_embedding.pdf) : motivation for
  frequency controls; the direct evidence concerns static word embeddings.
- [de Heer Kloots, Bentum et al. (2026)](https://arxiv.org/abs/2604.02043) :
  linguistic structure across speech-model checkpoints and layers.
- [Pasad, Shi, and Livescu (2023)](https://arxiv.org/abs/2211.03929) :
  canonical-correlation analysis of acoustic and linguistic information.
- [Choi et al. (2026), vector arithmetic](https://arxiv.org/abs/2602.18899):
  reusable phonological feature directions for interpreting learned modes.
- [Choi et al. (2026), phonetic context](https://arxiv.org/abs/2603.12642):
  position-dependent subspaces for interpreting contextual representations.
