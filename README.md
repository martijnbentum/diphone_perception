# diphone_perception
Database of Dutch diphone perception
========================================================

This files describes the naming conventions of files.

--------------------------------------------------------------

Contents:

  1. Material selection
  2. Labeling and production of phonemes
  3. Gating
  4. File naming conventions

--------------------------------------------------------------


1. MATERIAL SELECTION


Diphones were generated automatically by combining the following Dutch
phonemes (CELEX DISC transcription):
	I E A O } @
	i a u y e | o K L M
	p b t d k g N m n l r f v s z S Z j x G h w _


2. LABELING and PRODUCTION OF DIPHONES

Label names were written in CELEX phonetic codes, except the following:
	| -> eu
	_ -> J
	@ -> V
	} -> U

Coding of labels: Extensions indicate stress status of diphone:
	_0 -> segment is unstressed;
	_1 -> segment is stressed;
	_2 -> a syllable boundary is present between 1st and 2nd phoneme of
	      the diphone, with stress on the 2nd syllable;
        _3 -> a syllable boundary is present between 1st and 2nd phoneme of
	      the diphone, with stress on the 1st syllable;
	_s -> segment contains two strong vowels (both stressed);
	_w -> segment contains two weak vowels (both unstressed).

	      Codes 0,1,2, and 3 may have extended coding:
	      a -> the speech segment is produced with prefix 'a'
	      b -> the speech segment is produced with prefix 'b'


3. GATING

DOS does not make a distinction between capitals and non-capitals.
Therefore, this distinction cannot be present in the diphone file names, and
they had to be renamed. The following format was used:

	x1_x2yz.sd  where x1 represents the first phoneme,
			  x2 the second (x1 and x2 may both have length of 2),
			  y  represents the syllabification type of the speech
			     segment in which the phoneme is a part of (see
			     coding of labels above),
			  z  is optional and denotes the presence of an
			     additional prefix (is always /a/ or /b/).

The following phonemes had to be recoded to provide MsDos compatible file names:

OLD   -> NEW
------------
I     -> ih
E     -> eh
A     -> ah
O     -> oh
U (}) -> uh
V (@) -> vh
K     -> ei
L     -> ui
M     -> au
N     -> ng
S     -> sh
Z     -> zh
G     -> gx
J (_) -> dj


4. FILE NAMING CONVENTIONS

a) Original recordings: e.g. a_x2b.sd

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)

	Phonemes are encoded according to CELEX conventions.


b) Gated files: e.g. a_x2b3.wav

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)
	5. gate: 3

	Phonemes are encoded according to CELEX conventions.


c) Label files: a_x2b.sd.lab

	1. phoneme 1 in diphone: a
	2. phoneme 2 in diphone: x
	3. stress condition: 2
	4. leading environment: b (optional)

	Phonemes are encoded according to CELEX conventions.


d1) Response files - rawdata: pp03_gate5

	1. Subject number: pp03
	2. gate: 5


d2) Response files - confusion matrices: phon1_conf_matrix_gate1.dat

	1. indicates if contents belong to first or second phoneme
	   in a diphone: phon1
	2. gate: 1


--------------------------------------------------------------

5. CODE: probing/

Links CGN phone-level metadata to a `phraser` LMDB store, and computes
wav2vec2 hidden-state embeddings for those phones into an `echoframe` store.
Not to be confused with `probing_scripts/`, an older, separate set of
standalone extraction/probing scripts left as-is.

a) metadata.py

Parses `data/metadata.csv` (phone rows) and `data/news_books_sentences_zs.tsv`
(sentence rows) into linked Speaker / Sentence / Phone objects, and matches
every Phone to its corresponding phone segment in a `phraser` LMDB store
(`cgn_lmdb`, default `/vol/mlusers/mbentum/phraser/data/cgn_awd_lmdb`).

	from probing.metadata import Phones

	phones = Phones()
	phones.print_stats()          # phone counts by IPA label

	phones.phraser_phones         # list of matched phraser Phone objects,
	                               # aligned with phones.phones - RAISES if
	                               # any phone failed to match (see below)

Matching is cached to `data/phraser_phone_keys.bin` via
`Phones.save_phraser_keys()` (built automatically on first access to
`phraser_phones` if the cache file doesn't exist yet). Match failures can be
inspected with `phones.analyze_phraser_failures()` after
`save_phraser_keys()`.

`Phones.load_phraser_phones()` loads directly from the original and optional
duplicate-replacement key files without parsing `metadata.csv`. Pass
`validate_against_metadata=True` to additionally validate replacement labels
against the aligned metadata phones.
After loading, `phones.duplicate_replacement_phones` contains the Phraser
phones loaded from the replacement key file, in replacement-file order.

Create the balanced Flemish Dutch key inventory directly from a Phraser
store with:

```python
from probing.select_flemish_phones import save_flemish_phraser_phone_keys

result = save_flemish_phraser_phone_keys(
    store,
    seed=42,
    overwrite=True,
)
```

The selector uses exact `comp-k`/`comp-o` plus `vl` path components, prints
the available count for all 31 labels, and writes 5,000 randomly selected
keys per label to `data/flemish_phraser_phone_keys.bin` only when every label
has enough eligible unique tokens.

Load that binary inventory without parsing the Netherlandic metadata files:

```python
from probing.metadata import FlemishPhones

flemish_phones = FlemishPhones()
flemish_phones.phraser_phones
flemish_phones.flemish_phraser_phones  # alias of the same cached list
```

The loader lazily opens the existing CGN Phraser store and strictly validates
the 31 label-major blocks of 5,000 unique phones.

`phones.phraser_phones` raises `ValueError` if any phone is unmatched, rather
than silently returning a list with holes - downstream consumers (the
embedding extraction below) rely on every phone having a phraser phone.

`_data_dir` (and everything derived from it - `metadata_file`,
`sentence_file`, `phraser_key_file`) resolves to a `data/` directory that is
a sibling of the repo, not inside it (`.../diphone/data`, not
`.../diphone/repo/data`). This is a pre-existing convention, not something
introduced by moving this file into `probing/` - just something to remember
if this file moves again, since the parent-count in `_data_dir` needs to
match its nesting depth under the repo.

b) extract_embeddings.py

One function, `extract_phone_embeddings(phones, ...)`, that computes and
stores wav2vec2 hidden-state embeddings *and* CNN frontend features for
every phone in a Phones instance into an `echoframe.Store`, via
`echoframe.batch_embeddings_cnn_features.compute_embeddings_and_cnn_features_batch`
- every frame overlapping each phone's own span is stored, for each
requested layer, plus one CNN frame per phone.

Routing is per-segment, not per-call: a phone still missing its
hidden_state runs a full forward pass through the model, which incidentally
produces the CNN frontend output as a byproduct, so CNN is stored "for
free" alongside it in the same pass; a phone that already has its
hidden_state but is missing CNN falls back to the cheap CNN-only path
instead of rerunning the full model. This replaces two previously separate
pipelines (hidden states and CNN features extracted independently, via two
full passes over the audio when both were needed) with one call into one
store per model.

	from probing.metadata import Phones
	from probing.extract_embeddings import extract_phone_embeddings

	phones = Phones()
	store = extract_phone_embeddings(phones)   # default model/layer, collar=2000ms

Defaults:

- model_name='wav2vec2_nl1_checkpoint-200000', resolved against
  `data/model_paths.json` and registered in the echoframe store on first use
  (idempotent - safe to call repeatedly).
- layers=[9] - pass a list of hidden-state layer indices to store more than
  one, e.g. layers=[9, 10, 11].
- collar=2000 (ms) - up to 2 seconds of audio context on each side of the
  phone before running the model. Only widens the model's input window; it
  does not change which frames are stored (only frames overlapping the
  phone's own span are kept).
- store_root defaults to `data/echoframe_store`, opened lazily if no
  store= is passed in.
- phraser_source_id='cgn-awd' - label the phones' phraser store is
  registered under in the echoframe store.
- batch_size=120 - compute_embeddings_and_cnn_features_batch only
  auto-computes a batch size when gpu=True; left at None with gpu=False it
  loads every segment's audio into a single batch before running anything.
  The default here avoids that for large phone sets.

Known limitation: no audio-file deduplication.
compute_embeddings_and_cnn_features_batch loads and runs the model on each
phone independently - it does not group phones by source audio file or
reuse hidden states across phones from the same sentence. With a 2000ms
collar, phones spoken close together end up with heavily overlapping
context windows that are recomputed from scratch for each phone. Compute
cost therefore scales with phone count, not sentence count (unlike the old
probing_scripts/extract_embeds.py, which ran each sentence through the
model once and sliced every phone out of that one pass). Not addressed
yet - flagged for awareness if extraction throughput becomes a problem at
scale.

For extracting many training checkpoints,
`extract_phone_embeddings_for_models(...)` opens or creates one store per
registered model under `data/echoframe_model_stores`. Each model is removed
from the store cache and CUDA memory before the next model is loaded:

	from probing.extract_embeddings import (
	    extract_phone_embeddings_for_models, model_store_path)

	store_paths = extract_phone_embeddings_for_models(
	    phones, checkpoint_model_names, layers=[9], gpu=True)

Probe functions can open the matching checkpoint store by passing
`store_root=model_store_path(model_name)`.

Embedding and CNN extraction is incremental at the model / phone / layer
level. `compute_embeddings_and_cnn_features_batch` checks the requested
store before inference, keeps existing phone-layer/CNN outputs, and
computes only missing outputs. Repeating a single-model or multi-model
extraction therefore fills missing embeddings and CNN features without
recomputing outputs already present under the same model, layer, and
collar settings.

The Flemish inventory has a parallel multi-model entry point whose stores
default to `data/echoframe_model_flemish_stores` while retaining the
`cgn-awd` Phraser source ID:

```python
from probing.extract_embeddings import (
    extract_flemish_phone_embeddings_for_models,
)
from probing.metadata import FlemishPhones

flemish_store_paths = extract_flemish_phone_embeddings_for_models(
    FlemishPhones(), checkpoint_model_names, layers=[9], gpu=True)
```

If Flemish embeddings were accidentally written into the model-specific
Netherlandic stores, move them into new Flemish stores using their Phraser
keys:

```python
from scripts.move_embeddings_based_on_phraser_key import move_flemish_data

report = move_flemish_data()
```

The lower-level function accepts an explicit key list and one store pair:

```python
from scripts.move_embeddings_based_on_phraser_key import (
    move_embeddings_based_on_phraser_keys,
)

report = move_embeddings_based_on_phraser_keys(
    flemish_keys, dutch_model_store, new_flemish_model_store)
```

The sweep skips and reports model-specific destinations that already exist
below `data/echoframe_model_flemish_stores/`. For every new destination, it
copies and exactly verifies each matching hidden-state payload before deleting
source metadata, then compacts only affected Dutch shards. Fully Flemish
shards are removed; mixed shards are rewritten with their remaining Dutch
embeddings. Per-store progress and a final aggregate report are printed and
returned. The lower-level one-store function still requires a new destination.

c) train_binary_embedding_probe.py

`train_binary_embedding_probe(phones, target_phoneme, ...)` trains and
evaluates a binary (target-phoneme-vs-other) logistic regression probe on
middle-frame embeddings read back from an echoframe store (written by
`extract_phone_embeddings` above), with 5-fold
`StratifiedKFold(shuffle=True, random_state=42)`.

	from probing.metadata import Phones
	from probing.train_binary_embedding_probe import train_binary_embedding_probe

	phones = Phones()
	result = train_binary_embedding_probe(phones, target_phoneme='p')
	result['mean_accuracy'], result['std_accuracy']

To train one binary embedding probe run for every Phraser phone label:

	from probing.train_binary_embedding_probe import train_binary_embedding_probes

	results = train_binary_embedding_probes(phones, layer=9)

To train probes across the available wav2vec2 checkpoints:

```python
from probing.train_binary_embedding_probe import (
    train_binary_embedding_probe_checkpoint_sweep,
)

report = train_binary_embedding_probe_checkpoint_sweep(
    phones,
    overwrite=False,
)
```

The sweep discovers `wav2vec2_checkpoint-0` and
`wav2vec2_nl1_checkpoint-<step>` stores below
`data/echoframe_model_stores`. It probes layers 1–12 for the random checkpoint
and checkpoint 200000, and layer 9 for other checkpoints.

The sweep runs sequentially in the current process. For each model and layer,
it checks embedding metadata for every phone without loading embedding arrays.
Incomplete inventories are skipped. Store, inventory-check, and training
failures are recorded, and the sweep continues with the next model or layer.
Each model-specific Echoframe store is closed before the next model is opened.

The returned report contains `runs`, `status_counts`, and `errors`. Completed
runs contain compact per-phone metrics, including mean and standard-deviation
accuracy, sample and missing counts, and cache status. The sweep does not write
a consolidated embedding report; persist the returned dictionary explicitly
when needed.

Probe and prediction artifacts retain the run-manifest cache behavior described
below. A repeated sweep reuses complete target-phoneme runs independently for
each model and layer unless `overwrite=True`. The caller owns the `Phones`
object and its Phraser store and should close that store after the complete
workflow.

Sampling: the target phoneme uses every available target-phoneme phone;
every other phoneme class gets an even split of that same count, so the two
binary classes ("target" vs "other") stay balanced overall. Sampling is
deterministic (seeded internally). If any class doesn't have enough phones
to fill its quota, this raises `ValueError` naming the class and the
shortfall, rather than silently training on less data than requested.

Loading (`_load_middle_frame_vectors`): embeddings for the sampled phones
are batch-loaded in one call (`store.phraser_keys_to_embeddings`, grouped
by shard) rather than one call per phone, then reduced to each phone's
middle frame via echoframe's own `Embedding.middle_frame_segment(...)`.
Phones with no stored embedding (e.g. extraction hasn't been run on them
yet) are dropped and counted (`result['n_missing']`), not backfilled with
replacements.

`model_name`, `layer`, and `collar` must match what
`extract_phone_embeddings` wrote for the embeddings to be found.
`save_probes`/`save_predictions` (both default `True`) dump each fold's
fitted probe / per-example predictions under `probe_save_dir`
(`data/phone_probes`) / `results_dir` (`data/probe_results`).

Skip / overwrite / cache reuse: probes and predictions are identified by
their representation-specific parameters (model, target phoneme, layer, and
collar for embeddings) via `probing.result.PhoneResult`. A 500ms run
therefore cannot satisfy a 2000ms request, and changing those parameters
creates a separate run.

A target-phoneme run is reusable only when every fold has a completion marker
and matching probe and prediction checksums. Artifacts are replaced atomically
and each marker is written after its artifacts. If any fold is missing or
invalid, the complete target-phoneme run is retrained; valid folds from that
incomplete run are not reused. Pass `overwrite=True` to retrain even a complete
run. Reuse only runs when both save flags are `True`; otherwise every fold
trains normally.

The plural target trainer and checkpoint sweep still resume at their outer
boundaries. Within each model/layer combination, complete target-phoneme runs
are reused and targets without a complete run are trained again. Every
model/layer combination is reconsidered on the next sweep. An incomplete
embedding inventory is skipped by the probe sweep rather than filled there;
rerun embedding extraction to fill missing phone-layer outputs, then rerun the
sweep.

A true cache hit (all folds already stored, not overwriting) touches
neither the store nor the preloaded embedding matrix at all.

Probe training raises if multiple metadata phones have the same Phraser key.
It does not deduplicate them, because deciding which metadata row is correct
belongs upstream.

d) train_binary_mfcc_probe.py

`train_binary_mfcc_probe(phones, target_phoneme, ...)` provides the same
sampling, cross-validation, saving, and cache behavior for stored phone
MFCCs. Its default store is `data/echoframe_mfcc_store`
(`/vol/mlusers/mbentum/diphone/data/echoframe_mfcc_store` in the cluster
checkout). The primary/default feature is the center frame of the current
39-dimensional MFCC representation: 13 static coefficients, 13 deltas, and
13 delta-deltas.

	from probing.train_binary_mfcc_probe import train_binary_mfcc_probe

	result = train_binary_mfcc_probe(phones, target_phoneme='p')
	result['mean_accuracy'], result['std_accuracy']
	result['results_path']

The MFCC equivalent for every label is:

	from probing.train_binary_mfcc_probe import train_binary_mfcc_probes

	results = train_binary_mfcc_probes(phones)

Each single-target call atomically writes a JSON-safe `results.json` beside
that run's fold predictions. The all-label trainer writes one consolidated
`data/probe_results/mfcc/mfcc_probe_results.json` after every target has
finished. Both files contain fold accuracies and aggregate metrics, but omit
the fitted estimators already stored by the fold cache. Set
`save_results=False` to disable these summary files, or pass `results_path`
to the plural trainer to select the consolidated file location.

Both plural trainers take their default target list from
`phones.label_to_phraser_phone`. Before opening the feature store, they
require every label in that mapping to contain exactly the same number of
items. They open one Echoframe store for the complete sweep, reuse the
single-target cache behavior, and report target progress with elapsed time
and ETA. Pass `target_phonemes=[...]` to train only a selected subset; the
full Phraser label inventory must still be balanced.

Both trainers always fit a plain `LogisticRegression` on raw features; there
is no feature-scaling option.

e) Shared probe utilities

The shared code is divided by responsibility: `probe_utils.py` contains
sampling, sweep progress, and validation; `probe_training.py` contains
cross-validation and classifier fitting; and `probe_run.py` contains run
identity, cache validation, and persistence.
Representation-specific key construction and feature loading stay in their two
trainer modules.

f) Import convention

`probing/` has no `__init__.py` - it's a Python 3 namespace package. From
ipython launched at the repo root (`repo/`), `from probing.metadata import
Phones` and `from probing.extract_embeddings import extract_phone_embeddings`
work as-is, since ipython puts the current directory on sys.path.

--------------------------------------------------------------

6. SYNTHETIC ACOUSTIC PROBES

`synthetic_acoustic_probes/` contains model-independent signal generators,
acoustic checks, and representation-structure metrics based on Choi and Yeo
(2022). It includes the paper's pure-tone, temporal, bias,
sinusoidal-component formant, and amplitude grids, plus a Praat source-filter
monophthong generator.

    from synthetic_acoustic_probes import (
        praat_vowel_stimulus,
        sinusoidal_component_formant_stimuli,
        structure_report,
    )

    paper_stimuli = sinusoidal_component_formant_stimuli()
    vowel = praat_vowel_stimulus(f0_hz=120, f1_hz=500, f2_hz=1500)

`vowel_formant_reference/` exposes separate Dutch literature tables and local
selected-phone measurements. No loader pools sources.

    from vowel_formant_reference import (
        adank_2004_formants,
        pols_1973_formants,
        van_nierop_1973_formants,
        weenink_1985_formants,
    )

    pols = pols_1973_formants()
    female = van_nierop_1973_formants()
    adult_anchors = weenink_1985_formants()
    northern = adank_2004_formants(
        population='Northern Standard Dutch',
    )

The checked-in literature artifacts and citation sidecars live under
`vowel_formant_reference/formants/`. The default gender anchors contain F0 and
F1--F3 for 12 full Dutch monophthongs from Weenink's adult groups; no
literature table supplies schwa. See `vowel_formant_reference/literature.md`
for references, URLs, source pages, and table numbers. Tabular artifacts are
CSV files with JSON metadata, and loader results expose their rows as
`list[dict]`. Local phone extraction requires access to the configured
Phraser/CGN audio store:

    from probing.metadata import Phones
    from vowel_formant_reference import measure_and_write_phone_formants

    phones = Phones()
    measure_and_write_phone_formants(phones.phraser_phones)

Only non-overlapping monophthongs are selected; stress is not inspected. The
function prints selected-vowel counts and written paths. Token rows retain only
the Phraser phone key, measurement gender, and acoustic results; group anchors
use the median of per-speaker medians.
