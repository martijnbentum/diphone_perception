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
stores wav2vec2 hidden-state embeddings for every phone in a Phones instance
into an `echoframe.Store`, via a vanilla (unmodified)
`echoframe.batch_segment_features.compute_embeddings_batch` call - every
frame overlapping each phone's own span is stored, for each requested layer.

	from probing.metadata import Phones
	from probing.extract_embeddings import extract_phone_embeddings

	phones = Phones()
	store = extract_phone_embeddings(phones)   # default model, layers=[9], collar=500ms

Defaults:

- model_name='wav2vec2_nl1_checkpoint-200000', resolved against
  `data/model_paths.json` and registered in the echoframe store on first use
  (idempotent - safe to call repeatedly).
- layers=[9] - pass a list of hidden-state layer indices to store more than
  one, e.g. layers=[9, 10, 11].
- collar=500 (ms) - audio context padded around each phone before running
  the model. Only widens the model's input window; does not change what
  gets stored (only frames overlapping the phone's own span are kept).
- store_root defaults to `data/echoframe_store`, opened lazily if no
  store= is passed in.
- phraser_source_id='cgn-awd' - label the phones' phraser store is
  registered under in the echoframe store.
- batch_size=32 - compute_embeddings_batch only auto-computes a batch size
  when gpu=True; left at None with gpu=False it loads every segment's audio
  into a single batch before running anything. The default here avoids that
  for large phone sets.

Known limitation: no audio-file deduplication. compute_embeddings_batch
loads and runs the model on each phone independently - it does not group
phones by source audio file or reuse hidden states across phones from the
same sentence. With a 500ms collar, phones spoken close together end up
with heavily overlapping context windows that are recomputed from scratch
for each phone. Compute cost therefore scales with phone count, not
sentence count (unlike the old probing_scripts/extract_embeds.py, which ran
each sentence through the model once and sliced every phone out of that one
pass). Not addressed yet - flagged for awareness if extraction throughput
becomes a problem at scale.

c) train_binary_probe.py

One function, `train_binary_probe(phones, target_phoneme, ...)`, that trains
and evaluates a binary (target-phoneme-vs-other) logistic regression probe
on middle-frame embeddings read back from an echoframe store (written by
extract_phone_embeddings above), with 5-fold
`StratifiedKFold(shuffle=True, random_state=42)`.

	from probing.metadata import Phones
	from probing.train_binary_probe import train_binary_probe

	phones = Phones()
	result = train_binary_probe(phones, target_phoneme='p')
	result['mean_accuracy'], result['std_accuracy']

Sampling (`_select_phones`): the target phoneme gets `n_embeds` phones
(default `None` - every available target-phoneme phone, not an arbitrary
cap); every other phoneme class gets `n_embeds // (number of other phoneme
classes)`, so the two binary classes ("target" vs "other") stay balanced
overall. Sampling is deterministic (seeded by `random_state`). If any class
doesn't have enough phones to fill its quota, this raises `ValueError`
naming the class and the shortfall, rather than silently training on less
data than requested.

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

Skip / overwrite / gap-filling: before (re)training a fold, its probe and
predictions files are checked for on disk. A fold is only treated as
"already done" if **both** files are present (an orphaned single file - a
probe with no matching predictions, or vice versa - is not trusted, and
that fold is retrained, regenerating both files together so they can never
belong to different runs). If every fold for a (`model_name`,
`target_phoneme`, `layer`) combination is already complete, the whole call
is skipped - embeddings aren't even loaded - and the saved probes and
accuracies (read back from the predictions files) are returned as-is. Pass
`overwrite=True` to force every fold to (re)train regardless of what's on
disk. Because `StratifiedKFold(shuffle=True, random_state=random_state)`
produces the same fold splits every time for the same data, a partially
complete set (e.g. fold 3 saved, the rest missing) safely retrains only the
missing folds and reuses the saved one - `result['accuracies']` comes back
complete either way, never with gaps. This check only runs when both
`save_probes` and `save_predictions` are `True` and `overwrite` is `False`;
otherwise every fold always (re)trains.

The returned dict's `probes`/`accuracies` are always in memory regardless
of the save settings; `result['skipped']` is `True` only when the whole
call was served from disk without loading any embeddings.

d) Import convention

`probing/` has no `__init__.py` - it's a Python 3 namespace package. From
ipython launched at the repo root (`repo/`), `from probing.metadata import
Phones` and `from probing.extract_embeddings import extract_phone_embeddings`
work as-is, since ipython puts the current directory on sys.path.

