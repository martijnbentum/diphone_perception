import bisect
import csv
import warnings
from collections import Counter
from pathlib import Path

from phone_mapper import cgn
from phraser import SEGMENT_KEY_LENGTH, Store
from progressbar import progressbar

import locations
from probing import select_flemish_phones

_boundary_tokens = ('SOS', 'EOS')
_phraser_phones_per_label = 13_500
_phraser_label_count = 31


# Primary inventory interfaces

class Phones:
    '''all Phone objects linked to a phraser store.'''

    def __init__(self, store=None, path=locations.metadata_file,
        sentence_path=locations.sentence_file,
        phraser_key_path=locations.phraser_key_file,
        duplicate_replacement_phraser_key_path=(
            locations.duplicate_replacement_phraser_key_file),
    ):
        self._store = store
        self.path = path
        self.sentence_path = sentence_path
        self.phraser_key_path = phraser_key_path
        self.duplicate_replacement_phraser_key_path = (
            duplicate_replacement_phraser_key_path)
        self.duplicate_replacement_phones = []

    @property
    def store(self):
        if self._store is None:
            self._store = load_cgn()
        return self._store

    @property
    def phones(self):
        if hasattr(self, '_phones'):
            return self._phones
        phones = load_phones(self.path, self.sentence_path)
        for phone in phones:
            phone.parent = self
        self._phones = phones
        return self._phones

    @property
    def phraser_phones(self):
        '''phraser Phone objects aligned with self.phones. Uses the cached
        key file at self.phraser_key_path if present, building it on first
        use.

        Raises ValueError if any phone failed to match - callers (e.g. the
        echoframe embedding step) rely on every phone having a phraser
        phone, so a hole here should surface loudly rather than silently
        drop phones. Run save_phraser_keys() then analyze_phraser_failures()
        to diagnose.
        '''
        if hasattr(self, '_phraser_phones'):
            return self._phraser_phones
        if not Path(self.phraser_key_path).exists():
            self.save_phraser_keys()
        phraser_phones = self.load_phraser_phones()
        missing = sum(1 for p in phraser_phones if p is None)
        if missing:
            raise ValueError(
                f'{missing} / {len(phraser_phones)} phones have no matched '
                'phraser phone - run save_phraser_keys() and '
                'analyze_phraser_failures() to diagnose'
            )
        self._phraser_phones = phraser_phones
        return self._phraser_phones

    @property
    def phoneme_counts(self):
        if hasattr(self, '_phoneme_counts'):
            return self._phoneme_counts
        self._phoneme_counts = Counter(
            phone.phoneme_ipa for phone in self.phones)
        return self._phoneme_counts

    @property
    def label_to_phraser_phone(self):
        '''dict mapping each phraser phone label to the list of matched
        phraser phones with that label (unmatched phones are skipped).
        '''
        if hasattr(self, '_label_to_phraser_phone'):
            return self._label_to_phraser_phone
        grouped = {}
        for phraser_phone in self.phraser_phones:
            if phraser_phone is None:
                continue
            grouped.setdefault(phraser_phone.label, []).append(phraser_phone)
        self._label_to_phraser_phone = grouped
        return self._label_to_phraser_phone

    @property
    def audio_index(self):
        if hasattr(self, '_audio_index'):
            return self._audio_index
        self._audio_index = build_audio_index(self.store)
        return self._audio_index

    def print_stats(self):
        counts = self.phoneme_counts
        total = sum(counts.values())
        print(f'{total} phones, {len(counts)} phone types')
        for phoneme_ipa, count in counts.most_common():
            print(f'  {phoneme_ipa:<4} {count}')

    def save_phraser_keys(self, path=None, tolerance_ms=25):
        '''match every phone to phraser and save its key to path.

        keys are stored as fixed 22-byte records, one per phone, in the
        same order as self.phones. a phone that could not be matched gets
        a zero-byte placeholder instead, so later indices still line up
        with self.phones. phones that failed to match, along with the
        error raised for each, are stored on self.phraser_match_failures
        (a list of (phone, error) pairs) rather than returned - see
        analyze_phraser_failures().

        uses self.audio_index (one bulk load of every Audio) instead of
        looking up the audio per phone, which would otherwise reload every
        Audio in the store on every single phone.
        '''
        path = path or self.phraser_key_path
        audio_index = self.audio_index
        failures = []
        keys = bytearray()
        placeholder = bytes(SEGMENT_KEY_LENGTH)
        for phone in progressbar(self.phones):
            try:
                phraser_phone = phone.phraser_phone(
                    self.store, tolerance_ms=tolerance_ms,
                    audio_index=audio_index)
                keys += phraser_phone.key
            except ValueError as error:
                failures.append((phone, error))
                keys += placeholder
        with open(path, 'wb') as f:
            f.write(keys)
        self.phraser_match_failures = failures

    def load_phraser_phones(
        self, path=None, validate_against_metadata=False,
    ):
        '''bulk-load the phraser Phone objects saved by save_phraser_keys.

        returns a list in key-file order (None where a phone wasn't matched
        during save_phraser_keys). If an applicable duplicate replacement key
        file exists, repeated keys are replaced in place before loading so
        positional alignment is retained. By default only the key files and
        Phraser store are used. Set validate_against_metadata=True to also
        load metadata phones and validate replacement labels against them.
        '''
        path = path or self.phraser_key_path
        original_keys = load_phraser_keys(path)
        keys = original_keys
        replacement_indices = []
        replacement_path = self._applicable_replacement_path(path)
        if replacement_path is not None:
            replacement_keys = load_phraser_keys(replacement_path)
            keys, replacement_indices = _replace_duplicate_phraser_keys(
                keys, replacement_keys)
        real_keys = [key for key in keys if key is not None]
        loaded = iter(self.store.load_many(real_keys))
        phraser_phones = [
            next(loaded) if key is not None else None for key in keys]
        self._validate_replacement_labels_against_original(
            original_keys, phraser_phones, replacement_indices)
        expected_labels = None
        if validate_against_metadata:
            self._validate_replacement_labels_against_metadata(
                phraser_phones, replacement_indices)
            expected_labels = {
                phone.phoneme_ipa for phone in self.phones}
        self.duplicate_replacement_phones = [
            phraser_phones[index] for index in replacement_indices]
        self._warn_phraser_inventory(
            phraser_phones, expected_labels=expected_labels)
        return phraser_phones

    def analyze_phraser_failures(self):
        '''summarize self.phraser_match_failures: counts by error type,
        phoneme label, overlap, comp, position within the sentence
        (first/last/interior phone), and the label of the phraser phone
        closest in time (any label, not just the expected one) - useful to
        tell "right phone, just outside the tolerance window" from "wrong
        phone entirely" failures. also breaks the closest-label counts
        down per expected label, to spot systematic label confusions.
        prints the summary and returns it as a dict. raises if
        save_phraser_keys has not been run yet.
        '''
        if not hasattr(self, 'phraser_match_failures'):
            raise ValueError(
                'no phraser_match_failures - run save_phraser_keys first')

        failures = self.phraser_match_failures
        total_phones = len(self.phones)
        by_type = Counter(type(error).__name__ for _, error in failures)
        by_label = Counter(phone.phoneme_ipa for phone, _ in failures)
        by_overlap = Counter(phone.overlap for phone, _ in failures)
        by_comp = Counter(phone.comp for phone, _ in failures)
        by_sentence_edge = Counter(
            _sentence_edge_position(phone) for phone, _ in failures)
        closest_labels = [
            self._closest_phraser_label(phone)
            for phone, _ in progressbar(failures)
        ]
        by_closest_label = Counter(closest_labels)
        by_label_closest_label = {}
        for (phone, _), closest in zip(failures, closest_labels):
            by_label_closest_label.setdefault(
                phone.phoneme_ipa, Counter())[closest] += 1
        closest_matches_expected = sum(
            phone.phoneme_ipa == closest
            for (phone, _), closest in zip(failures, closest_labels)
        )
        stats = {
            'total_failures': len(failures),
            'total_phones': total_phones,
            'by_type': by_type,
            'by_label': by_label,
            'by_overlap': by_overlap,
            'by_comp': by_comp,
            'by_sentence_edge': by_sentence_edge,
            'by_closest_label': by_closest_label,
            'by_label_closest_label': by_label_closest_label,
            'closest_matches_expected': closest_matches_expected,
        }

        rate = 100 * stats['total_failures'] / total_phones if total_phones else 0
        print(
            f"{stats['total_failures']} / {total_phones} phones failed "
            f"to match ({rate:.1f}%)"
        )
        print('by error type:')
        for name, count in by_type.most_common():
            print(f'  {name:<20} {count}')
        print('by phoneme label:')
        for label, count in by_label.most_common():
            print(f'  {label:<4} {count}')
        print('by overlap:')
        for overlap, count in by_overlap.most_common():
            print(f'  {overlap!s:<6} {count}')
        print('by comp:')
        for comp, count in by_comp.most_common():
            print(f'  {comp:<4} {count}')
        print('by sentence position:')
        for position, count in by_sentence_edge.most_common():
            print(f'  {position:<10} {count}')
        print(
            'closest phone in time has the expected label: '
            f'{closest_matches_expected} / {len(failures)}'
        )
        print('by closest-in-time phraser label:')
        for label, count in by_closest_label.most_common(10):
            print(f'  {label!s:<4} {count}')
        print('by label -> closest-in-time label:')
        for label, _ in by_label.most_common():
            breakdown = ', '.join(
                f'{closest!s}={count}'
                for closest, count in by_label_closest_label[label].most_common(5)
            )
            print(f'  {label:<4} {breakdown}')

        return stats

    def _applicable_replacement_path(self, phraser_key_path):
        replacement_path = self.duplicate_replacement_phraser_key_path
        if replacement_path is None:
            warnings.warn(
                'duplicate replacement Phraser key file is disabled; loading '
                'the original keys without replacements',
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        uses_default_keys = (
            Path(phraser_key_path).resolve()
            == locations.phraser_key_file.resolve())
        uses_default_replacements = (
            Path(replacement_path).resolve()
            == locations.duplicate_replacement_phraser_key_file.resolve()
        )
        if not uses_default_keys and uses_default_replacements:
            warnings.warn(
                'not applying the default duplicate replacement Phraser key '
                'file: it was generated from the duplicate history of the '
                'default phraser_phone_keys.bin and is not valid for a custom '
                'phraser_key_path',
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        path = Path(replacement_path)
        if path.exists():
            return path
        warnings.warn(
            f'duplicate replacement Phraser key file is not available: '
            f'{path}; loading the original keys without replacements',
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    def _validate_replacement_labels_against_original(
        self, original_keys, phraser_phones, indices,
    ):
        if not indices:
            return
        duplicate_keys = {original_keys[index] for index in indices}
        first_indices = {}
        for index, key in enumerate(original_keys):
            if key in duplicate_keys and key not in first_indices:
                first_indices[key] = index

        for index in indices:
            original_index = first_indices[original_keys[index]]
            expected = phraser_phones[original_index].label
            observed = phraser_phones[index].label
            if observed != expected:
                raise ValueError(
                    f'duplicate replacement at index {index} has Phraser '
                    f'label {observed!r}, expected original label '
                    f'{expected!r}')

    def _validate_replacement_labels_against_metadata(
        self, phraser_phones, indices,
    ):
        for index in indices:
            expected = self.phones[index].phoneme_ipa
            observed = phraser_phones[index].label
            if observed != expected:
                raise ValueError(
                    f'duplicate replacement at index {index} has Phraser '
                    f'label {observed!r}, expected {expected!r}')

    def _warn_phraser_inventory(self, phraser_phones, expected_labels=None):
        keys = [bytes(phone.key) for phone in phraser_phones if phone is not None]
        duplicate_count = len(keys) - len(set(keys))
        if duplicate_count:
            warnings.warn(
                f'loaded Phraser phones contain {duplicate_count} duplicate '
                'key occurrences; generate and load the duplicate replacement '
                'Phraser key file to obtain a unique inventory',
                RuntimeWarning,
                stacklevel=3,
            )

        keys_by_label = {}
        for phone in phraser_phones:
            if phone is None:
                continue
            keys_by_label.setdefault(phone.label, set()).add(bytes(phone.key))

        if expected_labels is None:
            observed_label_count = len(keys_by_label)
            if observed_label_count != _phraser_label_count:
                warnings.warn(
                    f'loaded Phraser phone inventory contains '
                    f'{observed_label_count} distinct labels, expected '
                    f'{_phraser_label_count}',
                    RuntimeWarning,
                    stacklevel=3,
                )
            expected_labels = set(keys_by_label)
        counts = {
            label: len(keys_by_label.get(label, set()))
            for label in expected_labels.union(keys_by_label)
        }
        invalid = {
            label: count for label, count in counts.items()
            if count != _phraser_phones_per_label
        }
        if invalid:
            details = ', '.join(
                f'{label!r}={count}' for label, count in sorted(invalid.items()))
            warnings.warn(
                'Phraser phone inventory does not contain exactly '
                f'{_phraser_phones_per_label:,} unique keys per label: '
                f'{details}',
                RuntimeWarning,
                stacklevel=3,
            )

    def _closest_phraser_label(self, phone):
        '''label of the phraser phone closest in time to `phone` (by start
        time) within the same audio, regardless of label. None if that
        audio has no phones at all.

        uses bisect on a start-sorted copy of the audio's phones instead
        of a full linear scan (min()), since this runs once per failure
        and a recording can have hundreds to thousands of phones.
        '''
        audio = self.audio_index[phone.audio_filename_id]
        phones = _audio_phones_sorted(audio)
        if not phones:
            return None
        index = bisect.bisect_left(phones, phone.start, key=lambda p: p.start)
        neighbors = phones[max(index - 1, 0):index + 1]
        closest = min(neighbors, key=lambda p: abs(p.start - phone.start))
        return closest.label


class FlemishPhones:
    '''Metadata-free owner of the selected Flemish Phraser phones.'''

    def __init__(
        self, store=None,
        phraser_key_path=locations.flemish_phraser_phone_key_file,
    ):
        self._store = store
        self.phraser_key_path = phraser_key_path

    @property
    def store(self):
        if self._store is None:
            self._store = load_cgn()
        return self._store

    @property
    def phraser_phones(self):
        if not hasattr(self, '_phraser_phones'):
            self._phraser_phones = self.load_phraser_phones()
        return self._phraser_phones

    @property
    def flemish_phraser_phones(self):
        return self.phraser_phones

    def load_phraser_phones(self, path=None):
        '''Load and validate the label-major Flemish Phraser inventory.'''
        path = path or self.phraser_key_path
        keys = load_phraser_keys(path)
        _validate_flemish_phraser_keys(keys)
        try:
            phraser_phones = list(self.store.load_many(keys))
        except KeyError as error:
            raise ValueError(
                'Flemish Phraser key file refers to a missing object'
            ) from error
        if len(phraser_phones) != select_flemish_phones.flemish_phone_count:
            raise ValueError(
                f'loaded {len(phraser_phones):,} Flemish Phraser phones, '
                f'expected {select_flemish_phones.flemish_phone_count:,}')
        missing = sum(phone is None for phone in phraser_phones)
        if missing:
            raise ValueError(
                f'{missing:,} Flemish Phraser keys have no stored object')
        missing_labels = sum(
            not hasattr(phone, 'label') for phone in phraser_phones)
        if missing_labels:
            raise ValueError(
                f'{missing_labels:,} Flemish Phraser phones have no label')
        mismatched_keys = sum(
            not hasattr(phone, 'key') or phone.key != key
            for key, phone in zip(keys, phraser_phones, strict=True)
        )
        if mismatched_keys:
            raise ValueError(
                f'{mismatched_keys:,} loaded Flemish Phraser phones do not '
                'match their requested keys')
        _validate_flemish_phraser_labels(phraser_phones)
        return phraser_phones


# Metadata row models

class Phone:
    '''one row from metadata.csv, linked to its Sentence and Speaker'''
    def __init__(self, row, sentence_by_identifier, speakers, parent=None):
        self.parent = parent
        self.audio_filename = row['audio_filename']
        self.audio_filename_id = self.audio_filename.split('_')[0]

        self.start_seconds_from_sentence = float(row['start_time'])
        self.end_seconds_from_sentence = float(row['end_time'])
        self.duration_seconds = float(row['duration'])
        self.start_from_sentence = round(self.start_seconds_from_sentence * 1000)
        self.end_from_sentence = round(self.end_seconds_from_sentence * 1000)
        self.duration = round(self.duration_seconds * 1000)

        self.phoneme_sampa = row['phoneme']
        self.previous_phoneme_sampa = row['previous_phoneme']
        self.next_phoneme_sampa = row['next_phoneme']
        self.phoneme_ipa = _sampa_to_ipa(self.phoneme_sampa)
        self.previous_phoneme_ipa = _sampa_to_ipa(self.previous_phoneme_sampa)
        self.next_phoneme_ipa = _sampa_to_ipa(self.next_phoneme_sampa)

        self.speaker_id = row['speaker_id']
        self.overlap = {'True': True, 'False': False}[row['overlap']]
        self.comp = row['comp']
        self.ipa_phoneme = row['ipa_phoneme']

        if self.phoneme_ipa != self.ipa_phoneme:
            raise ValueError(
                'phoneme_ipa mismatch: mapped '
                f'{self.phoneme_sampa!r} -> {self.phoneme_ipa!r}, '
                f'expected {self.ipa_phoneme!r} '
                f'(audio_filename={self.audio_filename})'
            )

        if self.audio_filename not in sentence_by_identifier:
            raise ValueError(
                f'no sentence found for identifier {self.audio_filename!r}'
            )
        self.sentence = sentence_by_identifier[self.audio_filename]
        self.speaker = _get_speaker(self.speaker_id, speakers)

        if self.speaker not in self.sentence.speakers:
            raise ValueError(
                f'speaker {self.speaker.speaker_id!r} not among sentence '
                f'speakers {self.sentence.speaker_ids!r} '
                f'(audio_filename={self.audio_filename})'
            )

    @property
    def start(self):
        return self.start_from_sentence + self.sentence.start

    @property
    def end(self):
        return self.end_from_sentence + self.sentence.start

    @property
    def start_seconds(self):
        return round(self.start_seconds_from_sentence + self.sentence.start_seconds, 3)

    @property
    def end_seconds(self):
        return round(self.end_seconds_from_sentence + self.sentence.start_seconds, 3)

    def phraser_phone(self, store=None, tolerance_ms=25, audio_index=None):
        if hasattr(self, '_phraser_phone'):
            return self._phraser_phone
        if store is None:
            if self.parent is None:
                raise ValueError(
                    'no store given and no parent set on this Phone')
            store = self.parent.store
        self._phraser_phone = get_phraser_phone(
            store, self, tolerance_ms=tolerance_ms, audio_index=audio_index)
        return self._phraser_phone


class Sentence:
    '''one row from news_sentences_zs.tsv'''
    def __init__(self, row, speakers):
        self.audio_filename = row['audio_filename']

        self.start_seconds = float(row['start_time'])
        self.end_seconds = float(row['end_time'])
        self.duration_seconds = float(row['duration'])
        self.start = round(self.start_seconds * 1000)
        self.end = round(self.end_seconds * 1000)
        self.duration = round(self.duration_seconds * 1000)

        self.text = row['text']
        self.identifier = row['identifier']

        self.speaker_ids = row['speaker_ids']
        self.speakers = [
            _get_speaker(speaker_id, speakers)
            for speaker_id in self.speaker_ids.split(',')
        ]

        self.comp = row['comp']


class Speaker:
    '''a speaker parsed from a speaker_id like N02003_male_38'''
    def __init__(self, speaker_id):
        self.speaker_id = speaker_id
        parts = speaker_id.split('_')
        self.id = parts[0]
        self.gender = parts[1] if len(parts) > 1 else None
        self.age = int(parts[2]) if len(parts) > 2 else None


# Public loading and matching functions

def load_cgn(path=locations.cgn_lmdb):
    return Store(path=path)


def load_phones(
    path=locations.metadata_file,
    sentence_path=locations.sentence_file,
):
    sentences, speakers = load_sentences(sentence_path)
    sentence_by_identifier = {s.identifier: s for s in sentences}
    with open(path, newline='') as f:
        return [
            Phone(row, sentence_by_identifier, speakers)
            for row in progressbar(csv.DictReader(f))
        ]


def load_sentences(path=locations.sentence_file):
    '''load sentences and the shared speaker registry built along the way.

    returns (sentences, speakers), where speakers maps speaker_id ->
    Speaker, one object per id, reused across all sentences.
    '''
    speakers = {}
    with open(path, newline='') as f:
        sentences = [
            Sentence(row, speakers)
            for row in progressbar(csv.DictReader(f, delimiter='\t'))
        ]
    return sentences, speakers


def load_phraser_keys(path=locations.phraser_key_file):
    '''load phraser phone keys saved by Phones.save_phraser_keys.

    returns a list aligned with the Phones.phones order used when saving;
    a phone that could not be matched at save time is None.
    '''
    data = Path(path).read_bytes()
    if len(data) % SEGMENT_KEY_LENGTH:
        raise ValueError(
            f'{path} size is not a multiple of {SEGMENT_KEY_LENGTH} bytes')
    keys = [
        data[i:i + SEGMENT_KEY_LENGTH]
        for i in range(0, len(data), SEGMENT_KEY_LENGTH)
    ]
    placeholder = bytes(SEGMENT_KEY_LENGTH)
    return [None if key == placeholder else key for key in keys]


def build_audio_index(store):
    '''map audio_filename_id (the filename stem) -> Audio, for every Audio
    in the store. Load once and reuse across many get_phraser_phone calls
    instead of paying store.audios.get()'s full-table scan per call.
    '''
    index = {}
    for audio in store.audios.all():
        stem = Path(audio.filename).stem
        if stem in index and index[stem].filename != audio.filename:
            raise ValueError(
                f'duplicate audio stem {stem!r}: '
                f'{index[stem].filename!r} and {audio.filename!r}'
            )
        index[stem] = audio
    return index


def get_phraser_phone(store, phone_object, tolerance_ms=25, audio_index=None):
    '''find the phraser Phone corresponding to a metadata Phone.

    matches by audio, ipa label, and a start/end tolerance window (forced
    alignment timing may not match exactly between the two pipelines). if
    more than one candidate remains, disambiguates using the labels of the
    neighboring phones in audio.phones.

    audio_index, if given (see build_audio_index), is used instead of
    store.audios.get() to look up the audio - much faster over many calls,
    since store.audios.get() reloads every Audio in the store on each call.
    '''
    if audio_index is not None:
        audio = audio_index[phone_object.audio_filename_id]
    else:
        audio = store.audios.get(
            filename__contains=phone_object.audio_filename_id)

    same_label = _audio_label_index(audio).get(phone_object.phoneme_ipa, [])
    candidates = [
        p for p in same_label
        if p.start > phone_object.start - tolerance_ms
        and p.end < phone_object.end + tolerance_ms
    ]

    if not candidates:
        raise NoCandidateError(
            f'no phraser phone found for {phone_object.audio_filename!r} '
            f'label={phone_object.phoneme_ipa!r} '
            f'start={phone_object.start} end={phone_object.end}'
        )
    if len(candidates) == 1:
        return candidates[0]

    phones = _audio_phones(audio)
    refined = [
        candidate for candidate in candidates
        if _phraser_siblings_match(phones, candidate, phone_object)
    ]
    if len(refined) == 1:
        return refined[0]

    raise AmbiguousMatchError(
        f'ambiguous phraser phone match for {phone_object.audio_filename!r} '
        f'label={phone_object.phoneme_ipa!r}: {len(candidates)} candidates '
        f'in window, {len(refined)} after neighbor-label check'
    )


# Matching errors

class NoCandidateError(ValueError):
    '''raised when no phraser phone falls within the tolerance window.'''


class AmbiguousMatchError(ValueError):
    '''raised when more than one phraser phone remains after matching.'''


# Internal helpers

def _sampa_to_ipa(sampa):
    if sampa in _boundary_tokens:
        return sampa
    return cgn.cgn_to_ipa.get(sampa)


def _get_speaker(speaker_id, speakers):
    if speaker_id not in speakers:
        speakers[speaker_id] = Speaker(speaker_id)
    return speakers[speaker_id]


def _phraser_neighbor_labels(phones, candidate):
    index = phones.index(candidate)
    prev_label = phones[index - 1].label if index > 0 else None
    next_label = phones[index + 1].label if index < len(phones) - 1 else None
    return prev_label, next_label


def _phraser_siblings_match(phones, candidate, phone_object):
    prev_label, next_label = _phraser_neighbor_labels(phones, candidate)
    # SOS/EOS mark a sentence edge, not a real phoneme, and audio.phones
    # is only guaranteed time-ordered within a phrase, not across phrases:
    # skip the side rather than risk comparing across a phrase boundary.
    prev_ok = (
        phone_object.previous_phoneme_ipa in _boundary_tokens
        or prev_label == phone_object.previous_phoneme_ipa
    )
    next_ok = (
        phone_object.next_phoneme_ipa in _boundary_tokens
        or next_label == phone_object.next_phoneme_ipa
    )
    return prev_ok and next_ok


def _sentence_edge_position(phone):
    '''classify a phone as first/last/both/interior within its sentence.'''
    is_first = phone.previous_phoneme_ipa == 'SOS'
    is_last = phone.next_phoneme_ipa == 'EOS'
    if is_first and is_last:
        return 'both'
    if is_first:
        return 'first'
    if is_last:
        return 'last'
    return 'interior'


def _audio_phones(audio):
    '''Audio.phones rebuilds the whole phrase/word/syllable/phone tree on
    every access (unlike Audio.phrases, it is not cached by phraser) -
    cache it here so repeated lookups against the same audio don't repeat
    that walk.
    '''
    cached = getattr(audio, '_metadata_phones', None)
    if cached is None:
        cached = audio.phones
        audio._metadata_phones = cached
    return cached


def _audio_phones_sorted(audio):
    '''audio's phones sorted by start time. audio.phones is only ordered
    within each phrase, not necessarily across phrases (see
    _phraser_siblings_match), so this sorts explicitly rather than relying
    on tree-walk order - needed for the bisect search in
    Phones._closest_phraser_label.
    '''
    cached = getattr(audio, '_metadata_phones_sorted', None)
    if cached is None:
        cached = sorted(_audio_phones(audio), key=lambda p: p.start)
        audio._metadata_phones_sorted = cached
    return cached


def _audio_label_index(audio):
    '''label -> list of phones with that label, for this audio. Built once
    per audio (cached on it) so matching a phone only scans the phones
    that already share its label, instead of every phone in the recording.
    '''
    cached = getattr(audio, '_metadata_label_index', None)
    if cached is None:
        cached = {}
        for p in _audio_phones(audio):
            cached.setdefault(p.label, []).append(p)
        audio._metadata_label_index = cached
    return cached


def _replace_duplicate_phraser_keys(keys, replacement_keys):
    '''replace repeated real keys in place and return keys and changed indices.'''
    duplicate_indices = []
    seen = set()
    for index, key in enumerate(keys):
        if key is None:
            continue
        if key in seen:
            duplicate_indices.append(index)
        else:
            seen.add(key)

    if len(replacement_keys) != len(duplicate_indices):
        raise ValueError(
            f'expected {len(duplicate_indices)} duplicate replacement keys, '
            f'found {len(replacement_keys)}')
    if any(key is None for key in replacement_keys):
        raise ValueError('duplicate replacement keys cannot contain placeholders')

    replacements = set(replacement_keys)
    if len(replacements) != len(replacement_keys):
        raise ValueError('duplicate replacement key file contains repeated keys')
    reused = seen.intersection(replacements)
    if reused:
        raise ValueError(
            f'{len(reused)} duplicate replacement keys already occur in the '
            'original Phraser key file')

    output = list(keys)
    for index, replacement in zip(
        duplicate_indices, replacement_keys, strict=True,
    ):
        output[index] = replacement
    real_keys = [key for key in output if key is not None]
    if len(real_keys) != len(set(real_keys)):
        raise ValueError('duplicate Phraser keys remain after replacement')
    return output, duplicate_indices


def _validate_flemish_phraser_keys(keys):
    if len(keys) != select_flemish_phones.flemish_phone_count:
        raise ValueError(
            f'Flemish Phraser key file contains {len(keys):,} records, '
            f'expected {select_flemish_phones.flemish_phone_count:,}')
    if any(key is None for key in keys):
        raise ValueError('Flemish Phraser key file contains placeholders')
    if len(set(keys)) != len(keys):
        raise ValueError('Flemish Phraser key file contains duplicate keys')


def _validate_flemish_phraser_labels(phraser_phones):
    labels = [phone.label for phone in phraser_phones]
    for index, label in enumerate(labels):
        if label not in select_flemish_phones.flemish_phone_labels:
            raise ValueError(
                f'Flemish Phraser phone at index {index} has unexpected '
                f'label {label!r}')

    counts = Counter(labels)
    invalid_counts = {
        label: counts.get(label, 0)
        for label in set(select_flemish_phones.flemish_phone_labels).union(
            counts)
        if counts.get(label, 0)
        != select_flemish_phones.flemish_phones_per_label
    }
    if invalid_counts:
        details = ', '.join(
            f'{label!r}={count}'
            for label, count in sorted(invalid_counts.items()))
        raise ValueError(
            'Flemish Phraser phone inventory does not contain exactly '
            f'{select_flemish_phones.flemish_phones_per_label:,} phones '
            'per label: '
            f'{details}')

    for index, phraser_phone in enumerate(phraser_phones):
        label_index = (
            index // select_flemish_phones.flemish_phones_per_label)
        expected_label = select_flemish_phones.flemish_phone_labels[
            label_index]
        if phraser_phone.label != expected_label:
            raise ValueError(
                f'Flemish Phraser phone at index {index} has label '
                f'{phraser_phone.label!r}, expected {expected_label!r}')
