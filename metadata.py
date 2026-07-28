import csv
from collections import Counter
from pathlib import Path
from progressbar import progressbar
from phone_mapper import cgn
from phraser import Store

_data_dir = Path(__file__).resolve().parent.parent / 'data'
metadata_file = _data_dir / 'metadata.csv'
sentence_file = _data_dir / 'news_books_sentences_zs.tsv'
phraser_key_file = _data_dir / 'phraser_phone_keys.bin'
cgn_lmdb = Path('/vol/mlusers/mbentum/phraser/data/cgn_lmdb')
_boundary_tokens = ('SOS', 'EOS')
_bool = {'True': True, 'False': False}
_phraser_key_len = 22
_phraser_key_placeholder = b'\x00' * _phraser_key_len


def load_cgn(path=cgn_lmdb):
    return Store(path=path)


def _sampa_to_ipa(sampa):
    if sampa in _boundary_tokens:
        return sampa
    return cgn.cgn_to_ipa.get(sampa)


class Speaker:
    '''a speaker parsed from a speaker_id like N02003_male_38'''
    def __init__(self, speaker_id):
        self.speaker_id = speaker_id
        parts = speaker_id.split('_')
        self.id = parts[0]
        self.gender = parts[1] if len(parts) > 1 else None
        self.age = int(parts[2]) if len(parts) > 2 else None


def _get_speaker(speaker_id, speakers):
    if speaker_id not in speakers:
        speakers[speaker_id] = Speaker(speaker_id)
    return speakers[speaker_id]


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


def load_sentences(path=sentence_file):
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
        self.overlap = _bool[row['overlap']]
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

    def phraser_phone(self, store=None, tolerance_ms=25):
        if hasattr(self, '_phraser_phone'):
            return self._phraser_phone
        if store is None:
            if self.parent is None:
                raise ValueError(
                    'no store given and no parent set on this Phone')
            store = self.parent.store
        self._phraser_phone = get_phraser_phone(
            store, self, tolerance_ms=tolerance_ms)
        return self._phraser_phone


def load_phones(path=metadata_file, sentence_path=sentence_file):
    sentences, speakers = load_sentences(sentence_path)
    sentence_by_identifier = {s.identifier: s for s in sentences}
    with open(path, newline='') as f:
        return [
            Phone(row, sentence_by_identifier, speakers)
            for row in progressbar(csv.DictReader(f))
        ]


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


def get_phraser_phone(store, phone_object, tolerance_ms=25):
    '''find the phraser Phone corresponding to a metadata Phone.

    matches by audio, ipa label, and a start/end tolerance window (forced
    alignment timing may not match exactly between the two pipelines). if
    more than one candidate remains, disambiguates using the labels of the
    neighboring phones in audio.phones.
    '''
    audio = store.audios.get(filename__contains=phone_object.audio_filename_id)
    candidates = list(audio.phones_query.filter(
        label=phone_object.phoneme_ipa,
        start__gt=phone_object.start - tolerance_ms,
        end__lt=phone_object.end + tolerance_ms,
    ))

    if not candidates:
        raise ValueError(
            f'no phraser phone found for {phone_object.audio_filename!r} '
            f'label={phone_object.phoneme_ipa!r} '
            f'start={phone_object.start} end={phone_object.end}'
        )
    if len(candidates) == 1:
        return candidates[0]

    phones = audio.phones
    refined = [
        candidate for candidate in candidates
        if _phraser_siblings_match(phones, candidate, phone_object)
    ]
    if len(refined) == 1:
        return refined[0]

    raise ValueError(
        f'ambiguous phraser phone match for {phone_object.audio_filename!r} '
        f'label={phone_object.phoneme_ipa!r}: {len(candidates)} candidates '
        f'in window, {len(refined)} after neighbor-label check'
    )


def load_phraser_keys(path=phraser_key_file):
    '''load phraser phone keys saved by Phones.save_phraser_keys.

    returns a list aligned with the Phones.phones order used when saving;
    a phone that could not be matched at save time is None.
    '''
    data = Path(path).read_bytes()
    keys = [
        data[i:i + _phraser_key_len]
        for i in range(0, len(data), _phraser_key_len)
    ]
    return [None if key == _phraser_key_placeholder else key for key in keys]


class Phones:
    '''all Phone objects linked to a phraser store.'''
    def __init__(self, store=None, path=metadata_file,
        sentence_path=sentence_file, phraser_key_path=phraser_key_file):
        self._store = store
        self.path = path
        self.sentence_path = sentence_path
        self.phraser_key_path = phraser_key_path

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
    def phoneme_counts(self):
        if hasattr(self, '_phoneme_counts'):
            return self._phoneme_counts
        self._phoneme_counts = Counter(
            phone.phoneme_ipa for phone in self.phones)
        return self._phoneme_counts

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
        with self.phones. returns the list of phones that failed to match.
        '''
        path = path or self.phraser_key_path
        failed = []
        with open(path, 'wb') as f:
            for phone in progressbar(self.phones):
                try:
                    phraser_phone = phone.phraser_phone(
                        self.store, tolerance_ms=tolerance_ms)
                    f.write(phraser_phone.key)
                except ValueError:
                    failed.append(phone)
                    f.write(_phraser_key_placeholder)
        return failed

    def load_phraser_phones(self, path=None):
        '''bulk-load the phraser Phone objects saved by save_phraser_keys.

        returns a list aligned with self.phones (None where the phone
        wasn't matched during save_phraser_keys).
        '''
        path = path or self.phraser_key_path
        keys = load_phraser_keys(path)
        real_keys = [key for key in keys if key is not None]
        loaded = iter(self.store.load_many(real_keys))
        return [next(loaded) if key is not None else None for key in keys]

    @property
    def phraser_phones(self):
        '''phraser Phone objects aligned with self.phones (None where a
        phone couldn't be matched). Uses the cached key file at
        self.phraser_key_path if present, building it on first use.
        '''
        if hasattr(self, '_phraser_phones'):
            return self._phraser_phones
        if not Path(self.phraser_key_path).exists():
            self.save_phraser_keys()
        self._phraser_phones = self.load_phraser_phones()
        return self._phraser_phones

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
