import warnings
from collections import Counter
from types import SimpleNamespace

import pytest
from phone_mapper import cgn
from phraser import SEGMENT_KEY_LENGTH

import locations
from probing import metadata


def make_sentence_row(**overrides):
    row = {
        'audio_filename': 'data/audio/wav/comp-k/nl/fn001651.wav',
        'start_time': '10.000',
        'end_time': '20.000',
        'duration': '10.000',
        'text': 'a sentence',
        'identifier': 'fn001651_sentence-0.wav',
        'speaker_ids': 'N02003_male_38',
        'comp': 'k',
    }
    row.update(overrides)
    return row


def make_phone_row(**overrides):
    row = {
        'audio_filename': 'fn001651_sentence-0.wav',
        'start_time': '0.500',
        'end_time': '0.560',
        'duration': '0.060',
        'phoneme': 'd',
        'previous_phoneme': 'SOS',
        'next_phoneme': 'e',
        'speaker_id': 'N02003_male_38',
        'overlap': 'False',
        'comp': 'k',
        'ipa_phoneme': 'd',
    }
    row.update(overrides)
    return row


def build_linked_phone(**overrides):
    speakers = {}
    sentence = metadata.Sentence(make_sentence_row(), speakers)
    sentence_by_identifier = {sentence.identifier: sentence}
    row = make_phone_row(**overrides)
    phone = metadata.Phone(row, sentence_by_identifier, speakers)
    return phone, sentence, speakers


# -- Speaker ------------------------------------------------------------

def test_speaker_parses_id_gender_age():
    speaker = metadata.Speaker('N02003_male_38')
    assert speaker.id == 'N02003'
    assert speaker.gender == 'male'
    assert speaker.age == 38


def test_speaker_missing_age_is_none():
    speaker = metadata.Speaker('N02010_male')
    assert speaker.id == 'N02010'
    assert speaker.gender == 'male'
    assert speaker.age is None


# -- _get_speaker registry ------------------------------------------------

def test_get_speaker_dedups_by_id():
    speakers = {}
    a = metadata._get_speaker('N01_male_30', speakers)
    b = metadata._get_speaker('N01_male_30', speakers)
    c = metadata._get_speaker('N02_female_40', speakers)
    assert a is b
    assert a is not c


# -- Sentence -------------------------------------------------------------

def test_sentence_parses_row():
    sentence = metadata.Sentence(make_sentence_row(), {})
    assert sentence.start == 10000
    assert sentence.end == 20000
    assert sentence.duration == 10000
    assert [s.speaker_id for s in sentence.speakers] == ['N02003_male_38']


def test_sentence_multi_speaker_shares_registry():
    speakers = {}
    sentence_a = metadata.Sentence(
        make_sentence_row(speaker_ids='N01_male_30,N02_female_40'), speakers)
    sentence_b = metadata.Sentence(
        make_sentence_row(
            identifier='fn001651_sentence-1.wav', speaker_ids='N01_male_30'),
        speakers,
    )
    assert len(sentence_a.speakers) == 2
    assert sentence_a.speakers[0] is sentence_b.speakers[0]


# -- Phone: field parsing & mapping ---------------------------------------

def test_phone_audio_filename_id_extraction():
    phone, _, _ = build_linked_phone()
    assert phone.audio_filename_id == 'fn001651'


def test_phone_sampa_to_ipa_mapping():
    phone, _, _ = build_linked_phone(
        phoneme='e', ipa_phoneme=cgn.cgn_to_ipa['e'],
        previous_phoneme='SOS', next_phoneme='EOS',
    )
    assert phone.phoneme_ipa == cgn.cgn_to_ipa['e']
    assert phone.previous_phoneme_ipa == 'SOS'
    assert phone.next_phoneme_ipa == 'EOS'


def test_phone_ipa_mismatch_raises():
    speakers = {}
    sentence = metadata.Sentence(make_sentence_row(), speakers)
    sentence_by_identifier = {sentence.identifier: sentence}
    row = make_phone_row(ipa_phoneme='q')
    with pytest.raises(ValueError):
        metadata.Phone(row, sentence_by_identifier, speakers)


# -- Phone: linking to Sentence/Speaker ------------------------------------

def test_phone_no_matching_sentence_raises():
    row = make_phone_row()
    with pytest.raises(ValueError):
        metadata.Phone(row, {}, {})


def test_phone_speaker_not_in_sentence_raises():
    speakers = {}
    sentence = metadata.Sentence(make_sentence_row(), speakers)
    sentence_by_identifier = {sentence.identifier: sentence}
    row = make_phone_row(speaker_id='N99999_male_99')
    with pytest.raises(ValueError):
        metadata.Phone(row, sentence_by_identifier, speakers)


def test_phone_absolute_start_end_properties():
    phone, _, _ = build_linked_phone()
    assert phone.start == 10500
    assert phone.end == 10560
    assert phone.start_seconds == 10.5
    assert phone.end_seconds == 10.56


# -- load_sentences / load_phones (tmp-file integration) -------------------

def test_load_phones_links_and_shares_speakers(tmp_path):
    sentence_path = tmp_path / 'sentences.tsv'
    metadata_path = tmp_path / 'metadata.csv'

    sentence_path.write_text(
        'audio_filename\tstart_time\tend_time\tduration\ttext\tidentifier'
        '\tspeaker_ids\tcomp\n'
        'aud1.wav\t10.000\t20.000\t10.000\ttext one\tfn1_sentence-0.wav'
        '\tN01_male_30\tk\n'
        'aud1.wav\t20.000\t30.000\t10.000\ttext two\tfn1_sentence-1.wav'
        '\tN01_male_30,N02_female_40\tk\n'
    )
    metadata_path.write_text(
        'audio_filename,start_time,end_time,duration,phoneme,'
        'previous_phoneme,next_phoneme,speaker_id,overlap,comp,ipa_phoneme\n'
        'fn1_sentence-0.wav,0.500,0.560,0.060,d,SOS,e,N01_male_30,False,k,d\n'
        'fn1_sentence-1.wav,0.100,0.160,0.060,d,SOS,e,N02_female_40,False,k,d\n'
    )

    phones = metadata.load_phones(metadata_path, sentence_path)

    assert len(phones) == 2
    assert phones[0].sentence.identifier == 'fn1_sentence-0.wav'
    assert phones[1].sentence.identifier == 'fn1_sentence-1.wav'
    assert phones[0].speaker is phones[0].sentence.speakers[0]
    assert phones[1].speaker is phones[1].sentence.speakers[1]
    # N01_male_30 is shared across both sentences via the same registry
    assert phones[0].sentence.speakers[0] is phones[1].sentence.speakers[0]


# -- get_phraser_phone (stubbed phraser API) -------------------------------

class StubPhraserPhone:
    def __init__(self, label, start, end, key=None):
        self.label = label
        self.start = start
        self.end = end
        self.key = key


class StubAudio:
    def __init__(self, phones, filename='fn1.wav'):
        self.phones = phones
        self.filename = filename


class StubAudioLookup:
    def __init__(self, audio):
        self._audio = audio

    def get(self, filename__contains):
        return self._audio

    def all(self):
        return [self._audio]


class StubStore:
    def __init__(self, audio):
        self.audios = StubAudioLookup(audio)


class StubBulkStore(StubStore):
    '''StubStore that also supports load_many, for Phones-level tests.'''
    def __init__(self, audio):
        super().__init__(audio)
        self._by_key = {phone.key: phone for phone in audio.phones}
        self.load_many_calls = []

    def load_many(self, keys):
        self.load_many_calls.append(list(keys))
        return [self._by_key[key] for key in keys]


def make_phraser_key(value):
    return value.to_bytes(SEGMENT_KEY_LENGTH, byteorder='big')


def write_phraser_keys(path, keys):
    placeholder = bytes(SEGMENT_KEY_LENGTH)
    path.write_bytes(b''.join(
        placeholder if key is None else key for key in keys))


def make_phone_object(**overrides):
    fields = dict(
        audio_filename='fn1_sentence-0.wav',
        audio_filename_id='fn1',
        phoneme_ipa='d',
        start=100,
        end=150,
        previous_phoneme_ipa='SOS',
        next_phoneme_ipa='EOS',
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_get_phraser_phone_unique_match():
    target = StubPhraserPhone('d', 100, 150)
    store = StubStore(StubAudio([target]))

    result = metadata.get_phraser_phone(
        store, make_phone_object(), tolerance_ms=10)

    assert result is target


def test_get_phraser_phone_no_candidates_raises():
    store = StubStore(StubAudio([StubPhraserPhone('z', 100, 150)]))

    with pytest.raises(ValueError):
        metadata.get_phraser_phone(store, make_phone_object(), tolerance_ms=10)


def test_get_phraser_phone_disambiguates_via_neighbors():
    phones = [
        StubPhraserPhone('a', 0, 50),
        StubPhraserPhone('d', 100, 150),
        StubPhraserPhone('b', 200, 250),
        StubPhraserPhone('x', 300, 350),
        StubPhraserPhone('d', 400, 450),
    ]
    store = StubStore(StubAudio(phones))
    phone_object = make_phone_object(
        start=100, end=150, previous_phoneme_ipa='a', next_phoneme_ipa='b')

    result = metadata.get_phraser_phone(store, phone_object, tolerance_ms=1000)

    assert result is phones[1]


def test_get_phraser_phone_still_ambiguous_raises():
    phones = [
        StubPhraserPhone('a', 0, 50),
        StubPhraserPhone('d', 100, 150),
        StubPhraserPhone('b', 200, 250),
        StubPhraserPhone('a', 300, 350),
        StubPhraserPhone('d', 400, 450),
        StubPhraserPhone('b', 500, 550),
    ]
    store = StubStore(StubAudio(phones))
    phone_object = make_phone_object(
        start=100, end=150, previous_phoneme_ipa='a', next_phoneme_ipa='b')

    with pytest.raises(ValueError):
        metadata.get_phraser_phone(store, phone_object, tolerance_ms=1000)


def test_phone_phraser_phone_method_delegates(monkeypatch):
    phone, _, _ = build_linked_phone()
    captured = {}

    def fake_get_phraser_phone(store, phone_object, tolerance_ms=25, audio_index=None):
        captured['store'] = store
        captured['phone_object'] = phone_object
        captured['tolerance_ms'] = tolerance_ms
        return 'sentinel-result'

    monkeypatch.setattr(metadata, 'get_phraser_phone', fake_get_phraser_phone)

    result = phone.phraser_phone('fake-store', tolerance_ms=99)

    assert result == 'sentinel-result'
    assert captured['store'] == 'fake-store'
    assert captured['phone_object'] is phone
    assert captured['tolerance_ms'] == 99


def test_phraser_phone_caches_result(monkeypatch):
    phone, _, _ = build_linked_phone()
    calls = {'n': 0}

    def fake_get_phraser_phone(store, phone_object, tolerance_ms=25, audio_index=None):
        calls['n'] += 1
        return f'result-{calls["n"]}'

    monkeypatch.setattr(metadata, 'get_phraser_phone', fake_get_phraser_phone)

    first = phone.phraser_phone('some-store')
    second = phone.phraser_phone('some-store')

    assert first == second == 'result-1'
    assert calls['n'] == 1


def test_phraser_phone_uses_parent_store_when_store_none(monkeypatch):
    phone, _, _ = build_linked_phone()
    phone.parent = SimpleNamespace(store='parent-store')
    captured = {}

    def fake_get_phraser_phone(store, phone_object, tolerance_ms=25, audio_index=None):
        captured['store'] = store
        return 'result'

    monkeypatch.setattr(metadata, 'get_phraser_phone', fake_get_phraser_phone)

    phone.phraser_phone()

    assert captured['store'] == 'parent-store'


def test_phraser_phone_raises_without_store_or_parent():
    phone, _, _ = build_linked_phone()
    assert phone.parent is None
    with pytest.raises(ValueError):
        phone.phraser_phone()


# -- FlemishPhones ---------------------------------------------------------

def configure_small_flemish_inventory(monkeypatch):
    monkeypatch.setattr(
        metadata.select_flemish_phones,
        'flemish_phone_labels',
        ('d', 'f'),
    )
    monkeypatch.setattr(
        metadata.select_flemish_phones, 'flemish_phones_per_label', 2)
    monkeypatch.setattr(
        metadata.select_flemish_phones, 'flemish_phone_count', 4)


def make_flemish_inventory(labels=('d', 'd', 'f', 'f')):
    return [
        StubPhraserPhone(label, index, index + 1, make_phraser_key(index + 1))
        for index, label in enumerate(labels)
    ]


def test_flemish_phones_defaults_to_selected_key_file():
    phones = metadata.FlemishPhones(store='unused')

    assert phones.phraser_key_path == locations.flemish_phraser_phone_key_file


def test_flemish_phones_store_lazy_loads_cgn(monkeypatch):
    calls = []
    monkeypatch.setattr(
        metadata, 'load_cgn', lambda: calls.append(True) or 'cgn-store')

    phones = metadata.FlemishPhones()

    assert calls == []
    assert phones.store == 'cgn-store'
    assert phones.store == 'cgn-store'
    assert calls == [True]


def test_flemish_phones_loads_valid_inventory_without_metadata(
    tmp_path, monkeypatch,
):
    configure_small_flemish_inventory(monkeypatch)
    inventory = make_flemish_inventory()
    store = StubBulkStore(StubAudio(inventory))
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, [phone.key for phone in inventory])
    monkeypatch.setattr(
        metadata, 'load_phones',
        lambda *args, **kwargs: pytest.fail('metadata should not be loaded'),
    )
    phones = metadata.FlemishPhones(store=store, phraser_key_path=key_path)

    loaded = phones.phraser_phones

    assert loaded == inventory
    assert phones.flemish_phraser_phones is loaded
    assert phones.phraser_phones is loaded
    assert store.load_many_calls == [[phone.key for phone in inventory]]


@pytest.mark.parametrize(
    ('keys', 'match'),
    [
        ([make_phraser_key(1)] * 4, 'duplicate keys'),
        ([None, make_phraser_key(2), make_phraser_key(3),
            make_phraser_key(4)], 'placeholders'),
        ([make_phraser_key(1), make_phraser_key(2)], '2 records'),
    ],
)
def test_flemish_phones_rejects_invalid_key_inventory(
    tmp_path, monkeypatch, keys, match,
):
    configure_small_flemish_inventory(monkeypatch)
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, keys)
    phones = metadata.FlemishPhones(store='unused', phraser_key_path=key_path)

    with pytest.raises(ValueError, match=match):
        phones.phraser_phones


def test_flemish_phones_rejects_malformed_key_record(tmp_path):
    key_path = tmp_path / 'flemish.bin'
    key_path.write_bytes(b'incomplete')
    phones = metadata.FlemishPhones(store='unused', phraser_key_path=key_path)

    with pytest.raises(ValueError, match='not a multiple'):
        phones.phraser_phones


def test_flemish_phones_rejects_missing_store_object(tmp_path, monkeypatch):
    configure_small_flemish_inventory(monkeypatch)
    inventory = make_flemish_inventory()
    store = StubBulkStore(StubAudio(inventory[:-1]))
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, [phone.key for phone in inventory])
    phones = metadata.FlemishPhones(store=store, phraser_key_path=key_path)

    with pytest.raises(ValueError, match='missing object'):
        phones.phraser_phones


def test_flemish_phones_rejects_none_store_object(tmp_path, monkeypatch):
    configure_small_flemish_inventory(monkeypatch)
    inventory = make_flemish_inventory()
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, [phone.key for phone in inventory])

    class MissingObjectStore:
        def load_many(self, keys):
            return inventory[:-1] + [None]

    phones = metadata.FlemishPhones(
        store=MissingObjectStore(), phraser_key_path=key_path)

    with pytest.raises(ValueError, match='no stored object'):
        phones.phraser_phones


def test_flemish_phones_rejects_mismatched_store_object(
    tmp_path, monkeypatch,
):
    configure_small_flemish_inventory(monkeypatch)
    inventory = make_flemish_inventory()
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, [phone.key for phone in inventory])

    class ReorderedStore:
        def load_many(self, keys):
            return list(reversed(inventory))

    phones = metadata.FlemishPhones(
        store=ReorderedStore(), phraser_key_path=key_path)

    with pytest.raises(ValueError, match='requested keys'):
        phones.phraser_phones


@pytest.mark.parametrize(
    ('labels', 'match'),
    [
        (('d', 'f', 'd', 'f'), 'index 1'),
        (('d', 'd', 'd', 'f'), 'exactly 2 phones per label'),
        (('d', 'd', 'q', 'q'), 'unexpected label'),
    ],
)
def test_flemish_phones_rejects_invalid_label_inventory(
    tmp_path, monkeypatch, labels, match,
):
    configure_small_flemish_inventory(monkeypatch)
    inventory = make_flemish_inventory(labels)
    store = StubBulkStore(StubAudio(inventory))
    key_path = tmp_path / 'flemish.bin'
    write_phraser_keys(key_path, [phone.key for phone in inventory])
    phones = metadata.FlemishPhones(store=store, phraser_key_path=key_path)

    with pytest.raises(ValueError, match=match):
        phones.phraser_phones


# -- Phones -----------------------------------------------------------------

def write_dataset(tmp_path, sentence_rows, phone_rows):
    sentence_path = tmp_path / 'sentences.tsv'
    metadata_path = tmp_path / 'metadata.csv'

    sentence_columns = (
        'audio_filename', 'start_time', 'end_time', 'duration', 'text',
        'identifier', 'speaker_ids', 'comp',
    )
    sentence_path.write_text(
        '\t'.join(sentence_columns) + '\n'
        + ''.join(
            '\t'.join(str(row[c]) for c in sentence_columns) + '\n'
            for row in sentence_rows
        )
    )

    phone_columns = (
        'audio_filename', 'start_time', 'end_time', 'duration', 'phoneme',
        'previous_phoneme', 'next_phoneme', 'speaker_id', 'overlap', 'comp',
        'ipa_phoneme',
    )
    metadata_path.write_text(
        ','.join(phone_columns) + '\n'
        + ''.join(
            ','.join(str(row[c]) for c in phone_columns) + '\n'
            for row in phone_rows
        )
    )

    return sentence_path, metadata_path


def build_three_phone_dataset(tmp_path):
    '''one sentence with three phones: d, e (long/tense), f.

    only d and f get a stub phraser match built for them in the tests
    below; e is left to also be matched so exactly one phone (f) can be
    made to fail, exercising save_phraser_keys' failure handling.
    '''
    sentence_row = make_sentence_row(
        audio_filename='aud1.wav', identifier='fn1_sentence-0.wav',
        speaker_ids='N01_male_30',
    )
    e_ipa = cgn.cgn_to_ipa['e']
    phone_rows = [
        make_phone_row(
            audio_filename='fn1_sentence-0.wav', speaker_id='N01_male_30',
            phoneme='d', ipa_phoneme='d',
            previous_phoneme='SOS', next_phoneme='e',
            start_time='0.500', end_time='0.560', duration='0.060',
        ),
        make_phone_row(
            audio_filename='fn1_sentence-0.wav', speaker_id='N01_male_30',
            phoneme='e', ipa_phoneme=e_ipa,
            previous_phoneme='d', next_phoneme='f',
            start_time='1.000', end_time='1.060', duration='0.060',
        ),
        make_phone_row(
            audio_filename='fn1_sentence-0.wav', speaker_id='N01_male_30',
            phoneme='f', ipa_phoneme='f',
            previous_phoneme='e', next_phoneme='EOS',
            start_time='2.000', end_time='2.060', duration='0.060',
        ),
    ]
    sentence_path, metadata_path = write_dataset(
        tmp_path, [sentence_row], phone_rows)
    return sentence_path, metadata_path, e_ipa


def test_phones_phones_sets_parent(tmp_path):
    sentence_path, metadata_path, _ = build_three_phone_dataset(tmp_path)
    phones_obj = metadata.Phones(path=metadata_path, sentence_path=sentence_path)

    phones = phones_obj.phones

    assert len(phones) == 3
    assert all(phone.parent is phones_obj for phone in phones)


def test_phones_phoneme_counts(tmp_path):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    phones_obj = metadata.Phones(path=metadata_path, sentence_path=sentence_path)

    assert phones_obj.phoneme_counts == Counter({'d': 1, e_ipa: 1, 'f': 1})


def test_phones_store_lazy_loads_cgn(monkeypatch):
    calls = {'n': 0}

    def fake_load_cgn(path=locations.cgn_lmdb):
        calls['n'] += 1
        return 'cgn-store'

    monkeypatch.setattr(metadata, 'load_cgn', fake_load_cgn)

    phones_obj = metadata.Phones()
    assert calls['n'] == 0

    assert phones_obj.store == 'cgn-store'
    assert phones_obj.store == 'cgn-store'
    assert calls['n'] == 1


def test_phones_save_and_load_phraser_keys_roundtrip(tmp_path):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    key_path = tmp_path / 'keys.bin'
    phones_obj = metadata.Phones(
        path=metadata_path, sentence_path=sentence_path, phraser_key_path=key_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    d_key = b'\x01' * SEGMENT_KEY_LENGTH
    e_key = b'\x02' * SEGMENT_KEY_LENGTH
    stub_audio = StubAudio([
        StubPhraserPhone('d', d_phone.start, d_phone.end, d_key),
        StubPhraserPhone(e_ipa, e_phone.start, e_phone.end, e_key),
        # deliberately no stub phone for 'f' -> f_phone must fail to match
    ])
    phones_obj._store = StubBulkStore(stub_audio)

    phones_obj.save_phraser_keys()

    failed_phones = [phone for phone, error in phones_obj.phraser_match_failures]
    assert failed_phones == [f_phone]
    assert isinstance(
        phones_obj.phraser_match_failures[0][1], metadata.NoCandidateError)
    assert key_path.exists()
    assert metadata.load_phraser_keys(key_path) == [d_key, e_key, None]


def test_phones_phraser_phones_raises_when_incomplete(tmp_path):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    key_path = tmp_path / 'keys.bin'
    phones_obj = metadata.Phones(
        path=metadata_path, sentence_path=sentence_path, phraser_key_path=key_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    stub_audio = StubAudio([
        StubPhraserPhone(
            'd', d_phone.start, d_phone.end,
            b'\x01' * SEGMENT_KEY_LENGTH,
        ),
        StubPhraserPhone(
            e_ipa, e_phone.start, e_phone.end,
            b'\x02' * SEGMENT_KEY_LENGTH,
        ),
        # deliberately no stub phone for 'f' -> phraser_phones must raise
    ])
    phones_obj._store = StubBulkStore(stub_audio)

    assert not key_path.exists()
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match='1 / 3'):
            phones_obj.phraser_phones
    # save_phraser_keys still ran (and wrote the key file) before the check
    assert key_path.exists()


def test_phones_phraser_phones_builds_then_reuses_key_file(tmp_path):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    key_path = tmp_path / 'keys.bin'
    phones_obj = metadata.Phones(
        path=metadata_path, sentence_path=sentence_path, phraser_key_path=key_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    d_key = b'\x01' * SEGMENT_KEY_LENGTH
    e_key = b'\x02' * SEGMENT_KEY_LENGTH
    f_key = b'\x03' * SEGMENT_KEY_LENGTH
    stub_audio = StubAudio([
        StubPhraserPhone('d', d_phone.start, d_phone.end, d_key),
        StubPhraserPhone(e_ipa, e_phone.start, e_phone.end, e_key),
        StubPhraserPhone('f', f_phone.start, f_phone.end, f_key),
    ])
    phones_obj._store = StubBulkStore(stub_audio)

    assert not key_path.exists()
    with pytest.warns(RuntimeWarning):
        matched = phones_obj.phraser_phones
    assert key_path.exists()
    assert [p.key for p in matched] == [d_key, e_key, f_key]

    # a fresh Phones instance reuses the cached key file directly
    phones_obj2 = metadata.Phones(
        store=phones_obj.store, path=metadata_path, sentence_path=sentence_path,
        phraser_key_path=key_path)
    with pytest.warns(RuntimeWarning):
        matched2 = phones_obj2.phraser_phones
    assert [p.key for p in matched2] == [d_key, e_key, f_key]


def test_load_phraser_phones_replaces_without_loading_metadata(
    tmp_path, monkeypatch,
):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    original_key = make_phraser_key(1)
    middle_key = make_phraser_key(2)
    replacement_key = make_phraser_key(3)
    final_key = make_phraser_key(4)
    original_keys = [original_key, middle_key, original_key, final_key]
    write_phraser_keys(key_path, original_keys)
    write_phraser_keys(replacement_path, [replacement_key])

    phraser_phones = [
        StubPhraserPhone('d', 0, 1, original_key),
        StubPhraserPhone('e', 1, 2, middle_key),
        StubPhraserPhone('d', 2, 3, replacement_key),
        StubPhraserPhone('f', 3, 4, final_key),
    ]
    store = StubBulkStore(StubAudio(phraser_phones))
    phones_obj = metadata.Phones(
        store=store,
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )
    monkeypatch.setattr(
        metadata,
        'load_phones',
        lambda *args, **kwargs: pytest.fail('metadata phones were loaded'),
    )

    with pytest.warns(RuntimeWarning) as caught:
        loaded = phones_obj.load_phraser_phones()

    messages = [str(item.message) for item in caught]
    assert any('3 distinct labels, expected 31' in message
        for message in messages)
    assert any('13,500 unique keys per label' in message
        for message in messages)
    expected_keys = [
        original_key, middle_key, replacement_key, final_key]
    assert len(loaded) == len(original_keys)
    assert [phone.key for phone in loaded] == expected_keys
    assert store.load_many_calls == [expected_keys]
    assert metadata.load_phraser_keys(key_path) == original_keys
    assert not hasattr(phones_obj, '_phones')
    assert phones_obj.duplicate_replacement_phones == [phraser_phones[2]]


def test_load_phraser_phones_retains_duplicates_without_replacements(
    tmp_path,
):
    key_path = tmp_path / 'keys.bin'
    missing_replacement_path = tmp_path / 'missing-replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    other_key = make_phraser_key(2)
    write_phraser_keys(key_path, [duplicate_key, duplicate_key, other_key])

    duplicate_phone = StubPhraserPhone('d', 0, 1, duplicate_key)
    other_phone = StubPhraserPhone('f', 1, 2, other_key)
    store = StubBulkStore(StubAudio([duplicate_phone, other_phone]))
    phones_obj = metadata.Phones(
        store=store,
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=missing_replacement_path,
    )
    phones_obj._phones = [
        SimpleNamespace(phoneme_ipa=label) for label in ('d', 'd', 'f')]

    with pytest.warns(RuntimeWarning) as caught:
        loaded = phones_obj.load_phraser_phones()

    assert loaded == [duplicate_phone, duplicate_phone, other_phone]
    assert phones_obj.duplicate_replacement_phones == []
    assert any(
        'contain 1 duplicate key occurrence' in str(item.message)
        for item in caught
    )
    assert any(
        'replacement Phraser key file is not available' in str(item.message)
        for item in caught
    )


def test_custom_key_path_does_not_use_default_replacement_history(
    tmp_path, monkeypatch,
):
    key_path = tmp_path / 'custom-keys.bin'
    default_replacement_path = tmp_path / 'default-replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    replacement_key = make_phraser_key(2)
    write_phraser_keys(key_path, [duplicate_key, duplicate_key])
    write_phraser_keys(default_replacement_path, [replacement_key])
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file',
        default_replacement_path,
    )

    duplicate_phone = StubPhraserPhone('d', 0, 1, duplicate_key)
    store = StubBulkStore(StubAudio([duplicate_phone]))
    phones_obj = metadata.Phones(
        store=store,
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=default_replacement_path,
    )
    phones_obj._phones = [
        SimpleNamespace(phoneme_ipa='d'),
        SimpleNamespace(phoneme_ipa='d'),
    ]

    with pytest.warns(RuntimeWarning) as caught:
        loaded = phones_obj.load_phraser_phones()

    messages = [str(item.message) for item in caught]
    assert loaded == [duplicate_phone, duplicate_phone]
    assert any('duplicate history of the default' in message
        for message in messages)
    assert store.load_many_calls == [[duplicate_key, duplicate_key]]


@pytest.mark.parametrize(
    ('keys', 'replacement_keys', 'message'),
    [
        ([1, 1], [], 'expected 1 duplicate replacement key'),
        ([1, 1], [None], 'cannot contain placeholders'),
        ([1, 1], [1], 'already occur in the original'),
        ([1, 1, 2, 2], [3, 3], 'contains repeated keys'),
    ],
    ids=['wrong-count', 'placeholder', 'reused-original', 'repeated'],
)
def test_load_phraser_phones_rejects_malformed_replacement_inventory(
    tmp_path, keys, replacement_keys, message,
):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    write_phraser_keys(key_path, [make_phraser_key(key) for key in keys])
    write_phraser_keys(
        replacement_path,
        [None if key is None else make_phraser_key(key)
         for key in replacement_keys],
    )
    phones_obj = metadata.Phones(
        store=StubBulkStore(StubAudio([])),
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )

    with pytest.raises(ValueError, match=message):
        phones_obj.load_phraser_phones()


def test_load_phraser_phones_rejects_non_record_replacement_data(tmp_path):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    write_phraser_keys(key_path, [duplicate_key, duplicate_key])
    replacement_path.write_bytes(b'not-a-22-byte-record')
    phones_obj = metadata.Phones(
        store=StubBulkStore(StubAudio([])),
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )

    with pytest.raises(ValueError, match='not a multiple of 22 bytes'):
        phones_obj.load_phraser_phones()


def test_load_phraser_phones_rejects_final_duplicate_keys(tmp_path):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    colliding_key = make_phraser_key(2)
    write_phraser_keys(
        key_path, [duplicate_key, duplicate_key, colliding_key])
    write_phraser_keys(replacement_path, [colliding_key])
    phones_obj = metadata.Phones(
        store=StubBulkStore(StubAudio([])),
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )

    with pytest.raises(
        ValueError, match='already occur in the original|remain after replacement',
    ):
        phones_obj.load_phraser_phones()


def test_load_phraser_phones_rejects_wrong_replacement_label(tmp_path):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    replacement_key = make_phraser_key(2)
    write_phraser_keys(key_path, [duplicate_key, duplicate_key])
    write_phraser_keys(replacement_path, [replacement_key])

    phraser_phones = [
        StubPhraserPhone('d', 0, 1, duplicate_key),
        StubPhraserPhone('x', 1, 2, replacement_key),
    ]
    phones_obj = metadata.Phones(
        store=StubBulkStore(StubAudio(phraser_phones)),
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )
    phones_obj._phones = [
        SimpleNamespace(phoneme_ipa='d'),
        SimpleNamespace(phoneme_ipa='d'),
    ]

    with pytest.raises(
        ValueError, match="label 'x', expected original label 'd'",
    ):
        phones_obj.load_phraser_phones()


def test_load_phraser_phones_can_validate_replacements_against_metadata(
    tmp_path,
):
    key_path = tmp_path / 'keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    duplicate_key = make_phraser_key(1)
    replacement_key = make_phraser_key(2)
    write_phraser_keys(key_path, [duplicate_key, duplicate_key])
    write_phraser_keys(replacement_path, [replacement_key])

    phraser_phones = [
        StubPhraserPhone('d', 0, 1, duplicate_key),
        StubPhraserPhone('d', 1, 2, replacement_key),
    ]
    phones_obj = metadata.Phones(
        store=StubBulkStore(StubAudio(phraser_phones)),
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
    )
    phones_obj._phones = [
        SimpleNamespace(phoneme_ipa='d'),
        SimpleNamespace(phoneme_ipa='x'),
    ]

    with pytest.raises(ValueError, match="label 'd', expected 'x'"):
        phones_obj.load_phraser_phones(validate_against_metadata=True)


def test_warn_phraser_inventory_reports_duplicates_and_unbalanced_labels():
    duplicate_key = make_phraser_key(1)
    phones_obj = metadata.Phones(store='unused')
    phones_obj._phones = [
        SimpleNamespace(phoneme_ipa='d'),
        SimpleNamespace(phoneme_ipa='d'),
        SimpleNamespace(phoneme_ipa='f'),
    ]
    inventory = [
        StubPhraserPhone('d', 0, 1, duplicate_key),
        StubPhraserPhone('d', 1, 2, duplicate_key),
        StubPhraserPhone('f', 2, 3, make_phraser_key(2)),
    ]

    with pytest.warns(RuntimeWarning) as caught:
        phones_obj._warn_phraser_inventory(inventory)

    messages = [str(item.message) for item in caught]
    assert any('contain 1 duplicate key occurrence' in message
        for message in messages)
    assert any('2 distinct labels, expected 31' in message
        for message in messages)
    assert any("'d'=1" in message and "'f'=1" in message
        for message in messages)


def test_warn_phraser_inventory_accepts_balanced_unique_labels(monkeypatch):
    monkeypatch.setattr(metadata, '_phraser_label_count', 2)
    monkeypatch.setattr(metadata, '_phraser_phones_per_label', 2)
    phones_obj = metadata.Phones(store='unused')
    inventory = [
        StubPhraserPhone(label, index, index + 1, make_phraser_key(key))
        for label_index, label in enumerate(('d', 'f'))
        for index, key in enumerate(
            range(label_index * 2, (label_index + 1) * 2))
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        phones_obj._warn_phraser_inventory(inventory)


def test_phones_label_to_phraser_phone_groups_by_label(tmp_path):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    key_path = tmp_path / 'keys.bin'
    phones_obj = metadata.Phones(
        path=metadata_path, sentence_path=sentence_path, phraser_key_path=key_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    d_key = b'\x01' * SEGMENT_KEY_LENGTH
    e_key = b'\x02' * SEGMENT_KEY_LENGTH
    f_key = b'\x03' * SEGMENT_KEY_LENGTH
    stub_audio = StubAudio([
        StubPhraserPhone('d', d_phone.start, d_phone.end, d_key),
        StubPhraserPhone(e_ipa, e_phone.start, e_phone.end, e_key),
        StubPhraserPhone('f', f_phone.start, f_phone.end, f_key),
    ])
    phones_obj._store = StubBulkStore(stub_audio)

    with pytest.warns(RuntimeWarning):
        grouped = phones_obj.label_to_phraser_phone

    assert set(grouped.keys()) == {'d', e_ipa, 'f'}
    assert [p.key for p in grouped['d']] == [d_key]
    assert [p.key for p in grouped[e_ipa]] == [e_key]
    assert [p.key for p in grouped['f']] == [f_key]


def test_sentence_edge_position(tmp_path):
    sentence_path, metadata_path, _ = build_three_phone_dataset(tmp_path)
    phones_obj = metadata.Phones(path=metadata_path, sentence_path=sentence_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    # d_phone: previous=SOS, next=e -> sentence-first
    # e_phone: previous=d, next=f -> interior
    # f_phone: previous=e, next=EOS -> sentence-last
    assert metadata._sentence_edge_position(d_phone) == 'first'
    assert metadata._sentence_edge_position(e_phone) == 'interior'
    assert metadata._sentence_edge_position(f_phone) == 'last'


def test_phones_analyze_phraser_failures(tmp_path, capsys):
    sentence_path, metadata_path, e_ipa = build_three_phone_dataset(tmp_path)
    phones_obj = metadata.Phones(path=metadata_path, sentence_path=sentence_path)
    d_phone, e_phone, f_phone = phones_obj.phones

    phones_obj.phraser_match_failures = [
        (e_phone, metadata.NoCandidateError('no candidate')),
        (f_phone, metadata.AmbiguousMatchError('ambiguous')),
    ]

    # closest-in-time to e_phone is the mislabeled 'x' phone (off by 5ms);
    # closest-in-time to f_phone is an exact-position 'f' phone (a match)
    stub_audio = StubAudio([
        StubPhraserPhone('d', d_phone.start, d_phone.end),
        StubPhraserPhone('x', e_phone.start + 5, e_phone.end + 5),
        StubPhraserPhone('f', f_phone.start, f_phone.end),
    ])
    phones_obj._store = StubStore(stub_audio)

    stats = phones_obj.analyze_phraser_failures()

    assert stats['total_failures'] == 2
    assert stats['total_phones'] == 3
    assert stats['by_type'] == Counter(
        {'NoCandidateError': 1, 'AmbiguousMatchError': 1})
    assert stats['by_label'] == Counter({e_ipa: 1, 'f': 1})
    assert stats['by_overlap'] == Counter({False: 2})
    assert stats['by_comp'] == Counter({'k': 2})
    # e_phone is interior (prev=d, next=f); f_phone is sentence-last (next=EOS)
    assert stats['by_sentence_edge'] == Counter({'interior': 1, 'last': 1})
    assert stats['by_closest_label'] == Counter({'x': 1, 'f': 1})
    assert stats['by_label_closest_label'] == {
        e_ipa: Counter({'x': 1}),
        'f': Counter({'f': 1}),
    }
    assert stats['closest_matches_expected'] == 1

    printed = capsys.readouterr().out
    assert 'NoCandidateError' in printed
    assert 'AmbiguousMatchError' in printed


def test_phones_analyze_phraser_failures_raises_before_save(tmp_path):
    sentence_path, metadata_path, _ = build_three_phone_dataset(tmp_path)
    phones_obj = metadata.Phones(path=metadata_path, sentence_path=sentence_path)
    with pytest.raises(ValueError):
        phones_obj.analyze_phraser_failures()
