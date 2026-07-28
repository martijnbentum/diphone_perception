import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from phone_mapper import cgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metadata


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
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end


class StubAudio:
    def __init__(self, phones):
        self.phones = phones

    @property
    def phones_query(self):
        return self

    def filter(self, label, start__gt, end__lt):
        return [
            p for p in self.phones
            if p.label == label and p.start > start__gt and p.end < end__lt
        ]


class StubAudioLookup:
    def __init__(self, audio):
        self._audio = audio

    def get(self, filename__contains):
        return self._audio


class StubStore:
    def __init__(self, audio):
        self.audios = StubAudioLookup(audio)


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

    def fake_get_phraser_phone(store, phone_object, tolerance_ms=25):
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
