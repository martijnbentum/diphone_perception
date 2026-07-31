import csv
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import vowel_formant_reference.selected_phones as selected_phone_module
from synthetic_acoustic_probes.formants import praat_vowel_stimulus
from vowel_formant_reference.aggregation import (
    aggregate_gender_measurements,
    aggregate_speaker_measurements,
)
from vowel_formant_reference.formant_tables import load_formant_table
from vowel_formant_reference.selected_phones import (
    PhoneFormantMeasurement,
    _load_phone_audio,
    is_monophthong,
    measure_and_write_phone_formants,
)


class FakeSpeaker:
    def __init__(self, gender):
        self._gender = gender

    def gender(self):
        return self._gender


class FakePhone:
    def __init__(
        self,
        label,
        gender,
        key,
        speaker_id,
        overlap=False,
    ):
        self.label = label
        self.speaker = FakeSpeaker(gender)
        self.key = key
        self.speaker_id = speaker_id
        self.overlap = overlap
        self.start = 999999
        self.end = 999999
        self.start_seconds = 0.0
        self.end_seconds = 0.2

    @property
    def stress(self):
        raise AssertionError('phone stress must not be inspected')


def _audio_loader(_phone):
    stimulus = praat_vowel_stimulus(
        f0_hz=120,
        f1_hz=500,
        f2_hz=1500,
        duration=0.2,
    )
    return stimulus.waveform, stimulus.sample_rate


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def test_monophthong_inventory_includes_schwa_and_long_mid_vowels():
    for label in ('eː', 'øː', 'oː', 'ə', 'ɑ'):
        assert is_monophthong(label)
    for label in ('ɛi', 'œy', 'ɑu', 't'):
        assert not is_monophthong(label)


def test_one_call_uses_phraser_phones_and_writes_flat_csv_files(
    tmp_path,
    capsys,
):
    phones = [
        FakePhone('eː', 'male', b'phone-one', b'speaker-one'),
        FakePhone('ə', 'female', b'phone-two', b'speaker-two'),
        FakePhone('ɛi', 'male', b'diphthong', b'speaker-one'),
        FakePhone(
            'ɑ', 'male', b'overlap', b'speaker-one', overlap=True
        ),
    ]

    paths = measure_and_write_phone_formants(
        phones,
        data_root=tmp_path,
        audio_loader=_audio_loader,
        n_bootstrap=20,
        seed=4,
    )

    assert paths['phone_formants'].name == 'phone_formants.csv'
    assert paths['phone_formants_metadata'].name == (
        'phone_formants_metadata.json'
    )
    assert paths['gender_formants'].name == 'gender_formants.csv'
    assert all(path.exists() for path in paths.values())

    output = capsys.readouterr().out
    assert "Selected vowels: {'eː': 1, 'ə': 1}" in output
    for path in paths.values():
        assert str(path.resolve()) in output

    phone_rows = _read_csv(paths['phone_formants'])
    assert len(phone_rows) == 2
    assert {row['gender'] for row in phone_rows} == {'male', 'female'}
    assert {row['phone_key'] for row in phone_rows} == {
        b'phone-one'.hex(),
        b'phone-two'.hex(),
    }
    duplicated_metadata = {
        'ipa', 'speaker_id', 'audio_filename', 'start_seconds',
        'end_seconds', 'duration_seconds', 'stress', 'age',
    }
    assert duplicated_metadata.isdisjoint(phone_rows[0])

    metadata = json.loads(paths['phone_formants_metadata'].read_text())
    assert metadata['selection']['check_stress'] is False
    assert metadata['selection']['exclude_overlap'] is True
    assert metadata['selection']['selected_vowel_counts'] == {
        'eː': 1,
        'ə': 1,
    }

    anchors = _read_csv(paths['gender_formants'])
    assert {(row['gender'], row['ipa']) for row in anchors} == {
        ('male', 'eː'),
        ('female', 'ə'),
    }


def test_written_phone_tables_load_without_pandas(tmp_path):
    phones = [FakePhone('ɑ', 'male', b'phone', b'speaker')]
    measure_and_write_phone_formants(
        phones,
        data_root=tmp_path,
        audio_loader=_audio_loader,
        n_bootstrap=10,
    )

    tokens = load_formant_table('phone_formants', data_root=tmp_path).data
    anchors = load_formant_table('gender_formants', data_root=tmp_path).data
    assert tokens[0]['phone_key'] == b'phone'.hex()
    assert tokens[0]['success'] is True
    assert isinstance(tokens[0]['f1_hz'], float)
    assert anchors[0]['ipa'] == 'ɑ'
    assert anchors[0]['n_speakers'] == 1


def test_measurement_progress_bar_receives_every_selected_phone(
    tmp_path,
    monkeypatch,
):
    phones = [
        FakePhone('ɑ', 'male', b'one', b'speaker'),
        FakePhone('ə', 'female', b'two', b'speaker'),
    ]
    progress_items = []

    def fake_progressbar(items):
        progress_items.extend(items)
        return items

    monkeypatch.setattr(
        selected_phone_module,
        'progressbar',
        fake_progressbar,
    )

    measure_and_write_phone_formants(
        phones,
        data_root=tmp_path,
        audio_loader=_audio_loader,
        n_bootstrap=10,
    )

    assert progress_items == phones


def test_audio_loader_uses_phraser_phone_second_properties(monkeypatch):
    calls = []

    def load_audio_samples(filename, start_sample, stop_sample):
        calls.append((filename, start_sample, stop_sample))
        return [0.1] * (stop_sample - start_sample), 16_000

    package = ModuleType('phraser')
    audio_module = ModuleType('phraser.audio')
    audio_module.load_audio_samples = load_audio_samples
    package.audio = audio_module
    monkeypatch.setitem(sys.modules, 'phraser', package)
    monkeypatch.setitem(sys.modules, 'phraser.audio', audio_module)

    phone = FakePhone('ɑ', 'male', b'phone', b'speaker')
    phone.start_seconds = 0.125
    phone.end_seconds = 0.375
    phone.audio = SimpleNamespace(filename='recording.wav', sample_rate=16_000)

    waveform, sample_rate = _load_phone_audio(phone)

    assert calls == [(Path('recording.wav'), 2000, 6000)]
    assert len(waveform) == 4000
    assert sample_rate == 16_000


def test_phone_measurement_contains_gender_as_analysis_provenance():
    measurement = PhoneFormantMeasurement(
        phone_key=b'phone',
        gender='female',
        success=False,
        rejection_reason='test',
    )

    assert measurement.gender == 'female'
    assert measurement.to_csv_record()['phone_key'] == b'phone'.hex()


def test_speaker_first_aggregation_equalizes_unequal_token_counts():
    tokens = [
        *[
            {
                'speaker_id': 'many',
                'gender': 'female',
                'ipa': 'ɑ',
                'f1_hz': 500,
                'f2_hz': 1500,
                'f3_hz': 2500,
                'success': True,
            }
            for _ in range(101)
        ],
        {
            'speaker_id': 'few',
            'gender': 'female',
            'ipa': 'ɑ',
            'f1_hz': 900,
            'f2_hz': 1900,
            'f3_hz': 2900,
            'success': True,
        },
    ]
    speakers = aggregate_speaker_measurements(tokens)
    genders = aggregate_gender_measurements(
        speakers,
        n_bootstrap=100,
        seed=9,
    )

    assert len(speakers) == 2
    assert genders[0]['f1_hz'] == 700
    assert genders[0]['n_speakers'] == 2
    assert genders[0]['n_tokens'] == 102


def test_bootstrap_is_deterministic():
    speakers = [
        {
            'speaker_id': speaker_id,
            'gender': 'male',
            'ipa': 'ɑ',
            'f1_hz': f1_hz,
            'n_tokens': 1,
        }
        for speaker_id, f1_hz in zip(('a', 'b', 'c'), (500, 600, 700))
    ]

    left = aggregate_gender_measurements(
        speakers,
        n_bootstrap=100,
        seed=4,
    )
    right = aggregate_gender_measurements(
        speakers,
        n_bootstrap=100,
        seed=4,
    )

    assert left == right
