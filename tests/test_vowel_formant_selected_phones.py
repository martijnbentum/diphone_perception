from types import SimpleNamespace

import pandas as pd

from synthetic_acoustic_probes.formants import praat_vowel_stimulus
from vowel_formant_reference.aggregation import (
    aggregate_gender_measurements,
    aggregate_speaker_measurements,
)
from vowel_formant_reference.formant_tables import load_formant_table
from vowel_formant_reference.selected_phones import (
    is_monophthong,
    measure_selected_phones,
    select_monophthong_rows,
    write_selected_phone_measurements,
)


def _phone(label, gender, speaker_id, overlap=False):
    return SimpleNamespace(
        phoneme_ipa=label,
        overlap=overlap,
        audio_filename=f'{speaker_id}.wav',
        speaker_id=speaker_id,
        speaker=SimpleNamespace(gender=gender, age=35),
        duration_seconds=0.2,
        start_seconds=0.0,
        end_seconds=0.2,
    )


def _segment(stress):
    return SimpleNamespace(stress=stress, duration=200)


def _audio_loader(_segment):
    stimulus = praat_vowel_stimulus(
        f0_hz=120,
        f1_hz=500,
        f2_hz=1500,
        duration=0.2,
    )
    return stimulus.waveform, stimulus.sample_rate


def test_monophthong_selection_includes_long_mid_vowels_not_diphthongs():
    for label in ('eː', 'øː', 'oː', 'ə', 'ɑ'):
        assert is_monophthong(label)
    for label in ('ɛi', 'œy', 'ɑu', 't'):
        assert not is_monophthong(label)

    data = pd.DataFrame({
        'ipa_phoneme': ['eː', 'ɛi', 'ɑ', 't', 'oː'],
        'overlap': ['False', 'False', 'True', 'False', 'False'],
    })
    selected = select_monophthong_rows(data)
    assert selected['ipa_phoneme'].tolist() == ['eː', 'oː']


def test_selected_phone_measurement_applies_stress_policy_and_keeps_failures():
    phones = [
        _phone('eː', 'male', 'm1'),
        _phone('ə', 'female', 'f1'),
        _phone('ɛi', 'male', 'm1'),
        _phone('ɑ', 'male', 'm1'),
    ]
    segments = [
        _segment('primary'),
        _segment('unstressed'),
        _segment('primary'),
        _segment('unstressed'),
    ]
    data = measure_selected_phones(
        phones,
        segments,
        audio_loader=_audio_loader,
    )
    assert data['ipa'].tolist() == ['eː', 'ə', 'ɑ']
    assert data['success'].tolist() == [True, True, False]
    assert 'requires primary stress' in data.iloc[-1]['rejection_reason']


def test_speaker_first_aggregation_equalizes_unequal_token_counts():
    tokens = pd.DataFrame([
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
    ])
    speakers = aggregate_speaker_measurements(tokens)
    assert len(speakers) == 2
    genders = aggregate_gender_measurements(
        speakers, n_bootstrap=100, seed=9
    )
    assert genders.iloc[0]['f1_hz'] == 700
    assert genders.iloc[0]['n_speakers'] == 2
    assert genders.iloc[0]['n_tokens'] == 102


def test_bootstrap_is_deterministic():
    speakers = pd.DataFrame({
        'speaker_id': ['a', 'b', 'c'],
        'gender': ['male'] * 3,
        'ipa': ['ɑ'] * 3,
        'f1_hz': [500, 600, 700],
        'n_tokens': [1, 1, 1],
    })
    left = aggregate_gender_measurements(
        speakers, n_bootstrap=100, seed=4
    )
    right = aggregate_gender_measurements(
        speakers, n_bootstrap=100, seed=4
    )
    pd.testing.assert_frame_equal(left, right)


def test_selected_tables_round_trip_separately(tmp_path):
    tokens = pd.DataFrame({
        'speaker_id': ['a', 'b'],
        'gender': ['male', 'male'],
        'ipa': ['ɑ', 'ɑ'],
        'f0_hz': [110, 120],
        'f1_hz': [500, 600],
        'f2_hz': [1500, 1600],
        'f3_hz': [2500, 2600],
        'success': [True, True],
    })
    paths = write_selected_phone_measurements(
        tokens,
        data_root=tmp_path,
        n_bootstrap=20,
        seed=2,
    )
    assert paths['tokens'] != paths['speakers'] != paths['genders']
    loaded = load_formant_table(
        'selected_phone_genders',
        data_root=tmp_path,
    )
    assert loaded.source.record_level == 'group_summary'
    assert loaded.data.iloc[0]['n_speakers'] == 2
