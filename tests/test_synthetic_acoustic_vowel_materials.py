import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from synthetic_acoustic_probes import (
    DEFAULT_SOURCE_IDS,
    vowel_anchor_stimuli,
    write_vowel_anchor_materials,
)
from synthetic_acoustic_probes.stimuli import Stimulus
import synthetic_acoustic_probes.vowel_materials as vowel_materials


SOURCE_COUNTS = {
    'cgn_selected_phones': 22,
    'weenink_1985': 24,
    'adank_2004_nsd': 24,
    'adank_2004_ssd': 24,
}


@pytest.fixture
def deterministic_synthesizer(monkeypatch):
    calls = []

    def synthesize(**kwargs):
        calls.append(kwargs.copy())
        n_samples = round(kwargs['duration'] * kwargs['sample_rate'])
        value = (
            kwargs['f0_hz'] + kwargs['f1_hz'] + kwargs['f2_hz']
        ) / 100_000
        waveform = np.full(n_samples, value, dtype=np.float32)
        bandwidth_1, bandwidth_2 = kwargs['bandwidths_hz']
        parameters = {
            'generator': 'test_praat_source_filter',
            'family': 'praat_formants',
            'f0_hz': float(kwargs['f0_hz']),
            'f1_hz': float(kwargs['f1_hz']),
            'f2_hz': float(kwargs['f2_hz']),
            'bandwidth_1_hz': float(bandwidth_1),
            'bandwidth_2_hz': float(bandwidth_2),
            'duration_seconds': float(kwargs['duration']),
            'target_rms': float(kwargs['target_rms']),
            'fade_duration_seconds': 0.01,
        }
        return Stimulus(
            waveform,
            kwargs['sample_rate'],
            parameters,
            kwargs['stimulus_id'],
        )

    monkeypatch.setattr(
        vowel_materials,
        'praat_vowel_stimulus',
        synthesize,
    )
    return calls


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(stimulus):
    parameters = stimulus.parameters
    return parameters['gender'], parameters['ipa']


def test_default_sources_cover_94_unique_source_gender_vowel_anchors(
    deterministic_synthesizer,
):
    assert DEFAULT_SOURCE_IDS == tuple(SOURCE_COUNTS)

    all_stimuli = []
    for source_id, expected_count in SOURCE_COUNTS.items():
        stimuli = vowel_anchor_stimuli(source_id)
        assert len(stimuli) == expected_count
        assert len({_identity(item) for item in stimuli}) == expected_count
        all_stimuli.extend(stimuli)

    assert len(all_stimuli) == 94
    assert len({item.stimulus_id for item in all_stimuli}) == 94


def test_source_inventories_keep_local_and_literature_only_vowels(
    deterministic_synthesizer,
):
    cgn = vowel_anchor_stimuli('cgn_selected_phones')
    cgn_vowels = {item.parameters['ipa'] for item in cgn}
    assert 'ə' in cgn_vowels
    assert not {'yː', 'øː'} & cgn_vowels

    for source_id in DEFAULT_SOURCE_IDS[1:]:
        stimuli = vowel_anchor_stimuli(source_id)
        vowels = {item.parameters['ipa'] for item in stimuli}
        assert {'yː', 'øː'} <= vowels
        assert 'ə' not in vowels


def test_only_source_f0_f1_f2_and_requested_bandwidths_reach_synthesizer(
    deterministic_synthesizer,
):
    stimuli = vowel_anchor_stimuli(
        'cgn_selected_phones',
        bandwidths_hz=(80, 100),
    )

    assert len(deterministic_synthesizer) == 22
    for call, stimulus in zip(deterministic_synthesizer, stimuli):
        assert set(call) == {
            'f0_hz', 'f1_hz', 'f2_hz', 'bandwidths_hz',
            'duration', 'sample_rate', 'target_rms', 'stimulus_id',
        }
        assert call['bandwidths_hz'] == (80, 100)
        assert 'f3_hz' not in call
        assert 'f3_hz' not in stimulus.parameters
        assert stimulus.parameters['source_id'] == 'cgn_selected_phones'
        assert stimulus.parameters['aggregation']
        provenance = stimulus.parameters['anchor_provenance']
        assert provenance['citation']
        assert len(provenance['table_sha256']) == 64


def test_ids_and_waveforms_are_deterministic():
    left = vowel_anchor_stimuli(
        'weenink_1985',
        duration=0.03,
    )
    right = vowel_anchor_stimuli(
        'weenink_1985',
        duration=0.03,
    )

    assert [item.stimulus_id for item in left] == [
        item.stimulus_id for item in right
    ]
    assert all(
        np.array_equal(a.waveform, b.waveform)
        for a, b in zip(left, right)
    )
    assert left[0].stimulus_id == 'vowel_weenink_1985_female_ɪ'


@pytest.mark.parametrize(
    'source_id',
    ('pols_1973_male', 'van_nierop_1973_female'),
)
def test_comparison_only_sources_explain_that_f0_is_unavailable(source_id):
    with pytest.raises(ValueError, match=r'F0 is unavailable.*comparison-only'):
        vowel_anchor_stimuli(source_id)


def test_writer_separates_sources_and_records_round_trip_provenance(
    tmp_path,
    deterministic_synthesizer,
):
    written = write_vowel_anchor_materials(output_root=tmp_path)

    assert set(written) == set(DEFAULT_SOURCE_IDS)
    for source_id, expected_count in SOURCE_COUNTS.items():
        source_root = tmp_path / source_id
        audio_paths = sorted((source_root / 'audio').glob('*.wav'))
        manifest = json.loads(
            (source_root / 'manifest.json').read_text(encoding='utf-8')
        )
        assert len(audio_paths) == expected_count
        assert len(manifest['stimuli']) == expected_count
        assert manifest['source_id'] == source_id
        assert manifest['source_citation']
        assert manifest['population']
        assert len(manifest['anchor_table']['sha256']) == 64
        assert manifest['synthesis_settings']['formants_hz'] == ['F1', 'F2']
        assert manifest['synthesis_settings']['bandwidths_hz'] == [80.0, 100.0]
        assert manifest['software_versions']['python']
        for row in manifest['stimuli']:
            path = source_root / row['path']
            assert path.parent == source_root / 'audio'
            assert row['sha256'] == _sha256(path)

        sample_rate, waveform = wavfile.read(audio_paths[0])
        assert sample_rate == 16_000
        assert waveform.dtype == np.float32

    assert not list(tmp_path.glob('.vowel-formants-*'))


def test_writer_refuses_existing_sources_unless_overwrite_is_enabled(
    tmp_path,
    deterministic_synthesizer,
):
    source_ids = ('cgn_selected_phones',)
    write_vowel_anchor_materials(source_ids, output_root=tmp_path)
    manifest_path = tmp_path / source_ids[0] / 'manifest.json'
    first_checksum = _sha256(manifest_path)

    with pytest.raises(FileExistsError, match='overwrite=True'):
        write_vowel_anchor_materials(source_ids, output_root=tmp_path)
    assert _sha256(manifest_path) == first_checksum

    write_vowel_anchor_materials(
        source_ids,
        output_root=tmp_path,
        overwrite=True,
    )
    assert _sha256(manifest_path) == first_checksum
    assert not list(tmp_path.glob('.vowel-formants-*'))
