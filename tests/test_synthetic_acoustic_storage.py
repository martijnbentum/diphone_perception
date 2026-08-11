import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from synthetic_acoustic_probes import pure_tone_stimuli, write_stimuli
from synthetic_acoustic_probes.stimuli import sum_of_sinusoids
import synthetic_acoustic_probes.storage as storage
import synthetic_acoustic_probes.stimuli as stimuli_module


def test_write_stimuli_writes_audio_and_manifest(tmp_path):
    '''Audio and manifest data round-trip through one package.'''

    stimuli = [_stimulus(10, 'tone-10'), _stimulus(20, 'tone-20')]
    output_root = tmp_path / 'tones'

    stimulus_iterator = iter(stimuli)
    written = write_stimuli(stimulus_iterator, output_root)

    assert written == output_root
    manifest_path = output_root / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    assert manifest['schema_version'] == 1
    assert manifest['stimulus_count'] == 2
    expected_format = {'container': 'WAV', 'sample_format': 'float32'}
    assert manifest['audio_format'] == expected_format
    assert [row['stimulus_id'] for row in manifest['stimuli']] == [
        'tone-10', 'tone-20'
    ]

    for stimulus, row in zip(stimuli, manifest['stimuli']):
        path = output_root / row['path']
        sample_rate, waveform = wavfile.read(path)
        assert sample_rate == stimulus.sample_rate
        assert waveform.dtype == np.float32
        assert np.array_equal(waveform, stimulus.waveform)
        assert row['n_samples'] == stimulus.waveform.size
        assert row['sha256'] == _sha256(path)
        assert row['parameters']['test_frequency_hz'] == (
            stimulus.parameters['test_frequency_hz']
        )


def test_existing_output_requires_explicit_overwrite(tmp_path):
    '''Existing packages require opt-in replacement and remain whole.'''

    output_root = tmp_path / 'tones'
    write_stimuli([_stimulus(10, 'original')], output_root)
    original = _file_snapshot(output_root)

    with pytest.raises(FileExistsError, match='overwrite=True'):
        write_stimuli([_stimulus(20, 'replacement')], output_root)

    assert _file_snapshot(output_root) == original
    write_stimuli(
        [_stimulus(20, 'replacement')],
        output_root,
        overwrite=True,
    )
    manifest_path = output_root / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    assert [row['stimulus_id'] for row in manifest['stimuli']] == [
        'replacement'
    ]
    assert not (output_root / 'audio' / 'original.wav').exists()


def test_invalid_stimulus_collections_are_rejected(tmp_path):
    '''Invalid collections fail before a destination is created.'''

    valid = _stimulus(10, 'valid')
    unsafe = _stimulus(20, '../escape')

    with pytest.raises(ValueError, match='at least one'):
        write_stimuli([], tmp_path / 'empty')
    with pytest.raises(TypeError, match='not a Stimulus'):
        write_stimuli([object()], tmp_path / 'wrong-type')
    with pytest.raises(ValueError, match='duplicate stimulus IDs'):
        write_stimuli([valid, valid], tmp_path / 'duplicates')
    with pytest.raises(ValueError, match='safe filename'):
        write_stimuli([unsafe], tmp_path / 'unsafe')


def test_failed_write_preserves_output_and_cleans_staging(
    tmp_path,
    monkeypatch,
):
    '''A partial staging failure leaves no output fragments behind.

    tmp_path:  Temporary output root supplied by pytest.
    monkeypatch:  Pytest fixture used to simulate the WAV failure.
    '''

    output_root = tmp_path / 'tones'
    write_stimuli([_stimulus(10, 'original')], output_root)
    original = _file_snapshot(output_root)
    real_write = storage.wavfile.write
    call_count = 0

    def fail_on_second_write(filename, rate, data):
        nonlocal call_count
        call_count += 1
        if call_count == 2: raise RuntimeError('simulated WAV write failure')
        return real_write(filename, rate, data)

    monkeypatch.setattr(storage.wavfile, 'write', fail_on_second_write)
    replacements = [
        _stimulus(20, 'replacement-20'),
        _stimulus(30, 'replacement-30'),
    ]
    with pytest.raises(RuntimeError, match='simulated WAV write failure'):
        write_stimuli(replacements, output_root, overwrite=True)

    assert _file_snapshot(output_root) == original
    assert not list(tmp_path.glob('.tones-staging-*'))
    assert not list(tmp_path.glob('.tones-backup-*'))


def test_pure_tone_save_flag_writes_default_package(tmp_path, monkeypatch):
    '''The convenience flag delegates to storage and still returns stimuli.

    tmp_path:  Temporary output root supplied by pytest.
    monkeypatch:  Pytest fixture used to replace the default data path.
    '''

    output_root = tmp_path / 'f0_pure_tones'
    monkeypatch.setattr(
        stimuli_module,
        '_DEFAULT_PURE_TONE_OUTPUT_ROOT',
        output_root,
    )

    stimuli = pure_tone_stimuli(
        frequencies=(10, 20),
        duration=0.01,
        save=True,
    )

    assert len(stimuli) == 2
    manifest_path = output_root / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    assert manifest['stimulus_count'] == 2
    assert [row['stimulus_id'] for row in manifest['stimuli']] == [
        'pure-tone_f-10', 'pure-tone_f-20'
    ]


def _stimulus(frequency, stimulus_id):
    return sum_of_sinusoids(
        frequency,
        amplitudes=0.2,
        duration=0.01,
        stimulus_id=stimulus_id,
        extra_parameters={'test_frequency_hz': np.int64(frequency)},
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_snapshot(root):
    snapshot = {}
    for path in root.rglob('*'):
        if path.is_file():
            relative_path = path.relative_to(root).as_posix()
            snapshot[relative_path] = path.read_bytes()
    return snapshot
