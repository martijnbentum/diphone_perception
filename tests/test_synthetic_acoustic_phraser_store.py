import json

from phraser import Phrase, Store
import pytest

from synthetic_acoustic_probes import pure_tone_stimuli, write_stimuli
from synthetic_acoustic_probes.phraser_store import add_stimuli, load_stimuli
from synthetic_acoustic_probes.stimuli import sum_of_sinusoids


@pytest.fixture
def phraser_store(tmp_path):
    '''Return an open temporary Phraser store and close it after each test.'''

    store = Store(tmp_path / 'phraser')
    yield store
    store.close()


def test_add_stimuli_creates_phraser_objects(tmp_path, phraser_store):
    '''A package produces one Audio and full-duration Phrase per stimulus.

    tmp_path:  Temporary directory supplied by pytest.
    phraser_store:  Empty temporary Phraser store.
    '''

    package = _pure_tone_package(tmp_path)

    result = add_stimuli(package, phraser_store)

    assert result is None
    assert len(phraser_store.speakers) == 1
    assert len(phraser_store.audios) == 2
    assert len(phraser_store.phrases) == 2
    for audio in phraser_store.audios:
        assert audio.sample_rate == 16_000
        assert audio.duration == 10
        assert audio.n_channels == 1
        assert audio.dataset == package.name
    for phrase in phraser_store.phrases:
        assert phrase.start == 0
        assert phrase.end == 10


def test_add_stimuli_assigns_labels(tmp_path, phraser_store):
    '''Every Phrase label is its manifest stimulus ID.

    tmp_path:  Temporary directory supplied by pytest.
    phraser_store:  Empty temporary Phraser store.
    '''

    pure_tone = pure_tone_stimuli(frequencies=(10,), duration=0.01)[0]
    other = sum_of_sinusoids(
        20,
        duration=0.01,
        stimulus_id='other-stimulus',
        extra_parameters={'family': 'other'},
    )
    package = tmp_path / 'labels'
    write_stimuli((pure_tone, other), package)

    add_stimuli(package, phraser_store)

    labels = {phrase.label for phrase in phraser_store.phrases}
    assert labels == {'pure-tone_f-10', 'other-stimulus'}


def test_load_stimuli_returns_native_phrases(tmp_path, phraser_store):
    '''Loading returns every native Phrase held by the experiment store.

    tmp_path:  Temporary directory supplied by pytest.
    phraser_store:  Empty temporary Phraser store.
    '''

    package = _pure_tone_package(tmp_path)
    add_stimuli(package, phraser_store)

    phrases = load_stimuli(phraser_store)

    assert isinstance(phrases, tuple)
    assert all(isinstance(phrase, Phrase) for phrase in phrases)
    expected_keys = {phrase.key for phrase in phraser_store.phrases}
    assert {phrase.key for phrase in phrases} == expected_keys


@pytest.mark.parametrize(
    ('failure', 'error', 'match'),
    [
        ('missing-manifest', FileNotFoundError, 'manifest not found'),
        ('missing-wav', FileNotFoundError, 'WAV not found'),
        ('duplicate-id', ValueError, 'duplicate stimulus ID'),
        ('sample-rate', ValueError, 'sample rate does not match'),
        ('sample-count', ValueError, 'sample count does not match'),
        ('non-empty-store', FileExistsError, 'requires an empty'),
    ],
)
def test_add_stimuli_rejects_invalid_input(
    tmp_path,
    phraser_store,
    failure,
    error,
    match,
):
    '''Invalid packages and non-empty stores fail without another addition.

    tmp_path:  Temporary directory supplied by pytest.
    phraser_store:  Empty temporary Phraser store.
    failure:  Name of the invalid-input mutation.
    error:  Exception type expected from the mutation.
    match:  Text expected in the exception message.
    '''

    package = _pure_tone_package(tmp_path)
    manifest_path = package / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)

    if failure == 'missing-manifest': manifest_path.unlink()
    elif failure == 'missing-wav':
        audio_path = package / manifest['stimuli'][0]['path']
        audio_path.unlink()
    elif failure == 'duplicate-id':
        manifest['stimuli'][1]['stimulus_id'] = (
            manifest['stimuli'][0]['stimulus_id']
        )
        _write_manifest(manifest_path, manifest)
    elif failure == 'sample-rate':
        manifest['stimuli'][0]['sample_rate_hz'] += 1
        _write_manifest(manifest_path, manifest)
    elif failure == 'sample-count':
        manifest['stimuli'][0]['n_samples'] += 1
        _write_manifest(manifest_path, manifest)
    elif failure == 'non-empty-store':
        add_stimuli(package, phraser_store)

    with pytest.raises(error, match=match):
        add_stimuli(package, phraser_store)


def _pure_tone_package(tmp_path):
    package = tmp_path / 'f0_pure_tones'
    stimuli = pure_tone_stimuli(frequencies=(10, 20), duration=0.01)
    write_stimuli(stimuli, package)
    return package


def _write_manifest(path, manifest):
    text = json.dumps(manifest, indent=2) + '\n'
    path.write_text(text, encoding='utf-8')
