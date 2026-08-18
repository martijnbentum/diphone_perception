'''Map persisted synthetic-stimulus packages to a Phraser store.'''

from dataclasses import dataclass
import json
from pathlib import Path

from phraser import Audio, Phrase, Speaker
from scipy.io import wavfile


def add_stimuli(stimulus_dir, store):
    '''Add a stored stimulus package to a Phraser store.
    stimulus_dir:  Directory written by ``write_stimuli``.
    store:  Empty Phraser Store dedicated to this experiment.
    Adds one Audio and one full-duration Phrase per manifest row. Returns
    None; call ``load_stimuli`` to retrieve the persisted entries.
    '''
    package_root, stimuli = _load_manifest(stimulus_dir)
    store.refresh_query_roots()
    existing_audios = list(store.audios)
    existing_phrases = list(store.phrases)
    if existing_audios or existing_phrases:
        message = 'add_stimuli requires an empty experiment Phraser store'
        raise FileExistsError(message)
    speaker = store.create(Speaker, name='synthetic-stimuli')
    dataset = package_root.name
    objects = [speaker]
    audios = []
    for stimulus in stimuli:
        audio, phrase = _add_stimulus(stimulus, dataset, speaker, store)
        objects.extend((audio, phrase))
        audios.append(audio)
    store.save_many(objects)
    for audio in audios:
        speaker.add_audio(audio)
    store.refresh_query_roots()


def load_stimuli(store):
    '''Load every native Phraser Phrase from an experiment store.
    store:  Phraser Store dedicated to one synthetic-stimulus experiment.
    '''
    store.refresh_query_roots()
    return tuple(store.phrases)


def _add_stimulus(stimulus, dataset, speaker, store):
    '''Stage one Audio and full-duration Phrase for the bulk save.'''
    filename = str(stimulus.audio_path)
    audio = store.create(Audio, filename=filename,
        sample_rate=stimulus.sample_rate, duration=stimulus.duration_ms,
        n_channels=1, dataset=dataset)
    phrase = store.create(Phrase, label=stimulus.stimulus_id, start=0,
        end=stimulus.duration_ms, audio_id=audio.identifier,
        speaker_id=speaker.identifier)
    return audio, phrase


def _load_manifest(stimulus_dir):
    package_root = Path(stimulus_dir).expanduser().resolve()
    manifest_path = package_root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'stimulus manifest not found: {manifest_path}')
    manifest = _read_json(manifest_path)
    rows = manifest.get('stimuli')
    if manifest.get('schema_version') != 1 or not isinstance(rows, list):
        raise ValueError(f'invalid stimulus manifest: {manifest_path}')
    if manifest.get('stimulus_count') != len(rows):
        raise ValueError('stimulus_count does not match the manifest rows')
    seen = set()
    stimuli = []
    for row in rows:
        stimulus = _manifest_stimulus(row, package_root)
        if stimulus.stimulus_id in seen:
            message = f'duplicate stimulus ID: {stimulus.stimulus_id!r}'
            raise ValueError(message)
        seen.add(stimulus.stimulus_id)
        stimuli.append(stimulus)
    if not stimuli: raise ValueError('stimulus manifest contains no stimuli')
    return package_root, tuple(stimuli)


@dataclass(frozen=True)
class _ManifestStimulus:
    stimulus_id: str
    audio_path: Path
    sample_rate: int
    duration_ms: int


def _manifest_stimulus(row, package_root):
    stimulus_id = row['stimulus_id']
    audio_path = (package_root / row['path']).resolve()
    try: audio_path.relative_to(package_root)
    except ValueError as error:
        message = f'stimulus path escapes package: {audio_path}'
        raise ValueError(message) from error
    if not audio_path.is_file():
        raise FileNotFoundError(f'stimulus WAV not found: {audio_path}')
    sample_rate, waveform = wavfile.read(audio_path, mmap=True)
    if waveform.ndim != 1:
        raise ValueError(f'stimulus WAV must be mono: {audio_path}')
    if sample_rate != row['sample_rate_hz']:
        raise ValueError(f'sample rate does not match manifest: {audio_path}')
    if len(waveform) != row['n_samples']:
        raise ValueError(f'sample count does not match manifest: {audio_path}')
    duration_ms = round(len(waveform) / sample_rate * 1000)
    return _ManifestStimulus(stimulus_id, audio_path, sample_rate,
        duration_ms)


def _read_json(path):
    try:
        text = path.read_text(encoding='utf-8')
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid JSON in {path}: {error}') from error
