'''Map persisted synthetic-stimulus packages to a Phraser store.'''

from dataclasses import dataclass
from pathlib import Path

from phraser import Audio, Phrase, Speaker
from scipy.io import wavfile

from .storage import read_manifest


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


def align_phrases_to_manifest(phrases, rows, extract_target):
    '''Order phrases by manifest row and extract one target per row.
    phrases:         Native Phraser Phrase objects, each labeled by its
                      manifest stimulus_id.
    rows:            Manifest stimuli rows, e.g. from storage.read_manifest.
    extract_target:  Callable mapping one manifest row to its target value.
    Every row's stimulus_id must match exactly one Phrase label, and every
    Phrase's label must appear in rows. Returns ordered_phrases, row_ids,
    and targets, all aligned to the row order.
    '''
    phrases_by_id = {}
    for phrase in phrases:
        if phrase.label in phrases_by_id:
            raise ValueError(f'duplicate stimulus ID: {phrase.label!r}')
        phrases_by_id[phrase.label] = phrase
    row_ids = []
    ordered_phrases = []
    targets = []
    for row in rows:
        stimulus_id = row['stimulus_id']
        if stimulus_id in row_ids:
            message = f'duplicate manifest stimulus ID: {stimulus_id!r}'
            raise ValueError(message)
        if stimulus_id not in phrases_by_id:
            raise ValueError(f'Phrase not found for {stimulus_id!r}')
        row_ids.append(stimulus_id)
        ordered_phrases.append(phrases_by_id[stimulus_id])
        targets.append(extract_target(row))
    extras = set(phrases_by_id) - set(row_ids)
    if extras:
        extras = sorted(extras)
        raise ValueError(f'Phrases missing from manifest: {extras!r}')
    return tuple(ordered_phrases), row_ids, targets


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
    rows = read_manifest(package_root)
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
