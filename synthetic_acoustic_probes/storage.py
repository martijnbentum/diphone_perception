'''Persist generated stimuli as WAV packages with JSON manifests.'''

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import scipy
from scipy.io import wavfile

from .stimuli import Stimulus


def write_stimuli(stimuli, output_root, *, overwrite=False):
    '''Write stimuli as float32 WAV files with a provenance manifest.
    stimuli:  Iterable of ``Stimulus`` objects to persist in input order.
    output_root:  Destination package directory.
    overwrite:  Replace an existing package only when explicitly enabled.
    The package is assembled in a staging directory and moved into place only
    after every WAV file and the manifest have been written. Existing output
    is preserved unless ``overwrite`` is true.
    '''
    stimuli = _validated_stimuli(stimuli)
    output_root = _validated_output_root(Path(output_root), overwrite)
    staging_path = tempfile.mkdtemp(prefix=f'.{output_root.name}-staging-',
        dir=output_root.parent)
    staging_root = Path(staging_path)
    try:
        _write_package(staging_root, stimuli)
        _commit_package(staging_root, output_root, overwrite)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return output_root


def _validated_stimuli(stimuli):
    '''Validate stimuli as a non-empty tuple of unique Stimulus objects.'''
    if isinstance(stimuli, Stimulus):
        raise TypeError('stimuli must be an iterable of Stimulus objects')
    try:
        stimuli = tuple(stimuli)
    except TypeError as error:
        message = 'stimuli must be an iterable of Stimulus objects'
        raise TypeError(message) from error
    if not stimuli: raise ValueError('at least one stimulus is required')
    identifiers = []
    for index, stimulus in enumerate(stimuli):
        if not isinstance(stimulus, Stimulus):
            message = f'stimuli[{index}] is not a Stimulus object'
            raise TypeError(message)
        _validate_stimulus_id(stimulus.stimulus_id)
        identifiers.append(stimulus.stimulus_id)
    seen = set()
    duplicates = set()
    for identifier in identifiers:
        if identifier in seen: duplicates.add(identifier)
        seen.add(identifier)
    duplicates = sorted(duplicates)
    if duplicates: raise ValueError(f'duplicate stimulus IDs: {duplicates!r}')
    return stimuli


def _validate_stimulus_id(stimulus_id):
    '''Raise if stimulus_id is not a safe, non-empty filename component.'''
    if not isinstance(stimulus_id, str) or not stimulus_id:
        message = 'stimulus_id must be a non-empty string'
        raise ValueError(message)
    bad_chars = ('/', '\\', '\x00')
    if stimulus_id in {'.', '..'} or any(c in stimulus_id for c in bad_chars):
        message = 'stimulus_id is not a safe filename component: '
        message += repr(stimulus_id)
        raise ValueError(message)


def _validated_output_root(output_root, overwrite):
    '''Validate output_root and ensure its parent directory exists.'''
    if not output_root.name:
        message = 'output_root must name a package directory'
        raise ValueError(message)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(output_root) and not overwrite:
        message = f'refusing to replace existing output {output_root}; '
        message += 'pass overwrite=True to replace it'
        raise FileExistsError(message)
    return output_root


def _write_package(package_root, stimuli):
    '''Write every stimulus WAV and the package manifest under package_root.'''
    audio_root = package_root / 'audio'
    audio_root.mkdir()
    rows = []
    for stimulus in stimuli:
        relative_path = Path('audio') / f'{stimulus.stimulus_id}.wav'
        path = package_root / relative_path
        waveform = stimulus.waveform.astype(np.float32, copy=False)
        wavfile.write(path, stimulus.sample_rate, waveform)
        row = _manifest_row(stimulus, relative_path, path, waveform)
        rows.append(row)
    software_versions = {'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__}
    manifest = {'schema_version': 1, 'stimulus_count': len(stimuli),
        'audio_format': {'container': 'WAV', 'sample_format': 'float32'},
        'software_versions': software_versions, 'stimuli': rows}
    _write_json(package_root / 'manifest.json', manifest)


def _manifest_row(stimulus, relative_path, path, waveform):
    '''Return one manifest entry for a persisted stimulus WAV.'''
    row = {'stimulus_id': stimulus.stimulus_id,
        'path': relative_path.as_posix(), 'sha256': _file_sha256(path),
        'sample_rate_hz': stimulus.sample_rate,
        'dtype': str(waveform.dtype), 'n_samples': int(waveform.size),
        'duration_seconds': waveform.size / stimulus.sample_rate,
        'parameters': _json_value(stimulus.parameters)}
    return row


def _commit_package(staging_root, output_root, overwrite):
    '''Atomically replace output_root with the staged package.'''
    backup_path = tempfile.mkdtemp(prefix=f'.{output_root.name}-backup-',
        dir=output_root.parent)
    backup_root = Path(backup_path)
    backup = backup_root / output_root.name
    had_output = _path_exists(output_root)
    try:
        if had_output:
            message = f'refusing to replace existing output {output_root}; '
            message += 'pass overwrite=True to replace it'
            if not overwrite: raise FileExistsError(message)
            os.replace(output_root, backup)
        try:
            os.replace(staging_root, output_root)
        except Exception:
            if had_output: os.replace(backup, output_root)
            raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _write_json(path, value):
    '''Write value as indented JSON with a trailing newline.'''
    text = json.dumps(value, indent=2, ensure_ascii=False,
        allow_nan=False) + '\n'
    path.write_text(text, encoding='utf-8')


def _json_value(value):
    '''Recursively convert value into JSON-serializable native types.'''
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                message = 'stimulus parameter names must be strings'
                raise TypeError(message)
            output[key] = _json_value(item)
        return output
    if isinstance(value, np.ndarray): return _json_value(value.tolist())
    if isinstance(value, np.generic): return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        output = []
        for item in value: output.append(_json_value(item))
        return output
    if value is None or isinstance(value, (str, bool, int, float)): return value
    message = 'stimulus parameter value is not JSON-compatible: '
    message += repr(value)
    raise TypeError(message)


def _file_sha256(path):
    '''Return the SHA-256 hex digest of a file's contents.'''
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _path_exists(path):
    '''Return whether path exists, including broken symlinks.'''
    return path.exists() or path.is_symlink()
