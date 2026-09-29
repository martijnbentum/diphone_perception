'''Extract marker acoustic measurements by recording into SQLite.'''

import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile
from phraser.audio.mfcc import WINDOW_SECONDS

import locations
from decomposition.audio.database import _prepare_database
from decomposition.audio.frequency_band import BANDS_HZ, frequency_band_power
from decomposition.audio.intensity import praat_intensity


def extract_marker_acoustics(markers, database=None, workers=16,
    pitch_floor=100, overwrite=False, verbose=True):
    '''Extract intensity and four band powers for saved markers.

    markers:      iterable of saved markers, with unique Phraser keys
    database:     SQLite path; None uses the decomposition acoustics database
    workers:      concurrent recording readers and acoustic computations
    pitch_floor:  Praat intensity floor in Hz; sets its analysis window
    overwrite:    recompute all rows, including after an implementation change
    verbose:      print recording and marker progress with an estimated time

    Read each recording once in a worker process. The parent alone writes
    SQLite. Existing markers are skipped when settings match. Changed settings
    or overwrite=True clear old rows before extraction. Return the database
    path. A missing full intensity window is stored as SQL NULL.
    '''
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not np.isfinite(pitch_floor) or pitch_floor <= 0:
        raise ValueError('pitch_floor must be positive')
    groups = _group_markers(markers)
    if database is None:
        database = locations.decomposition_random_frames_acoustics_db
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        'pitch_floor': float(pitch_floor),
        'frame_seconds': WINDOW_SECONDS,
        'bands_hz': BANDS_HZ,
        'intensity_method': 'praat_kaiser20_centered',
        'intensity_reference': 2e-5,
        'intensity_subtract_mean': True,
        'intensity_channels': 'mean_channel_power',
        'band_method': 'hann_parseval_mean_square',
        'band_channels': 'mono_channel_mean',
    }
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            _prepare_database(connection, settings, overwrite)
        existing_rows = connection.execute(
            'SELECT marker_key FROM marker_acoustics')
        existing = {row[0] for row in existing_rows}
        tasks = []
        for filename, (sample_rate, entries) in groups.items():
            missing = [entry for entry in entries if entry[0] not in existing]
            if missing: tasks.append((filename, sample_rate, missing,
                pitch_floor))
        if not tasks:
            if verbose: print('Acoustics: all markers already stored')
            return database
    _process_tasks(tasks, database, workers, verbose)
    return database


def _group_markers(markers):
    '''Group primitive marker information by recording filename.'''
    groups = {}
    keys = set()
    for marker in markers:
        key = bytes(marker.key)
        if key in keys: raise ValueError('markers must have unique keys')
        keys.add(key)
        audio = marker.audio
        sample_rate = audio.sample_rate
        if (not np.isfinite(sample_rate) or sample_rate <= 0
                or not np.isfinite(marker.start)
                or not np.isfinite(marker.end)
                or marker.start < 0 or marker.end <= marker.start
                or marker.end > audio.duration):
            raise ValueError('marker must have valid audio and timing')
        filename = str(audio.filename)
        if filename not in groups: groups[filename] = (sample_rate, [])
        group_rate, entries = groups[filename]
        if group_rate != sample_rate:
            raise ValueError('recording has inconsistent sample rates')
        entries.append((key, marker.start))
    if not groups: raise ValueError('markers must not be empty')
    return groups


def _extract_recording(task):
    '''Compute rows for one recording, reading its waveform only once.'''
    filename, sample_rate, entries, pitch_floor = task
    signal, loaded_rate = soundfile.read(filename, dtype='float32',
        always_2d=True)
    if loaded_rate != sample_rate:
        raise ValueError(f'audio sample rate changed for {filename}')
    mono = signal[:, 0] if signal.shape[1] == 1 else signal.mean(axis=1)
    window_samples = round(WINDOW_SECONDS * sample_rate)
    rows = []
    for key, start_ms in entries:
        start = round(start_ms / 1000 * sample_rate)
        stop = start + window_samples
        if start < 0 or stop > len(signal):
            raise ValueError(f'no complete marker frame in {filename}')
        center = start + window_samples // 2
        intensity = praat_intensity(signal, sample_rate, center,
            pitch_floor=pitch_floor)
        if np.isnan(intensity): intensity = None
        power = frequency_band_power(mono[start:stop], sample_rate)
        rows.append((key, intensity, *(float(value) for value in power)))
    return rows


def _process_tasks(tasks, database, workers, verbose):
    '''Run bounded recording tasks and write completed rows in the parent.'''
    total_markers = sum(len(task[2]) for task in tasks)
    completed_markers = completed_recordings = 0
    started = time.monotonic()
    task_iter = iter(tasks)
    context = get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=context) as executor:
        pending = {}
        initial_tasks = min(2 * workers, len(tasks))
        for _ in range(initial_tasks):
            task = next(task_iter)
            pending[executor.submit(_extract_recording, task)] = task
        with closing(sqlite3.connect(database)) as connection:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    rows = future.result()
                    with connection:
                        connection.executemany('''INSERT INTO marker_acoustics
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(marker_key) DO NOTHING''', rows)
                    completed_markers += len(rows)
                    completed_recordings += 1
                    if verbose and (completed_recordings % 100 == 0
                            or completed_recordings == len(tasks)):
                        elapsed = time.monotonic() - started
                        remaining = elapsed * (len(tasks) /
                            completed_recordings - 1)
                        print(f'Acoustics: {completed_recordings}/{len(tasks)} '
                            f'recordings, {completed_markers}/{total_markers} '
                            f'markers, ETA {remaining / 60:.1f} min',
                            flush=True)
                    task = next(task_iter, None)
                    if task is not None:
                        future = executor.submit(_extract_recording, task)
                        pending[future] = task
