'''Extract MFCCs at the exact start frame of each sampled marker.'''

import echoframe
import numpy as np
from echoframe.acoustic_features import make_acoustic_feature_item
from phraser.audio.audio import load_audio_samples
from phraser.audio.batch import mfcc_batch
from phraser.audio.mfcc import DELTA_CONTEXT, FEATURE_DIM
from phraser.audio.mfcc import HOP_SECONDS, WINDOW_SECONDS, _mfcc_matrix

import locations
from decomposition.load_embeddings import default_phraser_source_id


def extract_marker_mfcc(markers, store=None,
    phraser_source_id=default_phraser_source_id, workers=None,
    recordings_per_batch=30, tags=None, verbose=True):
    '''Store one 39-value MFCC row at each marker's exact frame start.

    markers:               saved markers with recording-relative times in ms
    store:                 open Echoframe store; None opens the decomposition
                           MFCC store
    phraser_source_id:     source label for the markers' Phraser store
    workers:               worker count passed to Phraser's mfcc_batch
    recordings_per_batch:  recordings processed in each batch
    tags:                  optional tags for newly stored features
    verbose:               print batch progress

    Phraser computes 13 static MFCCs, 13 deltas, and 13 delta-deltas with
    neighboring recording audio. When its recording grid matches marker.start,
    select that exact frame from a shared batch. Otherwise, compute a short
    MFCC sequence with its grid anchored at marker.start. Store one (1, 39)
    matrix per marker, keyed by its Phraser key. Existing features are skipped.
    Return the open store. The caller closes it before closing Phraser.
    '''
    markers = list(markers)
    if not markers: raise ValueError('markers must not be empty')
    if (isinstance(recordings_per_batch, bool)
            or not isinstance(recordings_per_batch, int)
            or recordings_per_batch < 1):
        raise ValueError('recordings_per_batch must be a positive integer')
    if len({marker.key for marker in markers}) != len(markers):
        raise ValueError('markers must have unique Phraser keys')
    phraser_store = markers[0].store
    if any(marker.store is not phraser_store for marker in markers):
        raise ValueError('all markers must belong to the same Phraser store')
    for marker in markers: _validate_marker(marker)

    owns_store = store is None
    if owns_store:
        root = locations.decomposition_random_frames_echoframe_mfcc_store
        store = echoframe.Store(str(root))
    try:
        store.attach_phraser_store(phraser_source_id, phraser_store)
        batches = _recording_batches(markers, recordings_per_batch)
        for batch_number, batch in enumerate(batches, start=1):
            missing = _missing_markers(batch, store)
            if verbose:
                print(f'MFCC batch {batch_number}: '
                    f'{len(missing)} missing of {len(batch)} markers',
                    flush=True)
            if not missing: continue
            aligned, shifted = [], []
            for marker in missing:
                position = _recording_grid_row(marker)
                if position is None: shifted.append(marker)
                else: aligned.append((marker, position))
            features = {}
            if aligned:
                aligned_markers = [marker for marker, _ in aligned]
                matrices = mfcc_batch(aligned_markers, workers=workers,
                    cache_on_segment=False)
                for (marker, position), matrix in zip(aligned, matrices,
                        strict=True):
                    row, n_rows = position
                    if np.shape(matrix) != (n_rows, FEATURE_DIM):
                        message = 'Phraser MFCC rows do not match frames'
                        raise ValueError(message)
                    features[marker.key] = matrix[row:row + 1].copy()
            for marker in shifted:
                features[marker.key] = _marker_start_mfcc(marker)
            items = []
            for marker in missing:
                key = store.make_echoframe_key('acoustic_feature',
                    feature_name='mfcc', phraser_key=marker.key)
                item = make_acoustic_feature_item(key, 'mfcc',
                    features[marker.key],
                    store, tags=tags, phraser_source_id=phraser_source_id)
                items.append(item)
            store.save_many(items)
    except Exception:
        if owns_store: store.close()
        raise
    return store


def marker_to_mfcc(marker):
    '''Return one (1, 39) MFCC row without storing it.

    marker:  marker with recording-relative start and end times in ms

    The 25 ms target window starts at marker.start. Neighboring 10 ms frames
    are included where audio is available to compute deltas and delta-deltas.
    '''
    _validate_marker(marker)
    return _marker_start_mfcc(marker)


def find_unaligned_markers(markers):
    '''Report and return markers whose starts miss Phraser's recording grid.

    markers:  list of markers with recording-relative starts in ms

    Alignment uses the same sample rounding and 20 ms frame grid as
    extract_marker_mfcc. This checks starts only, without reading audio or
    checking whether a full 25 ms window fits in the recording.
    '''
    markers = list(markers)
    unaligned = []
    for marker in markers:
        start_sample, hop_samples = _marker_grid_position(marker)
        if start_sample % (2 * hop_samples): unaligned.append(marker)
    print(f'Unaligned marker starts: {len(unaligned)} / {len(markers)}')
    return unaligned


def _recording_batches(markers, recordings_per_batch):
    '''Yield marker batches without splitting a recording across batches.'''
    groups = {}
    for marker in markers:
        audio = marker.audio
        key = str(audio.filename), audio.sample_rate, audio.duration
        if key not in groups: groups[key] = []
        groups[key].append(marker)
    recordings = list(groups.values())
    n_recordings = len(recordings)
    for start in range(0, n_recordings, recordings_per_batch):
        selected = recordings[start:start + recordings_per_batch]
        batch = []
        for group in selected:
            batch.extend(group)
        yield batch


def _missing_markers(markers, store):
    '''Return markers without an MFCC payload, checking stored row shapes.'''
    keys = []
    for marker in markers:
        key = store.make_echoframe_key('acoustic_feature',
            feature_name='mfcc', phraser_key=marker.key)
        keys.append(key)
    metadatas = store.load_many_metadata(keys, keep_missing=True)
    missing = []
    for marker, metadata in zip(markers, metadatas, strict=True):
        if metadata is None:
            missing.append(marker)
        elif tuple(metadata.shape) != (1, FEATURE_DIM):
            raise ValueError('stored marker MFCC must have shape (1, 39)')
    return missing


def _validate_marker(marker):
    '''Require enough recording audio for a window beginning at marker.start.'''
    audio = marker.audio
    sample_rate = audio.sample_rate
    if sample_rate <= 0 or marker.start < 0 or marker.end <= marker.start:
        raise ValueError('marker must have valid audio and positive timing')
    if marker.end > audio.duration:
        raise ValueError('marker must fit within its audio recording')
    window_samples = round(WINDOW_SECONDS * sample_rate)
    hop_samples = round(HOP_SECONDS * sample_rate)
    if window_samples < 1 or hop_samples < 1:
        raise ValueError('audio sample rate is too low for the MFCC grid')
    start_sample = round(marker.start / 1000 * sample_rate)
    audio_samples = round(audio.duration / 1000 * sample_rate)
    if start_sample + window_samples > audio_samples:
        raise ValueError('no complete MFCC window starts at marker.start')


def _recording_grid_row(marker):
    '''Return exact row and output length when Phraser's grid already aligns.'''
    audio = marker.audio
    sample_rate = audio.sample_rate
    window_samples = round(WINDOW_SECONDS * sample_rate)
    start_sample, hop_samples = _marker_grid_position(marker)
    if start_sample % (2 * hop_samples): return None
    end_sample = round(marker.end / 1000 * sample_rate)
    audio_samples = round(audio.duration / 1000 * sample_rate)
    last_complete = (audio_samples - window_samples) // hop_samples
    first = max(0, (start_sample - window_samples) // hop_samples + 1)
    if first % 2: first += 1
    last = min(last_complete, (end_sample - 1) // hop_samples)
    indices = range(first, last + 1, 2)
    target = start_sample // hop_samples
    if target not in indices: return None
    return (target - first) // 2, len(indices)


def _marker_grid_position(marker):
    '''Return the start and Phraser hop in recording sample coordinates.'''
    sample_rate = marker.audio.sample_rate
    if sample_rate <= 0 or marker.start < 0:
        raise ValueError('marker must have valid audio and start time')
    hop_samples = round(HOP_SECONDS * sample_rate)
    if hop_samples < 1:
        raise ValueError('audio sample rate is too low for the MFCC grid')
    start_sample = round(marker.start / 1000 * sample_rate)
    return start_sample, hop_samples


def _marker_start_mfcc(marker):
    '''Compute one MFCC row on a grid anchored to marker.start.'''
    audio = marker.audio
    sample_rate = audio.sample_rate
    start_sample = round(marker.start / 1000 * sample_rate)
    audio_samples = round(audio.duration / 1000 * sample_rate)
    window_samples = round(WINDOW_SECONDS * sample_rate)
    hop_samples = round(HOP_SECONDS * sample_rate)
    before = min(DELTA_CONTEXT, start_sample // hop_samples)
    remaining = audio_samples - start_sample - window_samples
    after = min(DELTA_CONTEXT, remaining // hop_samples)
    crop_start = start_sample - before * hop_samples
    crop_stop = start_sample + window_samples + after * hop_samples
    signal, loaded_rate = load_audio_samples(audio.filename,
        start_sample=crop_start, stop_sample=crop_stop)
    if loaded_rate != sample_rate or len(signal) != crop_stop - crop_start:
        raise ValueError('audio samples do not match marker metadata')
    matrix = _mfcc_matrix(signal, sample_rate)
    expected_shape = FEATURE_DIM, before + 1 + after
    if matrix.shape != expected_shape:
        raise ValueError('anchored MFCC frames do not match selected audio')
    return matrix[:, before][None, :].copy()
