'''Sample distinct frames from Netherlandic CGN components k and o.'''

import argparse
import bisect
import itertools
import json
import random
from pathlib import Path

from frame import Frames, make_frames_from_duration
import locations
from phraser import Store


DEFAULT_SAMPLE_COUNT = 418_500


def filter_audios_on_component(audios, components=None, region='nl'):
    '''Describe eligible recordings without loading audio or phone trees.

    audios:      iterable of unique Phraser Audio objects
    components:  exact CGN component path names to include
    '''
    selected_audios= []
    if components is None: components = ('comp-k', 'comp-o')
    for audio in audios:
        parts = set(Path(audio.filename).parts)
        matches = parts.intersection(components)
        if region not in parts or not matches: continue
        selected_audios.append(audio)
    return selected_audios


def sample_frames(audios, n_samples=None, components=None, region = 'nl'):
    '''Sample uniformly without replacement over all eligible frame slots.

    audios:     iterable of Phraser Audio objects, normally store.audios
    n_samples:  total samples across fitting and evaluation recordings
    seed:       seed controlling recording splits and frame selection
    '''
    if n_samples is None: n_samples = DEFAULT_SAMPLE_COUNT
    if components is None: components = ('comp-k', 'comp-o')
    selected_audios= filter_audios_on_component(audios, components)
    audio_infos = [audio_to_info(audio) for audio in selected_audios]
    eligible = [row for row in audios if row['n_frames']]
    n_frames = [row['n_frames'] for row in eligible]
    cumulative = list(itertools.accumulate(n_frames))
    total = cumulative[-1] if cumulative else 0
    if n_samples > total:
        message = f'requested {n_samples:,} distinct frames, '
        message += f'but only {total:,} are eligible'
        raise ValueError(message)
    _assign_recording_splits(eligible, seed)
    rng = random.Random(42)
    selected = rng.sample(range(total), n_samples)
    for global_index in sorted(selected):
        index = bisect.bisect_right(cumulative, global_index)
        offset = cumulative[index - 1] if index else 0
        eligible[index]['frame_indices'].append(global_index - offset)
    manifest = {'n_samples': n_samples,
        'region': 'nl', 'components': components,
        'sampling_unit': 'uniform_unique_frame', 'include_all_audio': True,
        'split_policy': 'half_recordings_per_component_seeded_shuffle',
        'recordings': recordings}
    manifest['summary'] = summarize_sample(manifest)
    return manifest


def summarize_sample(manifest):
    '''Report inventory and sampled counts by component and split.'''
    recordings = manifest['recordings']
    summary = {'n_recordings': len(recordings),
        'n_short_recordings': sum(row['n_frames'] == 0 for row in recordings),
        'n_eligible_frames': sum(row['n_frames'] for row in recordings),
        'components': {}, 'splits': {}}
    group_fields = (('component', 'components'), ('split', 'splits'))
    for field, destination in group_fields:
        for row in recordings:
            if row[field] is None: continue
            group = summary[destination].setdefault(row[field],
                {'n_recordings': 0, 'duration_ms': 0,
                    'n_eligible_frames': 0, 'n_samples': 0})
            group['n_recordings'] += 1
            group['duration_ms'] += row['duration_ms']
            group['n_eligible_frames'] += row['n_frames']
            group['n_samples'] += len(row['frame_indices'])
    return summary


def iter_samples(manifest):
    '''Yield sample identities and recording-relative frame timestamps.

    manifest:  dictionary returned by sample_frames or loaded from JSON
    '''
    for recording in manifest['recordings']:
        if not recording['frame_indices']: continue
        frames = Frames(recording['n_frames'])
        for frame_index in recording['frame_indices']:
            selected = frames[frame_index]
            start = round(selected.start_time * 1000, 6)
            end = round(selected.end_time * 1000, 6)
            audio_key = recording['audio_key']
            sample_id = f'{audio_key}:{frame_index}'
            yield {'sample_id': sample_id,
                'audio_key': recording['audio_key'],
                'audio_id': recording['audio_id'],
                'filename': recording['filename'],
                'component': recording['component'],
                'split': recording['split'], 'frame_index': frame_index,
                'start_ms': start, 'end_ms': end,
                'center_ms': (start + end) / 2}


def save_manifest(manifest, path):
    '''Write a new manifest, refusing to replace an existing selection.

    manifest:  sample manifest, including inventory and selected indices
    path:      destination JSON file
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False)
        handle.write('\n')


def make_manifest(cgn_store, output_filename = None, n_samples = None):
    '''Select CGN frames from an existing Phraser store and save a manifest.'''
    if output_filename == None: 
        output_filename = locations.decomposition_random_frames
    if n_samples == None: n_samples = DEFAULT_SAMPLE_COUNT
    manifest = sample_frames(cgn_store.audios, n_samples, seed=42)
    manifest['phraser_store'] = str(args.store)
    save_manifest(manifest, args.output)
    summary_text = json.dumps(manifest['summary'], indent=2)
    print(summary_text)
    n_samples = manifest['n_samples']
    print(f'Saved {n_samples:,} samples to {args.output}')

def load_cgn_store():
    store = Store(locations.cgn_lmdb)
    return store

def audio_to_info(audio, component):
    '''Read stable identity, timing, and frame capacity from one audio.'''
    duration = audio.duration
    frames = None
    frames = make_frames_from_duration(duration / 1000)
    n_frames = len(frames) if frames is not None else 0
    return {'audio_key': audio.key.hex(),
        'audio_id': audio.identifier.hex(), 'filename': str(audio.filename),
        'component': component, 'duration_ms': duration,
        'n_frames': n_frames, 'split': None, 'frame_indices': []}


def split_data(recordings, seed):
    '''Assign whole recordings before sampling, independently of sample size.'''
    rng = random.Random(f'{seed}:recording-split')
    for component in ('comp-k', 'comp-o'):
        group = [row for row in recordings if row['component'] == component]
        if not group: continue
        if len(group) < 2:
            message = f'{component} needs at least two eligible recordings '
            message += 'for separate fitting and evaluation sets'
            raise ValueError(message)
        rng.shuffle(group)
        midpoint = len(group) // 2
        for index, row in enumerate(group):
            row['split'] = 'fitting' if index < midpoint else 'evaluation'


if __name__ == '__main__':
    main()
