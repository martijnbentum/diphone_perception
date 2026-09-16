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

def make_manifest(cgn_store, components = None, region = None, n_samples = None):
    '''Select CGN frames from an existing Phraser store and save a manifest.'''
    if n_samples == None: n_samples = DEFAULT_SAMPLE_COUNT
    if components == None: components = ('comp-k', 'comp-o')
    if region == None: region = 'nl'
    f = locations.decomposition_random_frames_base
    f.mkdir(parents=True, exist_ok=True)
    comps = '_'.join([x.split('-') for x in components])
    output_filename = f / f'region-{region}_comps-{comps}.json'
    selected_audios = filter_audios_on_component(cgn_store.audios, components)
    audio_infos= sample_frames(selected_audios, n_samples)
    manifest = {'n_samples': n_samples, 'region': region,
        'components': components,'sampling_unit': 'uniform_unique_frame',
        'include_all_audio': True,'audio_infos': audio_infos
        'phraser_store' : str(args.store)}
    save_manifest(manifest, output_filename)
    print(f'manifest saved to {output_filename}')


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


def sample_frames(audios, n_samples=None):
    '''Sample uniformly without replacement over all eligible frame slots.

    audios:     iterable of Phraser Audio objects, normally store.audios
    n_samples:  total samples across fitting and evaluation recordings
    seed:       seed controlling recording splits and frame selection
    '''
    if n_samples is None: n_samples = DEFAULT_SAMPLE_COUNT
    audio_infos = [audio_to_info(audio) for audio in selected_audios]
    n_frames = [row['n_frames'] for row in audio_infos]
    cumulative = list(itertools.accumulate(n_frames))
    total = cumulative[-1] if cumulative else 0
    rng = random.Random(42)
    selected = rng.sample(range(total), n_samples)
    for global_index in sorted(selected):
        index = bisect.bisect_right(cumulative, global_index)
        offset = cumulative[index - 1] if index else 0
        audio_infos[index]['frame_indices'].append(global_index - offset)
    return audio_infos

def iter_samples(manifest, collar = 2):
    '''Yield sample identities and recording-relative frame timestamps.

    manifest:  dictionary returned by sample_frames or loaded from JSON
    '''
    for info in manifest['audio_infos']:
        frames = Frames(info['n_frames'])
        for frame_index in info['frame_indices']:
            selected = frames[frame_index]
            audio_key = recording['audio_key']
            sample_id = f'{audio_key}:{frame_index}'
            d = {'sample_id': sample_id, 'audio_key': recording['audio_key'],
                'filename': recording['filename'],
                'component': recording['component'],
                'split': recording['split'], 'frame_index': frame_index,
                'start_second': selected.start,
                'collar_start_second': max(0, selected.start - collar),}
                'collar_end_second': selected.end + collar}
            yield d

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


def audio_to_info(audio):
    '''Read stable identity, timing, and frame capacity from one audio.'''
    duration = audio.duration
    component = audio_to_component(audio)
    frames = make_frames_from_duration(duration / 1000)
    n_frames = len(frames) 
    return {'audio_key': audio.key.hex(), 'filename': str(audio.filename),
        'component': component, 'duration_ms': duration,
        'n_frames': n_frames, 'split': None, 'frame_indices': []}

def audio_to_component(audio):
    filename = Path(audio.filename)
    parts = set(filename.parts)
    component = [x for x in parts if x.startswith('comp-')][0]
    return component

def load_cgn_store():
    store = Store(locations.cgn_lmdb)
    return store


