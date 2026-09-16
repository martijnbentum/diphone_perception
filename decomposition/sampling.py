'''Sample distinct frames from Netherlandic CGN components k and o.'''

import argparse
import bisect
import itertools
import json
import random
from pathlib import Path
from progressbar import progressbar

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
    comps = '_'.join([x.split('-')[-1] for x in components])
    output_filename = f / f'region-{region}_comps-{comps}.json'
    if output_filename.exists():
        raise FileExistsError(f'manifest already exists at {output_filename}. '
            'delete it first if you want to replace it.')
    sa = filter_audios_on_component(cgn_store.audios, components, region)
    audio_infos= sample_frames(sa, n_samples)
    manifest = {'n_samples': n_samples, 'region': region,
        'components': components,'sampling_unit': 'uniform_unique_frame',
        'include_all_audio': True,'audio_infos': audio_infos,
        'phraser_store_path' : str(cgn_store.path)}
    save_manifest(manifest, output_filename)
    print(f'manifest saved to {output_filename}')
    return manifest


def filter_audios_on_component(audios, components=None, region='nl',
    min_duration_ms=1000):
    '''Describe eligible recordings without loading audio or phone trees.

    audios:      iterable of unique Phraser Audio objects
    components:  exact CGN component path names to include
    '''
    selected_audios= []
    if components is None: components = ('comp-k', 'comp-o')
    for audio in audios:
        if audio.duration < min_duration_ms: continue
        parts = set(Path(audio.filename).parts)
        matches = parts.intersection(components)
        if region not in parts or not matches: continue
        selected_audios.append(audio)
    return selected_audios


def assign_fit_and_eval_splits(audio_infos, seed=42):
    '''Assign fitting/evaluation per recording, within each component.

    Updates audio_infos in place and returns it. Odd counts give the extra
    recording to evaluation.
    '''
    groups = {}
    for info in audio_infos:
        groups.setdefault(info['component'], []).append(info)

    rng = random.Random(seed)
    for component in sorted(groups):
        rows = sorted(groups[component],
            key=lambda row: row['audio_key'])
        rng.shuffle(rows)
        midpoint = len(rows) // 2
        for index, row in enumerate(rows):
            row['split'] = (
                'fitting' if index < midpoint else 'evaluation')
    return audio_infos


def sample_frames(audios, n_samples=None):
    '''Sample uniformly without replacement over all eligible frame slots.

    audios:     iterable of Phraser Audio objects, normally store.audios
    n_samples:  total samples across fitting and evaluation recordings
    '''
    if n_samples is None: n_samples = DEFAULT_SAMPLE_COUNT
    audio_infos = []
    print('making audio infos...')
    for audio in progressbar(audios):
        audio_infos.append(audio_to_info(audio))
    audio_infos.sort(key=lambda info: info['audio_key'])
    assign_fit_and_eval_splits(audio_infos)
    n_frames = [row['n_frames'] for row in audio_infos]
    cumulative = list(itertools.accumulate(n_frames))
    total = cumulative[-1] if cumulative else 0
    rng = random.Random(42)
    selected = rng.sample(range(total), n_samples)
    print('sampling frames...')
    for global_index in progressbar(sorted(selected)):
        index = bisect.bisect_right(cumulative, global_index)
        offset = cumulative[index - 1] if index else 0
        audio_infos[index]['frame_indices'].append(global_index - offset)
    return audio_infos


def save_manifest(manifest, path):
    '''write a new manifest, refusing to replace an existing selection.

    manifest:  sample manifest, including inventory and selected indices
    path:      destination json file
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False)
        handle.write('\n')

def load_manifest(region = 'nl', components = ('comp-k', 'comp-o')):
    f = locations.decomposition_random_frames_base
    comps = '_'.join([x.split('-')[-1] for x in components])
    input_filename = f / f'region-{region}_comps-{comps}.json'
    if not input_filename.exists():
        m = f'no manifest found at {input_filename}. please run:\n' 
        m += f'cgn_store = load_cgn_store()\n make_manifest('
        m += f'cgn_store, region="{region}", components={components})\n first.'
        raise FileNotFoundError(m)
    with input_filename.open('r', encoding='utf-8') as handle:
        d = json.load(handle)
    return d

def iter_marker_info(manifest, label = 'decomp_random_frames', collar_ms= 2000):
    '''Yield sample identities and recording-relative frame timestamps.

    manifest:  dictionary returned by sample_frames or loaded from JSON
    '''
    for info in manifest['audio_infos']:
        frames = Frames(info['n_frames'])
        duration = info['duration_ms'] 
        for frame_index in info['frame_indices']:
            selected = frames[frame_index]
            start, end = selected.start_time/1000, selected.end_time/1000
            audio_key = info['audio_key']
            sample_id = f'{audio_key}:{frame_index}'
            d = {'audio_key': info['audio_key'],'start': start, 'end': end,
                'label': label}
            yield d


def audio_to_info(audio):
    '''read stable identity, timing, and frame capacity from one audio.'''
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
