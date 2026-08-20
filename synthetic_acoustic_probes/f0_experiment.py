'''Experiment entry points for the pure-tone F0 probe.'''

import echoframe
import numpy as np
from phraser import Store

import locations

from .cnn_extraction import extract_cnn_checkpoints
from .echoframe_store import create_store, make_x_y
from .echoframe_store import select_wav2vec2_nl1_checkpoints
from .phraser_store import add_stimuli, align_phrases_to_manifest
from .phraser_store import load_stimuli
from .stimuli import pure_tone_stimuli
from .storage import read_manifest
from .umap_projection import project_umap


F0_PHRASER_SOURCE_ID = 'f0-pure-tones'


def create_auditory_stimuli():
    '''Generate and save the complete pure-tone F0 stimulus grid.
    Returns the created stimuli.
    '''
    output_root = locations.f0_pure_tone_stimuli
    stimuli = pure_tone_stimuli(save=True, output_root=output_root,
        overwrite=False)
    return stimuli


def create_f0_pure_tone_phraser_store():
    '''Create and fill the experiment-specific Phraser store.
    Uses the stimuli from ``create_auditory_stimuli``. Returns the new
    Phraser store.
    '''
    store = Store(locations.f0_pure_tone_phraser_store)
    add_stimuli(locations.f0_pure_tone_stimuli, store)
    return store


def load_f0_pure_tone_phraser_store():
    '''Open and return the existing experiment-specific Phraser store.
    Returns the store created by ``create_f0_pure_tone_phraser_store``.
    '''
    store_path = locations.f0_pure_tone_phraser_store
    if not store_path.is_dir():
        raise FileNotFoundError(f'F0 Phraser store not found: {store_path}')
    store = Store(store_path)
    return store


def create_f0_echoframe_store():
    '''Create the shared synthetic Echoframe store initialized for F0.
    Registers the complete wav2vec2 checkpoint set and attaches the existing
    F0 Phraser store. Returns the native Echoframe Store.
    '''
    model_names = select_wav2vec2_nl1_checkpoints()
    store_path = locations.f0_echoframe_store
    store = create_store(store_path, model_names)
    _attach_f0_phraser_store(store)
    return store


def load_f0_echoframe_store():
    '''Load the shared Echoframe store and attach the F0 Phraser store.
    Returns the native Echoframe Store.
    '''
    store_path = locations.f0_echoframe_store
    if not store_path.is_dir():
        raise FileNotFoundError(f'Echoframe store not found: {store_path}')
    store = echoframe.Store(store_path)
    _attach_f0_phraser_store(store)
    return store


def extract_f0_cnn_features(echoframe_store, model_names=None, *,
    gpu=False, overwrite=False):
    '''Extract F0 CNN features for registered checkpoints.
    echoframe_store:  Loaded F0 Echoframe Store.
    model_names:      Optional checkpoint iterable; defaults to the complete
                       set.
    gpu:              Whether Echoframe should run models on a GPU.
    overwrite:        Whether Echoframe should replace stored CNN features.
    Extraction always uses zero collar and returns None.
    '''
    if model_names is None:
        model_names = select_wav2vec2_nl1_checkpoints()
    phraser_store = echoframe_store.load_phraser_store(F0_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    extract_cnn_checkpoints(phrases, model_names, echoframe_store, collar=0,
        gpu=gpu, overwrite=overwrite)


def make_f0_x_y(model_name, store, *, aggregation):
    '''Return CNN representations and numeric F0 targets in manifest order.
    model_name:   Registered Echoframe model name.
    store:        Loaded F0 Echoframe Store.
    aggregation:  ``center`` or ``mean`` frame reduction.
    '''
    rows = read_manifest(locations.f0_pure_tone_stimuli)
    phraser_store = store.load_phraser_store(F0_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    ordered_phrases, row_ids, targets = align_phrases_to_manifest(phrases,
        rows, _f0_frequency)
    X, stimulus_ids = make_x_y(ordered_phrases, model_name, store,
        aggregation=aggregation, collar=0)
    if stimulus_ids.tolist() != row_ids:
        raise ValueError('F0 stimulus IDs are not aligned with the manifest')
    y = np.asarray(targets, dtype=float)
    return X, y


def save_f0_checkpoint_result(model_name, store):
    '''Save mean CNN features and their F0 UMAP for one checkpoint.
    model_name:  Registered Echoframe model name.
    store:       Loaded F0 Echoframe Store.
    Existing checkpoint files are skipped. Returns the output path.
    '''
    output_directory = locations.f0_output_data
    output_path = output_directory / f'{model_name}.npz'
    if output_path.exists(): return output_path
    X, y = make_f0_x_y(model_name, store, aggregation='mean')
    coordinates = project_umap(X, metric='cosine', random_state=42)
    output_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as stream:
        np.savez_compressed(stream, mean_cnn_features=np.asarray(X),
            coordinates=coordinates, frequencies=np.asarray(y),
            random_state=42, metric='cosine', model_name=model_name,
            aggregation='mean')
    return output_path


def save_f0_checkpoint_results(store):
    '''Save F0 result bundles for the complete wav2vec2 checkpoint set.
    Existing checkpoint files are skipped. Returns paths grouped under
    ``saved`` and ``skipped``.
    '''
    output_directory = locations.f0_output_data
    saved = []
    skipped = []
    for model_name in select_wav2vec2_nl1_checkpoints():
        output_path = output_directory / f'{model_name}.npz'
        if output_path.exists():
            skipped.append(output_path)
            continue
        path = save_f0_checkpoint_result(model_name, store)
        saved.append(path)
    return {'saved': tuple(saved), 'skipped': tuple(skipped)}


def _attach_f0_phraser_store(store):
    phraser_store = load_f0_pure_tone_phraser_store()
    store.attach_phraser_store(F0_PHRASER_SOURCE_ID, phraser_store)


def _f0_frequency(row):
    parameters = row.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('F0 manifest row has invalid parameters')
    frequencies = parameters.get('frequencies_hz')
    if not isinstance(frequencies, list) or len(frequencies) != 1:
        raise ValueError('F0 manifest row must contain one frequency')
    frequency = frequencies[0]
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
        raise ValueError('F0 manifest frequency must be numeric')
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('F0 manifest frequency must be finite and positive')
    frequency = float(frequency)
    return frequency
