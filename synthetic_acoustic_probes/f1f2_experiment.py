'''Experiment entry points for the F1/F2 formant-grid probe.'''

import echoframe
import numpy as np
from phraser import Store

import locations

from .cnn_extraction import extract_cnn_checkpoints
from .echoframe_store import create_store, make_x_y
from .echoframe_store import select_wav2vec2_nl1_checkpoints
from .phraser_store import add_stimuli, align_phrases_to_manifest
from .phraser_store import load_stimuli
from .stimuli import sinusoidal_component_formant_stimuli
from .storage import read_manifest
from .umap_projection import project_umap


F1F2_PHRASER_SOURCE_ID = 'f1f2-formant-grid'


def create_auditory_stimuli():
    '''Generate and save the complete F1/F2 formant-grid stimulus set.
    Returns the created stimuli.
    '''
    output_root = locations.f1f2_stimuli
    stimuli = sinusoidal_component_formant_stimuli(save=True,
        output_root=output_root, overwrite=False)
    return stimuli


def create_f1f2_phraser_store():
    '''Create and fill the experiment-specific Phraser store.
    Uses the stimuli from ``create_auditory_stimuli``. Returns the new
    Phraser store.
    '''
    store = Store(locations.f1f2_phraser_store)
    add_stimuli(locations.f1f2_stimuli, store)
    return store


def load_f1f2_phraser_store():
    '''Open and return the existing experiment-specific Phraser store.
    Returns the store created by ``create_f1f2_phraser_store``.
    '''
    store_path = locations.f1f2_phraser_store
    if not store_path.is_dir():
        message = f'F1/F2 Phraser store not found: {store_path}'
        raise FileNotFoundError(message)
    store = Store(store_path)
    return store


def create_f1f2_echoframe_store():
    '''Create the shared synthetic Echoframe store initialized for F1/F2.
    Registers the complete wav2vec2 checkpoint set and attaches the existing
    F1/F2 Phraser store. Returns the native Echoframe Store.
    '''
    model_names = select_wav2vec2_nl1_checkpoints()
    store_path = locations.f1f2_echoframe_store
    store = create_store(store_path, model_names)
    _attach_f1f2_phraser_store(store)
    return store


def load_f1f2_echoframe_store():
    '''Load the shared Echoframe store and attach the F1/F2 Phraser store.
    Returns the native Echoframe Store.
    '''
    store_path = locations.f1f2_echoframe_store
    if not store_path.is_dir():
        raise FileNotFoundError(f'Echoframe store not found: {store_path}')
    store = echoframe.Store(store_path)
    _attach_f1f2_phraser_store(store)
    return store


def extract_f1f2_cnn_features(echoframe_store, model_names=None, *,
    gpu=False, overwrite=False):
    '''Extract F1/F2 CNN features for registered checkpoints.
    echoframe_store:  Loaded F1/F2 Echoframe Store.
    model_names:      Optional checkpoint iterable; defaults to the complete
                       set.
    gpu:              Whether Echoframe should run models on a GPU.
    overwrite:        Whether Echoframe should replace stored CNN features.
    Extraction always uses zero collar and returns None.
    '''
    if model_names is None:
        model_names = select_wav2vec2_nl1_checkpoints()
    phraser_store = echoframe_store.load_phraser_store(
        F1F2_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    extract_cnn_checkpoints(phrases, model_names, echoframe_store, collar=0,
        gpu=gpu, overwrite=overwrite)


def make_f1f2_x_y(model_name, store):
    '''Return CNN representations and F1/F2 targets in manifest order.
    model_name:  Registered Echoframe model name.
    store:       Loaded F1/F2 Echoframe Store.
    Always aggregates frames by their mean. Returns X, f1_hz, and f2_hz.
    '''
    rows = read_manifest(locations.f1f2_stimuli)
    phraser_store = store.load_phraser_store(F1F2_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    ordered_phrases, row_ids, targets = align_phrases_to_manifest(phrases,
        rows, _f1f2_targets)
    X, stimulus_ids = make_x_y(ordered_phrases, model_name, store,
        aggregation='mean', collar=0)
    if stimulus_ids.tolist() != row_ids:
        message = 'F1/F2 stimulus IDs are not aligned with the manifest'
        raise ValueError(message)
    f1_hz = np.asarray([target[0] for target in targets], dtype=float)
    f2_hz = np.asarray([target[1] for target in targets], dtype=float)
    return X, f1_hz, f2_hz


def save_f1f2_checkpoint_result(model_name, store):
    '''Save mean CNN features and their F1/F2 UMAP for one checkpoint.
    model_name:  Registered Echoframe model name.
    store:       Loaded F1/F2 Echoframe Store.
    Existing checkpoint files are skipped. Returns the output path.
    '''
    output_directory = locations.f1f2_output_data
    output_path = output_directory / f'{model_name}.npz'
    if output_path.exists(): return output_path
    X, f1_hz, f2_hz = make_f1f2_x_y(model_name, store)
    coordinates = project_umap(X, metric='cosine', random_state=42)
    output_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as stream:
        np.savez_compressed(stream, cnn=np.asarray(X),
            umap_coordinates=coordinates, umap_metric='cosine',
            umap_random_state=42, model_name=model_name,
            f1_hz=f1_hz, f2_hz=f2_hz)
    return output_path


def save_f1f2_checkpoint_results(store):
    '''Save F1/F2 result bundles for the complete wav2vec2 checkpoint set.
    Existing checkpoint files are skipped. Returns paths grouped under
    ``saved`` and ``skipped``.
    '''
    output_directory = locations.f1f2_output_data
    saved = []
    skipped = []
    for model_name in select_wav2vec2_nl1_checkpoints():
        output_path = output_directory / f'{model_name}.npz'
        if output_path.exists():
            skipped.append(output_path)
            continue
        path = save_f1f2_checkpoint_result(model_name, store)
        saved.append(path)
    return {'saved': tuple(saved), 'skipped': tuple(skipped)}


def _attach_f1f2_phraser_store(store):
    phraser_store = load_f1f2_phraser_store()
    store.attach_phraser_store(F1F2_PHRASER_SOURCE_ID, phraser_store)


def _f1f2_targets(row):
    parameters = row.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('F1/F2 manifest row has invalid parameters')
    f1_hz, f2_hz = parameters.get('f1_hz'), parameters.get('f2_hz')
    for name, value in (('f1_hz', f1_hz), ('f2_hz', f2_hz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'F1/F2 manifest {name} must be numeric')
        if not np.isfinite(value) or value <= 0:
            message = f'F1/F2 manifest {name} must be finite and positive'
            raise ValueError(message)
    return float(f1_hz), float(f2_hz)
