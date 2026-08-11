'''Experiment entry points for the pure-tone F0 probe.'''

import json
from pathlib import Path

import echoframe
import numpy as np
from phraser import Store

import locations

from .cnn_extraction import extract_cnn_checkpoints
from .echoframe_store import create_store, make_x_y
from .echoframe_store import select_wav2vec2_nl1_checkpoints
from .phraser_store import add_stimuli, load_stimuli
from .stimuli import pure_tone_stimuli


F0_PHRASER_SOURCE_ID = 'f0-pure-tones'


def create_auditory_stimuli(output_root=locations.f0_pure_tone_stimuli,
    overwrite=False):
    '''Generate and save the complete pure-tone F0 stimulus grid.

    output_root:  Destination package directory.
    overwrite:  Replace an existing stimulus package when true.
    '''

    return pure_tone_stimuli(
        save=True,
        output_root=output_root,
        overwrite=overwrite,
    )


def create_f0_pure_tone_phraser_store(
    stimulus_package=locations.f0_pure_tone_stimuli,
    store_path=locations.f0_pure_tone_phraser_store,
):
    '''Create and fill the experiment-specific Phraser store.

    stimulus_package:  Package created by ``create_auditory_stimuli``.
    store_path:  Destination of the dedicated Phraser store.
    '''

    store = Store(store_path)
    try: add_stimuli(stimulus_package, store)
    except Exception:
        store.close()
        raise
    return store


def load_f0_pure_tone_phraser_store(
    store_path=locations.f0_pure_tone_phraser_store,
):
    '''Open and return the existing experiment-specific Phraser store.

    store_path:  Path created by ``create_f0_pure_tone_phraser_store``.
    '''

    store_path = Path(store_path)
    if not store_path.is_dir():
        raise FileNotFoundError(f'F0 Phraser store not found: {store_path}')
    return Store(store_path)


def create_f0_echoframe_store():
    '''Create the shared synthetic Echoframe store initialized for F0.

    Registers the complete wav2vec2 checkpoint set and attaches the existing
    F0 Phraser store. Returns the native Echoframe Store.
    '''

    model_names = select_wav2vec2_nl1_checkpoints()
    store = create_store(
        locations.synthetic_acoustic_probes_echoframe_store,
        model_names,
    )
    try: _attach_f0_phraser_store(store)
    except Exception:
        store.close()
        raise
    return store


def load_f0_echoframe_store():
    '''Load the shared Echoframe store and attach the F0 Phraser store.'''

    store_path = locations.synthetic_acoustic_probes_echoframe_store
    if not store_path.is_dir():
        raise FileNotFoundError(f'Echoframe store not found: {store_path}')
    store = echoframe.Store(store_path)
    try: _attach_f0_phraser_store(store)
    except Exception:
        store.close()
        raise
    return store


def extract_f0_cnn_features(
    store,
    model_names=None,
    *,
    gpu=False,
    overwrite=False,
):
    '''Extract F0 CNN features for registered checkpoints.

    store:        Loaded F0 Echoframe Store.
    model_names:  Optional checkpoint iterable; defaults to the complete set.
    gpu:          Whether Echoframe should run models on a GPU.
    overwrite:    Whether Echoframe should replace stored CNN features.

    Extraction always uses zero collar and returns None.
    '''

    if model_names is None:
        model_names = select_wav2vec2_nl1_checkpoints()
    phraser_store = store.load_phraser_store(F0_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    extract_cnn_checkpoints(
        phrases,
        model_names,
        store,
        collar=0,
        gpu=gpu,
        overwrite=overwrite,
    )


def make_f0_x_y(model_name, store, *, aggregation):
    '''Return CNN representations and numeric F0 targets in manifest order.

    model_name:   Registered Echoframe model name.
    store:        Loaded F0 Echoframe Store.
    aggregation:  ``center`` or ``mean`` frame reduction.
    '''

    rows = _f0_manifest_rows()
    phraser_store = store.load_phraser_store(F0_PHRASER_SOURCE_ID)
    phrases = load_stimuli(phraser_store)
    phrases_by_id = {}
    for phrase in phrases:
        if phrase.label in phrases_by_id:
            raise ValueError(f'duplicate F0 stimulus ID: {phrase.label!r}')
        phrases_by_id[phrase.label] = phrase

    row_ids = []
    ordered_phrases = []
    targets = []
    for row in rows:
        stimulus_id = row['stimulus_id']
        if stimulus_id in row_ids:
            raise ValueError(f'duplicate manifest stimulus ID: {stimulus_id!r}')
        if stimulus_id not in phrases_by_id:
            raise ValueError(f'F0 Phrase not found for {stimulus_id!r}')
        row_ids.append(stimulus_id)
        ordered_phrases.append(phrases_by_id[stimulus_id])
        targets.append(_f0_frequency(row))
    extras = set(phrases_by_id) - set(row_ids)
    if extras:
        extras = sorted(extras)
        raise ValueError(f'F0 Phrases missing from manifest: {extras!r}')

    X, stimulus_ids = make_x_y(
        ordered_phrases,
        model_name,
        store,
        aggregation=aggregation,
        collar=0,
    )
    if stimulus_ids.tolist() != row_ids:
        raise ValueError('F0 stimulus IDs are not aligned with the manifest')
    y = np.asarray(targets, dtype=float)
    return X, y


def _attach_f0_phraser_store(store):
    phraser_store = load_f0_pure_tone_phraser_store()
    try: store.attach_phraser_store(F0_PHRASER_SOURCE_ID, phraser_store)
    except Exception:
        phraser_store.close()
        raise


def _f0_manifest_rows():
    manifest_path = locations.f0_pure_tone_stimuli / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'F0 manifest not found: {manifest_path}')
    try:
        text = manifest_path.read_text(encoding='utf-8')
        manifest = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid F0 manifest: {error}') from error
    rows = manifest.get('stimuli')
    if manifest.get('schema_version') != 1 or not isinstance(rows, list):
        raise ValueError(f'invalid F0 manifest: {manifest_path}')
    if manifest.get('stimulus_count') != len(rows):
        raise ValueError('F0 manifest stimulus_count does not match rows')
    return tuple(rows)


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
    return float(frequency)
