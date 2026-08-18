'''Store and load synthetic-stimulus CNN features with Echoframe.'''

import json
from pathlib import Path
import re

import echoframe
import numpy as np

import locations


def create_store(store_path, model_names):
    '''Create an Echoframe store and register its initial models.

    store_path:   Destination for the new Echoframe store.
    model_names:  Iterable of model names to register.

    Existing paths are rejected. Returns the native Echoframe Store.
    '''
    store_path = Path(store_path)
    if store_path.exists():
        raise FileExistsError(f'Echoframe store path exists: {store_path}')
    model_names = _model_names(model_names)
    entries = _model_entries(model_names, locations.model_paths_file)
    store = echoframe.Store(store_path)
    try: _register_models(model_names, entries, store)
    except Exception:
        store.close()
        raise
    return store


def load_cnn_features(phrases, model_name, store, *, collar=0):
    '''Load native Echoframe CNN features for Phraser Phrases.

    phrases:     Iterable of native Phraser Phrase objects.
    model_name:  Registered Echoframe model name.
    store:       Echoframe Store containing the CNN features.
    collar:      Context in milliseconds used during extraction.

    Results follow the order of ``phrases``.
    '''
    phraser_keys = [phrase.key for phrase in phrases]
    return store.phraser_keys_to_cnn_features(phraser_keys, model_name,
        collar=collar)


def make_x_y(phrases, model_name, store, *, aggregation, collar=0):
    '''Return aggregated CNN representations and aligned stimulus IDs.

    phrases:      Iterable of native Phraser Phrase objects.
    model_name:   Registered Echoframe model name.
    store:        Echoframe Store containing the CNN features.
    aggregation:  ``center`` selects the middle frame; ``mean`` averages
                  frames over each Phrase.
    collar:       Context in milliseconds used during extraction.

    ``y`` contains Phrase labels. Synthetic Phraser stores define those labels
    as manifest stimulus IDs, for joining experiment-specific targets rather
    than treating them as the final modeling target.
    '''
    phrases = tuple(phrases)
    if not phrases: raise ValueError('phrases must not be empty')
    if aggregation not in {'center', 'mean'}:
        raise ValueError("aggregation must be 'center' or 'mean'")

    features = load_cnn_features(phrases, model_name, store, collar=collar)
    expected_keys = tuple(phrase.key for phrase in phrases)
    if features.phraser_keys != expected_keys:
        message = 'CNN features are missing or not aligned with phrases'
        raise ValueError(message)

    method = 'middle' if aggregation == 'center' else 'mean'
    vectors = []
    pairs = zip(phrases, features.cnn_features, strict=True)
    for phrase, feature in pairs:
        vector = feature.aggregate_segment(phrase, method=method)
        vectors.append(vector)
    X = np.stack(vectors)
    y = np.asarray([phrase.label for phrase in phrases])
    return X, y


def select_wav2vec2_nl1_checkpoints():
    '''Return the validated checkpoint set used by the probe experiments.

    The random checkpoint is returned first, followed by exactly 121 NL1
    checkpoints ordered by numeric training step.
    '''
    trained_checkpoint_count = 121
    path, catalog = _model_catalog(locations.model_paths_file)
    random_name = locations.wav2vec2_random_checkpoint_name
    pattern = re.compile(locations.wav2vec2_nl1_checkpoint_pattern)
    random_count = 0
    trained = []
    for entry in catalog:
        model_name = entry.get('model_name')
        if model_name == random_name: random_count += 1
        if not isinstance(model_name, str): continue
        match = pattern.fullmatch(model_name)
        if match is None: continue
        trained.append((int(match.group(1)), model_name))

    if random_count != 1:
        message = f'expected one {random_name!r} in {path}, found '
        message += str(random_count)
        raise ValueError(message)
    if len(trained) != trained_checkpoint_count:
        message = f'expected {trained_checkpoint_count} NL1 checkpoints '
        message += f'in {path}, found {len(trained)}'
        raise ValueError(message)
    steps = [step for step, _ in trained]
    if len(set(steps)) != len(steps):
        raise ValueError(f'duplicate NL1 checkpoint steps in {path}')

    trained.sort(key=lambda item: item[0])
    trained_names = tuple(model_name for _, model_name in trained)
    return (random_name, *trained_names)


def _register_models(model_names, entries, store):
    '''Register each model, raising if any name is already registered.'''
    existing = []
    for model_name in model_names:
        metadata = store.load_model_metadata(model_name)
        if metadata is not None: existing.append(model_name)
    if existing:
        names = ', '.join(repr(name) for name in existing)
        raise ValueError(f'models already registered in store: {names}')

    for model_name in model_names:
        entry = entries[model_name]
        local_path = entry.get('local_path')
        huggingface_id = entry.get('huggingface_id')
        language = entry.get('language')
        size = entry.get('size')
        architecture = entry.get('architecture')
        store.register_model(model_name, local_path=local_path,
            huggingface_id=huggingface_id, language=language, size=size,
            architecture=architecture)


def _model_names(model_names):
    '''Validate model_names as a non-empty iterable of unique strings.'''
    if isinstance(model_names, str):
        raise TypeError('model_names must be an iterable of strings')
    try: names = tuple(model_names)
    except TypeError as error:
        message = 'model_names must be an iterable of strings'
        raise TypeError(message) from error
    if not names: raise ValueError('model_names must not be empty')

    seen = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('model names must be non-empty strings')
        if name in seen: raise ValueError(f'duplicate model name: {name!r}')
        seen.add(name)
    return names


def _model_entries(model_names, model_paths_file):
    '''Return one catalog entry per requested model name.'''
    path, catalog = _model_catalog(model_paths_file)
    matches = {name: [] for name in model_names}
    for entry in catalog:
        model_name = entry.get('model_name')
        if model_name in matches: matches[model_name].append(entry)

    selected = {}
    for model_name in model_names:
        entries = matches[model_name]
        if not entries:
            message = f'{model_name!r} not found in {path}'
            raise ValueError(message)
        if len(entries) > 1:
            message = f'multiple entries for {model_name!r} in {path}'
            raise ValueError(message)
        selected[model_name] = entries[0]
    return selected


def _model_catalog(model_paths_file):
    '''Load and validate the model-paths JSON catalog.'''
    path = Path(model_paths_file)
    try:
        text = path.read_text(encoding='utf-8')
        catalog = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid JSON in {path}: {error}') from error
    if not isinstance(catalog, list):
        raise ValueError(f'model paths file must contain a JSON list: {path}')
    for entry in catalog:
        if not isinstance(entry, dict):
            raise ValueError(f'model paths file contains a non-object: {path}')
    return path, catalog
