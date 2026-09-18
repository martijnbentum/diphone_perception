'''Extract embeddings for saved random-frame markers.'''

from echoframe.batch_segment_features import compute_embeddings_batch

import locations
import model_store
from decomposition.load_embeddings import default_model_name
from decomposition.load_embeddings import default_phraser_source_id
from decomposition.load_embeddings import load_markers


def extract_marker_embeddings(markers, model_name=default_model_name,
    layers=[9], collar=2000, store=None,
    phraser_source_id=default_phraser_source_id, gpu=False, batch_size=120,
    tags=None, verbose=True):
    '''Extract hidden states and CNN features within each marker's span.

    markers:            iterable of saved Phraser markers
    model_name:         model registered in locations.model_paths_file
    layers:             hidden-state layers; CNN is always included
    collar:             context in milliseconds on each side of each marker
    store:              output store; None opens a model-specific store
    phraser_source_id:  source label for the attached Phraser store
    gpu:                whether to run the model on GPU
    batch_size:         markers per inference batch
    tags:               optional tags for newly stored outputs
    verbose:            print extraction progress

    Assumes a non-empty iterable of markers bound to the same open Phraser
    store, obtained from the first marker.

    Returns an open output store owned by the caller. Existing outputs are
    skipped by Echoframe. Marker timestamps and frame selection are passed
    through unchanged; context is clipped at recording boundaries.
    '''
    markers = list(markers)
    phraser_store = markers[0].store
    if store is None:
        p = locations.decomposition_random_frames_echoframe_model_stores
        store = model_store.open_model_store(model_name, stores_root=p)
    else:
        mpf = locations.model_paths_file
        model_store.ensure_model_registered(store, model_name, mpf)
    store.attach_phraser_store(phraser_source_id, phraser_store)
    requested_layers = list(layers)
    if 'cnn' not in requested_layers: requested_layers.append('cnn')
    compute_embeddings_batch(markers, requested_layers, model_name, store,
        collar=collar, gpu=gpu, tags=tags, batch_size=batch_size,
        verbose=verbose)
    return store


def extract_marker_embeddings_for_models(markers, model_names,
    layers=[9], collar=2000, phraser_source_id=default_phraser_source_id,
    gpu=False, batch_size=120, tags=None, verbose=True):
    '''Extract the same markers into a dedicated decomposition store per model.

    markers:            iterable of saved markers; materialized for reuse
    model_names:        iterable of registered model names, not a string
    layers:             hidden-state layers; CNN is always included
    collar:             context in milliseconds on each side of each marker
    phraser_source_id:  source label for the attached Phraser store
    gpu:                whether to run the model on GPU
    batch_size:         markers per inference batch
    tags:               optional tags for newly stored outputs
    verbose:            print extraction progress

    Assumes non-empty markers bound to the same open Phraser store.
    Uses locations.decomposition_random_frames_echoframe_model_stores.
    Returns model names mapped to store paths. Each model is unloaded and its
    output store closed even on failure; the markers' Phraser store stays open.
    '''
    if isinstance(model_names, str):
        raise TypeError('model_names must be an iterable, not a string')
    markers = list(markers)
    layers = list(layers)
    p = locations.decomposition_random_frames_echoframe_model_stores
    store_paths = {}
    for model_name in model_names:
        store = model_store.open_model_store(model_name, stores_root=p)
        try:
            extract_marker_embeddings(markers,
                model_name=model_name, layers=layers, collar=collar,
                store=store,
                phraser_source_id=phraser_source_id, gpu=gpu,
                batch_size=batch_size, tags=tags, verbose=verbose)
            store_paths[model_name] = store.root
        finally:
            try:
                store.remove_cached_model()
            finally:
                try:
                    store.close()
                finally:
                    if gpu: model_store.release_cuda_memory()
    return store_paths
