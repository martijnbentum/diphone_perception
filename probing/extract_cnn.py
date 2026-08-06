from echoframe.batch_cnn_features import compute_cnn_features_batch

import locations
from probing.model_store import (
    _ensure_model_registered,
    _release_cuda_memory,
    model_store_path,
    open_model_store,
)

default_model_name = 'wav2vec2_nl1_checkpoint-200000'
default_phraser_source_id = 'cgn-awd'
default_collar = 500


def extract_phone_cnn_features(
    phones,
    store,
    model_name=default_model_name,
    collar=default_collar,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute and store CNN frontend features for every phone in `phones`
    into an already-open echoframe Store, via echoframe's
    compute_cnn_features_batch (stores every CNN frame overlapping each
    phone's own span).

    phones:             probing.metadata.Phones - phones.phraser_phones must
                         be complete (raises otherwise)
    store:              open echoframe.Store to write into. CNN features are
                         always written to a per-model store (see
                         extract_phone_cnn_features_for_models), so unlike
                         extract_phone_embeddings there is no store_root
                         convenience path here
    model_name:          registered echoframe model_name; looked up in
                         model_paths_file and registered on first use
    collar:              ms of audio context padded around each phone before
                         running the CNN (does not affect what is stored;
                         only widens the CNN's input window)
    model_paths_file:    JSON file of {model_name, local_path/huggingface_id,
                         language, size} records
    phraser_source_id:   label to register phones.store under in this store
    gpu:                 whether to run the model on GPU
    batch_size:          segments per forward-pass batch. compute_cnn_features_batch
                         loads every segment's audio into one batch when this
                         is left None and gpu=False - an explicit default
                         avoids loading all of phones' audio into memory at once
    tags:                optional tags stored on new metadata
    verbose:             print batch progress
    '''
    _ensure_model_registered(store, model_name, model_paths_file)
    store.attach_phraser_store(phraser_source_id, phones.store)

    segments = phones.phraser_phones
    compute_cnn_features_batch(
        segments, model_name, store,
        collar=collar, gpu=gpu, tags=tags, batch_size=batch_size,
        verbose=verbose,
    )
    return store


def extract_phone_cnn_features_for_models(
    phones,
    model_names,
    collar=default_collar,
    store_root=locations.echoframe_model_cnn_stores,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute phone CNN features in a dedicated store for every model.

    Accepts the extraction options from `extract_phone_cnn_features`,
    replacing `model_name` with `model_names` and managing each model's
    store. Stores are opened below `store_root`, then the cached model is
    unloaded and the store is closed after each extraction. When `gpu` is
    true, unreferenced CUDA allocations are also released before the next
    model is loaded.

    Returns a dictionary mapping each model name to its store path.
    '''
    return _extract_phone_cnn_features_for_models(
        phones,
        model_names,
        collar=collar,
        store_root=store_root,
        model_paths_file=model_paths_file,
        phraser_source_id=phraser_source_id,
        gpu=gpu,
        batch_size=batch_size,
        tags=tags,
        verbose=verbose,
    )


def extract_flemish_phone_cnn_features_for_models(
    flemish_phones,
    model_names,
    collar=default_collar,
    store_root=locations.echoframe_model_cnn_flemish_stores,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute Flemish phone CNN features in a dedicated store per model.

    The model stores are opened below ``store_root``. The validated inventory
    exposed by ``flemish_phones.phraser_phones`` is extracted through the same
    single-model workflow used for the Netherlandic phone inventory. Each
    cached model is unloaded and its store is closed after extraction.

    Returns a dictionary mapping each model name to its store path.
    '''
    return _extract_phone_cnn_features_for_models(
        flemish_phones,
        model_names,
        collar=collar,
        store_root=store_root,
        model_paths_file=model_paths_file,
        phraser_source_id=phraser_source_id,
        gpu=gpu,
        batch_size=batch_size,
        tags=tags,
        verbose=verbose,
    )


def _extract_phone_cnn_features_for_models(
    phones,
    model_names,
    collar,
    store_root,
    model_paths_file,
    phraser_source_id,
    gpu,
    batch_size,
    tags,
    verbose,
):
    '''Run the shared dedicated-store lifecycle for a phone inventory.'''
    if isinstance(model_names, str):
        raise TypeError('model_names must be an iterable, not a string')

    store_paths = {}
    for model_name in model_names:
        store = open_model_store(
            model_name,
            stores_root=store_root,
            model_paths_file=model_paths_file,
        )
        try:
            extract_phone_cnn_features(
                phones,
                store,
                model_name=model_name,
                collar=collar,
                model_paths_file=model_paths_file,
                phraser_source_id=phraser_source_id,
                gpu=gpu,
                batch_size=batch_size,
                tags=tags,
                verbose=verbose,
            )
            store_paths[model_name] = model_store_path(model_name, store_root)
        finally:
            try:
                store.remove_cached_model()
            finally:
                try:
                    store.close()
                finally:
                    if gpu:
                        _release_cuda_memory()
    return store_paths
