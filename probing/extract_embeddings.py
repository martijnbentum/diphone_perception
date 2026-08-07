import echoframe
from echoframe.batch_embeddings_cnn_features import (
    compute_embeddings_and_cnn_features_batch,
)

import locations
from probing import model_store

default_model_name = 'wav2vec2_nl1_checkpoint-200000'
default_phraser_source_id = 'cgn-awd'


def extract_phone_embeddings(
    phones,
    model_name=default_model_name,
    layers=[9],
    collar=2000,
    store=None,
    store_root=locations.echoframe_store,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute and store wav2vec2 hidden-state embeddings and CNN frontend
    features for every phone in `phones` into an echoframe Store, via
    compute_embeddings_and_cnn_features_batch (stores every frame overlapping
    each phone's own span, for each requested layer, plus one CNN frame per
    phone). Routing is per-segment: a phone missing its hidden_state runs a
    full forward pass, which incidentally produces the CNN features as a
    byproduct, so CNN is stored "for free" alongside it; a phone that already
    has its hidden_state but is missing CNN falls back to the cheap CNN-only
    path instead of rerunning the full model.

    phones:             probing.metadata.Phones - phones.phraser_phones must
                         be complete (raises otherwise)
    model_name:          registered echoframe model_name; looked up in
                         model_paths_file and registered on first use
    layers:              list of hidden_state layer indices to store
    collar:              ms of audio context padded around each phone before
                         running the model (does not affect what is stored;
                         only widens the model's input window)
    store:               existing echoframe.Store to write into; if None,
                         one is opened at store_root
    store_root:          path for a new echoframe.Store, used only when
                         store is None
    model_paths_file:    JSON file of {model_name, local_path/huggingface_id,
                         language, size} records
    phraser_source_id:   label to register phones.store under in this store
    gpu:                 whether to run the model on GPU
    batch_size:          segments per forward-pass batch.
                         compute_embeddings_and_cnn_features_batch loads every
                         segment's audio into one batch when this
                         is left None and gpu=False - an explicit default
                         avoids loading all of phones' audio into memory at once
    tags:                optional tags stored on new metadata
    verbose:             print batch progress
    '''
    if store is None:
        store = echoframe.Store(str(store_root))
    model_store.ensure_model_registered(store, model_name, model_paths_file)
    store.attach_phraser_store(phraser_source_id, phones.store)

    segments = phones.phraser_phones
    compute_embeddings_and_cnn_features_batch(
        segments, layers, model_name, store,
        collar=collar, gpu=gpu, tags=tags, batch_size=batch_size,
        verbose=verbose,
    )
    return store


def extract_phone_embeddings_for_models(
    phones,
    model_names,
    layers=[9],
    collar=2000,
    store_root=locations.echoframe_model_stores,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute phone embeddings and CNN features in a dedicated store for
    every model.

    Accepts the extraction options from `extract_phone_embeddings`, replacing
    `model_name` with `model_names` and managing each model's store. Stores are
    opened below `store_root`, then the cached model is unloaded and the store
    is closed after each extraction. When `gpu` is true, unreferenced CUDA
    allocations are also released before the next model is loaded.

    Returns a dictionary mapping each model name to its store path.
    '''
    return _extract_phone_embeddings_for_models(
        phones,
        model_names,
        layers=layers,
        collar=collar,
        store_root=store_root,
        model_paths_file=model_paths_file,
        phraser_source_id=phraser_source_id,
        gpu=gpu,
        batch_size=batch_size,
        tags=tags,
        verbose=verbose,
    )


def extract_flemish_phone_embeddings_for_models(
    flemish_phones,
    model_names,
    layers=[9],
    collar=2000,
    store_root=locations.echoframe_model_flemish_stores,
    model_paths_file=locations.model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=120,
    tags=None,
    verbose=True,
):
    '''Compute Flemish phone embeddings and CNN features in a dedicated store
    per model.

    The model stores are opened below ``store_root``. The validated inventory
    exposed by ``flemish_phones.phraser_phones`` is extracted through the same
    single-model workflow used for the Netherlandic phone inventory. Each
    cached model is unloaded and its store is closed after extraction.

    Returns a dictionary mapping each model name to its store path.
    '''
    return _extract_phone_embeddings_for_models(
        flemish_phones,
        model_names,
        layers=layers,
        collar=collar,
        store_root=store_root,
        model_paths_file=model_paths_file,
        phraser_source_id=phraser_source_id,
        gpu=gpu,
        batch_size=batch_size,
        tags=tags,
        verbose=verbose,
    )


def _extract_phone_embeddings_for_models(
    phones,
    model_names,
    layers,
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
        store = model_store.open_model_store(
            model_name,
            stores_root=store_root,
            model_paths_file=model_paths_file,
        )
        try:
            extract_phone_embeddings(
                phones,
                model_name=model_name,
                layers=layers,
                collar=collar,
                store=store,
                model_paths_file=model_paths_file,
                phraser_source_id=phraser_source_id,
                gpu=gpu,
                batch_size=batch_size,
                tags=tags,
                verbose=verbose,
            )
            store_paths[model_name] = model_store.model_store_path(
                model_name, store_root)
        finally:
            try:
                store.remove_cached_model()
            finally:
                try:
                    store.close()
                finally:
                    if gpu:
                        model_store.release_cuda_memory()
    return store_paths
