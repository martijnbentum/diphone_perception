import json
from pathlib import Path

import echoframe
from echoframe.batch_segment_features import compute_embeddings_batch

from probing.metadata import _data_dir

default_model_paths_file = _data_dir / 'model_paths.json'
default_store_root = _data_dir / 'echoframe_store'
default_model_name = 'wav2vec2_nl1_checkpoint-200000'
default_phraser_source_id = 'cgn-awd'


def _find_model_entry(model_name, model_paths_file):
    '''look up model_name's registration record in model_paths_file.'''
    entries = json.loads(Path(model_paths_file).read_text())
    matches = [e for e in entries if e['model_name'] == model_name]
    if not matches:
        raise ValueError(f'{model_name!r} not found in {model_paths_file}')
    if len(matches) > 1:
        raise ValueError(
            f'multiple entries for {model_name!r} in {model_paths_file}')
    return matches[0]


def _ensure_model_registered(store, model_name, model_paths_file):
    '''register model_name from model_paths_file, unless already registered.'''
    if store.load_model_metadata(model_name) is not None:
        return
    entry = _find_model_entry(model_name, model_paths_file)
    store.register_model(
        entry['model_name'],
        local_path=entry.get('local_path'),
        huggingface_id=entry.get('huggingface_id'),
        language=entry.get('language'),
        size=entry.get('size'),
    )


def extract_phone_embeddings(
    phones,
    model_name=default_model_name,
    layers=[9],
    collar=500,
    store=None,
    store_root=default_store_root,
    model_paths_file=default_model_paths_file,
    phraser_source_id=default_phraser_source_id,
    gpu=False,
    batch_size=32,
    tags=None,
    verbose=True,
):
    '''Compute and store wav2vec2 hidden-state embeddings for every phone in
    `phones` into an echoframe Store, via vanilla compute_embeddings_batch
    (stores every frame overlapping each phone's own span, for each layer).

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
    batch_size:          segments per forward-pass batch. compute_embeddings_batch
                         loads every segment's audio into one batch when this
                         is left None and gpu=False - an explicit default
                         avoids loading all of phones' audio into memory at once
    tags:                optional tags stored on new metadata
    verbose:             print batch progress
    '''
    if store is None:
        store = echoframe.Store(str(store_root))
    _ensure_model_registered(store, model_name, model_paths_file)
    store.attach_phraser_store(phraser_source_id, phones.store)

    segments = phones.phraser_phones
    compute_embeddings_batch(
        segments, layers, model_name, store,
        collar=collar, gpu=gpu, tags=tags, batch_size=batch_size,
        verbose=verbose,
    )
    return store
