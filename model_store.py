import gc
import json
from pathlib import Path
from urllib.parse import quote

import echoframe

import locations


def find_model_entry(model_name, model_paths_file):
    '''Look up model_name's registration record in model_paths_file.

    model_name:        model name to look up
    model_paths_file:  JSON file of {model_name, local_path/huggingface_id,
                       language, size} records
    '''
    model_paths_path = Path(model_paths_file)
    entries = json.loads(model_paths_path.read_text())
    matches = [e for e in entries if e['model_name'] == model_name]
    if not matches:
        raise ValueError(f'{model_name!r} not found in {model_paths_file}')
    if len(matches) > 1:
        message = f'multiple entries for {model_name!r} in {model_paths_file}'
        raise ValueError(message)
    return matches[0]


def ensure_model_registered(store, model_name, model_paths_file):
    '''Register model_name from model_paths_file, unless already registered.

    store:              echoframe Store to register the model in
    model_name:         model name to register
    model_paths_file:   JSON file of {model_name, local_path/huggingface_id,
                        language, size} records
    '''
    if store.load_model_metadata(model_name) is not None: return
    entry = find_model_entry(model_name, model_paths_file)
    local_path = entry.get('local_path')
    huggingface_id = entry.get('huggingface_id')
    language = entry.get('language')
    size = entry.get('size')
    store.register_model(entry['model_name'], local_path=local_path,
        huggingface_id=huggingface_id, language=language, size=size)


def model_store_path(model_name, stores_root):
    '''Return the dedicated Echoframe store path for one model.

    model_name:   model to build a store path for
    stores_root:  directory containing one subdirectory per model
    '''
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('model_name must be a non-empty string')
    directory_name = quote(model_name, safe='._-')
    return Path(stores_root) / directory_name


def open_model_store(
    model_name,
    stores_root,
    max_shard_size_bytes=100_000_000,
):
    '''Open or create and register the dedicated store for one model.

    model_name:             model to open or create a store for
    stores_root:            directory containing one subdirectory per model
    max_shard_size_bytes:   shard size passed to echoframe.Store
    '''
    store_path = model_store_path(model_name, stores_root)
    store_path = str(store_path)
    store = echoframe.Store(store_path,
        max_shard_size_bytes=max_shard_size_bytes)
    ensure_model_registered(store, model_name, locations.model_paths_file)
    return store


def release_cuda_memory():
    '''Release unreferenced CUDA allocations after unloading a model.'''
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
