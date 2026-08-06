import gc
import json
from pathlib import Path
from urllib.parse import quote

import echoframe

import locations


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


def model_store_path(model_name, stores_root):
    '''Return the dedicated Echoframe store path for one model.'''
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('model_name must be a non-empty string')
    directory_name = quote(model_name, safe='._-')
    return Path(stores_root) / directory_name


def open_model_store(
    model_name,
    stores_root,
    model_paths_file=locations.model_paths_file,
    max_shard_size_bytes=100_000_000,
):
    '''Open or create and register the dedicated store for one model.'''
    store = echoframe.Store(
        str(model_store_path(model_name, stores_root)),
        max_shard_size_bytes=max_shard_size_bytes,
    )
    _ensure_model_registered(store, model_name, model_paths_file)
    return store


def _release_cuda_memory():
    '''Release unreferenced CUDA allocations after unloading a model.'''
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
