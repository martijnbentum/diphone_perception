import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import extract_embeddings


MODEL_PATHS = [
    {'model_name': 'model-a', 'local_path': '/models/a',
        'language': 'Dutch', 'size': 'base'},
    {'model_name': 'model-b', 'huggingface_id': 'facebook/wav2vec2-base'},
]


def write_model_paths(tmp_path, entries):
    path = tmp_path / 'model_paths.json'
    path.write_text(json.dumps(entries))
    return path


# -- _find_model_entry ------------------------------------------------------

def test_find_model_entry_returns_match(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    entry = extract_embeddings._find_model_entry('model-a', path)
    assert entry == MODEL_PATHS[0]


def test_find_model_entry_raises_when_missing(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    with pytest.raises(ValueError, match='not found'):
        extract_embeddings._find_model_entry('missing-model', path)


def test_find_model_entry_raises_when_multiple_matches(tmp_path):
    duplicated = MODEL_PATHS + [MODEL_PATHS[0]]
    path = write_model_paths(tmp_path, duplicated)
    with pytest.raises(ValueError, match='multiple entries'):
        extract_embeddings._find_model_entry('model-a', path)


# -- _ensure_model_registered ------------------------------------------------

class FakeStore:
    def __init__(self, registered=None):
        self._registered = dict(registered or {})
        self.register_model_calls = []
        self.attach_phraser_store_calls = []

    def load_model_metadata(self, model_name):
        return self._registered.get(model_name)

    def register_model(self, model_name, local_path=None, huggingface_id=None,
        language=None, size=None):
        self.register_model_calls.append(dict(
            model_name=model_name, local_path=local_path,
            huggingface_id=huggingface_id, language=language, size=size))
        self._registered[model_name] = object()

    def attach_phraser_store(self, source_id, phraser_store):
        self.attach_phraser_store_calls.append((source_id, phraser_store))


def test_ensure_model_registered_skips_when_already_registered(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore(registered={'model-a': object()})

    extract_embeddings._ensure_model_registered(store, 'model-a', path)

    assert store.register_model_calls == []


def test_ensure_model_registered_registers_from_file(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()

    extract_embeddings._ensure_model_registered(store, 'model-a', path)

    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]


def test_ensure_model_registered_passes_huggingface_id(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()

    extract_embeddings._ensure_model_registered(store, 'model-b', path)

    assert store.register_model_calls == [dict(
        model_name='model-b', local_path=None,
        huggingface_id='facebook/wav2vec2-base', language=None, size=None)]


# -- extract_phone_embeddings ------------------------------------------------

class FakePhones:
    def __init__(self, phraser_phones, store):
        self.phraser_phones = phraser_phones
        self.store = store


def test_extract_phone_embeddings_registers_model_and_computes(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()
    phraser_phones = ['phone-a', 'phone-b']
    phones = FakePhones(phraser_phones, store='cgn-store')

    calls = []

    def fake_compute_embeddings_batch(segments, layers, model_name, store_arg,
        collar=None, gpu=None, tags=None, batch_size=None, verbose=None):
        calls.append(dict(segments=segments, layers=layers,
            model_name=model_name, store=store_arg, collar=collar, gpu=gpu,
            tags=tags, batch_size=batch_size, verbose=verbose))

    monkeypatch.setattr(extract_embeddings, 'compute_embeddings_batch',
        fake_compute_embeddings_batch)

    result = extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', layers=[9, 10], collar=500,
        store=store, model_paths_file=path, gpu=False, batch_size=16,
        tags=['exp-a'], verbose=False)

    assert result is store
    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]
    assert store.attach_phraser_store_calls == [
        (extract_embeddings.default_phraser_source_id, 'cgn-store')]
    assert calls == [dict(segments=phraser_phones, layers=[9, 10],
        model_name='model-a', store=store, collar=500, gpu=False,
        tags=['exp-a'], batch_size=16, verbose=False)]


def test_extract_phone_embeddings_skips_registration_when_already_registered(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore(registered={'model-a': object()})
    phones = FakePhones(['phone-a'], store='cgn-store')

    monkeypatch.setattr(extract_embeddings, 'compute_embeddings_batch',
        lambda *a, **k: None)

    extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', store=store, model_paths_file=path,
        verbose=False)

    assert store.register_model_calls == []


def test_extract_phone_embeddings_uses_default_layers_and_model(
    tmp_path, monkeypatch):
    entries = MODEL_PATHS + [{
        'model_name': extract_embeddings.default_model_name,
        'local_path': '/models/default', 'language': 'Dutch', 'size': 'base',
    }]
    path = write_model_paths(tmp_path, entries)
    store = FakeStore()
    phones = FakePhones(['phone-a'], store='cgn-store')

    calls = []
    monkeypatch.setattr(extract_embeddings, 'compute_embeddings_batch',
        lambda *a, **k: calls.append((a, k)))

    extract_embeddings.extract_phone_embeddings(
        phones, store=store, model_paths_file=path, verbose=False)

    (segments, layers, model_name, store_arg), kwargs = calls[0]
    assert layers == [9]
    assert model_name == extract_embeddings.default_model_name
    assert kwargs['collar'] == 2000


def test_extract_phone_embeddings_opens_store_when_none_given(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    phones = FakePhones(['phone-a'], store='cgn-store')
    opened_store = FakeStore()
    store_roots = []

    def fake_store_constructor(root):
        store_roots.append(root)
        return opened_store

    monkeypatch.setattr(extract_embeddings.echoframe, 'Store',
        fake_store_constructor)
    monkeypatch.setattr(extract_embeddings, 'compute_embeddings_batch',
        lambda *a, **k: None)

    result = extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', model_paths_file=path,
        store_root=tmp_path / 'store', verbose=False)

    assert result is opened_store
    assert store_roots == [str(tmp_path / 'store')]
