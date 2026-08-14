import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import model_store


MODEL_PATHS = [
    {'model_name': 'model-a', 'local_path': '/models/a',
        'language': 'Dutch', 'size': 'base'},
    {'model_name': 'model-b', 'huggingface_id': 'facebook/wav2vec2-base'},
]


def write_model_paths(tmp_path, entries):
    path = tmp_path / 'model_paths.json'
    path.write_text(json.dumps(entries))
    return path


# -- find_model_entry --------------------------------------------------------

def test_find_model_entry_returns_match(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    entry = model_store.find_model_entry('model-a', path)
    assert entry == MODEL_PATHS[0]


def test_find_model_entry_raises_when_missing(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    with pytest.raises(ValueError, match='not found'):
        model_store.find_model_entry('missing-model', path)


def test_find_model_entry_raises_when_multiple_matches(tmp_path):
    duplicated = MODEL_PATHS + [MODEL_PATHS[0]]
    path = write_model_paths(tmp_path, duplicated)
    with pytest.raises(ValueError, match='multiple entries'):
        model_store.find_model_entry('model-a', path)


# -- ensure_model_registered --------------------------------------------------

class FakeStore:
    def __init__(self, registered=None):
        self._registered = dict(registered or {})
        self.register_model_calls = []

    def load_model_metadata(self, model_name):
        return self._registered.get(model_name)

    def register_model(self, model_name, local_path=None, huggingface_id=None,
        language=None, size=None):
        self.register_model_calls.append(dict(
            model_name=model_name, local_path=local_path,
            huggingface_id=huggingface_id, language=language, size=size))
        self._registered[model_name] = object()


def test_ensure_model_registered_skips_when_already_registered(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore(registered={'model-a': object()})

    model_store.ensure_model_registered(store, 'model-a', path)

    assert store.register_model_calls == []


def test_ensure_model_registered_registers_from_file(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()

    model_store.ensure_model_registered(store, 'model-a', path)

    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]


def test_ensure_model_registered_passes_huggingface_id(tmp_path):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()

    model_store.ensure_model_registered(store, 'model-b', path)

    assert store.register_model_calls == [dict(
        model_name='model-b', local_path=None,
        huggingface_id='facebook/wav2vec2-base', language=None, size=None)]


# -- model_store_path ---------------------------------------------------------

def test_model_store_path_uses_model_name(tmp_path):
    result = model_store.model_store_path('model-a', tmp_path)

    assert result == tmp_path / 'model-a'


def test_model_store_path_escapes_path_separators(tmp_path):
    result = model_store.model_store_path('owner/model', tmp_path)

    assert result == tmp_path / 'owner%2Fmodel'


@pytest.mark.parametrize('model_name', ['', '   ', None, 42])
def test_model_store_path_rejects_invalid_model_name(tmp_path, model_name):
    with pytest.raises(ValueError, match='non-empty string'):
        model_store.model_store_path(model_name, tmp_path)


# -- open_model_store ---------------------------------------------------------

def test_open_model_store_opens_and_registers_dedicated_store(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    opened = []

    def fake_store_constructor(root, max_shard_size_bytes):
        store = FakeStore()
        opened.append((root, max_shard_size_bytes, store))
        return store

    monkeypatch.setattr(model_store.echoframe, 'Store',
        fake_store_constructor)
    monkeypatch.setattr(locations, 'model_paths_file', path)

    result = model_store.open_model_store(
        'model-a', stores_root=tmp_path / 'stores',
        max_shard_size_bytes=1234)

    root, max_shard_size_bytes, store = opened[0]
    assert result is store
    assert root == str(tmp_path / 'stores' / 'model-a')
    assert max_shard_size_bytes == 1234
    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]
