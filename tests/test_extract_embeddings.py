import json
import inspect
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
        self.remove_cached_model_calls = 0
        self.close_calls = 0

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

    def remove_cached_model(self):
        self.remove_cached_model_calls += 1

    def close(self):
        self.close_calls += 1


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


# -- per-model stores --------------------------------------------------------

def test_model_store_path_uses_model_name(tmp_path):
    result = extract_embeddings.model_store_path('model-a', tmp_path)

    assert result == tmp_path / 'model-a'


def test_model_store_path_escapes_path_separators(tmp_path):
    result = extract_embeddings.model_store_path('owner/model', tmp_path)

    assert result == tmp_path / 'owner%2Fmodel'


@pytest.mark.parametrize('model_name', ['', '   ', None, 42])
def test_model_store_path_rejects_invalid_model_name(tmp_path, model_name):
    with pytest.raises(ValueError, match='non-empty string'):
        extract_embeddings.model_store_path(model_name, tmp_path)


def test_open_model_store_opens_and_registers_dedicated_store(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    opened = []

    def fake_store_constructor(root, max_shard_size_bytes):
        store = FakeStore()
        opened.append((root, max_shard_size_bytes, store))
        return store

    monkeypatch.setattr(extract_embeddings.echoframe, 'Store',
        fake_store_constructor)

    result = extract_embeddings.open_model_store(
        'model-a', stores_root=tmp_path / 'stores', model_paths_file=path,
        max_shard_size_bytes=1234)

    root, max_shard_size_bytes, store = opened[0]
    assert result is store
    assert root == str(tmp_path / 'stores' / 'model-a')
    assert max_shard_size_bytes == 1234
    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]


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


# -- extract_phone_embeddings_for_models ------------------------------------

def test_extract_phone_embeddings_for_models_opens_extracts_and_closes(
    tmp_path, monkeypatch):
    stores = {name: FakeStore() for name in ('model-a', 'model-b')}
    open_calls = []
    extract_calls = []
    cuda_release_calls = []

    def fake_open_model_store(
        model_name, stores_root, model_paths_file):
        open_calls.append((model_name, stores_root, model_paths_file))
        return stores[model_name]

    def fake_extract_phone_embeddings(phones, **kwargs):
        extract_calls.append((phones, kwargs))

    monkeypatch.setattr(extract_embeddings, 'open_model_store',
        fake_open_model_store)
    monkeypatch.setattr(extract_embeddings, 'extract_phone_embeddings',
        fake_extract_phone_embeddings)
    monkeypatch.setattr(extract_embeddings, '_release_cuda_memory',
        lambda: cuda_release_calls.append(True))

    path = tmp_path / 'model_paths.json'
    phones = object()
    result = extract_embeddings.extract_phone_embeddings_for_models(
        phones,
        ['model-a', 'model-b'],
        layers=[8, 9],
        collar=500,
        store_root=tmp_path / 'stores',
        model_paths_file=path,
        phraser_source_id='phones',
        gpu=True,
        batch_size=16,
        tags=['experiment'],
        verbose=False,
    )

    assert open_calls == [
        ('model-a', tmp_path / 'stores', path),
        ('model-b', tmp_path / 'stores', path),
    ]
    assert [call[1] for call in extract_calls] == [
        dict(
            model_name='model-a', layers=[8, 9], collar=500,
            store=stores['model-a'], model_paths_file=path,
            phraser_source_id='phones', gpu=True, batch_size=16,
            tags=['experiment'], verbose=False,
        ),
        dict(
            model_name='model-b', layers=[8, 9], collar=500,
            store=stores['model-b'], model_paths_file=path,
            phraser_source_id='phones', gpu=True, batch_size=16,
            tags=['experiment'], verbose=False,
        ),
    ]
    assert all(store.remove_cached_model_calls == 1
        for store in stores.values())
    assert all(store.close_calls == 1 for store in stores.values())
    assert cuda_release_calls == [True, True]
    assert result == {
        'model-a': tmp_path / 'stores' / 'model-a',
        'model-b': tmp_path / 'stores' / 'model-b',
    }


def test_extract_phone_embeddings_for_models_cleans_up_after_failure(
    tmp_path, monkeypatch):
    store = FakeStore()
    cuda_release_calls = []

    monkeypatch.setattr(extract_embeddings, 'open_model_store',
        lambda *args, **kwargs: store)

    def fail_extraction(*args, **kwargs):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(extract_embeddings, 'extract_phone_embeddings',
        fail_extraction)
    monkeypatch.setattr(extract_embeddings, '_release_cuda_memory',
        lambda: cuda_release_calls.append(True))

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_embeddings.extract_phone_embeddings_for_models(
            object(), ['model-a'], store_root=tmp_path, gpu=True)

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1
    assert cuda_release_calls == [True]


def test_extract_phone_embeddings_for_models_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_embeddings.extract_phone_embeddings_for_models(
            object(), 'model-a')


# -- Flemish embedding extraction ------------------------------------------

def test_flemish_model_store_root_and_batch_defaults():
    assert (
        extract_embeddings.default_flemish_model_stores_root
        == extract_embeddings._data_dir / 'echoframe_model_flemish_stores'
    )
    assert extract_embeddings.model_store_path(
        'owner/model',
        extract_embeddings.default_flemish_model_stores_root,
    ) == (
        extract_embeddings.default_flemish_model_stores_root
        / 'owner%2Fmodel'
    )
    functions = (
        extract_embeddings.extract_phone_embeddings,
        extract_embeddings.extract_phone_embeddings_for_models,
        extract_embeddings.extract_flemish_phone_embeddings_for_models,
    )
    assert all(
        inspect.signature(function).parameters['batch_size'].default == 120
        for function in functions
    )


def test_extract_flemish_phone_embeddings_for_models_lifecycle(
    tmp_path, monkeypatch,
):
    stores = {name: FakeStore() for name in ('model-a', 'model-b')}
    open_calls = []
    extract_calls = []

    def fake_open_model_store(model_name, stores_root, model_paths_file):
        open_calls.append((model_name, stores_root, model_paths_file))
        return stores[model_name]

    monkeypatch.setattr(
        extract_embeddings, 'open_model_store', fake_open_model_store)
    monkeypatch.setattr(
        extract_embeddings, 'extract_phone_embeddings',
        lambda phones, **kwargs: extract_calls.append((phones, kwargs)),
    )

    model_paths_file = tmp_path / 'model_paths.json'
    flemish_phones = object()
    store_root = tmp_path / 'flemish-stores'
    result = (
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            flemish_phones,
            ['model-a', 'model-b'],
            layers=[8, 9],
            collar=500,
            store_root=store_root,
            model_paths_file=model_paths_file,
            gpu=False,
            tags=['flemish'],
            verbose=False,
        )
    )

    assert open_calls == [
        ('model-a', store_root, model_paths_file),
        ('model-b', store_root, model_paths_file),
    ]
    assert [phones for phones, _ in extract_calls] == [
        flemish_phones, flemish_phones]
    assert [kwargs for _, kwargs in extract_calls] == [
        dict(
            model_name=model_name,
            layers=[8, 9],
            collar=500,
            store=stores[model_name],
            model_paths_file=model_paths_file,
            phraser_source_id='cgn-awd',
            gpu=False,
            batch_size=120,
            tags=['flemish'],
            verbose=False,
        )
        for model_name in ('model-a', 'model-b')
    ]
    assert all(store.remove_cached_model_calls == 1
        for store in stores.values())
    assert all(store.close_calls == 1 for store in stores.values())
    assert result == {
        'model-a': store_root / 'model-a',
        'model-b': store_root / 'model-b',
    }


def test_extract_flemish_phone_embeddings_for_models_cleans_up_after_failure(
    tmp_path, monkeypatch,
):
    store = FakeStore()

    def fail_extraction(*args, **kwargs):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(
        extract_embeddings, 'open_model_store',
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        extract_embeddings, 'extract_phone_embeddings',
        fail_extraction,
    )

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            object(), ['model-a'], store_root=tmp_path)

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1


def test_extract_flemish_phone_embeddings_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            object(), 'model-a')
