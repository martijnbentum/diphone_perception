import json
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import extract_embeddings, model_store


MODEL_PATHS = [
    {'model_name': 'model-a', 'local_path': '/models/a',
        'language': 'Dutch', 'size': 'base'},
    {'model_name': 'model-b', 'huggingface_id': 'facebook/wav2vec2-base'},
]


def write_model_paths(tmp_path, entries):
    path = tmp_path / 'model_paths.json'
    path.write_text(json.dumps(entries))
    return path


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

    def fake_compute_embeddings_batch(
        segments, layers, model_name, store_arg,
        collar=None, gpu=None, tags=None, batch_size=None, verbose=None):
        calls.append(dict(segments=segments, layers=layers,
            model_name=model_name, store=store_arg, collar=collar, gpu=gpu,
            tags=tags, batch_size=batch_size, verbose=verbose))

    monkeypatch.setattr(extract_embeddings,
        'compute_embeddings_batch',
        fake_compute_embeddings_batch)
    monkeypatch.setattr(locations, 'model_paths_file', path)

    result = extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', layers=[9, 10], collar=500,
        store=store, gpu=False, batch_size=16,
        tags=['exp-a'], verbose=False)

    assert result is store
    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]
    assert store.attach_phraser_store_calls == [
        (extract_embeddings.default_phraser_source_id, 'cgn-store')]
    assert calls == [dict(segments=phraser_phones, layers=[9, 10, 'cnn'],
        model_name='model-a', store=store, collar=500, gpu=False,
        tags=['exp-a'], batch_size=16, verbose=False)]


def test_extract_phone_embeddings_skips_registration_when_already_registered(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore(registered={'model-a': object()})
    phones = FakePhones(['phone-a'], store='cgn-store')

    monkeypatch.setattr(extract_embeddings,
        'compute_embeddings_batch', lambda *a, **k: None)
    monkeypatch.setattr(locations, 'model_paths_file', path)

    extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', store=store,
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
    monkeypatch.setattr(extract_embeddings,
        'compute_embeddings_batch',
        lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(locations, 'model_paths_file', path)

    extract_embeddings.extract_phone_embeddings(
        phones, store=store, verbose=False)

    (segments, layers, model_name, store_arg), kwargs = calls[0]
    assert layers == [9, 'cnn']
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
    monkeypatch.setattr(extract_embeddings,
        'compute_embeddings_batch', lambda *a, **k: None)
    monkeypatch.setattr(locations, 'model_paths_file', path)
    monkeypatch.setattr(locations, 'echoframe_store', tmp_path / 'store')

    result = extract_embeddings.extract_phone_embeddings(
        phones, model_name='model-a', verbose=False)

    assert result is opened_store
    assert store_roots == [str(tmp_path / 'store')]


# -- extract_phone_embeddings_for_models ------------------------------------

def test_extract_phone_embeddings_for_models_opens_extracts_and_closes(
    tmp_path, monkeypatch):
    stores = {name: FakeStore() for name in ('model-a', 'model-b')}
    open_calls = []
    extract_calls = []
    cuda_release_calls = []

    def fake_open_model_store(model_name, stores_root):
        open_calls.append((model_name, stores_root))
        return stores[model_name]

    def fake_extract_phone_embeddings(phones, **kwargs):
        extract_calls.append((phones, kwargs))

    monkeypatch.setattr(model_store, 'open_model_store',
        fake_open_model_store)
    monkeypatch.setattr(extract_embeddings, 'extract_phone_embeddings',
        fake_extract_phone_embeddings)
    monkeypatch.setattr(model_store, 'release_cuda_memory',
        lambda: cuda_release_calls.append(True))
    monkeypatch.setattr(locations, 'echoframe_model_stores', tmp_path / 'stores')

    phones = object()
    result = extract_embeddings.extract_phone_embeddings_for_models(
        phones,
        ['model-a', 'model-b'],
        layers=[8, 9],
        collar=500,
        phraser_source_id='phones',
        gpu=True,
        batch_size=16,
        tags=['experiment'],
        verbose=False,
    )

    assert open_calls == [
        ('model-a', tmp_path / 'stores'),
        ('model-b', tmp_path / 'stores'),
    ]
    assert [call[1] for call in extract_calls] == [
        dict(
            model_name='model-a', layers=[8, 9], collar=500,
            store=stores['model-a'],
            phraser_source_id='phones', gpu=True, batch_size=16,
            tags=['experiment'], verbose=False,
        ),
        dict(
            model_name='model-b', layers=[8, 9], collar=500,
            store=stores['model-b'],
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

    monkeypatch.setattr(model_store, 'open_model_store',
        lambda *args, **kwargs: store)

    def fail_extraction(*args, **kwargs):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(extract_embeddings, 'extract_phone_embeddings',
        fail_extraction)
    monkeypatch.setattr(model_store, 'release_cuda_memory',
        lambda: cuda_release_calls.append(True))

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_embeddings.extract_phone_embeddings_for_models(
            object(), ['model-a'], gpu=True)

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1
    assert cuda_release_calls == [True]


def test_extract_phone_embeddings_for_models_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_embeddings.extract_phone_embeddings_for_models(
            object(), 'model-a')


# -- Flemish embedding extraction ------------------------------------------

def test_flemish_model_store_path_and_batch_defaults():
    assert model_store.model_store_path(
        'owner/model',
        locations.echoframe_model_flemish_stores,
    ) == (
        locations.echoframe_model_flemish_stores
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

    def fake_open_model_store(model_name, stores_root):
        open_calls.append((model_name, stores_root))
        return stores[model_name]

    monkeypatch.setattr(
        model_store, 'open_model_store', fake_open_model_store)
    monkeypatch.setattr(
        extract_embeddings, 'extract_phone_embeddings',
        lambda phones, **kwargs: extract_calls.append((phones, kwargs)),
    )

    flemish_phones = object()
    store_root = tmp_path / 'flemish-stores'
    monkeypatch.setattr(
        locations, 'echoframe_model_flemish_stores', store_root)
    result = (
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            flemish_phones,
            ['model-a', 'model-b'],
            layers=[8, 9],
            collar=500,
            gpu=False,
            tags=['flemish'],
            verbose=False,
        )
    )

    assert open_calls == [
        ('model-a', store_root),
        ('model-b', store_root),
    ]
    assert [phones for phones, _ in extract_calls] == [
        flemish_phones, flemish_phones]
    assert [kwargs for _, kwargs in extract_calls] == [
        dict(
            model_name=model_name,
            layers=[8, 9],
            collar=500,
            store=stores[model_name],
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
        model_store, 'open_model_store',
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        extract_embeddings, 'extract_phone_embeddings',
        fail_extraction,
    )

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            object(), ['model-a'])

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1


def test_extract_flemish_phone_embeddings_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_embeddings.extract_flemish_phone_embeddings_for_models(
            object(), 'model-a')
