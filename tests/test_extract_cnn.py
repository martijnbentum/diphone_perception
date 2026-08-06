import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import extract_cnn


MODEL_PATHS = [
    {'model_name': 'model-a', 'local_path': '/models/a',
        'language': 'Dutch', 'size': 'base'},
    {'model_name': 'model-b', 'huggingface_id': 'facebook/wav2vec2-base'},
]


def write_model_paths(tmp_path, entries):
    import json
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


class FakePhones:
    def __init__(self, phraser_phones, store):
        self.phraser_phones = phraser_phones
        self.store = store


# -- extract_phone_cnn_features ----------------------------------------------

def test_extract_phone_cnn_features_registers_model_and_computes(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore()
    phraser_phones = ['phone-a', 'phone-b']
    phones = FakePhones(phraser_phones, store='cgn-store')

    calls = []

    def fake_compute_cnn_features_batch(segments, model_name, store_arg,
        collar=None, gpu=None, tags=None, batch_size=None, verbose=None):
        calls.append(dict(segments=segments, model_name=model_name,
            store=store_arg, collar=collar, gpu=gpu, tags=tags,
            batch_size=batch_size, verbose=verbose))

    monkeypatch.setattr(extract_cnn, 'compute_cnn_features_batch',
        fake_compute_cnn_features_batch)

    result = extract_cnn.extract_phone_cnn_features(
        phones, store, model_name='model-a', collar=500,
        model_paths_file=path, gpu=False, batch_size=16,
        tags=['exp-a'], verbose=False)

    assert result is store
    assert store.register_model_calls == [dict(
        model_name='model-a', local_path='/models/a', huggingface_id=None,
        language='Dutch', size='base')]
    assert store.attach_phraser_store_calls == [
        (extract_cnn.default_phraser_source_id, 'cgn-store')]
    assert calls == [dict(segments=phraser_phones, model_name='model-a',
        store=store, collar=500, gpu=False, tags=['exp-a'], batch_size=16,
        verbose=False)]


def test_extract_phone_cnn_features_skips_registration_when_already_registered(
    tmp_path, monkeypatch):
    path = write_model_paths(tmp_path, MODEL_PATHS)
    store = FakeStore(registered={'model-a': object()})
    phones = FakePhones(['phone-a'], store='cgn-store')

    monkeypatch.setattr(extract_cnn, 'compute_cnn_features_batch',
        lambda *a, **k: None)

    extract_cnn.extract_phone_cnn_features(
        phones, store, model_name='model-a', model_paths_file=path,
        verbose=False)

    assert store.register_model_calls == []


def test_extract_phone_cnn_features_uses_default_collar_and_model(
    tmp_path, monkeypatch):
    entries = MODEL_PATHS + [{
        'model_name': extract_cnn.default_model_name,
        'local_path': '/models/default', 'language': 'Dutch', 'size': 'base',
    }]
    path = write_model_paths(tmp_path, entries)
    store = FakeStore()
    phones = FakePhones(['phone-a'], store='cgn-store')

    calls = []
    monkeypatch.setattr(extract_cnn, 'compute_cnn_features_batch',
        lambda *a, **k: calls.append((a, k)))

    extract_cnn.extract_phone_cnn_features(
        phones, store, model_paths_file=path, verbose=False)

    (segments, model_name, store_arg), kwargs = calls[0]
    assert model_name == extract_cnn.default_model_name
    assert kwargs['collar'] == 500


# -- extract_phone_cnn_features_for_models -----------------------------------

def test_extract_phone_cnn_features_for_models_opens_extracts_and_closes(
    tmp_path, monkeypatch):
    stores = {name: FakeStore() for name in ('model-a', 'model-b')}
    open_calls = []
    extract_calls = []
    cuda_release_calls = []

    def fake_open_model_store(model_name, stores_root, model_paths_file):
        open_calls.append((model_name, stores_root, model_paths_file))
        return stores[model_name]

    def fake_extract_phone_cnn_features(phones, store, **kwargs):
        extract_calls.append((phones, store, kwargs))

    monkeypatch.setattr(extract_cnn, 'open_model_store',
        fake_open_model_store)
    monkeypatch.setattr(extract_cnn, 'extract_phone_cnn_features',
        fake_extract_phone_cnn_features)
    monkeypatch.setattr(extract_cnn, '_release_cuda_memory',
        lambda: cuda_release_calls.append(True))

    path = tmp_path / 'model_paths.json'
    phones = object()
    result = extract_cnn.extract_phone_cnn_features_for_models(
        phones,
        ['model-a', 'model-b'],
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
    assert [(p, s, kwargs) for p, s, kwargs in extract_calls] == [
        (phones, stores['model-a'], dict(
            model_name='model-a', collar=500, model_paths_file=path,
            phraser_source_id='phones', gpu=True, batch_size=16,
            tags=['experiment'], verbose=False,
        )),
        (phones, stores['model-b'], dict(
            model_name='model-b', collar=500, model_paths_file=path,
            phraser_source_id='phones', gpu=True, batch_size=16,
            tags=['experiment'], verbose=False,
        )),
    ]
    assert all(store.remove_cached_model_calls == 1
        for store in stores.values())
    assert all(store.close_calls == 1 for store in stores.values())
    assert cuda_release_calls == [True, True]
    assert result == {
        'model-a': tmp_path / 'stores' / 'model-a',
        'model-b': tmp_path / 'stores' / 'model-b',
    }


def test_extract_phone_cnn_features_for_models_cleans_up_after_failure(
    tmp_path, monkeypatch):
    store = FakeStore()
    cuda_release_calls = []

    monkeypatch.setattr(extract_cnn, 'open_model_store',
        lambda *args, **kwargs: store)

    def fail_extraction(*args, **kwargs):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(extract_cnn, 'extract_phone_cnn_features',
        fail_extraction)
    monkeypatch.setattr(extract_cnn, '_release_cuda_memory',
        lambda: cuda_release_calls.append(True))

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_cnn.extract_phone_cnn_features_for_models(
            object(), ['model-a'], store_root=tmp_path, gpu=True)

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1
    assert cuda_release_calls == [True]


def test_extract_phone_cnn_features_for_models_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_cnn.extract_phone_cnn_features_for_models(
            object(), 'model-a')


# -- Flemish CNN extraction ---------------------------------------------------

def test_flemish_model_store_root_and_batch_defaults():
    store_root_default = inspect.signature(
        extract_cnn.extract_flemish_phone_cnn_features_for_models,
    ).parameters['store_root'].default
    assert store_root_default == locations.echoframe_model_cnn_flemish_stores
    default_root = inspect.signature(
        extract_cnn.extract_phone_cnn_features_for_models,
    ).parameters['store_root'].default
    assert default_root == locations.echoframe_model_cnn_stores
    functions = (
        extract_cnn.extract_phone_cnn_features,
        extract_cnn.extract_phone_cnn_features_for_models,
        extract_cnn.extract_flemish_phone_cnn_features_for_models,
    )
    assert all(
        inspect.signature(function).parameters['batch_size'].default == 120
        for function in functions
    )
    assert all(
        inspect.signature(function).parameters['collar'].default == 500
        for function in functions
    )


def test_extract_flemish_phone_cnn_features_for_models_lifecycle(
    tmp_path, monkeypatch):
    stores = {name: FakeStore() for name in ('model-a', 'model-b')}
    open_calls = []
    extract_calls = []

    def fake_open_model_store(model_name, stores_root, model_paths_file):
        open_calls.append((model_name, stores_root, model_paths_file))
        return stores[model_name]

    monkeypatch.setattr(
        extract_cnn, 'open_model_store', fake_open_model_store)
    monkeypatch.setattr(
        extract_cnn, 'extract_phone_cnn_features',
        lambda phones, store, **kwargs: extract_calls.append(
            (phones, store, kwargs)),
    )

    model_paths_file = tmp_path / 'model_paths.json'
    flemish_phones = object()
    store_root = tmp_path / 'flemish-stores'
    result = extract_cnn.extract_flemish_phone_cnn_features_for_models(
        flemish_phones,
        ['model-a', 'model-b'],
        collar=500,
        store_root=store_root,
        model_paths_file=model_paths_file,
        gpu=False,
        tags=['flemish'],
        verbose=False,
    )

    assert open_calls == [
        ('model-a', store_root, model_paths_file),
        ('model-b', store_root, model_paths_file),
    ]
    assert [phones for phones, _, _ in extract_calls] == [
        flemish_phones, flemish_phones]
    assert [(store, kwargs) for _, store, kwargs in extract_calls] == [
        (stores[model_name], dict(
            model_name=model_name,
            collar=500,
            model_paths_file=model_paths_file,
            phraser_source_id='cgn-awd',
            gpu=False,
            batch_size=120,
            tags=['flemish'],
            verbose=False,
        ))
        for model_name in ('model-a', 'model-b')
    ]
    assert all(store.remove_cached_model_calls == 1
        for store in stores.values())
    assert all(store.close_calls == 1 for store in stores.values())
    assert result == {
        'model-a': store_root / 'model-a',
        'model-b': store_root / 'model-b',
    }


def test_extract_flemish_phone_cnn_features_for_models_cleans_up_after_failure(
    tmp_path, monkeypatch):
    store = FakeStore()

    def fail_extraction(*args, **kwargs):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(
        extract_cnn, 'open_model_store',
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        extract_cnn, 'extract_phone_cnn_features',
        fail_extraction,
    )

    with pytest.raises(RuntimeError, match='extraction failed'):
        extract_cnn.extract_flemish_phone_cnn_features_for_models(
            object(), ['model-a'], store_root=tmp_path)

    assert store.remove_cached_model_calls == 1
    assert store.close_calls == 1


def test_extract_flemish_phone_cnn_features_rejects_string_model_names():
    with pytest.raises(TypeError, match='iterable, not a string'):
        extract_cnn.extract_flemish_phone_cnn_features_for_models(
            object(), 'model-a')
