import json
from types import SimpleNamespace

import numpy as np
import pytest

from synthetic_acoustic_probes import echoframe_store


class _ModelStore:
    def __init__(self, existing=()):
        '''Create a fake model registry with optional existing names.'''

        self.existing = set(existing)
        self.registered = []

    def load_model_metadata(self, model_name):
        '''Return a marker when model_name is registered.'''

        if model_name in self.existing: return object()
        return None

    def register_model(
        self,
        model_name,
        local_path=None,
        huggingface_id=None,
        language=None,
        size=None,
        architecture=None,
    ):
        '''Record one model registration.

        model_name:      Model name being registered.
        local_path:      Optional local model path.
        huggingface_id:  Optional Hugging Face identifier.
        language:        Optional language metadata.
        size:            Optional model size metadata.
        architecture:    Optional architecture metadata.
        '''

        metadata = {}
        metadata['local_path'] = local_path
        metadata['huggingface_id'] = huggingface_id
        metadata['language'] = language
        metadata['size'] = size
        metadata['architecture'] = architecture
        self.registered.append((model_name, metadata))


class _CNNFeature:
    def __init__(self, middle, mean):
        '''Create a fake feature with fixed aggregate vectors.

        middle:  Vector returned for middle-frame selection.
        mean:    Vector returned for mean aggregation.
        '''

        self.middle = np.asarray(middle)
        self.mean = np.asarray(mean)
        self.calls = []

    def aggregate_segment(self, phrase, method='mean'):
        '''Return the configured vector and record the aggregation call.

        phrase:  Phrase passed by make_x_y.
        method:  Native Echoframe aggregation method.
        '''

        self.calls.append((phrase, method))
        if method == 'middle': return self.middle
        return self.mean


def test_add_models_registers_selected_models_in_requested_order(tmp_path):
    '''Selected catalog metadata is forwarded in requested order.

    tmp_path:  Temporary directory supplied by pytest.
    '''

    catalog = [
        {'model_name': 'unused', 'local_path': '/models/unused'},
        {
            'model_name': 'second',
            'local_path': '/models/second',
            'language': 'Dutch',
            'size': 'base',
            'architecture': 'wav2vec2',
        },
        {'model_name': 'first', 'huggingface_id': 'example/first'},
    ]
    catalog_path = tmp_path / 'models.json'
    catalog_text = json.dumps(catalog)
    catalog_path.write_text(catalog_text, encoding='utf-8')
    store = _ModelStore()

    result = echoframe_store.add_models(
        ('first', 'second'),
        catalog_path,
        store,
    )

    assert result is None
    assert [name for name, _ in store.registered] == ['first', 'second']
    first = store.registered[0][1]
    second = store.registered[1][1]
    assert first['huggingface_id'] == 'example/first'
    assert first['local_path'] is None
    assert second == {
        'local_path': '/models/second',
        'huggingface_id': None,
        'language': 'Dutch',
        'size': 'base',
        'architecture': 'wav2vec2',
    }


@pytest.mark.parametrize(
    ('model_names', 'catalog_names', 'existing', 'match'),
    [
        (('available', 'missing'), ('available',), (), 'not found'),
        (('duplicate', 'duplicate'), ('duplicate',), (), 'duplicate model'),
        (('available', 'existing'), ('available', 'existing'),
            ('existing',), 'already registered'),
    ],
)
def test_add_models_rejects_invalid_input_before_registration(
    tmp_path,
    model_names,
    catalog_names,
    existing,
    match,
):
    '''Invalid model selections cause no partial registration.

    tmp_path:       Temporary directory supplied by pytest.
    model_names:    Requested model names.
    catalog_names:  Model names available in the catalog fixture.
    existing:       Model names already registered in the fake store.
    match:          Text expected in the error message.
    '''

    catalog = []
    for name in catalog_names:
        catalog.append({'model_name': name, 'local_path': f'/models/{name}'})
    catalog_path = tmp_path / 'models.json'
    catalog_text = json.dumps(catalog)
    catalog_path.write_text(catalog_text, encoding='utf-8')
    store = _ModelStore(existing)

    with pytest.raises((TypeError, ValueError), match=match):
        echoframe_store.add_models(model_names, catalog_path, store)

    assert store.registered == []


def test_add_cnn_features_delegates_sequentially(monkeypatch):
    '''Every Phrase is passed to compute_cnn in input order.

    monkeypatch:  Pytest fixture used to replace Echoframe extraction.
    '''

    phrases = (SimpleNamespace(key=b'a'), SimpleNamespace(key=b'b'))
    store = object()
    calls = []

    def compute_cnn(
        phrase,
        model_name,
        output_store,
        collar=0,
        gpu=False,
        overwrite=False,
    ):
        options = {'collar': collar, 'gpu': gpu, 'overwrite': overwrite}
        calls.append((phrase, model_name, output_store, options))

    monkeypatch.setattr(
        echoframe_store.segment_features,
        'compute_cnn',
        compute_cnn,
        raising=False,
    )

    result = echoframe_store.add_cnn_features(
        phrases,
        'checkpoint',
        store,
        collar=25,
        gpu=True,
        overwrite=True,
    )

    assert result is None
    assert [call[0] for call in calls] == list(phrases)
    assert all(call[1] == 'checkpoint' for call in calls)
    assert all(call[2] is store for call in calls)
    expected = {'collar': 25, 'gpu': True, 'overwrite': True}
    assert all(call[3] == expected for call in calls)


def test_load_cnn_features_preserves_phrase_order():
    '''Phrase keys are forwarded in order and the native result is returned.'''

    phrases = (SimpleNamespace(key=b'b'), SimpleNamespace(key=b'a'))
    native_features = object()

    class Store:
        def phraser_keys_to_cnn_features(
            self,
            phraser_keys,
            model_name,
            collar=0,
        ):
            self.call = (phraser_keys, model_name, collar)
            return native_features

    store = Store()
    result = echoframe_store.load_cnn_features(
        phrases,
        'checkpoint',
        store,
        collar=15,
    )

    assert result is native_features
    assert store.call == ([b'b', b'a'], 'checkpoint', 15)


@pytest.mark.parametrize(
    ('aggregation', 'method', 'expected'),
    [
        ('center', 'middle', [[1, 2], [5, 6]]),
        ('mean', 'mean', [[3, 4], [7, 8]]),
    ],
)
def test_make_x_y_aggregates_features_and_aligns_stimulus_ids(
    monkeypatch,
    aggregation,
    method,
    expected,
):
    '''Aggregation produces aligned representation and stimulus-ID arrays.

    monkeypatch:  Pytest fixture used to replace feature loading.
    aggregation:  Public aggregation option under test.
    method:       Expected native CNNFeature aggregation method.
    expected:     Expected representation matrix.
    '''

    phrases = (
        SimpleNamespace(key=b'a', label='pure-tone_f-10'),
        SimpleNamespace(key=b'b', label='pure-tone_f-20'),
    )
    cnn_features = (
        _CNNFeature([1, 2], [3, 4]),
        _CNNFeature([5, 6], [7, 8]),
    )
    features = SimpleNamespace(
        phraser_keys=(b'a', b'b'),
        cnn_features=cnn_features,
    )
    monkeypatch.setattr(
        echoframe_store,
        'load_cnn_features',
        lambda *args, **kwargs: features,
    )
    store = object()

    X, y = echoframe_store.make_x_y(
        phrases,
        'checkpoint',
        store,
        aggregation=aggregation,
    )

    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(
        y,
        ['pure-tone_f-10', 'pure-tone_f-20'],
    )
    for phrase, feature in zip(phrases, cnn_features, strict=True):
        assert feature.calls == [(phrase, method)]


@pytest.mark.parametrize('phrases', [(), []])
def test_make_x_y_rejects_empty_phrases(phrases):
    '''At least one Phrase is required.

    phrases:  Empty iterable under test.
    '''

    store = object()
    with pytest.raises(ValueError, match='must not be empty'):
        echoframe_store.make_x_y(
            phrases,
            'checkpoint',
            store,
            aggregation='mean',
        )


def test_make_x_y_rejects_invalid_aggregation():
    '''Only center and mean are accepted aggregation modes.'''

    phrases = (SimpleNamespace(key=b'a', label='stimulus-a'),)
    store = object()

    with pytest.raises(ValueError, match='center.*mean'):
        echoframe_store.make_x_y(
            phrases,
            'checkpoint',
            store,
            aggregation='first',
        )


def test_make_x_y_rejects_missing_or_misaligned_features(monkeypatch):
    '''Skipped Echoframe results cannot silently misalign X and y.

    monkeypatch:  Pytest fixture used to replace feature loading.
    '''

    phrases = (
        SimpleNamespace(key=b'a', label='stimulus-a'),
        SimpleNamespace(key=b'b', label='stimulus-b'),
    )
    features = SimpleNamespace(
        phraser_keys=(b'a',),
        cnn_features=(_CNNFeature([1], [2]),),
    )
    monkeypatch.setattr(
        echoframe_store,
        'load_cnn_features',
        lambda *args, **kwargs: features,
    )
    store = object()

    with pytest.raises(ValueError, match='missing or not aligned'):
        echoframe_store.make_x_y(
            phrases,
            'checkpoint',
            store,
            aggregation='mean',
        )
