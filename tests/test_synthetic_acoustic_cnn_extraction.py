from types import SimpleNamespace

import pytest

from synthetic_acoustic_probes import cnn_extraction


def test_extract_cnn_checkpoint_delegates_sequentially(monkeypatch):
    '''Every Phrase is passed to compute_cnn in input order.

    monkeypatch:  Pytest fixture used to replace Echoframe extraction.
    '''

    phrases = (SimpleNamespace(key=b'a'), SimpleNamespace(key=b'b'))
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
        cnn_extraction.segment_features,
        'compute_cnn',
        compute_cnn,
        raising=False,
    )
    remove_calls = []
    store = SimpleNamespace(
        remove_cached_model=lambda: remove_calls.append(True),
    )

    result = cnn_extraction.extract_cnn_checkpoint(
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
    assert remove_calls == [True]


def test_extract_cnn_checkpoint_rejects_empty_phrases():
    '''A checkpoint extraction requires at least one Phrase.'''

    store = object()
    with pytest.raises(ValueError, match='phrases must not be empty'):
        cnn_extraction.extract_cnn_checkpoint((), 'checkpoint', store)


def test_extract_cnn_checkpoint_releases_model_after_failure(monkeypatch):
    '''The cached model is released when extraction raises.

    monkeypatch:  Pytest fixture used to replace Echoframe extraction.
    '''

    phrase = SimpleNamespace(key=b'a')

    def compute_cnn(
        phrase,
        model_name,
        output_store,
        collar=0,
        gpu=False,
        overwrite=False,
    ):
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(
        cnn_extraction.segment_features,
        'compute_cnn',
        compute_cnn,
        raising=False,
    )
    remove_calls = []
    store = SimpleNamespace(
        remove_cached_model=lambda: remove_calls.append(True),
    )

    with pytest.raises(RuntimeError, match='extraction failed'):
        cnn_extraction.extract_cnn_checkpoint((phrase,), 'checkpoint', store)

    assert remove_calls == [True]


def test_extract_cnn_checkpoints_processes_models_in_order(monkeypatch):
    '''The sweep reuses Phrases and releases every checkpoint in order.

    monkeypatch:  Pytest fixture used to replace Echoframe extraction.
    '''

    phrases = (SimpleNamespace(key=b'a'), SimpleNamespace(key=b'b'))
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
        calls.append((phrase.key, model_name, output_store, options))

    monkeypatch.setattr(
        cnn_extraction.segment_features,
        'compute_cnn',
        compute_cnn,
        raising=False,
    )
    remove_calls = []
    store = SimpleNamespace(
        remove_cached_model=lambda: remove_calls.append(True),
    )
    phrase_generator = (phrase for phrase in phrases)

    result = cnn_extraction.extract_cnn_checkpoints(
        phrase_generator,
        ('model-b', 'model-a'),
        store,
        collar=10,
        gpu=True,
        overwrite=True,
    )

    assert result is None
    observed = [(key, model_name) for key, model_name, _, _ in calls]
    assert observed == [
        (b'a', 'model-b'),
        (b'b', 'model-b'),
        (b'a', 'model-a'),
        (b'b', 'model-a'),
    ]
    assert all(call[2] is store for call in calls)
    expected = {'collar': 10, 'gpu': True, 'overwrite': True}
    assert all(call[3] == expected for call in calls)
    assert remove_calls == [True, True]


@pytest.mark.parametrize(
    ('model_names', 'error', 'match'),
    [
        ((), ValueError, 'model_names must not be empty'),
        ('checkpoint', TypeError, 'iterable of strings'),
    ],
)
def test_extract_cnn_checkpoints_rejects_invalid_model_names(
    model_names,
    error,
    match,
):
    '''Invalid model-name collections fail before extraction.

    model_names:  Invalid model-name input.
    error:        Expected exception class.
    match:        Text expected in the error message.
    '''

    phrases = (SimpleNamespace(key=b'a'),)
    store = object()
    with pytest.raises(error, match=match):
        cnn_extraction.extract_cnn_checkpoints(
            phrases,
            model_names,
            store,
        )
