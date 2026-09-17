from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
import model_store
from decomposition import extract_embeddings as extraction


def test_load_markers_uses_label_prefix():
    store = Mock()
    markers = [object(), object()]
    store.markers.filter.return_value = iter(markers)
    assert extraction.load_markers(store) == markers
    store.markers.filter.assert_called_once_with(
        label__startswith='decomp_random_frames')


def test_single_model_uses_decomposition_store_and_forwards_options(monkeypatch):
    store, phraser_store = Mock(), Mock()
    constructor, register, compute = Mock(return_value=store), Mock(), Mock()
    monkeypatch.setattr(model_store, 'open_model_store', constructor)
    monkeypatch.setattr(model_store, 'ensure_model_registered', register)
    monkeypatch.setattr(extraction, 'compute_embeddings_batch', compute)
    markers, layers = [Mock(store=phraser_store)], [8, 9]
    result = extraction.extract_marker_embeddings(iter(markers),
        model_name='model-a', layers=layers, collar=500, gpu=True,
        batch_size=16, tags=['sample'], verbose=False)
    assert result is store
    constructor.assert_called_once_with('model-a',
        stores_root=locations.decomposition_random_frames_echoframe_model_stores)
    register.assert_not_called()
    store.attach_phraser_store.assert_called_once_with('cgn-awd', phraser_store)
    compute.assert_called_once_with(markers, [8, 9, 'cnn'], 'model-a', store,
        collar=500, gpu=True, tags=['sample'], batch_size=16, verbose=False)
    assert layers == [8, 9]
    store.close.assert_not_called()


def test_supplied_store_and_cnn_are_preserved(monkeypatch):
    store, compute = Mock(), Mock()
    register = Mock()
    monkeypatch.setattr(model_store, 'ensure_model_registered', register)
    monkeypatch.setattr(extraction, 'compute_embeddings_batch', compute)
    constructor = Mock(side_effect=AssertionError('unexpected store creation'))
    monkeypatch.setattr(model_store, 'open_model_store', constructor)
    extraction.extract_marker_embeddings([Mock()], store=store,
        layers=[9, 'cnn'])
    assert compute.call_args.args[1] == [9, 'cnn']
    register.assert_called_once_with(
        store, extraction.default_model_name, locations.model_paths_file)


@pytest.mark.parametrize('failure', [None, 'extraction', 'unload'])
def test_models_reuse_markers_and_clean_up(monkeypatch, tmp_path, failure):
    stores = [Mock(root=tmp_path / 'model-a'),
        Mock(root=tmp_path / 'owner%2Fmodel-b')]
    opened = Mock(side_effect=stores)
    extract, release = Mock(), Mock()
    if failure == 'extraction': extract.side_effect = RuntimeError('extraction')
    if failure == 'unload':
        stores[0].remove_cached_model.side_effect = RuntimeError('unload')
    monkeypatch.setattr(model_store, 'open_model_store', opened)
    monkeypatch.setattr(model_store, 'release_cuda_memory', release)
    monkeypatch.setattr(extraction, 'extract_marker_embeddings', extract)
    monkeypatch.setattr(locations,
        'decomposition_random_frames_echoframe_model_stores', tmp_path)
    phraser_store = Mock()
    markers = [Mock(store=phraser_store), Mock(store=phraser_store)]
    args = (iter(markers), ['model-a', 'owner/model-b'])
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            extraction.extract_marker_embeddings_for_models(*args, gpu=True)
        stores[0].close.assert_called_once()
        release.assert_called_once()
        assert opened.call_count == 1
    else:
        result = extraction.extract_marker_embeddings_for_models(*args, gpu=True)
        assert result == {'model-a': tmp_path / 'model-a',
            'owner/model-b': tmp_path / 'owner%2Fmodel-b'}
        assert opened.call_count == 2
        for call in opened.call_args_list:
            assert call.kwargs['stores_root'] == tmp_path
        for call, store in zip(extract.call_args_list, stores):
            assert call.args == (markers,)
            assert call.kwargs['store'] is store
            store.remove_cached_model.assert_called_once()
            store.close.assert_called_once()
        assert release.call_count == 2
    phraser_store.close.assert_not_called()


def test_model_names_reject_string_before_opening_store(monkeypatch):
    opened = Mock()
    monkeypatch.setattr(model_store, 'open_model_store', opened)
    with pytest.raises(TypeError, match='not a string'):
        extraction.extract_marker_embeddings_for_models([], 'model-a')
    opened.assert_not_called()
