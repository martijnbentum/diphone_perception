'''Check stored marker MFCC loading and alignment.'''

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import locations
from decomposition import load_mfcc


def make_store(vectors):
    '''Return a mock store with payloads mapped by marker key.

    vectors:  marker keys mapped to stored vectors or None
    '''
    store = Mock()
    def make_key(kind, feature_name, phraser_key):
        return kind, feature_name, phraser_key
    def load_frames(keys, frame, keep_missing):
        rows = []
        for key in keys: rows.append(vectors.get(key[2]))
        return rows
    store.make_echoframe_key.side_effect = make_key
    store.load_many_frames.side_effect = load_frames
    return store


def test_load_mfcc_preserves_marker_order_and_keeps_supplied_store_open():
    '''Return one row per marker in input order.'''
    markers = [SimpleNamespace(key=b'b'), SimpleNamespace(key=b'a')]
    store = make_store({b'a': np.ones(39), b'b': np.full(39, 2.0)})

    matrix = load_mfcc.load_mfcc(markers, store=store)

    assert matrix.shape == (2, 39)
    np.testing.assert_array_equal(matrix[:, 0], [2.0, 1.0])
    store.load_many_frames.assert_called_once()
    kwargs = {'frame': 'center', 'keep_missing': True}
    assert store.load_many_frames.call_args.kwargs == kwargs
    store.close.assert_not_called()


def test_load_mfcc_closes_owned_store_on_missing_row(monkeypatch):
    '''Report missing MFCCs without leaving the opened store alive.'''
    store = make_store({})
    constructor = Mock(return_value=store)
    monkeypatch.setattr(load_mfcc.echoframe, 'Store', constructor)
    marker = SimpleNamespace(key=b'missing')

    with pytest.raises(ValueError, match='missing MFCC for marker 0'):
        load_mfcc.load_mfcc([marker])

    root = locations.decomposition_random_frames_echoframe_mfcc_store
    root_string = str(root)
    constructor.assert_called_once_with(root_string)
    store.close.assert_called_once_with()


def test_load_mfcc_rejects_malformed_vector():
    '''Reject a stored row with a feature count other than 39.'''
    marker = SimpleNamespace(key=b'wrong-shape')
    store = make_store({marker.key: np.zeros(38)})

    with pytest.raises(ValueError, match='invalid MFCC shape'):
        load_mfcc.load_mfcc([marker], store=store)


def test_load_mfcc_rejects_empty_input():
    '''Require at least one marker before opening a store.'''
    with pytest.raises(ValueError, match='markers must not be empty'):
        load_mfcc.load_mfcc([])
