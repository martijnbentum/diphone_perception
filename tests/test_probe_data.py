from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_data


class FakePhone:
    def __init__(self, phoneme_ipa):
        self.phoneme_ipa = phoneme_ipa


class FakePhraserPhone:
    def __init__(self, key):
        self.key = key


class FakePhones:
    def __init__(self, labels):
        self.phones = [FakePhone(label) for label in labels]
        self.phraser_phones = [
            FakePhraserPhone(index) for index in range(len(labels))]


class FakeEmbedding:
    def __init__(self, phraser_key, vector):
        self.phraser_key = phraser_key
        self._vector = vector

    def middle_frame_segment(self, phraser_phone):
        return self._vector


class FakeEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeStore:
    def __init__(self, vectors_by_key):
        self.vectors_by_key = vectors_by_key
        self.calls = []

    def phraser_keys_to_embeddings(self, phraser_keys, model_name, layer,
        collar=500):
        self.calls.append(dict(phraser_keys=list(phraser_keys),
            model_name=model_name, layer=layer, collar=collar))
        embeddings = [
            FakeEmbedding(key, self.vectors_by_key[key])
            for key in phraser_keys if key in self.vectors_by_key]
        return FakeEmbeddings(embeddings)


def test_build_probe_matrix_returns_aligned_arrays():
    phones = FakePhones(['p', 't', 'p', 't'])
    store = FakeStore({
        0: np.array([0., 1.]), 1: np.array([1., 0.]),
        2: np.array([2., 2.]), 3: np.array([3., 3.])})

    matrix = probe_data.build_probe_matrix(
        phones, store, 'model-a', layer=9, collar=2000,
        expected_target_count=2)

    assert matrix.X.shape == (4, 2)
    np.testing.assert_array_equal(matrix.X[0], [0., 1.])
    assert list(matrix.phone_labels) == ['p', 't', 'p', 't']
    assert matrix.phraser_keys == [0, 1, 2, 3]
    assert matrix.missing == []
    call = store.calls[0]
    assert call['phraser_keys'] == [0, 1, 2, 3]
    assert call['model_name'] == 'model-a'
    assert call['layer'] == 9
    assert call['collar'] == 2000


def test_build_probe_matrix_tracks_missing_embeddings():
    phones = FakePhones(['p', 'p', 't', 't'])
    store = FakeStore({0: np.array([0., 1.]), 2: np.array([2., 2.])})
    # key 1 ('p') and key 3 ('t') have no stored embedding

    matrix = probe_data.build_probe_matrix(phones, store, 'model-a', layer=9,
        expected_target_count=1)

    assert matrix.X.shape == (2, 2)
    assert list(matrix.phone_labels) == ['p', 't']
    assert matrix.phraser_keys == [0, 2]
    assert matrix.missing == [1, 3]


def test_build_probe_matrix_rejects_duplicate_phraser_keys():
    phones = FakePhones(['p', 't'])
    phones.phraser_phones[1] = phones.phraser_phones[0]
    store = FakeStore({0: np.array([0., 1.])})

    with pytest.raises(ValueError, match='duplicate Phraser key'):
        probe_data.build_probe_matrix(phones, store, 'model-a', layer=9)


def test_build_probe_matrix_rejects_label_count_mismatch():
    phones = FakePhones(['p', 'p', 't'])
    store = FakeStore({
        0: np.array([0., 1.]), 1: np.array([1., 1.]), 2: np.array([2., 2.])})

    with pytest.raises(ValueError,
        match=r"expected 2 tokens per label.*'t': 1"):
        probe_data.build_probe_matrix(phones, store, 'model-a', layer=9,
            expected_target_count=2)


def test_build_probe_matrix_rejects_entirely_missing_label():
    phones = FakePhones(['p', 'p', 't'])
    store = FakeStore({0: np.array([0., 1.]), 1: np.array([1., 1.])})
    # 't' (key 2) has no stored embedding at all

    with pytest.raises(ValueError,
        match=r"expected 2 tokens per label.*'t': 0"):
        probe_data.build_probe_matrix(phones, store, 'model-a', layer=9,
            expected_target_count=2)


def test_token_counts_counts_labels():
    counts = probe_data.token_counts(['p', 't', 'p', 'k', 'p'])
    assert counts == {'p': 3, 't': 1, 'k': 1}


def test_token_counts_empty():
    assert probe_data.token_counts([]) == {}


def test_describe_probe_run_returns_expected_dict():
    description = probe_data.describe_probe_run(
        ['p', 't', 'p', 'k'], 'p', 'embedding', expected_token_count=2)

    assert description == {'target_phoneme': 'p', 'n_target': 2,
        'n_other': 2, 'representation': 'embedding'}


def test_describe_probe_run_target_with_zero_occurrences():
    description = probe_data.describe_probe_run(
        ['t', 'k'], 'p', 'mfcc', expected_token_count=0)

    assert description == {'target_phoneme': 'p', 'n_target': 0,
        'n_other': 2, 'representation': 'mfcc'}


def test_describe_probe_run_rejects_invalid_target_phoneme():
    with pytest.raises(TypeError, match='non-empty string'):
        probe_data.describe_probe_run(
            ['p', 't'], '', 'embedding', expected_token_count=0)


def test_describe_probe_run_rejects_mismatched_expected_token_count():
    with pytest.raises(ValueError, match='expected 3 tokens.*found 2'):
        probe_data.describe_probe_run(
            ['p', 't', 'p', 'k'], 'p', 'embedding', expected_token_count=3)


def _matrix_with_labels(labels, dim=2):
    X = np.arange(len(labels) * dim, dtype=float).reshape(len(labels), dim)
    phraser_keys = list(range(len(labels)))
    return probe_data.ProbeMatrix(X, np.array(labels), phraser_keys, [])


def test_select_balanced_vectors_balances_target_and_other():
    matrix = _matrix_with_labels(['p'] * 30 + ['a'] * 20 + ['t'] * 20)

    X, y, true_labels, missing = probe_data.select_balanced_vectors(
        matrix, 'p')

    counts = Counter(y.tolist())
    assert counts == {'target': 30, 'other': 30}  # 15 each from 'a' and 't'
    assert X.shape == (60, 2)
    assert len(true_labels) == 60
    assert missing == []


def test_select_balanced_vectors_is_deterministic():
    matrix = _matrix_with_labels(['p'] * 20 + ['a'] * 20 + ['t'] * 20)

    first = probe_data.select_balanced_vectors(matrix, 'p')
    second = probe_data.select_balanced_vectors(matrix, 'p')

    np.testing.assert_array_equal(first[0], second[0])
    assert list(first[1]) == list(second[1])


def test_select_balanced_vectors_uses_every_available_target_phone():
    matrix = _matrix_with_labels(
        ['p'] * 13500 + ['a'] * 13500 + ['t'] * 13500)

    _, y, _, _ = probe_data.select_balanced_vectors(matrix, 'p')

    counts = Counter(y.tolist())
    assert counts['target'] == 13500
    assert counts['other'] == 13500  # 6750 each from 'a' and 't'


def test_select_balanced_vectors_raises_when_target_missing():
    matrix = _matrix_with_labels(['a'] * 20 + ['t'] * 20)

    with pytest.raises(ValueError, match='not found'):
        probe_data.select_balanced_vectors(matrix, 'p')


def test_select_balanced_vectors_raises_when_other_class_underfilled():
    matrix = _matrix_with_labels(['p'] * 20 + ['a'] * 3 + ['t'] * 20)
    # target 'p' has 20 -> n_per_other = 20 // 2 = 10, but 'a' only has 3

    with pytest.raises(ValueError, match="'a' has only 3.*need 10"):
        probe_data.select_balanced_vectors(matrix, 'p')


def test_select_balanced_vectors_raises_when_no_other_classes():
    matrix = _matrix_with_labels(['p'] * 20)

    with pytest.raises(ValueError, match='no other phoneme classes'):
        probe_data.select_balanced_vectors(matrix, 'p')


def test_select_balanced_vectors_raises_when_too_small_to_split():
    matrix = _matrix_with_labels(
        ['p'] * 2 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20)
    # target 'p' has 2 -> n_per_other = 2 // 3 = 0 across three other labels

    with pytest.raises(ValueError, match='too small to split'):
        probe_data.select_balanced_vectors(matrix, 'p')
