from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import train_binary_mfcc_probe as tbp


class FakePhone:
    def __init__(self, phoneme_ipa):
        self.phoneme_ipa = phoneme_ipa


class FakePhraserPhone:
    def __init__(self, key, label=None):
        self.key = key
        self.label = label


class FakePhones:
    def __init__(self, labels):
        self.phones = [FakePhone(label) for label in labels]
        self.phraser_phones = [
            FakePhraserPhone(index, label)
            for index, label in enumerate(labels)
        ]

    @property
    def label_to_phraser_phone(self):
        grouped = {}
        for phone in self.phraser_phones:
            grouped.setdefault(phone.label, []).append(phone)
        return grouped


class FakeMetadata:
    def __init__(self, key, matrix):
        self.created_at = '2026-01-01T00:00:00+00:00'
        self.dataset_path = f'mfcc/{key[2]}'
        self.shape = matrix.shape
        self.shard_id = 0


class FakeStore:
    def __init__(self, matrices_by_key):
        self.matrices_by_key = matrices_by_key
        self.load_many_frames_calls = []

    def make_echoframe_key(
        self, output_type, feature_name, phraser_key,
    ):
        return output_type, feature_name, phraser_key

    def load_many_metadata(self, keys, keep_missing=False):
        output = []
        for key in keys:
            matrix = self.matrices_by_key.get(key[2])
            metadata = (
                None if matrix is None else FakeMetadata(key, matrix)
            )
            if metadata is not None or keep_missing:
                output.append(metadata)
        return output

    def load_many_frames(self, keys, frame='center', keep_missing=False):
        self.load_many_frames_calls.append({
            'keys': list(keys),
            'frame': frame,
            'keep_missing': keep_missing,
        })
        output = []
        for key in keys:
            matrix = self.matrices_by_key.get(key[2])
            if matrix is None:
                vector = None
            elif frame == 'center':
                vector = matrix[len(matrix) // 2]
            elif frame == 'mean':
                vector = matrix.mean(axis=0)
            elif frame == 'first':
                vector = matrix[0]
            elif frame == 'last':
                vector = matrix[-1]
            if vector is not None or keep_missing:
                output.append(vector)
        return output


def _make_separable_dataset(
    rng, n_target=30, n_other_each=15, other_labels=('a', 't'), dim=39,
):
    labels = ['p'] * n_target
    for label in other_labels:
        labels += [label] * n_other_each
    phones = FakePhones(labels)

    matrices_by_key = {}
    for phone, phraser_phone in zip(
        phones.phones, phones.phraser_phones, strict=True,
    ):
        center = 0.0 if phone.phoneme_ipa == 'p' else 5.0
        matrices_by_key[phraser_phone.key] = (
            center + rng.normal(scale=.01, size=(3, dim))
        )
    return phones, FakeStore(matrices_by_key)


def test_load_mfcc_vectors_uses_center_frame_and_reports_missing():
    selected = [
        (FakePhone('p'), FakePhraserPhone(0), 'target'),
        (FakePhone('p'), FakePhraserPhone(1), 'target'),
        (FakePhone('a'), FakePhraserPhone(2), 'other'),
    ]
    matrices = {
        0: np.array([[0., 1.], [2., 3.], [4., 5.]]),
        2: np.array([[6., 7.], [8., 9.], [10., 11.]]),
    }
    store = FakeStore(matrices)

    X, y, true_labels, missing = tbp._load_mfcc_vectors(
        store, selected)

    np.testing.assert_array_equal(X, [[2., 3.], [8., 9.]])
    assert list(y) == ['target', 'other']
    assert list(true_labels) == ['p', 'a']
    assert [phone.phoneme_ipa for phone in missing] == ['p']
    assert store.load_many_frames_calls == [{
        'keys': [
            ('acoustic_feature', 'mfcc', 0),
            ('acoustic_feature', 'mfcc', 1),
            ('acoustic_feature', 'mfcc', 2),
        ],
        'frame': 'center',
        'keep_missing': True,
    }]


@pytest.mark.parametrize('standardize', [False, True])
def test_train_binary_mfcc_probe_end_to_end(standardize):
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    result = tbp.train_binary_mfcc_probe(
        phones, 'p', store=store, n_samples=30, standardize=standardize,
        verbose=False, save_probes=False, save_predictions=False)

    assert result['representation'] == 'mfcc'
    assert result['frame'] == 'center'
    assert result['n_samples'] == 60
    assert result['n_missing'] == 0
    assert result['mean_accuracy'] > .9
    probe_type = Pipeline if standardize else LogisticRegression
    assert all(isinstance(probe, probe_type) for probe in result['probes'])


def test_train_binary_mfcc_probe_scale_flag_changes_run_identity():
    phones, store = _make_separable_dataset(np.random.default_rng(0))
    arguments = {
        'store': store,
        'n_samples': 30,
        'verbose': False,
        'save_probes': False,
        'save_predictions': False,
    }

    raw = tbp.train_binary_mfcc_probe(
        phones, 'p', standardize=False, **arguments)
    scaled = tbp.train_binary_mfcc_probe(
        phones, 'p', standardize=True, **arguments)

    assert raw['run_id'] != scaled['run_id']
    assert raw['standardize'] is False
    assert scaled['standardize'] is True


def test_train_binary_mfcc_probes_trains_each_phraser_label():
    phones, store = _make_separable_dataset(
        np.random.default_rng(0), n_target=30, n_other_each=30)

    results = tbp.train_binary_mfcc_probes(
        phones,
        target_phonemes=['p', 't'],
        store=store,
        n_samples=30,
        verbose=False,
        save_probes=False,
        save_predictions=False,
    )

    assert list(results) == ['p', 't']
    assert all(
        result['representation'] == 'mfcc'
        for result in results.values()
    )
    assert len(store.load_many_frames_calls) == 2


def test_train_binary_mfcc_probe_opens_default_store(monkeypatch):
    phones, store = _make_separable_dataset(np.random.default_rng(0))
    opened_roots = []

    def fake_store_constructor(root):
        opened_roots.append(root)
        return store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)
    tbp.train_binary_mfcc_probe(
        phones, 'p', n_samples=30, verbose=False, save_probes=False,
        save_predictions=False)

    assert opened_roots == [str(tbp.default_mfcc_store_root)]
    assert str(tbp.default_mfcc_store_root).endswith(
        'data/echoframe_mfcc_store')


def test_train_binary_mfcc_probe_rejects_unknown_frame():
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    with pytest.raises(ValueError, match='frame must be one of'):
        tbp.train_binary_mfcc_probe(
            phones, 'p', store=store, frame='middle')
