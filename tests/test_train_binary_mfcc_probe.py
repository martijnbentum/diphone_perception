import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import probe_run
from probing import result as probe_result
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


class FakeStore:
    def __init__(self, matrices_by_key):
        self.matrices_by_key = matrices_by_key
        self.load_many_frames_calls = []

    def make_echoframe_key(
        self, output_type, feature_name, phraser_key,
    ):
        return output_type, feature_name, phraser_key

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
    rng, n_target=30, n_other_each=30, other_labels=('a', 't'), dim=39,
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


def _load_saved_probes(probe_dir, target_phoneme, frame, n_splits):
    probe_run_directory = tbp._run_directory(probe_dir, target_phoneme, frame)
    probes = []
    for fold_idx in range(n_splits):
        probe_path, _, _ = probe_run.fold_paths(probe_run_directory,
            probe_run_directory, fold_idx)
        probes.append(joblib.load(probe_path))
    return probes


def test_train_binary_mfcc_probe_end_to_end(tmp_path):
    phones, store = _make_separable_dataset(np.random.default_rng(0))
    probe_dir = tmp_path / 'probes'
    results_dir = tmp_path

    tbp.train_binary_mfcc_probe(
        phones, 'p', store=store, expected_target_count=30,
        verbose=False, probe_save_dir=probe_dir, results_dir=results_dir)

    phone_result = probe_result.PhoneResult.mfcc('p', 'center',
        root=results_dir)
    assert phone_result.run['actual_n_samples'] == 60
    assert phone_result.run['actual_n_missing'] == 0
    assert phone_result.mean_accuracy > .9
    probes = _load_saved_probes(probe_dir, 'p', 'center', n_splits=5)
    assert all(isinstance(probe, LogisticRegression) for probe in probes)

    results_path = phone_result.path / 'results.json'
    saved = json.loads(results_path.read_text(encoding='utf-8'))
    assert saved['kind'] == 'binary_mfcc_probe_results'
    assert saved['results']['p']['mean_accuracy'] == pytest.approx(
        phone_result.mean_accuracy)
    assert 'probes' not in saved['results']['p']


def test_train_binary_mfcc_probe_skips_when_all_folds_already_saved(
    tmp_path, capsys):
    phones, store = _make_separable_dataset(np.random.default_rng(0))
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_mfcc_probe(
        phones, 'p', store=store, expected_target_count=30, verbose=True,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert 'cache status: miss' in capsys.readouterr().out
    calls_after_first = len(store.load_many_frames_calls)
    assert calls_after_first == 1

    first_result = probe_result.PhoneResult.mfcc('p', 'center',
        root=results_dir)
    first_mean_accuracy = first_result.mean_accuracy

    tbp.train_binary_mfcc_probe(
        phones, 'p', store=store, expected_target_count=30, verbose=True,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert 'cache status: hit' in capsys.readouterr().out

    # a true hit touches nothing - zero store calls, not just zero extra
    assert len(store.load_many_frames_calls) == calls_after_first
    second_result = probe_result.PhoneResult.mfcc('p', 'center',
        root=results_dir)
    assert second_result.mean_accuracy == pytest.approx(first_mean_accuracy)


@pytest.mark.multicore
def test_train_binary_mfcc_probes_trains_each_phraser_label(
    tmp_path, capsys):
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    results = tbp.train_binary_mfcc_probes(
        phones,
        target_phonemes=['p', 't'],
        store=store,
        expected_target_count=30,
        verbose=True,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path,
        report=True,
    )

    # results are reordered to match target_phonemes regardless of which
    # worker process happens to finish first
    assert list(results) == ['p', 't']
    assert all(
        result['representation'] == 'mfcc' for result in results.values())
    # one batched load for the whole sweep, not one per label - the
    # concrete proof the redundant-reload problem is fixed
    assert len(store.load_many_frames_calls) == 1
    output = capsys.readouterr().out
    assert '[mfcc pool] 1/2 completed' in output
    assert '[mfcc pool] 2/2 completed' in output
    results_path = tmp_path / 'mfcc' / 'mfcc_probe_results.json'
    saved = json.loads(results_path.read_text(encoding='utf-8'))
    assert saved['target_phonemes'] == ['p', 't']
    assert set(saved['results']) == {'p', 't'}
    assert {
        result['results_path'] for result in results.values()
    } == {str(results_path.resolve())}


@pytest.mark.multicore
def test_train_binary_mfcc_probes_runs_in_a_process_pool(tmp_path, capsys):
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    results = tbp.train_binary_mfcc_probes(
        phones,
        target_phonemes=['p', 't'],
        store=store,
        expected_target_count=30,
        max_workers=2,
        verbose=True,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path,
        report=True,
    )

    assert set(results) == {'p', 't'}
    assert all('mean_accuracy' in result for result in results.values())
    output = capsys.readouterr().out
    assert '[mfcc pool] 1/2 completed' in output
    assert '[mfcc pool] 2/2 completed' in output

    # results are actually persisted to disk by the worker processes
    phone_result = probe_result.PhoneResult.mfcc('p', 'center', root=tmp_path)
    assert phone_result.complete is True


def test_train_binary_mfcc_probes_rejects_unknown_target_before_pooling(
    tmp_path):
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    with pytest.raises(ValueError,
        match='not found in label_to_phraser_phone'):
        tbp.train_binary_mfcc_probes(
            phones,
            target_phonemes=['p', 'nonexistent'],
            store=store,
            expected_target_count=30,
            max_workers=2,
            verbose=False,
            probe_save_dir=tmp_path / 'probes',
            results_dir=tmp_path,
        )

    # failed before the batched matrix load ever touched the store
    assert store.load_many_frames_calls == []


@pytest.mark.multicore
def test_train_binary_mfcc_probes_returns_none_without_report(tmp_path):
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    results = tbp.train_binary_mfcc_probes(
        phones,
        target_phonemes=['p', 't'],
        store=store,
        expected_target_count=30,
        verbose=False,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path,
        save_results=False,
    )

    assert results is None


def test_train_binary_mfcc_probe_opens_default_store(tmp_path, monkeypatch):
    phones, store = _make_separable_dataset(np.random.default_rng(0))
    opened_roots = []

    def fake_store_constructor(root):
        opened_roots.append(root)
        return store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)
    tbp.train_binary_mfcc_probe(
        phones, 'p', expected_target_count=30, verbose=False,
        save_results=False,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results')

    assert opened_roots == [str(locations.echoframe_mfcc_store)]


def test_train_binary_mfcc_probe_rejects_unknown_frame():
    phones, store = _make_separable_dataset(np.random.default_rng(0))

    with pytest.raises(ValueError, match='frame must be one of'):
        tbp.train_binary_mfcc_probe(
            phones, 'p', store=store, frame='middle')


def test_save_mfcc_probe_results_rejects_mismatched_target(tmp_path):
    result = {
        'target_phoneme': 't',
    }

    with pytest.raises(ValueError, match='does not match'):
        tbp.save_mfcc_probe_results(
            {'p': result}, results_dir=tmp_path, verbose=False)
