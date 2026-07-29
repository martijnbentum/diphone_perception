import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import train_binary_probe as tbp


class _Marker:
    '''Distinguishable stand-in for a saved probe, so a test can tell
    whether a fold's probe was loaded from disk (has .tag) or freshly
    trained (a real LogisticRegression, no .tag attribute).
    '''
    def __init__(self, tag):
        self.tag = tag


# -- fakes -------------------------------------------------------------

class FakePhone:
    def __init__(self, phoneme_ipa):
        self.phoneme_ipa = phoneme_ipa


class FakePhraserPhone:
    def __init__(self, key):
        self.key = key


class FakePhones:
    def __init__(self, labels):
        self.phones = [FakePhone(label) for label in labels]
        self.phraser_phones = [FakePhraserPhone(i) for i in range(len(labels))]


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
        self.phraser_keys_to_embeddings_calls = []

    def phraser_keys_to_embeddings(self, phraser_keys, model_name, layer,
        collar=500):
        self.phraser_keys_to_embeddings_calls.append(
            dict(phraser_keys=list(phraser_keys), model_name=model_name,
                layer=layer, collar=collar))
        embeddings = [
            FakeEmbedding(key, self.vectors_by_key[key])
            for key in phraser_keys if key in self.vectors_by_key
        ]
        return FakeEmbeddings(embeddings)


# -- _select_phones ------------------------------------------------------

def test_select_phones_balances_target_and_other():
    labels = ['p'] * 50 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20
    phones = FakePhones(labels)

    selected = tbp._select_phones(phones, 'p', n_embeds=30, seed=42)

    counts = Counter(label for _, _, label in selected)
    assert counts == {'target': 30, 'other': 30}
    assert len(selected) == 60


def test_select_phones_is_deterministic():
    labels = ['p'] * 50 + ['a'] * 20 + ['t'] * 20
    phones = FakePhones(labels)

    first = tbp._select_phones(phones, 'p', n_embeds=20, seed=42)
    second = tbp._select_phones(phones, 'p', n_embeds=20, seed=42)

    assert [pp.key for _, pp, _ in first] == [pp.key for _, pp, _ in second]


def test_select_phones_none_uses_all_available_target_phones():
    labels = ['p'] * 13500 + ['a'] * 13500 + ['t'] * 13500
    phones = FakePhones(labels)

    selected = tbp._select_phones(phones, 'p')

    counts = Counter(label for _, _, label in selected)
    assert counts['target'] == 13500
    assert counts['other'] == 13500  # 6750 each from 'a' and 't'


def test_select_phones_raises_when_target_missing():
    phones = FakePhones(['a'] * 20 + ['t'] * 20)
    with pytest.raises(ValueError, match='not found'):
        tbp._select_phones(phones, 'p', n_embeds=10)


def test_select_phones_raises_when_target_underfilled():
    phones = FakePhones(['p'] * 5 + ['a'] * 20 + ['t'] * 20)
    with pytest.raises(ValueError, match="'p' has only 5.*need 10"):
        tbp._select_phones(phones, 'p', n_embeds=10)


def test_select_phones_raises_when_other_class_underfilled():
    phones = FakePhones(['p'] * 20 + ['a'] * 3 + ['t'] * 20)
    # n_embeds=10 -> n_per_other = 10 // 2 = 5, but 'a' only has 3
    with pytest.raises(ValueError, match="'a' has only 3.*need 5"):
        tbp._select_phones(phones, 'p', n_embeds=10)


def test_select_phones_raises_when_no_other_classes():
    phones = FakePhones(['p'] * 20)
    with pytest.raises(ValueError, match='no other phoneme classes'):
        tbp._select_phones(phones, 'p', n_embeds=10)


def test_select_phones_raises_when_n_embeds_too_small_to_split():
    phones = FakePhones(['p'] * 20 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20)
    with pytest.raises(ValueError, match='too small to split'):
        tbp._select_phones(phones, 'p', n_embeds=2)


# -- _load_middle_frame_vectors -------------------------------------------

def test_load_middle_frame_vectors_reports_missing():
    selected = [
        (FakePhone('p'), FakePhraserPhone(0), 'target'),
        (FakePhone('p'), FakePhraserPhone(1), 'target'),
        (FakePhone('a'), FakePhraserPhone(2), 'other'),
    ]
    store = FakeStore({0: np.array([1.0, 2.0]), 2: np.array([3.0, 4.0])})
    # key 1 deliberately missing from the store

    X, y, true_labels, missing = tbp._load_middle_frame_vectors(
        store, selected, 'model-a', 9, 500)

    assert X.shape == (2, 2)
    assert list(y) == ['target', 'other']
    assert list(true_labels) == ['p', 'a']
    assert len(missing) == 1
    assert missing[0].phoneme_ipa == 'p'
    call = store.phraser_keys_to_embeddings_calls[0]
    assert call == dict(phraser_keys=[0, 1, 2], model_name='model-a',
        layer=9, collar=500)


# -- train_binary_probe ----------------------------------------------------

def _make_separable_dataset(rng, n_target, n_other_each, other_labels, dim=4):
    target_center = np.zeros(dim)
    other_center = np.full(dim, 5.0)

    labels = ['p'] * n_target
    for label in other_labels:
        labels += [label] * n_other_each
    phones = FakePhones(labels)

    vectors_by_key = {}
    for phone, phraser_phone in zip(phones.phones, phones.phraser_phones):
        center = target_center if phone.phoneme_ipa == 'p' else other_center
        vectors_by_key[phraser_phone.key] = center + rng.normal(
            scale=0.01, size=dim)

    store = FakeStore(vectors_by_key)
    return phones, store


def test_train_binary_probe_end_to_end():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    result = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        save_probes=False, save_predictions=False)

    assert result['target_phoneme'] == 'p'
    assert result['layer'] == 9
    assert result['n_samples'] == 60
    assert result['n_missing'] == 0
    assert len(result['accuracies']) == 5
    assert len(result['probes']) == 5
    assert result['mean_accuracy'] > 0.9  # clusters are well separated
    assert result['skipped'] is False


def test_train_binary_probe_opens_store_when_none_given(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    phones, opened_store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    store_roots = []

    def fake_store_constructor(root):
        store_roots.append(root)
        return opened_store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)

    result = tbp.train_binary_probe(
        phones, 'p', model_name='model-a', layer=9, collar=500,
        store_root=tmp_path / 'store', n_embeds=30, verbose=False,
        save_probes=False, save_predictions=False)

    assert store_roots == [str(tmp_path / 'store')]
    assert result['n_samples'] == 60


def test_train_binary_probe_saves_probes_and_predictions(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    probe_dir = tmp_path / 'probes'
    results_dir = tmp_path / 'results'

    tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        save_probes=True, probe_save_dir=probe_dir,
        save_predictions=True, results_dir=results_dir)

    probe_files = sorted((probe_dir / 'model-a' / 'p').glob('*.joblib'))
    pred_files = sorted((results_dir / 'model-a' / 'p').glob('*.txt'))
    assert len(probe_files) == 5
    assert len(pred_files) == 5

    header = pred_files[0].read_text().splitlines()[0]
    assert header == 'true_phoneme\tbinary_true\tbinary_pred\tcorrect'


# -- skip / overwrite / gap-filling behavior --------------------------------

def test_train_binary_probe_skips_when_all_folds_already_saved(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert first['skipped'] is False
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)
    assert calls_after_first == 1

    second = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert second['skipped'] is True
    assert second['n_samples'] is None
    assert second['n_missing'] is None
    # embeddings were never reloaded - proves the fast path skipped loading
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first
    assert second['accuracies'] == pytest.approx(first['accuracies'])
    assert second['mean_accuracy'] == pytest.approx(first['mean_accuracy'])
    assert all(isinstance(p, LogisticRegression) for p in second['probes'])


def test_train_binary_probe_overwrite_forces_retrain(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)

    second = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir, overwrite=True)

    assert second['skipped'] is False
    # overwrite bypasses the skip check, so embeddings get reloaded
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first + 1


def test_train_binary_probe_fills_gaps_for_partially_saved_folds(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    # pre-seed only fold 0 with a stub probe + a predictions file whose
    # accuracy (1.0, from 2 rows) could not have come from a real fold on
    # this dataset - so we can tell whether it was reused or retrained
    probe_path = tbp._probe_path(probe_dir, 'model-a', 'p', 9, 0)
    pred_path = tbp._predictions_path(results_dir, 'model-a', 'p', 9, 0)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_Marker('fold0-stub'), probe_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        'true_phoneme\tbinary_true\tbinary_pred\tcorrect\n'
        'p\ttarget\ttarget\t1\n'
        'p\ttarget\ttarget\t1\n')

    result = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert result['skipped'] is False
    assert result['accuracies'][0] == 1.0  # loaded from the seeded file
    assert result['probes'][0].tag == 'fold0-stub'  # reused, not retrained
    for probe in result['probes'][1:]:
        assert isinstance(probe, LogisticRegression)  # freshly trained
    # the other 4 folds got written to disk this run
    for fold_idx in range(1, 5):
        assert tbp._probe_path(
            probe_dir, 'model-a', 'p', 9, fold_idx).exists()
        assert tbp._predictions_path(
            results_dir, 'model-a', 'p', 9, fold_idx).exists()


def test_train_binary_probe_retrains_fold_with_orphaned_probe_file(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    # fold 0 has a leftover probe file but no matching predictions file -
    # an inconsistent pair that must not be trusted
    probe_path = tbp._probe_path(probe_dir, 'model-a', 'p', 9, 0)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_Marker('orphan'), probe_path)

    result = tbp.train_binary_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert isinstance(result['probes'][0], LogisticRegression)  # not the orphan
    pred_path = tbp._predictions_path(results_dir, 'model-a', 'p', 9, 0)
    assert pred_path.exists()  # predictions for fold 0 now written too


def test_read_accuracy_matches_saved_fraction(tmp_path):
    path = tmp_path / 'predictions.txt'
    path.write_text(
        'true_phoneme\tbinary_true\tbinary_pred\tcorrect\n'
        'p\ttarget\ttarget\t1\n'
        'a\tother\ttarget\t0\n'
        'p\ttarget\ttarget\t1\n'
        't\tother\tother\t1\n')

    assert tbp._read_accuracy(path) == pytest.approx(0.75)
