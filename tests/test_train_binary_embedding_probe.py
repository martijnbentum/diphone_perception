import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_utils
from probing import train_binary_embedding_probe as tbp


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


class FakeEmbedding:
    def __init__(self, phraser_key, vector):
        self.phraser_key = phraser_key
        self._vector = vector

    def middle_frame_segment(self, phraser_phone):
        return self._vector


class FakeEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeMetadata:
    def __init__(self, echoframe_key, vector):
        self.echoframe_key = echoframe_key
        self.created_at = '2026-01-01T00:00:00+00:00'
        self.dataset_path = f'embeddings/{echoframe_key[2]}'
        self.shape = vector.shape
        self.shard_id = 0


class FakeStore:
    def __init__(self, vectors_by_key):
        self.vectors_by_key = vectors_by_key
        self.phraser_keys_to_embeddings_calls = []
        self.closed = False

    def close(self):
        self.closed = True

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

    def make_echoframe_key(
        self, output_type, model_name, phraser_key, layer, collar,
    ):
        return output_type, model_name, phraser_key, layer, collar

    def load_many_metadata(self, echoframe_keys, keep_missing=False):
        metadatas = []
        for key in echoframe_keys:
            vector = self.vectors_by_key.get(key[2])
            metadata = (
                FakeMetadata(key, vector) if vector is not None else None)
            if metadata is not None or keep_missing:
                metadatas.append(metadata)
        return metadatas


# -- _select_phones ------------------------------------------------------

def test_select_phones_balances_target_and_other():
    labels = ['p'] * 50 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20
    phones = FakePhones(labels)

    selected = probe_utils.select_phones(
        phones, 'p', n_samples=30, seed=42)

    counts = Counter(label for _, _, label in selected)
    assert counts == {'target': 30, 'other': 30}
    assert len(selected) == 60


def test_select_phones_is_deterministic():
    labels = ['p'] * 50 + ['a'] * 20 + ['t'] * 20
    phones = FakePhones(labels)

    first = probe_utils.select_phones(
        phones, 'p', n_samples=20, seed=42)
    second = probe_utils.select_phones(
        phones, 'p', n_samples=20, seed=42)

    assert [pp.key for _, pp, _ in first] == [pp.key for _, pp, _ in second]


def test_select_phones_none_uses_all_available_target_phones():
    labels = ['p'] * 13500 + ['a'] * 13500 + ['t'] * 13500
    phones = FakePhones(labels)

    selected = probe_utils.select_phones(phones, 'p')

    counts = Counter(label for _, _, label in selected)
    assert counts['target'] == 13500
    assert counts['other'] == 13500  # 6750 each from 'a' and 't'


def test_select_phones_raises_when_target_missing():
    phones = FakePhones(['a'] * 20 + ['t'] * 20)
    with pytest.raises(ValueError, match='not found'):
        probe_utils.select_phones(phones, 'p', n_samples=10)


def test_select_phones_raises_when_target_underfilled():
    phones = FakePhones(['p'] * 5 + ['a'] * 20 + ['t'] * 20)
    with pytest.raises(ValueError, match="'p' has only 5.*need 10"):
        probe_utils.select_phones(phones, 'p', n_samples=10)


def test_select_phones_raises_when_other_class_underfilled():
    phones = FakePhones(['p'] * 20 + ['a'] * 3 + ['t'] * 20)
    # n_embeds=10 -> n_per_other = 10 // 2 = 5, but 'a' only has 3
    with pytest.raises(ValueError, match="'a' has only 3.*need 5"):
        probe_utils.select_phones(phones, 'p', n_samples=10)


def test_select_phones_raises_when_no_other_classes():
    phones = FakePhones(['p'] * 20)
    with pytest.raises(ValueError, match='no other phoneme classes'):
        probe_utils.select_phones(phones, 'p', n_samples=10)


def test_select_phones_raises_when_n_embeds_too_small_to_split():
    phones = FakePhones(['p'] * 20 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20)
    with pytest.raises(ValueError, match='too small to split'):
        probe_utils.select_phones(phones, 'p', n_samples=2)


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


# -- train_binary_embedding_probe ----------------------------------------------------

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


def _fold_paths(probe_dir, results_dir, result, fold_idx):
    probe_run = tbp._run_directory(
        probe_dir, 'model-a', 'p', 9, result['collar'], result['run_id'])
    predictions_run = tbp._run_directory(
        results_dir, 'model-a', 'p', 9, result['collar'], result['run_id'])
    return probe_utils.fold_paths(probe_run, predictions_run, fold_idx)


def test_train_binary_embedding_probe_end_to_end():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    result = tbp.train_binary_embedding_probe(
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
    assert result['cache_status'] == 'disabled'


def test_train_binary_embedding_probe_passes_default_collar_to_echoframe():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    result = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9,
        n_embeds=30, verbose=False, save_probes=False,
        save_predictions=False)

    assert store.phraser_keys_to_embeddings_calls[0]['collar'] == 2000
    assert result['collar'] == 2000


def test_train_binary_embedding_probe_rejects_too_few_splits():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    with pytest.raises(ValueError, match='at least 2'):
        tbp.train_binary_embedding_probe(phones, 'p', store=store, n_splits=0)


def test_train_binary_embedding_probe_rejects_nonboolean_scale_flag():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    with pytest.raises(TypeError, match='standardize must be a boolean'):
        tbp.train_binary_embedding_probe(
            phones, 'p', store=store, standardize='yes')


def test_train_binary_embedding_probe_rejects_duplicate_phraser_keys():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    phones.phraser_phones[1].key = phones.phraser_phones[0].key

    with pytest.raises(ValueError, match='duplicate Phraser key'):
        tbp.train_binary_embedding_probe(
            phones, 'p', store=store, verbose=False)


def test_train_binary_embedding_probe_standardization_is_fold_local():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    raw = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', n_embeds=30,
        standardize=False, verbose=False, save_probes=False,
        save_predictions=False)
    scaled = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', n_embeds=30,
        standardize=True, verbose=False, save_probes=False,
        save_predictions=False)

    assert raw['run_id'] != scaled['run_id']
    assert all(isinstance(probe, LogisticRegression) for probe in raw['probes'])
    assert all(isinstance(probe, Pipeline) for probe in scaled['probes'])
    assert all(
        hasattr(probe.named_steps['standardscaler'], 'mean_')
        for probe in scaled['probes']
    )
    assert scaled['standardize'] is True


def test_train_binary_embedding_probes_trains_each_phraser_label(capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    results = tbp.train_binary_embedding_probes(
        phones,
        target_phonemes=['p', 'a'],
        store=store,
        model_name='model-a',
        n_embeds=30,
        verbose=True,
        save_probes=False,
        save_predictions=False,
    )

    assert list(results) == ['p', 'a']
    assert all(
        result['representation'] == 'embedding'
        for result in results.values()
    )
    assert len(store.phraser_keys_to_embeddings_calls) == 2
    output = capsys.readouterr().out
    assert "[embedding probes] 1/2 starting 'p'" in output
    assert '[embedding probes] 2/2 completed' in output
    assert 'ETA ' in output


def test_train_binary_embedding_probes_rejects_unbalanced_inventory():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=29, other_labels=['a', 't'])

    with pytest.raises(ValueError, match='not balanced'):
        tbp.train_binary_embedding_probes(
            phones, store=store, verbose=False)

    assert store.phraser_keys_to_embeddings_calls == []


def test_train_binary_embedding_probes_opens_one_shared_store(
    tmp_path, monkeypatch,
):
    rng = np.random.default_rng(0)
    phones, opened_store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    opened_roots = []

    def fake_store_constructor(root):
        opened_roots.append(root)
        return opened_store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)
    tbp.train_binary_embedding_probes(
        phones,
        target_phonemes=['p', 'a'],
        store_root=tmp_path / 'store',
        model_name='model-a',
        n_embeds=30,
        verbose=False,
        save_probes=False,
        save_predictions=False,
    )

    assert opened_roots == [str(tmp_path / 'store')]
    assert opened_store.closed is True


def test_train_binary_embedding_probe_opens_store_when_none_given(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    phones, opened_store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    store_roots = []

    def fake_store_constructor(root):
        store_roots.append(root)
        return opened_store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)

    result = tbp.train_binary_embedding_probe(
        phones, 'p', model_name='model-a', layer=9, collar=500,
        store_root=tmp_path / 'store', n_embeds=30, verbose=False,
        save_probes=False, save_predictions=False)

    assert store_roots == [str(tmp_path / 'store')]
    assert result['n_samples'] == 60


def test_train_binary_embedding_probe_saves_probes_and_predictions(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])

    probe_dir = tmp_path / 'probes'
    results_dir = tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        save_probes=True, probe_save_dir=probe_dir,
        save_predictions=True, results_dir=results_dir)

    probe_files = sorted((probe_dir / 'model-a' / 'p').rglob('*.joblib'))
    pred_files = sorted((results_dir / 'model-a' / 'p').rglob('*.tsv'))
    assert len(probe_files) == 5
    assert len(pred_files) == 5

    header = pred_files[0].read_text().splitlines()[0]
    assert header == 'true_phoneme\tbinary_true\tbinary_pred\tcorrect'
    assert len(list(probe_dir.rglob('*_complete.json'))) == 5
    assert len(list(probe_dir.rglob('run.json'))) == 1
    assert len(list(results_dir.rglob('run.json'))) == 1


# -- skip / overwrite / gap-filling behavior --------------------------------

def test_train_binary_embedding_probe_skips_when_all_folds_already_saved(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert first['skipped'] is False
    assert first['cache_status'] == 'miss'
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)
    assert calls_after_first == 1

    second = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert second['skipped'] is True
    assert second['cache_status'] == 'hit'
    assert second['run_id'] == first['run_id']
    assert second['n_samples'] is None
    assert second['n_missing'] is None
    # embeddings were never reloaded - proves the fast path skipped loading
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first
    assert second['accuracies'] == pytest.approx(first['accuracies'])
    assert second['mean_accuracy'] == pytest.approx(first['mean_accuracy'])
    assert all(isinstance(p, LogisticRegression) for p in second['probes'])


def test_train_binary_embedding_probe_overwrite_forces_retrain(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)

    second = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir, overwrite=True)

    assert second['skipped'] is False
    assert second['cache_status'] == 'refresh'
    # overwrite bypasses the skip check, so embeddings get reloaded
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first + 1


def test_train_binary_embedding_probe_fills_gaps_for_partially_saved_folds(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    # Make fold 0 distinguishable, while preserving its valid completion
    # marker, then remove the other four folds to create a partial run.
    probe_path, _, marker_path = _fold_paths(
        probe_dir, results_dir, first, 0)
    joblib.dump(_Marker('fold0-stub'), probe_path)
    marker = probe_utils._read_json(marker_path)
    marker['probe_sha256'] = probe_utils._sha256_file(probe_path)
    probe_utils._write_json(marker_path, marker)
    for fold_idx in range(1, 5):
        for path in _fold_paths(probe_dir, results_dir, first, fold_idx):
            path.unlink()

    result = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert result['skipped'] is False
    assert result['cache_status'] == 'partial'
    assert result['probes'][0].tag == 'fold0-stub'  # reused, not retrained
    for probe in result['probes'][1:]:
        assert isinstance(probe, LogisticRegression)  # freshly trained
    # the other 4 folds got written to disk this run
    for fold_idx in range(1, 5):
        assert all(path.exists() for path in _fold_paths(
            probe_dir, results_dir, first, fold_idx))


def test_train_binary_embedding_probe_retrains_fold_with_orphaned_probe_file(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    # Fold 0 has a leftover probe but neither predictions nor a completion
    # marker, so it must be retrained while the other folds remain reusable.
    probe_path, pred_path, marker_path = _fold_paths(
        probe_dir, results_dir, first, 0)
    joblib.dump(_Marker('orphan'), probe_path)
    pred_path.unlink()
    marker_path.unlink()

    result = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert isinstance(result['probes'][0], LogisticRegression)  # not the orphan
    assert result['cache_status'] == 'partial'
    assert pred_path.exists()
    assert marker_path.exists()


def test_train_binary_embedding_probe_retrains_fold_with_bad_checksum(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    _, pred_path, marker_path = _fold_paths(
        probe_dir, results_dir, first, 0)
    pred_path.write_text(pred_path.read_text() + 'corrupt\n')

    result = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, n_splits=5, random_state=42, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    assert result['cache_status'] == 'partial'
    assert probe_utils._load_cached_fold(
        _fold_paths(probe_dir, results_dir, result, 0),
        result['run_id'], 0) is not None
    marker = probe_utils._read_json(marker_path)
    assert (
        marker['predictions_sha256']
        == probe_utils._sha256_file(pred_path)
    )


def test_failed_overwrite_invalidates_completion_marker(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'
    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, verbose=False, probe_save_dir=probe_dir,
        results_dir=results_dir)
    paths = _fold_paths(probe_dir, results_dir, first, 0)

    def fail_to_save(*args):
        raise RuntimeError('interrupted prediction write')

    monkeypatch.setattr(probe_utils, '_save_predictions', fail_to_save)
    with pytest.raises(RuntimeError, match='interrupted'):
        tbp.train_binary_embedding_probe(
            phones, 'p', store=store, model_name='model-a', layer=9,
            collar=500, n_embeds=30, verbose=False,
            probe_save_dir=probe_dir, results_dir=results_dir, overwrite=True)

    assert not paths[2].exists()
    assert probe_utils._load_cached_fold(
        paths, first['run_id'], 0) is None


def test_train_binary_embedding_probe_does_not_reuse_a_different_collar(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    first = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        n_embeds=30, verbose=False, probe_save_dir=probe_dir,
        results_dir=results_dir)
    second = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=2000,
        n_embeds=30, verbose=False, probe_save_dir=probe_dir,
        results_dir=results_dir)

    assert second['run_id'] != first['run_id']
    assert second['cache_status'] == 'miss'
    assert second['skipped'] is False
    assert len(store.phraser_keys_to_embeddings_calls) == 2


def test_train_binary_embedding_probe_run_id_tracks_embedding_availability(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    restored_vector = store.vectors_by_key.pop(0)
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'
    arguments = {
        'store': store,
        'model_name': 'model-a',
        'layer': 9,
        'collar': 500,
        'n_embeds': 30,
        'verbose': False,
        'probe_save_dir': probe_dir,
        'results_dir': results_dir,
    }

    first = tbp.train_binary_embedding_probe(phones, 'p', **arguments)
    store.vectors_by_key[0] = restored_vector
    second = tbp.train_binary_embedding_probe(phones, 'p', **arguments)

    assert first['n_missing'] == 1
    assert second['n_missing'] == 0
    assert second['run_id'] != first['run_id']
    assert second['cache_status'] == 'miss'


@pytest.mark.parametrize(
    'changed',
    [
        {'n_embeds': 20},
        {'n_splits': 3},
        {'random_state': 7},
    ],
)
def test_train_binary_embedding_probe_run_id_covers_training_settings(tmp_path, changed):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'
    arguments = {
        'store': store,
        'model_name': 'model-a',
        'layer': 9,
        'collar': 500,
        'n_embeds': 30,
        'n_splits': 5,
        'random_state': 42,
        'verbose': False,
        'probe_save_dir': probe_dir,
        'results_dir': results_dir,
    }

    first = tbp.train_binary_embedding_probe(phones, 'p', **arguments)
    second = tbp.train_binary_embedding_probe(phones, 'p', **(arguments | changed))

    assert second['run_id'] != first['run_id']
    assert second['cache_status'] == 'miss'
