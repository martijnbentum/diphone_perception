import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_run, probe_utils
from probing import result as probe_result
from probing import train_binary_embedding_probe as tbp


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
        self.store = object()
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


class FakeCNNFeatures:
    def __init__(self, features):
        self.cnn_features = features


class FakeStore:
    def __init__(self, vectors_by_key):
        self.vectors_by_key = vectors_by_key
        self.attached_phraser_stores = []
        self.phraser_keys_to_embeddings_calls = []
        self.phraser_keys_to_cnn_features_calls = []
        self.closed = False

    def close(self):
        self.closed = True

    def attach_phraser_store(self, source_id, phraser_store):
        self.attached_phraser_stores.append((source_id, phraser_store))

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

    def phraser_keys_to_cnn_features(self, phraser_keys, model_name,
        collar=500):
        self.phraser_keys_to_cnn_features_calls.append(
            dict(phraser_keys=list(phraser_keys), model_name=model_name,
                collar=collar))
        features = [
            FakeEmbedding(key, self.vectors_by_key[key])
            for key in phraser_keys if key in self.vectors_by_key
        ]
        return FakeCNNFeatures(features)

# -- checkpoint discovery -------------------------------------------------

def test_discover_wav2vec2_checkpoint_stores_filters_and_sorts(tmp_path):
    directory_names = (
        'wav2vec2_nl1_checkpoint-200000',
        'wav2vec2_nl1_checkpoint-1000',
        'wav2vec2_checkpoint-0',
        'wav2vec2_checkpoint-1000',
        'hubert_nl1_checkpoint-1000',
        'wav2vec2_nl1_not-a-checkpoint',
    )
    for name in directory_names:
        (tmp_path / name).mkdir()
    (tmp_path / 'wav2vec2_nl1_checkpoint-500').write_text('not a store')

    stores = tbp.discover_wav2vec2_checkpoint_stores(tmp_path)

    assert stores == [
        ('wav2vec2_checkpoint-0', tmp_path / 'wav2vec2_checkpoint-0'),
        ('wav2vec2_nl1_checkpoint-1000',
            tmp_path / 'wav2vec2_nl1_checkpoint-1000'),
        ('wav2vec2_nl1_checkpoint-200000',
            tmp_path / 'wav2vec2_nl1_checkpoint-200000'),
    ]


@pytest.mark.parametrize(
    ('model_name', 'layers'),
    [
        ('wav2vec2_checkpoint-0', (*range(1, 13), 'cnn')),
        ('wav2vec2_nl1_checkpoint-1000', (9, 'cnn')),
        ('wav2vec2_nl1_checkpoint-200000', (*range(1, 13), 'cnn')),
    ],
)
def test_checkpoint_probe_layers(model_name, layers):
    assert tbp.checkpoint_probe_layers(model_name) == layers


@pytest.mark.parametrize(
    'model_name',
    ['wav2vec2_checkpoint-1', 'hubert_nl1_checkpoint-1000', 'checkpoint-1'],
)
def test_checkpoint_probe_layers_rejects_unsupported_models(model_name):
    with pytest.raises(ValueError, match='unsupported checkpoint'):
        tbp.checkpoint_probe_layers(model_name)


def test_find_missing_checkpoint_probe_results_reports_folds_and_manifest(
    tmp_path, monkeypatch,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / 'stores' / model_name
    results_dir = tmp_path / 'results'
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    phones = FakePhones(['p', 'p', 'a', 'a'])

    complete_without_manifest = probe_result.PhoneResult.model_feature(
        'p', model_name, 9, 500, root=results_dir)
    for fold_number in range(1, 6):
        probe_result.Fold(complete_without_manifest, fold_number).save_results(
            [('p', 'target', 'target')])

    partial_with_manifest = probe_result.PhoneResult.model_feature(
        'a', model_name, 9, 500, root=results_dir)
    partial_with_manifest.save_run({'representation': 'embedding'})
    probe_result.Fold(partial_with_manifest, 1).save_results(
        [('a', 'target', 'target')])

    missing = tbp.find_missing_checkpoint_probe_results(
        phones, store_root=tmp_path / 'stores', collar=500,
        results_dir=results_dir)

    assert missing == {
        (model_name, 9): [{
            'target_phoneme': 'a',
            'missing_fold_numbers': [2, 3, 4, 5],
            'run_manifest_missing': False,
        }],
        (model_name, 'cnn'): [{
            'target_phoneme': 'a',
            'missing_fold_numbers': [1, 2, 3, 4, 5],
            'run_manifest_missing': True,
        }, {
            'target_phoneme': 'p',
            'missing_fold_numbers': [1, 2, 3, 4, 5],
            'run_manifest_missing': True,
        }],
    }


# -- _select_phones ------------------------------------------------------

def test_select_phones_balances_target_and_other():
    labels = ['p'] * 30 + ['a'] * 20 + ['t'] * 20
    phones = FakePhones(labels)

    selected = probe_utils.select_phones(phones, 'p')

    counts = Counter(label for _, _, label in selected)
    assert counts == {'target': 30, 'other': 30}  # 15 each from 'a' and 't'
    assert len(selected) == 60


def test_select_phones_is_deterministic():
    labels = ['p'] * 20 + ['a'] * 20 + ['t'] * 20
    phones = FakePhones(labels)

    first = probe_utils.select_phones(phones, 'p')
    second = probe_utils.select_phones(phones, 'p')

    assert [pp.key for _, pp, _ in first] == [pp.key for _, pp, _ in second]


def test_select_phones_uses_every_available_target_phone():
    labels = ['p'] * 13500 + ['a'] * 13500 + ['t'] * 13500
    phones = FakePhones(labels)

    selected = probe_utils.select_phones(phones, 'p')

    counts = Counter(label for _, _, label in selected)
    assert counts['target'] == 13500
    assert counts['other'] == 13500  # 6750 each from 'a' and 't'


def test_select_phones_raises_when_target_missing():
    phones = FakePhones(['a'] * 20 + ['t'] * 20)
    with pytest.raises(ValueError, match='not found'):
        probe_utils.select_phones(phones, 'p')


def test_select_phones_raises_when_other_class_underfilled():
    phones = FakePhones(['p'] * 20 + ['a'] * 3 + ['t'] * 20)
    # target 'p' has 20 -> n_per_other = 20 // 2 = 10, but 'a' only has 3
    with pytest.raises(ValueError, match="'a' has only 3.*need 10"):
        probe_utils.select_phones(phones, 'p')


def test_select_phones_raises_when_no_other_classes():
    phones = FakePhones(['p'] * 20)
    with pytest.raises(ValueError, match='no other phoneme classes'):
        probe_utils.select_phones(phones, 'p')


def test_select_phones_raises_when_too_small_to_split():
    phones = FakePhones(['p'] * 2 + ['a'] * 20 + ['t'] * 20 + ['e'] * 20)
    # target 'p' has 2 -> n_per_other = 2 // 3 = 0 across three other labels
    with pytest.raises(ValueError, match='too small to split'):
        probe_utils.select_phones(phones, 'p')


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


def _phone_result(target_phoneme, model_name, layer, collar, results_dir):
    return probe_result.PhoneResult.model_feature(target_phoneme, model_name,
        layer, collar, root=results_dir)


def test_train_binary_embedding_probe_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    phone_result = _phone_result('p', 'model-a', 9, 500, results_dir)
    assert phone_result.run['actual_n_samples'] == 60
    assert phone_result.run['actual_n_missing'] == 0
    assert len(phone_result.accuracies) == 5
    assert phone_result.mean_accuracy > 0.9  # clusters are well separated


def test_train_binary_embedding_probe_uses_cnn_features_and_paths(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer='cnn',
        collar=500, expected_target_count=30, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    phone_result = _phone_result(
        'p', 'model-a', 'cnn', 500, results_dir)
    assert phone_result.run['representation'] == 'cnn'
    assert phone_result.mean_accuracy > .9
    assert len(store.phraser_keys_to_cnn_features_calls) == 1
    assert store.phraser_keys_to_embeddings_calls == []
    assert phone_result.path.parent.name == 'layer-cnn'
    assert (probe_dir / 'model-a' / 'p' / 'layer-cnn').is_dir()


def test_train_binary_embedding_probe_passes_default_collar_to_echoframe(
    tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9,
        expected_target_count=30, verbose=False, probe_save_dir=probe_dir,
        results_dir=results_dir)

    assert store.phraser_keys_to_embeddings_calls[0]['collar'] == 2000
    phone_result = _phone_result('p', 'model-a', 9, 2000, results_dir)
    assert phone_result.run is not None
    assert phone_result.run['representation'] == 'embedding'


def test_train_binary_embedding_probe_rejects_duplicate_phraser_keys(
    tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    phones.phraser_phones[1].key = phones.phraser_phones[0].key

    with pytest.raises(ValueError, match='duplicate Phraser key'):
        tbp.train_binary_embedding_probe(
            phones, 'p', store=store, verbose=False,
            probe_save_dir=tmp_path / 'probes',
            results_dir=tmp_path / 'results')


@pytest.mark.multicore
def test_train_binary_embedding_probes_trains_each_phraser_label(
    tmp_path, capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    results = tbp.train_binary_embedding_probes(
        phones,
        target_phonemes=['p', 'a'],
        store=store,
        model_name='model-a',
        expected_target_count=30,
        verbose=True,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
        report=True,
    )

    # results are reordered to match target_phonemes regardless of which
    # worker process happens to finish first
    assert list(results) == ['p', 'a']
    assert all('mean_accuracy' in result for result in results.values())
    # one batched load for the whole sweep, not one per label - the
    # concrete proof the redundant-reload problem is fixed
    assert len(store.phraser_keys_to_embeddings_calls) == 1
    output = capsys.readouterr().out
    assert '[model-a layer 9 probes] 1/2 completed phone label' in output
    assert '[model-a layer 9 probes] 2/2 completed phone label' in output


@pytest.mark.multicore
def test_train_binary_embedding_probes_runs_in_a_process_pool(
    tmp_path, capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    results = tbp.train_binary_embedding_probes(
        phones,
        target_phonemes=['p', 'a'],
        store=store,
        model_name='model-a',
        layer='cnn',
        expected_target_count=30,
        max_workers=2,
        verbose=True,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
        report=True,
    )

    assert set(results) == {'p', 'a'}
    assert all('mean_accuracy' in result for result in results.values())
    output = capsys.readouterr().out
    assert '[model-a layer cnn probes] 1/2 completed phone label' in output
    assert '[model-a layer cnn probes] 2/2 completed phone label' in output
    assert len(store.phraser_keys_to_cnn_features_calls) == 1
    assert store.phraser_keys_to_embeddings_calls == []

    # results are actually persisted to disk by the worker processes
    phone_result = _phone_result(
        'p', 'model-a', 'cnn', 2000, tmp_path / 'results')
    assert phone_result.complete is True


def test_train_binary_embedding_probes_reports_probe_matrix_progress(
    monkeypatch, capsys,
):
    phones = FakePhones(['p', 'p', 'a', 'a'])
    vectors_by_key = {
        key: np.ones(2) for key in range(len(phones.phraser_phones))}
    store = FakeStore(vectors_by_key)
    monkeypatch.setattr(tbp, '_train_labels_in_pool', lambda *args: {})

    tbp.train_binary_embedding_probes(
        phones,
        store=store,
        model_name='model-a',
        layer=8,
        expected_target_count=2,
        max_workers=2,
        verbose=True,
        report=True,
    )

    output = capsys.readouterr().out
    assert '[model-a layer 8] loading probe matrix' in output
    assert ('[model-a layer 8] probe matrix loaded: 4 vectors, 0 missing; '
        'starting 2 phone probes with 2 workers') in output


def test_pool_worker_reports_phone_label_start(tmp_path, monkeypatch, capsys):
    probe_matrix = object()
    monkeypatch.setattr(tbp, '_pool_probe_matrix', probe_matrix)
    train_calls = []

    def fake_train(*args, **kwargs):
        train_calls.append((args, kwargs))

    monkeypatch.setattr(tbp, 'train_binary_embedding_probe', fake_train)
    monkeypatch.setattr(tbp, '_phone_report', lambda *args: {'done': True})

    result = tbp._train_one_label_in_pool(
        'p', 'model-a', 8, 2000, 13500, tmp_path / 'probes',
        tmp_path / 'results', False, True)

    assert result == ('p', {'done': True})
    assert train_calls[0][1]['probe_matrix'] is probe_matrix
    assert train_calls[0][1]['verbose'] is False
    output = capsys.readouterr().out
    assert '[model-a layer 8 probes] starting phone label \'p\'' in output


@pytest.mark.multicore
def test_train_binary_embedding_probes_pool_propagates_a_label_failure(
    tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    with pytest.raises(ValueError, match="'nonexistent' not found"):
        tbp.train_binary_embedding_probes(
            phones,
            target_phonemes=['p', 'nonexistent'],
            store=store,
            model_name='model-a',
            expected_target_count=30,
            max_workers=2,
            verbose=False,
            probe_save_dir=tmp_path / 'probes',
            results_dir=tmp_path / 'results',
        )


@pytest.mark.multicore
def test_train_binary_embedding_probes_returns_none_without_report(
    tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    results = tbp.train_binary_embedding_probes(
        phones,
        target_phonemes=['p', 'a'],
        store=store,
        model_name='model-a',
        expected_target_count=30,
        verbose=False,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
    )

    assert results is None


def test_train_binary_embedding_probes_rejects_label_count_mismatch():
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=29, other_labels=['a', 't'])

    with pytest.raises(ValueError, match='expected 30 tokens per label'):
        tbp.train_binary_embedding_probes(
            phones, store=store, expected_target_count=30, verbose=False)

    # the mismatch is only known once the matrix is actually loaded
    assert len(store.phraser_keys_to_embeddings_calls) == 1


@pytest.mark.multicore
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
        expected_target_count=30,
        verbose=False,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
    )

    assert opened_roots == [str(tmp_path / 'store')]
    assert opened_store.closed is True


# -- all-checkpoint probe sweep --------------------------------------------

def make_compact_probe_results():
    return {
        'p': {
            'mean_accuracy': .8,
            'std_accuracy': .1,
            'n_samples': 4,
            'n_missing': 0,
            'probes': [object()],
        },
        'a': {
            'mean_accuracy': .75,
            'std_accuracy': .05,
            'n_samples': None,
            'n_missing': None,
            'probes': [object()],
        },
    }


class SweepStore:
    def __init__(self, path):
        self.path = path
        self.attached_phraser_stores = []
        self.closed = False

    def close(self):
        self.closed = True

    def attach_phraser_store(self, source_id, phraser_store):
        self.attached_phraser_stores.append((source_id, phraser_store))


def _save_complete_checkpoint_layer_results(targets, model_name, layer,
    collar, results_dir):
    for target_phoneme in targets:
        phone_result = probe_result.PhoneResult.model_feature(
            target_phoneme, model_name, layer, collar, root=results_dir)
        for fold_number in range(1, 6):
            probe_result.Fold(phone_result, fold_number).save_results(
                [(target_phoneme, 'target', 'target')])


def test_checkpoint_probe_sweep_validates_source_inventory_first(
    monkeypatch,
):
    monkeypatch.setattr(
        tbp,
        'discover_wav2vec2_checkpoint_stores',
        lambda root: pytest.fail('invalid inventory must fail first'),
    )
    phones = FakePhones(['p', 'p', 'a'])

    with pytest.raises(ValueError, match='not balanced'):
        tbp.train_binary_embedding_probe_checkpoint_sweep(
            phones, verbose=False)


def test_checkpoint_probe_sweep_trains_and_returns_compact_report(
    tmp_path, monkeypatch, capsys,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    store = SweepStore(store_path)
    train_calls = []
    progressbar_calls = []

    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)

    def fake_progressbar(items, prefix):
        progressbar_calls.append((list(items), prefix))
        return items

    monkeypatch.setattr(tbp, 'progressbar', fake_progressbar)

    def fake_train(phones, **kwargs):
        train_calls.append((phones, kwargs))
        return make_compact_probe_results()

    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)
    phones = FakePhones(['p', 'p', 'a', 'a'])

    report = tbp.train_binary_embedding_probe_checkpoint_sweep(
        phones,
        store_root=tmp_path,
        collar=500,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
        overwrite=True,
        verbose=True,
    )

    expected_train_options = {
        'target_phonemes': None,
        'store': store,
        'model_name': model_name,
        'collar': 500,
        'expected_target_count': 13500,
        'max_workers': None,
        'probe_save_dir': tmp_path / 'probes',
        'results_dir': tmp_path / 'results',
        'overwrite': True,
        'verbose': True,
        'report': True,
    }
    assert train_calls == [
        (phones, {**expected_train_options, 'layer': layer})
        for layer in (9, 'cnn')]
    assert store.attached_phraser_stores == [
        (tbp.default_phraser_source_id, phones.store)]
    assert progressbar_calls == [
        ([(model_name, store_path)], 'Checkpoints: ')]
    assert store.closed is True
    assert report['status_counts'] == {
        'completed': 2, 'skipped': 0, 'failed': 0}
    assert [run['layer'] for run in report['runs']] == [9, 'cnn']
    for run in report['runs']:
        assert run['status'] == 'completed'
        assert run['n_total'] == 4
        assert run['n_available'] == 4
        assert run['n_missing'] == 0
        assert run['n_labels'] == 2
        assert run['mean_label_accuracy'] == pytest.approx(.775)
        assert run['labels']['p'] == {
            'mean_accuracy': .8,
            'std_accuracy': .1,
            'n_samples': 4,
            'n_missing': 0,
        }
        assert run['labels']['a']['n_samples'] == 4
        assert run['labels']['a']['n_missing'] == 0
        assert all(
            'probes' not in summary for summary in run['labels'].values())
    output = capsys.readouterr().out
    assert '2 completed, 0 skipped, 0 failed' in output
    assert '2 labels, mean label accuracy 0.7750' in output


def test_checkpoint_probe_sweep_threads_max_workers_through(
    tmp_path, monkeypatch,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    store = SweepStore(store_path)
    train_calls = []

    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)
    monkeypatch.setattr(
        tbp, 'progressbar',
        lambda *args, **kwargs: pytest.fail(
            'verbose=False must not create a progress bar'))

    def fake_train(phones, **kwargs):
        train_calls.append(kwargs['max_workers'])
        return make_compact_probe_results()

    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)

    phones = FakePhones(['p', 'p', 'a', 'a'])
    tbp.train_binary_embedding_probe_checkpoint_sweep(
        phones,
        store_root=tmp_path,
        max_workers=4,
        results_dir=tmp_path / 'results',
        verbose=False,
    )

    assert train_calls == [4, 4]


def test_checkpoint_probe_sweep_runs_only_layers_with_missing_results(
    tmp_path, monkeypatch,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    results_dir = tmp_path / 'results'
    store = SweepStore(store_path)
    train_calls = []
    _save_complete_checkpoint_layer_results(
        ('a', 'p'), model_name, 9, 500, results_dir)
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)

    def fake_train(phones, **kwargs):
        train_calls.append(kwargs['layer'])
        return make_compact_probe_results()

    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)

    phones = FakePhones(['p', 'p', 'a', 'a'])
    report = tbp.train_binary_embedding_probe_checkpoint_sweep(
        phones,
        store_root=tmp_path,
        collar=500,
        results_dir=results_dir,
        verbose=False,
    )

    assert train_calls == ['cnn']
    assert store.closed is True
    assert [run['layer'] for run in report['runs']] == [9, 'cnn']
    assert [run['status'] for run in report['runs']] == [
        'skipped', 'completed']
    assert report['status_counts'] == {
        'completed': 1, 'skipped': 1, 'failed': 0}


def test_checkpoint_probe_sweep_does_not_open_store_when_results_complete(
    tmp_path, monkeypatch,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    results_dir = tmp_path / 'results'
    for layer in (9, 'cnn'):
        _save_complete_checkpoint_layer_results(
            ('a', 'p'), model_name, layer, 500, results_dir)
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(
        tbp.echoframe, 'Store',
        lambda path: pytest.fail('complete checkpoint store must not open'),
    )

    phones = FakePhones(['p', 'p', 'a', 'a'])
    report = tbp.train_binary_embedding_probe_checkpoint_sweep(
        phones,
        store_root=tmp_path,
        collar=500,
        results_dir=results_dir,
        verbose=False,
    )

    assert [run['layer'] for run in report['runs']] == [9, 'cnn']
    assert all(run['status'] == 'skipped' for run in report['runs'])
    assert all(run['reason'] == 'all fold results already stored'
        for run in report['runs'])
    expected_counts = {'completed': 0, 'skipped': 2, 'failed': 0}
    assert report['status_counts'] == expected_counts


def test_checkpoint_probe_sweep_validates_inventory_while_loading_features(
    tmp_path, monkeypatch,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    store = FakeStore({0: np.ones(2), 2: np.ones(2), 3: np.ones(2)})
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)

    with pytest.warns(RuntimeWarning) as warning_records:
        phones = FakePhones(['p', 'p', 'a', 'a'])
        report = tbp.train_binary_embedding_probe_checkpoint_sweep(
            phones,
            store_root=tmp_path,
            expected_target_count=2,
            results_dir=tmp_path / 'results',
            verbose=False,
        )

    assert len(warning_records) == 2
    assert len(store.phraser_keys_to_embeddings_calls) == 1
    assert len(store.phraser_keys_to_cnn_features_calls) == 1
    assert store.closed is True
    expected_counts = {'completed': 0, 'skipped': 0, 'failed': 2}
    assert report['status_counts'] == expected_counts
    assert all(run['failure_stage'] == 'training' for run in report['runs'])
    assert all('expected 2 tokens per label' in run['error']
        for run in report['runs'])


def test_checkpoint_probe_sweep_records_failures_and_continues(
    tmp_path, monkeypatch,
):
    model_names = [
        f'wav2vec2_nl1_checkpoint-{checkpoint}'
        for checkpoint in (1000, 2000, 3000, 4000)
    ]
    stores = {
        model_name: SweepStore(tmp_path / model_name)
        for model_name in model_names[1:]
    }
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [
            (model_name, tmp_path / model_name)
            for model_name in model_names
        ],
    )

    def fake_store(path):
        model_name = Path(path).name
        if model_name == model_names[0]:
            raise OSError('cannot open')
        return stores[model_name]

    def fake_train(phones, model_name, **kwargs):
        if model_name == model_names[1]:
            raise RuntimeError('cannot load probe matrix')
        if model_name == model_names[2]:
            raise RuntimeError('cannot train')
        return make_compact_probe_results()

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store)
    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)

    with pytest.warns(RuntimeWarning) as warning_records:
        phones = FakePhones(['p', 'p', 'a', 'a'])
        report = tbp.train_binary_embedding_probe_checkpoint_sweep(
            phones,
            store_root=tmp_path,
            results_dir=tmp_path / 'results',
            verbose=False,
        )

    assert len(warning_records) == 5
    assert [run['status'] for run in report['runs']] == [
        'failed', 'failed', 'failed', 'failed',
        'failed', 'failed', 'completed', 'completed']
    assert [run.get('failure_stage') for run in report['runs'][:6]] == [
        'store', 'store', 'training', 'training',
        'training', 'training']
    assert report['status_counts'] == {
        'completed': 2, 'skipped': 0, 'failed': 6}
    assert report['errors'][0]['stage'] == 'store'
    assert all(store.closed for store in stores.values())


def test_train_binary_embedding_probe_opens_store_when_none_given(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    phones, opened_store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    store_roots = []

    def fake_store_constructor(root):
        store_roots.append(root)
        return opened_store

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store_constructor)

    tbp.train_binary_embedding_probe(
        phones, 'p', model_name='model-a', layer=9, collar=500,
        expected_target_count=30, store_root=tmp_path / 'store',
        verbose=False, probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results')

    assert store_roots == [str(tmp_path / 'store')]
    phone_result = _phone_result('p', 'model-a', 9, 500,
        tmp_path / 'results')
    assert phone_result.run['actual_n_samples'] == 60


def test_train_binary_embedding_probe_saves_probes_and_predictions(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])

    probe_dir = tmp_path / 'probes'
    results_dir = tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)

    probe_files = sorted((probe_dir / 'model-a' / 'p').rglob('*.joblib'))
    pred_files = sorted((results_dir / 'model-a' / 'p').rglob('*.tsv'))
    assert len(probe_files) == 5
    assert len(pred_files) == 5

    header = pred_files[0].read_text().splitlines()[0]
    assert header == 'true_phoneme\tbinary_true\tbinary_pred\tcorrect'
    assert len(list(probe_dir.rglob('*_complete.json'))) == 5
    # embedding no longer writes a separate manifest copy into the
    # probe-artifacts directory - only results_dir gets one now
    assert len(list(probe_dir.rglob('run.json'))) == 0
    assert len(list(results_dir.rglob('run.json'))) == 1

    phone_result = _phone_result('p', 'model-a', 9, 500, results_dir)
    assert phone_result.run['actual_n_samples'] == 60
    assert phone_result.run['actual_n_missing'] == 0


# -- skip / overwrite / gap-filling behavior --------------------------------

def test_train_binary_embedding_probe_skips_when_all_folds_already_saved(
    tmp_path, capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=True,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert 'cache status: miss' in capsys.readouterr().out
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)
    assert calls_after_first == 1

    first_result = _phone_result('p', 'model-a', 9, 500, results_dir)
    first_run = dict(first_result.run)
    first_accuracies = first_result.accuracies
    first_mean_accuracy = first_result.mean_accuracy

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=True,
        probe_save_dir=probe_dir, results_dir=results_dir)
    assert 'cache status: hit' in capsys.readouterr().out

    second_result = _phone_result('p', 'model-a', 9, 500, results_dir)
    assert second_result.run == first_run
    # a true hit touches nothing - zero store calls, not just zero extra
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first
    assert second_result.accuracies == pytest.approx(first_accuracies)
    assert second_result.mean_accuracy == pytest.approx(first_mean_accuracy)


def test_train_binary_embedding_probe_overwrite_forces_retrain(
    tmp_path, capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=False,
        probe_save_dir=probe_dir, results_dir=results_dir)
    calls_after_first = len(store.phraser_keys_to_embeddings_calls)

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=True,
        probe_save_dir=probe_dir, results_dir=results_dir, overwrite=True)

    assert 'cache status: refresh' in capsys.readouterr().out
    # overwrite bypasses the skip check, so embeddings get reloaded
    assert len(store.phraser_keys_to_embeddings_calls) == calls_after_first + 1


def test_train_binary_embedding_probe_does_not_reuse_a_different_collar(
    tmp_path, capsys):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=30, other_labels=['a', 't'])
    probe_dir, results_dir = tmp_path / 'probes', tmp_path / 'results'

    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=500,
        expected_target_count=30, verbose=True, probe_save_dir=probe_dir,
        results_dir=results_dir)
    assert 'cache status: miss' in capsys.readouterr().out

    # a different collar is a different PhoneResult path, so it's never a
    # cache hit - trains fresh and issues its own store call
    tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', layer=9, collar=2000,
        expected_target_count=30, verbose=True, probe_save_dir=probe_dir,
        results_dir=results_dir)

    assert 'cache status: miss' in capsys.readouterr().out
    assert len(store.phraser_keys_to_embeddings_calls) == 2
