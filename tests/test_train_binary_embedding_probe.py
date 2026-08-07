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
        self.load_many_metadata_calls = []
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
        self.load_many_metadata_calls.append(
            (list(echoframe_keys), keep_missing))
        metadatas = []
        for key in echoframe_keys:
            vector = self.vectors_by_key.get(key[2])
            metadata = (
                FakeMetadata(key, vector) if vector is not None else None)
            if metadata is not None or keep_missing:
                metadatas.append(metadata)
        return metadatas


# -- checkpoint discovery and inventory preflight -------------------------

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
        ('wav2vec2_checkpoint-0', tuple(range(1, 13))),
        ('wav2vec2_nl1_checkpoint-1000', (9,)),
        ('wav2vec2_nl1_checkpoint-200000', tuple(range(1, 13))),
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


def test_check_embedding_inventory_checks_every_phone_in_batches():
    phones = FakePhones(['p', 'p', 'a', 'a', 't'])
    store = FakeStore({key: np.ones(2) for key in (0, 1, 3, 4)})

    report = tbp.check_embedding_inventory(
        phones,
        store,
        'wav2vec2_nl1_checkpoint-1000',
        layer=9,
        collar=2000,
        batch_size=2,
        verbose=False,
    )

    assert report == {
        'n_total': 5,
        'n_available': 4,
        'n_missing': 1,
        'complete': False,
    }
    assert len(store.load_many_metadata_calls) == 3
    requested_keys = [
        key
        for keys, keep_missing in store.load_many_metadata_calls
        for key in keys
    ]
    assert all(
        keep_missing for _, keep_missing in store.load_many_metadata_calls)
    assert requested_keys == [
        ('hidden_state', 'wav2vec2_nl1_checkpoint-1000', phraser_key, 9, 2000)
        for phraser_key in range(5)
    ]


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_check_embedding_inventory_rejects_invalid_batch_size(batch_size):
    phones = FakePhones(['p'])
    store = FakeStore({0: np.ones(2)})

    with pytest.raises((TypeError, ValueError), match='positive integer'):
        tbp.check_embedding_inventory(
            phones, store, 'wav2vec2_nl1_checkpoint-1000', layer=9,
            batch_size=batch_size, verbose=False)


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
    return probe_result.PhoneResult.embedding(target_phoneme, model_name,
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
    assert '[embedding pool] 1/2 completed' in output
    assert '[embedding pool] 2/2 completed' in output


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
    assert '[embedding pool] 1/2 completed' in output
    assert '[embedding pool] 2/2 completed' in output

    # results are actually persisted to disk by the worker processes
    phone_result = _phone_result('p', 'model-a', 9, 2000, tmp_path / 'results')
    assert phone_result.complete is True


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
        self.closed = False

    def close(self):
        self.closed = True


def test_checkpoint_probe_sweep_trains_and_returns_compact_report(
    tmp_path, monkeypatch, capsys,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    store = SweepStore(store_path)
    preflight_calls = []
    train_calls = []

    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)

    def fake_check(phones, store_arg, model_name_arg, layer, **kwargs):
        preflight_calls.append(
            (phones, store_arg, model_name_arg, layer, kwargs))
        return {
            'n_total': 4,
            'n_available': 4,
            'n_missing': 0,
            'complete': True,
        }

    def fake_train(phones, **kwargs):
        train_calls.append((phones, kwargs))
        return make_compact_probe_results()

    monkeypatch.setattr(tbp, 'check_embedding_inventory', fake_check)
    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)
    phones = FakePhones(['p', 'p', 'a', 'a'])

    report = tbp.train_binary_embedding_probe_checkpoint_sweep(
        phones,
        store_root=tmp_path,
        collar=500,
        probe_save_dir=tmp_path / 'probes',
        results_dir=tmp_path / 'results',
        overwrite=True,
        metadata_batch_size=25,
        verbose=True,
    )

    assert preflight_calls == [(
        phones,
        store,
        model_name,
        9,
        {'collar': 500, 'batch_size': 25, 'verbose': True},
    )]
    assert train_calls == [(
        phones,
        {
            'target_phonemes': None,
            'store': store,
            'model_name': model_name,
            'layer': 9,
            'collar': 500,
            'expected_target_count': 13500,
            'max_workers': None,
            'probe_save_dir': tmp_path / 'probes',
            'results_dir': tmp_path / 'results',
            'overwrite': True,
            'verbose': True,
            'report': True,
        },
    )]
    assert store.closed is True
    assert report['status_counts'] == {
        'completed': 1, 'skipped': 0, 'failed': 0}
    run = report['runs'][0]
    assert run['status'] == 'completed'
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
    assert '1 completed, 0 skipped, 0 failed' in output
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
        tbp, 'check_embedding_inventory',
        lambda *args, **kwargs: {
            'n_total': 4, 'n_available': 4, 'n_missing': 0, 'complete': True,
        },
    )

    def fake_train(phones, **kwargs):
        train_calls.append(kwargs['max_workers'])
        return make_compact_probe_results()

    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)

    tbp.train_binary_embedding_probe_checkpoint_sweep(
        FakePhones(['p', 'p', 'a', 'a']),
        store_root=tmp_path,
        max_workers=4,
        verbose=False,
    )

    assert train_calls == [4]


def test_checkpoint_probe_sweep_skips_incomplete_inventory(
    tmp_path, monkeypatch, capsys,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    store_path = tmp_path / model_name
    store = SweepStore(store_path)
    monkeypatch.setattr(
        tbp, 'discover_wav2vec2_checkpoint_stores',
        lambda root: [(model_name, store_path)],
    )
    monkeypatch.setattr(tbp.echoframe, 'Store', lambda path: store)
    monkeypatch.setattr(
        tbp, 'check_embedding_inventory',
        lambda *args, **kwargs: {
            'n_total': 4,
            'n_available': 3,
            'n_missing': 1,
            'complete': False,
        },
    )
    monkeypatch.setattr(
        tbp, 'train_binary_embedding_probes',
        lambda *args, **kwargs: pytest.fail('training should be skipped'),
    )

    with pytest.warns(RuntimeWarning, match='1 of 4 embeddings are missing'):
        report = tbp.train_binary_embedding_probe_checkpoint_sweep(
            FakePhones(['p', 'p', 'a', 'a']),
            store_root=tmp_path,
            verbose=True,
        )

    assert store.closed is True
    assert report['status_counts'] == {
        'completed': 0, 'skipped': 1, 'failed': 0}
    assert report['runs'][0] == {
        'model_name': model_name,
        'layer': 9,
        'n_total': 4,
        'n_available': 3,
        'n_missing': 1,
        'status': 'skipped',
        'reason': 'incomplete embedding inventory',
    }
    assert '0 completed, 1 skipped, 0 failed' in capsys.readouterr().out


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

    def fake_check(phones, store, model_name, layer, **kwargs):
        if model_name == model_names[1]:
            raise RuntimeError('cannot check')
        return {
            'n_total': 4,
            'n_available': 4,
            'n_missing': 0,
            'complete': True,
        }

    def fake_train(phones, model_name, **kwargs):
        if model_name == model_names[2]:
            raise RuntimeError('cannot train')
        return make_compact_probe_results()

    monkeypatch.setattr(tbp.echoframe, 'Store', fake_store)
    monkeypatch.setattr(tbp, 'check_embedding_inventory', fake_check)
    monkeypatch.setattr(tbp, 'train_binary_embedding_probes', fake_train)

    with pytest.warns(RuntimeWarning) as warning_records:
        report = tbp.train_binary_embedding_probe_checkpoint_sweep(
            FakePhones(['p', 'p', 'a', 'a']),
            store_root=tmp_path,
            verbose=False,
        )

    assert len(warning_records) == 3
    assert [run['status'] for run in report['runs']] == [
        'failed', 'failed', 'failed', 'completed']
    assert [run.get('failure_stage') for run in report['runs'][:3]] == [
        'store', 'preflight', 'training']
    assert report['status_counts'] == {
        'completed': 1, 'skipped': 0, 'failed': 3}
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
