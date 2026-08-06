import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_run, probe_utils
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


def _load_saved_probes(probe_dir, model_name, target_phoneme, layer, collar,
    run_id, n_splits):
    probe_run_directory = tbp._run_directory(probe_dir, model_name,
        target_phoneme, layer, collar, run_id)
    probes = []
    for fold_idx in range(n_splits):
        probe_path, _, _ = probe_run.fold_paths(probe_run_directory,
            probe_run_directory, fold_idx)
        probes.append(joblib.load(probe_path))
    return probes


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


def test_train_binary_embedding_probe_standardization_is_fold_local(tmp_path):
    rng = np.random.default_rng(0)
    phones, store = _make_separable_dataset(
        rng, n_target=30, n_other_each=15, other_labels=['a', 't'])
    probe_dir = tmp_path / 'probes'

    raw = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', n_embeds=30,
        standardize=False, verbose=False, save_probes=True,
        probe_save_dir=probe_dir, save_predictions=False)
    scaled = tbp.train_binary_embedding_probe(
        phones, 'p', store=store, model_name='model-a', n_embeds=30,
        standardize=True, verbose=False, save_probes=True,
        probe_save_dir=probe_dir, save_predictions=False)

    assert raw['run_id'] != scaled['run_id']
    raw_probes = _load_saved_probes(probe_dir, 'model-a', 'p', 9, 2000,
        raw['run_id'], n_splits=5)
    scaled_probes = _load_saved_probes(probe_dir, 'model-a', 'p', 9, 2000,
        scaled['run_id'], n_splits=5)
    assert all(isinstance(probe, LogisticRegression) for probe in raw_probes)
    assert all(isinstance(probe, Pipeline) for probe in scaled_probes)
    assert all(
        hasattr(probe.named_steps['standardscaler'], 'mean_')
        for probe in scaled_probes
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


# -- all-checkpoint probe sweep --------------------------------------------

def make_compact_probe_results():
    return {
        'p': {
            'mean_accuracy': .8,
            'std_accuracy': .1,
            'n_samples': 4,
            'n_missing': 0,
            'skipped': False,
            'cache_status': 'miss',
            'probes': [object()],
        },
        'a': {
            'mean_accuracy': .75,
            'std_accuracy': .05,
            'n_samples': None,
            'n_missing': None,
            'skipped': True,
            'cache_status': 'hit',
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
        n_embeds=2,
        n_splits=2,
        random_state=7,
        standardize=True,
        save_probes=True,
        probe_save_dir=tmp_path / 'probes',
        save_predictions=True,
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
            'n_embeds': 2,
            'n_splits': 2,
            'random_state': 7,
            'standardize': True,
            'save_probes': True,
            'probe_save_dir': tmp_path / 'probes',
            'save_predictions': True,
            'results_dir': tmp_path / 'results',
            'overwrite': True,
            'verbose': True,
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
        'skipped': False,
        'cache_status': 'miss',
    }
    assert run['labels']['a']['n_samples'] == 4
    assert run['labels']['a']['n_missing'] == 0
    assert all(
        'probes' not in summary for summary in run['labels'].values())
    output = capsys.readouterr().out
    assert '1 completed, 0 skipped, 0 failed' in output
    assert '2 labels, mean label accuracy 0.7750' in output


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
