import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import phone_binary_probe as pbp
from probing import phone_probe_common as probe_common
from probing import phone_probe_metadata as probe_metadata
from probing import phone_probe_report as probe_report
from probing import phone_probe_sweep as probe_sweep
from probing import phone_probe_worker as probe_worker
from probing import probe_utils


MODEL_NAME = 'wav2vec2_nl1_checkpoint-1000'
OTHER_MODEL_NAME = 'wav2vec2_nl1_checkpoint-2000'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probe_result(*, skipped=False, run_id='run-1'):
    return {
        'run_id': run_id,
        'cache_status': 'complete' if skipped else 'trained',
        'accuracies': [0.8, 0.9],
        'mean_accuracy': 0.85,
        'std_accuracy': 0.05,
        'n_samples': 60,
        'n_missing': 0,
        'skipped': skipped,
    }


class _FakePhraserStore:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakePhones:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._store = _FakePhraserStore()
        self.phraser_phones = list(range(31 * 30))
        self.label_to_phraser_phone = {
            f'phone-{index:02d}': list(range(30))
            for index in range(31)
        }
        type(self).instances.append(self)

    @property
    def store(self):
        return self._store


class _FakeEchoframeStore:
    instances = []

    def __init__(self, path):
        self.path = path
        self.attach_calls = []
        self.close_calls = 0
        type(self).instances.append(self)

    def attach_phraser_store(self, source_id, store):
        self.attach_calls.append((source_id, store))

    def close(self):
        self.close_calls += 1


@pytest.fixture
def worker_fakes(monkeypatch):
    _FakePhones.instances = []
    _FakeEchoframeStore.instances = []
    monkeypatch.setattr(probe_worker.metadata, 'Phones', _FakePhones)
    monkeypatch.setattr(probe_worker.echoframe, 'Store', _FakeEchoframeStore)
    return _FakePhones, _FakeEchoframeStore


@pytest.mark.parametrize(
    ('skipped', 'expected_status'),
    [(False, 'completed'), (True, 'already_complete')],
)
def test_worker_records_status_and_uses_one_phraser_store(
    tmp_path, monkeypatch, worker_fakes, skipped, expected_status,
):
    calls = []

    def train(*args, **kwargs):
        calls.append((args, kwargs))
        return _probe_result(skipped=skipped)

    monkeypatch.setattr(probe_worker, 'train_binary_embedding_probe', train)
    status_path = tmp_path / 'task.json'
    probe_dir = tmp_path / 'probes'

    result = pbp.train_phone_binary_probe(
        'p', MODEL_NAME, 9,
        metadata_path=tmp_path / 'metadata.csv',
        sentence_path=tmp_path / 'sentences.csv',
        phraser_key_path=tmp_path / 'keys.bin',
        duplicate_replacement_phraser_key_path=None,
        model_store_path=tmp_path / 'model-store',
        probe_save_dir=probe_dir,
        results_dir=tmp_path / 'results',
        overwrite=True,
        verbose=False,
        task_status_path=status_path,
    )

    assert result['run_id'] == 'run-1'
    assert len(_FakePhones.instances) == 1
    assert len(_FakeEchoframeStore.instances) == 1
    phones = _FakePhones.instances[0]
    echoframe_store = _FakeEchoframeStore.instances[0]
    assert echoframe_store.attach_calls == [
        (pbp.default_phraser_source_id, phones._store)]
    assert phones._store.close_calls == 1
    assert echoframe_store.close_calls == 1
    assert calls[0][0][:2] == (phones, 'p')
    assert calls[0][1]['store'] is echoframe_store
    assert calls[0][1]['overwrite'] is True
    assert calls[0][1]['layer'] == 9

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == expected_status
    assert status['task'] == {
        'phone': 'p', 'model_name': MODEL_NAME, 'layer': 9}
    assert status['metrics']['run_id'] == 'run-1'
    pointer_path = Path(status['selected_run_pointer_path'])
    assert pointer_path.is_file()
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    assert pointer['run_id'] == 'run-1'
    assert pointer['worker_status'] == expected_status


def test_worker_records_failure_and_closes_each_store_once(
    tmp_path, monkeypatch, worker_fakes,
):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic training failure')

    monkeypatch.setattr(probe_worker, 'train_binary_embedding_probe', fail)
    status_path = tmp_path / 'task.json'

    with pytest.raises(RuntimeError, match='synthetic training failure'):
        pbp.train_phone_binary_probe(
            'p', MODEL_NAME, 9,
            model_store_path=tmp_path / 'model-store',
            probe_save_dir=tmp_path / 'probes',
            results_dir=tmp_path / 'results',
            verbose=False,
            task_status_path=status_path,
        )

    assert _FakePhones.instances[0]._store.close_calls == 1
    assert _FakeEchoframeStore.instances[0].close_calls == 1
    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['error'] == {
        'type': 'RuntimeError', 'message': 'synthetic training failure'}


def test_train_cli_uses_netherlandic_defaults_and_forwards_overwrite(
    monkeypatch,
):
    captured = {}

    def train(phone, model_name, layer, **kwargs):
        captured.update({
            'phone': phone, 'model_name': model_name, 'layer': layer,
            **kwargs,
        })

    monkeypatch.setattr(pbp, 'train_phone_binary_probe', train)
    exit_code = pbp.main([
        'train', '--phone', 'p', '--model-name', MODEL_NAME,
        '--layer', '9', '--overwrite', '--no-verbose',
    ])

    assert exit_code == 0
    assert captured['phone'] == 'p'
    assert captured['model_name'] == MODEL_NAME
    assert captured['layer'] == 9
    assert captured['metadata_path'] == pbp.metadata.metadata_file
    assert captured['sentence_path'] == pbp.metadata.sentence_file
    assert captured['phraser_key_path'] == pbp.metadata.phraser_key_file
    assert captured['duplicate_replacement_phraser_key_path'] == (
        pbp.metadata.duplicate_replacement_phraser_key_file)
    assert captured['model_stores_root'] == pbp.default_model_stores_root
    assert captured['overwrite'] is True


def _metadata_inputs(tmp_path):
    key_path = tmp_path / 'phone-keys.bin'
    replacement_path = tmp_path / 'replacement-keys.bin'
    key_path.write_bytes(b'keys-v1')
    replacement_path.write_bytes(b'replacements-v1')
    return key_path, replacement_path


def _patch_metadata_check(
    monkeypatch, model_root, inventories, *, store_error_names=(),
):
    _FakePhones.instances = []
    _FakeEchoframeStore.instances = []
    model_paths = []
    for model_name in inventories:
        path = model_root / model_name
        path.mkdir(parents=True, exist_ok=True)
        model_paths.append((model_name, path))

    monkeypatch.setattr(
        probe_metadata, 'discover_wav2vec2_checkpoint_stores',
        lambda root: model_paths,
    )
    monkeypatch.setattr(probe_metadata.metadata, 'Phones', _FakePhones)

    class Store(_FakeEchoframeStore):
        def __init__(self, path):
            if Path(path).name in store_error_names:
                raise RuntimeError(f'cannot open {Path(path).name}')
            super().__init__(path)

    monkeypatch.setattr(probe_metadata.echoframe, 'Store', Store)
    calls = []

    def check(phones, store, model_name, layer, **kwargs):
        calls.append((phones, store, model_name, layer, kwargs))
        value = inventories[model_name]
        if isinstance(value, BaseException):
            raise value
        return dict(value)

    monkeypatch.setattr(probe_metadata, 'check_embedding_inventory', check)
    return calls


def _run_metadata_check(
    tmp_path, monkeypatch, inventories, *, collar=2000,
    force_metadata_check=False, verbose=False, store_error_names=(),
):
    model_root = tmp_path / 'stores'
    key_path, replacement_path = _metadata_inputs(tmp_path)
    calls = _patch_metadata_check(
        monkeypatch, model_root, inventories,
        store_error_names=store_error_names,
    )
    report = pbp.check_phone_binary_probe_metadata(
        metadata_path=tmp_path / 'metadata.csv',
        sentence_path=tmp_path / 'sentences.csv',
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
        model_stores_root=model_root,
        collar=collar,
        batch_size=17,
        force_metadata_check=force_metadata_check,
        verbose=verbose,
    )
    return report, calls, key_path, replacement_path


def test_metadata_complete_cache_is_reused_and_parent_opens_phraser_once(
    tmp_path, monkeypatch,
):
    inventory = {
        MODEL_NAME: {
            'n_total': 31 * 30, 'n_available': 31 * 30,
            'n_missing': 0, 'complete': True,
        },
    }
    report, calls, key_path, replacement_path = _run_metadata_check(
        tmp_path, monkeypatch, inventory)
    assert report['status'] == 'complete'
    assert len(calls) == 1
    assert len(_FakePhones.instances) == 1
    assert _FakePhones.instances[0]._store.close_calls == 1

    second_calls = _patch_metadata_check(
        monkeypatch, tmp_path / 'stores', inventory)
    second = pbp.check_phone_binary_probe_metadata(
        metadata_path=tmp_path / 'metadata.csv',
        sentence_path=tmp_path / 'sentences.csv',
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
        model_stores_root=tmp_path / 'stores',
        verbose=False,
    )

    assert second['models'][0]['cache_status'] == 'cached'
    assert second['summary']['cached_layers'] == 1
    assert second_calls == []
    assert len(_FakePhones.instances) == 0


@pytest.mark.parametrize('mode', ['incomplete', 'force', 'stale'])
def test_metadata_rechecks_incomplete_forced_and_stale_entries(
    tmp_path, monkeypatch, mode,
):
    incomplete = {
        'n_total': 930, 'n_available': 929,
        'n_missing': 1, 'complete': False,
    }
    complete = {
        'n_total': 930, 'n_available': 930,
        'n_missing': 0, 'complete': True,
    }
    first_value = complete if mode in {'force', 'stale'} else incomplete
    warning_context = (
        pytest.warns(RuntimeWarning)
        if mode == 'incomplete' else _no_warning()
    )
    with warning_context:
        first, _, key_path, replacement_path = _run_metadata_check(
            tmp_path, monkeypatch, {MODEL_NAME: first_value})
    assert first['status'] in {'complete', 'incomplete'}
    if mode == 'stale':
        key_path.write_bytes(b'keys-v2')

    calls = _patch_metadata_check(
        monkeypatch, tmp_path / 'stores', {MODEL_NAME: complete})
    second = pbp.check_phone_binary_probe_metadata(
        metadata_path=tmp_path / 'metadata.csv',
        sentence_path=tmp_path / 'sentences.csv',
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
        model_stores_root=tmp_path / 'stores',
        force_metadata_check=mode == 'force',
        verbose=False,
    )

    assert len(calls) == 1
    assert second['models'][0]['cache_status'] == 'checked'
    assert second['status'] == 'complete'


class _no_warning:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_metadata_cache_retains_independent_collar_records(
    tmp_path, monkeypatch,
):
    inventory = {
        MODEL_NAME: {
            'n_total': 930, 'n_available': 930,
            'n_missing': 0, 'complete': True,
        },
    }
    _, _, key_path, replacement_path = _run_metadata_check(
        tmp_path, monkeypatch, inventory, collar=500)
    _patch_metadata_check(monkeypatch, tmp_path / 'stores', inventory)
    pbp.check_phone_binary_probe_metadata(
        phraser_key_path=key_path,
        duplicate_replacement_phraser_key_path=replacement_path,
        model_stores_root=tmp_path / 'stores',
        collar=2000,
        verbose=False,
    )

    status = json.loads((
        tmp_path / 'stores' / probe_metadata._metadata_status_filename
    ).read_text(encoding='utf-8'))
    collars = status['models'][MODEL_NAME]['layers']['9']
    assert set(collars) == {'500', '2000'}


def test_metadata_empty_root_records_failure_without_opening_stores(
    tmp_path, monkeypatch,
):
    model_root = tmp_path / 'stores'
    model_root.mkdir()
    key_path, replacement_path = _metadata_inputs(tmp_path)
    monkeypatch.setattr(
        probe_metadata, 'discover_wav2vec2_checkpoint_stores', lambda root: [])
    monkeypatch.setattr(
        probe_metadata.metadata, 'Phones',
        lambda **kwargs: pytest.fail('metadata should not be opened'))
    monkeypatch.setattr(
        probe_metadata.echoframe, 'Store',
        lambda path: pytest.fail('Echoframe should not be opened'))

    with pytest.warns(RuntimeWarning, match='no supported checkpoint'):
        report = pbp.check_phone_binary_probe_metadata(
            phraser_key_path=key_path,
            duplicate_replacement_phraser_key_path=replacement_path,
            model_stores_root=model_root,
            verbose=False,
        )

    assert report['status'] == 'failed'
    assert report['summary']['n_models'] == 0
    assert Path(report['status_path']).is_file()


def test_metadata_failure_continues_to_later_model_and_suppresses_progress(
    tmp_path, monkeypatch,
):
    complete = {
        'n_total': 930, 'n_available': 930,
        'n_missing': 0, 'complete': True,
    }
    inventories = {
        MODEL_NAME: complete,
        OTHER_MODEL_NAME: complete,
    }
    monkeypatch.setattr(
        probe_metadata, '_metadata_progress_bar',
        lambda *args, **kwargs: pytest.fail('progress bar was constructed'))
    with pytest.warns(RuntimeWarning, match='Could not open'):
        report, calls, _, _ = _run_metadata_check(
            tmp_path, monkeypatch, inventories,
            store_error_names={MODEL_NAME}, verbose=False,
        )

    assert [model['status'] for model in report['models']] == [
        'failed', 'complete']
    assert [call[2] for call in calls] == [OTHER_MODEL_NAME]
    assert len(_FakePhones.instances) == 1
    assert _FakePhones.instances[0]._store.close_calls == 1


def _command_options(tmp_path, **overrides):
    values = {
        'metadata_path': tmp_path / 'metadata.csv',
        'sentence_path': tmp_path / 'sentences.csv',
        'phraser_key_path': tmp_path / 'keys.bin',
        'duplicate_replacement_phraser_key_path': None,
        'model_stores_root': tmp_path / 'stores',
        'collar': 2000,
        'n_embeds': 100,
        'n_splits': 5,
        'random_state': 42,
        'standardize': False,
        'save_probes': True,
        'probe_save_dir': tmp_path / 'probes',
        'save_predictions': True,
        'results_dir': tmp_path / 'results',
        'overwrite': False,
        'verbose': False,
    }
    values.update(overrides)
    return values


def _task(phone='p', model_name=MODEL_NAME, layer=9, store_path=None):
    return {
        'phone': phone,
        'model_name': model_name,
        'layer': layer,
        'model_store_path': store_path or Path('/synthetic/store'),
    }


def test_worker_command_contains_exact_task_and_boolean_options(tmp_path):
    status_path = tmp_path / 'status.json'
    command = probe_sweep._build_train_subprocess_command(
        _task(store_path=tmp_path / 'model'),
        task_status_path=status_path,
        **_command_options(
            tmp_path, overwrite=True, standardize=True,
            save_predictions=False, verbose=True,
        ),
    )

    assert command[:5] == [
        probe_sweep.sys.executable, '-u', '-m',
        'probing.phone_binary_probe', 'train']
    assert command[command.index('--phone') + 1] == 'p'
    assert command[command.index('--model-name') + 1] == MODEL_NAME
    assert command[command.index('--layer') + 1] == '9'
    assert '--no-duplicate-replacement-phraser-key' in command
    assert '--standardize' in command
    assert '--no-save-predictions' in command
    assert '--overwrite' in command
    assert '--verbose' in command
    assert command[command.index('--task-status-path') + 1] == str(status_path)


class _ImmediateProcess:
    active = 0
    peak = 0
    launches = 0
    fail_indices = set()

    def __init__(self, command, **kwargs):
        self.command = command
        self.returncode = None
        self.index = type(self).launches
        type(self).launches += 1
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        status_path = Path(command[command.index('--task-status-path') + 1])
        phone = command[command.index('--phone') + 1]
        model = command[command.index('--model-name') + 1]
        layer = int(command[command.index('--layer') + 1])
        failed = self.index in type(self).fail_indices
        _write_json(status_path, {
            'status': 'failed' if failed else (
                'already_complete' if self.index % 2 else 'completed'),
            'task': {'phone': phone, 'model_name': model, 'layer': layer},
            **({'error': {'type': 'Synthetic', 'message': 'failed'}}
               if failed else {}),
        })

    def poll(self):
        if self.returncode is None:
            self.returncode = 1 if self.index in self.fail_indices else 0
            type(self).active -= 1
        return self.returncode


def test_scheduler_accepts_arbitrary_jobs_and_continues_after_failure(
    tmp_path, monkeypatch,
):
    _ImmediateProcess.active = 0
    _ImmediateProcess.peak = 0
    _ImmediateProcess.launches = 0
    _ImmediateProcess.fail_indices = {1}
    monkeypatch.setattr(probe_sweep.subprocess, 'Popen', _ImmediateProcess)
    tasks = [_task(phone=f'p-{index}', store_path=tmp_path / 'store')
             for index in range(7)]
    temporary = tmp_path / 'temporary'
    temporary.mkdir()

    result = probe_sweep._run_sweep_subprocesses(
        tasks,
        jobs=4,
        temporary_directory=temporary,
        command_options=_command_options(tmp_path),
        poll_interval=0.001,
        verbose=False,
    )

    assert _ImmediateProcess.peak == 4
    assert _ImmediateProcess.launches == 7
    assert [outcome['status'] for outcome in result['outcomes']] == [
        'completed', 'failed', 'completed', 'already_complete',
        'completed', 'already_complete', 'completed',
    ]
    assert result['n_not_started'] == 0
    assert result['interrupted'] is False


class _RunningProcess:
    instances = []

    def __init__(self, command, **kwargs):
        self.returncode = None
        self.terminated = False
        self.killed = False
        type(self).instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_scheduler_interrupt_terminates_active_workers(
    tmp_path, monkeypatch,
):
    _RunningProcess.instances = []
    monkeypatch.setattr(probe_sweep.subprocess, 'Popen', _RunningProcess)
    monkeypatch.setattr(
        probe_sweep.time, 'sleep',
        lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    temporary = tmp_path / 'temporary'
    temporary.mkdir()

    result = probe_sweep._run_sweep_subprocesses(
        [_task(phone=f'p-{index}', store_path=tmp_path / 'store')
         for index in range(5)],
        jobs=3,
        temporary_directory=temporary,
        command_options=_command_options(tmp_path),
        poll_interval=0.001,
        verbose=False,
    )

    assert result['interrupted'] is True
    assert result['n_not_started'] == 2
    assert len(result['outcomes']) == 3
    assert all(item['status'] == 'interrupted'
               for item in result['outcomes'])
    assert all(process.terminated for process in _RunningProcess.instances)
    assert not any(process.killed for process in _RunningProcess.instances)


def _metadata_cache(
    model_root, key_path, replacement_path, *, phone_labels=('p',),
):
    store_path = model_root / MODEL_NAME
    store_path.mkdir(parents=True, exist_ok=True)
    inventory = probe_common._phone_inventory_fingerprint(
        key_path, replacement_path)
    record = {
        'model_name': MODEL_NAME,
        'layer': 9,
        'collar': 2000,
        'phone_inventory_fingerprint': inventory['fingerprint'],
        'status': 'complete',
        'n_total': 930,
        'n_available': 930,
        'n_missing': 0,
        'complete': True,
        'checked_at': '2026-08-03T00:00:00Z',
    }
    cache = {
        'schema_version': probe_metadata._metadata_status_schema_version,
        'kind': 'phone_binary_probe_metadata_status',
        'phone_inventory': inventory,
        'phone_labels': list(phone_labels),
        'phones_per_label': 30,
        'created_at': '2026-08-03T00:00:00Z',
        'updated_at': '2026-08-03T00:00:00Z',
        'models': {
            MODEL_NAME: {
                'model_name': MODEL_NAME,
                'store_path': str(store_path.resolve()),
                'last_status': 'complete',
                'layers': {'9': {'2000': record}},
            },
        },
        'errors': [],
    }
    _write_json(model_root / probe_metadata._metadata_status_filename, cache)
    return cache


def _manifest(phone, *, random_state=42, discriminator='one'):
    return {
        'cache_schema_version': 2,
        'trainer_version': 2,
        'representation': 'embedding',
        'feature_parameters': {
            'model_name': MODEL_NAME,
            'layer': 9,
            'collar': 2000,
            'frame': 'middle',
        },
        'target_phoneme': phone,
        'n_samples': 30,
        'n_splits': 2,
        'random_state': random_state,
        'classifier': probe_utils.classifier_manifest(False),
        'selected_sample_count': 60,
        'selected_samples_hash': discriminator,
        'feature_set_hash': f'features-{discriminator}',
    }


def _write_probe_run(
    probe_root,
    results_root,
    phone,
    *,
    discriminator='one',
    random_state=42,
    complete_folds=2,
    malformed_fold=None,
    bad_checksum_fold=None,
):
    manifest = _manifest(
        phone, random_state=random_state, discriminator=discriminator)
    run_id = probe_utils.hash_run_manifest(manifest)
    probe_run = (
        probe_root / MODEL_NAME / phone / 'layer09' / 'collar2000ms'
        / run_id)
    results_run = (
        results_root / MODEL_NAME / phone / 'layer09' / 'collar2000ms'
        / run_id)
    _write_json(probe_run / 'run.json', manifest)
    _write_json(results_run / 'run.json', manifest)
    for fold_index in range(complete_folds):
        probe_path, predictions_path, marker_path = probe_utils.fold_paths(
            probe_run, results_run, fold_index)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_bytes(f'probe-{fold_index}'.encode())
        predictions_path.write_text('predictions\n', encoding='utf-8')
        marker = {
            'run_id': run_id,
            'fold': fold_index + 1,
            'accuracy': 0.75 + fold_index / 10,
            'n_predictions': 10,
            'probe_sha256': _sha256(probe_path),
            'predictions_sha256': _sha256(predictions_path),
        }
        if malformed_fold == fold_index:
            marker['accuracy'] = 'invalid'
        if bad_checksum_fold == fold_index:
            marker['probe_sha256'] = '0' * 64
        _write_json(marker_path, marker)
    return run_id


def _report_paths(tmp_path, phone_labels):
    key_path, replacement_path = _metadata_inputs(tmp_path)
    model_root = tmp_path / 'stores'
    _metadata_cache(
        model_root, key_path, replacement_path,
        phone_labels=phone_labels,
    )
    return {
        'metadata_path': tmp_path / 'metadata.csv',
        'sentence_path': tmp_path / 'sentences.csv',
        'phraser_key_path': key_path,
        'duplicate_replacement_phraser_key_path': replacement_path,
        'model_stores_root': model_root,
        'probe_save_dir': tmp_path / 'custom-probes',
        'results_dir': tmp_path / 'custom-results',
    }


def test_report_is_artifact_only_filters_settings_and_classifies_artifacts(
    tmp_path, monkeypatch,
):
    labels = ['complete', 'partial', 'missing', 'malformed', 'checksum']
    paths = _report_paths(tmp_path, labels)
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'complete')
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'partial',
        complete_folds=1)
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'malformed',
        malformed_fold=0)
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'checksum',
        bad_checksum_fold=1)
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'missing',
        random_state=7)
    monkeypatch.setattr(
        probe_report.metadata, 'Phones',
        lambda **kwargs: pytest.fail('report opened metadata'))

    report = pbp.build_phone_binary_probe_report(
        **paths,
        collar=2000,
        n_embeds=30,
        n_splits=2,
        phone_labels=labels,
        verbose=False,
    )

    statuses = {item['task']['phone']: item['status']
                for item in report['tasks']}
    assert statuses == {
        'complete': 'complete',
        'partial': 'partial',
        'missing': 'missing',
        'malformed': 'failed',
        'checksum': 'failed',
    }
    assert report['summary']['n_complete'] == 1
    assert report['summary']['n_partial'] == 1
    assert report['summary']['n_missing'] == 1
    assert report['summary']['n_failed'] == 2
    missing = next(
        task for task in report['tasks']
        if task['task']['phone'] == 'missing')
    assert missing['matching_run_ids'] == []
    checksum = next(
        task for task in report['tasks']
        if task['task']['phone'] == 'checksum')
    assert any(
        error['type'] == 'ChecksumMismatch'
        for fold in checksum['folds'] for error in fold['errors'])
    saved = json.loads(Path(report['report_path']).read_text(encoding='utf-8'))
    assert saved == report
    assert Path(report['report_path']) == (
        paths['probe_save_dir'].resolve() / probe_report._probe_report_filename)
    assert not list(paths['probe_save_dir'].glob(
        f'.{probe_report._probe_report_filename}.*.tmp'))


def test_report_uses_selected_run_pointer_and_flags_ambiguous_fallback(
    tmp_path,
):
    paths = _report_paths(tmp_path, ['p', 'a'])
    first = _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'p',
        discriminator='first')
    second = _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'p',
        discriminator='second')
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'a',
        discriminator='first')
    _write_probe_run(
        paths['probe_save_dir'], paths['results_dir'], 'a',
        discriminator='second')
    probe_common._write_selected_run_pointer(
        phone='p', model_name=MODEL_NAME, layer=9, collar=2000,
        n_embeds=30, n_splits=2, random_state=42, standardize=False,
        probe_save_dir=paths['probe_save_dir'], run_id=second,
        worker_status='completed',
    )

    report = pbp.build_phone_binary_probe_report(
        **paths,
        collar=2000,
        n_embeds=30,
        n_splits=2,
        phone_labels=['p', 'a'],
        verbose=False,
    )

    by_phone = {task['task']['phone']: task for task in report['tasks']}
    assert by_phone['p']['status'] == 'complete'
    assert by_phone['p']['run_id'] == second
    assert by_phone['p']['matching_run_ids'] == sorted([first, second])
    assert by_phone['a']['status'] == 'failed'
    assert by_phone['a']['run_id'] is None
    assert by_phone['a']['errors'][0]['type'] == 'AmbiguousMatchingRuns'


def _preflight_report(model_root, phone_labels=('p',)):
    store_path = (model_root / MODEL_NAME).resolve()
    layer = {
        'model_name': MODEL_NAME,
        'layer': 9,
        'collar': 2000,
        'status': 'complete',
        'complete': True,
        'n_total': 930,
        'n_available': 930,
        'n_missing': 0,
        'cache_status': 'cached',
    }
    return {
        'status': 'complete',
        'phone_labels': list(phone_labels),
        'phones_per_label': 30,
        'phraser_store_opened': False,
        'models': [{
            'model_name': MODEL_NAME,
            'store_path': str(store_path),
            'status': 'complete',
            'layers': [layer],
        }],
        'summary': {
            'n_models': 1, 'n_layers': 1,
            'complete_layers': 1, 'incomplete_layers': 0,
            'failed_layers': 0, 'cached_layers': 1,
            'checked_layers': 0,
        },
    }


def _patch_temporary_directory(monkeypatch, tmp_path):
    original = probe_sweep.tempfile.TemporaryDirectory
    temporary_root = tmp_path / 'temporary-runs'
    temporary_root.mkdir()
    created = []

    def temporary_directory(*, prefix, dir):
        context = original(prefix=prefix, dir=temporary_root)

        class Context:
            def __enter__(self):
                path = context.__enter__()
                created.append(Path(path))
                return path

            def __exit__(self, *args):
                return context.__exit__(*args)

        return Context()

    monkeypatch.setattr(
        probe_sweep.tempfile, 'TemporaryDirectory', temporary_directory)
    return created


def _patch_sweep_prerequisites(tmp_path, monkeypatch, *, interrupted=False):
    labels = ['p']
    paths = _report_paths(tmp_path, labels)
    preflight = _preflight_report(paths['model_stores_root'], labels)
    monkeypatch.setattr(
        probe_sweep, 'check_phone_binary_probe_metadata',
        lambda **kwargs: preflight)
    monkeypatch.setattr(
        probe_sweep, '_sweep_phone_labels_from_preflight',
        lambda report, **kwargs: labels,
    )

    def scheduler(tasks, *, temporary_directory, **kwargs):
        assert Path(temporary_directory).is_dir()
        if interrupted:
            outcomes = [{
                'task_index': 0,
                'task': {
                    'phone': 'p', 'model_name': MODEL_NAME, 'layer': 9},
                'model_store_path': str(
                    paths['model_stores_root'] / MODEL_NAME),
                'status': 'interrupted',
                'returncode': -15,
                'elapsed_seconds': 0.1,
                'worker_status': None,
                'command': ['synthetic'],
                'log_tail': '',
            }]
        else:
            outcomes = [{
                'task_index': 0,
                'task': {
                    'phone': 'p', 'model_name': MODEL_NAME, 'layer': 9},
                'model_store_path': str(
                    paths['model_stores_root'] / MODEL_NAME),
                'status': 'completed',
                'returncode': 0,
                'elapsed_seconds': 0.1,
                'worker_status': {
                    'metrics': _probe_result(),
                },
                'command': ['synthetic'],
            }]
        return {
            'outcomes': outcomes,
            'interrupted': interrupted,
            'n_not_started': 0,
            'elapsed_seconds': 0.1,
        }

    monkeypatch.setattr(probe_sweep, '_run_sweep_subprocesses', scheduler)
    created = _patch_temporary_directory(monkeypatch, tmp_path)
    return paths, created


def test_sweep_returns_exact_persisted_report_and_removes_temporary_directory(
    tmp_path, monkeypatch,
):
    paths, created = _patch_sweep_prerequisites(tmp_path, monkeypatch)

    report = pbp.run_phone_binary_probe_sweep(
        **paths,
        n_embeds=30,
        n_splits=2,
        save_probes=False,
        save_predictions=False,
        jobs=77,
        verbose=False,
    )

    assert report['status'] == 'complete'
    assert report['settings']['jobs'] == 77
    assert json.loads(
        Path(report['report_path']).read_text(encoding='utf-8')) == report
    assert len(created) == 1
    assert not created[0].exists()


def test_interrupted_sweep_persists_report_cleans_up_and_raises_with_report(
    tmp_path, monkeypatch,
):
    paths, created = _patch_sweep_prerequisites(
        tmp_path, monkeypatch, interrupted=True)

    with pytest.raises(pbp.PhoneBinaryProbeSweepInterrupted) as captured:
        pbp.run_phone_binary_probe_sweep(
            **paths,
            n_embeds=30,
            n_splits=2,
            save_probes=False,
            save_predictions=False,
            jobs=2,
            verbose=False,
        )

    report = captured.value.report
    assert report['status'] == 'interrupted'
    assert json.loads(
        Path(report['report_path']).read_text(encoding='utf-8')) == report
    assert len(created) == 1
    assert not created[0].exists()
