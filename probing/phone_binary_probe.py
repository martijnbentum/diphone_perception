'''Path-based worker and command-line interface for one phone probe task.'''

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import echoframe
from progressbar import (
    Bar, ETA, Percentage, ProgressBar, SimpleProgress, Variable,
)

from probing import metadata, probe_utils
from probing.extract_embeddings import (
    default_model_stores_root,
    default_phraser_source_id,
    model_store_path as model_store_directory,
)
from probing.probe_utils import default_probe_save_dir, default_results_dir
from probing.train_binary_embedding_probe import (
    check_embedding_inventory,
    checkpoint_probe_layers,
    discover_wav2vec2_checkpoint_stores,
    train_binary_embedding_probe,
)


_task_status_schema_version = 1
_metadata_status_schema_version = 2
_metadata_status_filename = 'phone_binary_probe_metadata_status.json'
_probe_report_schema_version = 1
_probe_report_filename = 'phone_binary_probe_report.json'
_default_metadata_batch_size = 1_000
_default_sweep_jobs = 31
_default_sweep_poll_interval = 0.2
_expected_phone_label_count = 31
_failure_log_tail_lines = 40
_maximum_error_message_length = 2_000
_hash_chunk_size = 1024 * 1024
_selected_run_pointer_schema_version = 1
_selected_run_pointer_prefix = 'selected_run_'


class PhoneBinaryProbeSweepInterrupted(KeyboardInterrupt):
    '''Raised after an interrupted sweep has stopped and recorded its work.'''

    def __init__(self, report):
        super().__init__('phone binary-probe sweep interrupted')
        self.report = report


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(text)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolved_path(path):
    return str(Path(path).expanduser().resolve())


def _file_fingerprint(path):
    path = Path(path).expanduser().resolve()
    output = {'path': str(path), 'exists': path.is_file()}
    if not output['exists']:
        return output

    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(_hash_chunk_size):
            digest.update(chunk)
    output.update({
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    })
    return output


def _replacement_fingerprint(phraser_key_path, replacement_path):
    if replacement_path is None:
        return {'mode': 'disabled', 'configured_path': None}

    key_path = Path(phraser_key_path).expanduser().resolve()
    replacement_path = Path(replacement_path).expanduser().resolve()
    default_key_path = Path(metadata.phraser_key_file).resolve()
    default_replacement_path = Path(
        metadata.duplicate_replacement_phraser_key_file).resolve()
    if (
        key_path != default_key_path
        and replacement_path == default_replacement_path
    ):
        return {
            'mode': 'ignored_default_for_custom_keys',
            'configured_path': str(replacement_path),
        }
    if not replacement_path.is_file():
        return {
            'mode': 'unavailable',
            'configured_path': str(replacement_path),
        }
    return {
        'mode': 'applied',
        'configured_path': str(replacement_path),
        'file': _file_fingerprint(replacement_path),
    }


def _phone_inventory_fingerprint(
    phraser_key_path, duplicate_replacement_phraser_key_path,
):
    inputs = {
        'phraser_keys': _file_fingerprint(phraser_key_path),
        'duplicate_replacements': _replacement_fingerprint(
            phraser_key_path, duplicate_replacement_phraser_key_path),
    }
    serialized = json.dumps(
        inputs, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')
    return {
        'fingerprint': hashlib.sha256(serialized).hexdigest(),
        'inputs': inputs,
    }


def _load_metadata_status(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open(encoding='utf-8') as file:
            status = json.load(file)
        if not isinstance(status, dict):
            raise TypeError('top-level value must be a JSON object')
        return status
    except (OSError, ValueError, TypeError) as error:
        warnings.warn(
            f'Ignoring unreadable metadata status cache {path}: '
            f'{type(error).__name__}: {error}',
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _matching_cached_layer(
    cache,
    model_name,
    store_path,
    layer,
    collar,
    inventory_fingerprint,
):
    if cache is None:
        return None
    if cache.get('schema_version') != _metadata_status_schema_version:
        return None
    cached_inventory = cache.get('phone_inventory', {})
    if cached_inventory.get('fingerprint') != inventory_fingerprint:
        return None
    model = cache.get('models', {}).get(model_name, {})
    if model.get('store_path') != _resolved_path(store_path):
        return None
    record = (
        model.get('layers', {}).get(str(layer), {}).get(str(collar)))
    if not isinstance(record, dict):
        return None
    expected = {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'phone_inventory_fingerprint': inventory_fingerprint,
        'status': 'complete',
        'complete': True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    if (
        not isinstance(record.get('n_total'), int)
        or record.get('n_available') != record['n_total']
        or record.get('n_missing') != 0
    ):
        return None
    return record


def _metadata_progress_bar(max_value, label):
    return ProgressBar(
        max_value=max(max_value, 1),
        widgets=[
            Variable('label', format='{formatted_value}', width=62),
            ' ', Bar(), ' ', SimpleProgress(), ' ', Percentage(), ' ', ETA(),
        ],
        variables={'label': label},
    ).start()


def _update_metadata_progress(bar, value, label):
    if bar is not None:
        bar.update(value, label=label)


def _finish_metadata_progress(bar, label):
    if bar is not None:
        bar.variables['label'] = label
        bar.finish()


class _MetadataProgressStore:
    def __init__(self, store, update):
        self.store = store
        self.update = update

    def make_echoframe_key(self, *args, **kwargs):
        return self.store.make_echoframe_key(*args, **kwargs)

    def load_many_metadata(self, keys, **kwargs):
        output = self.store.load_many_metadata(keys, **kwargs)
        self.update(len(keys))
        return output


def _metadata_layer_record(
    model_name, layer, collar, inventory_fingerprint, inventory,
):
    status = 'complete' if inventory['complete'] else 'incomplete'
    return {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'phone_inventory_fingerprint': inventory_fingerprint,
        'status': status,
        'n_total': int(inventory['n_total']),
        'n_available': int(inventory['n_available']),
        'n_missing': int(inventory['n_missing']),
        'complete': bool(inventory['complete']),
        'checked_at': _utc_timestamp(),
    }


def _failed_metadata_layer_record(
    model_name, layer, collar, inventory_fingerprint, error, n_total=None,
):
    return {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'phone_inventory_fingerprint': inventory_fingerprint,
        'status': 'failed',
        'n_total': n_total,
        'n_available': None,
        'n_missing': None,
        'complete': False,
        'checked_at': _utc_timestamp(),
        'error': _compact_error(error),
    }


def _metadata_model_status(layer_records):
    statuses = {record['status'] for record in layer_records}
    if 'failed' in statuses:
        return 'failed'
    if 'incomplete' in statuses:
        return 'incomplete'
    return 'complete'


def _metadata_cache_document(
    inventory, collar, started_at, existing_cache=None,
):
    created_at = started_at
    compatible = (
        isinstance(existing_cache, dict)
        and existing_cache.get('schema_version')
        == _metadata_status_schema_version
        and existing_cache.get('phone_inventory', {}).get('fingerprint')
        == inventory['fingerprint']
    )
    if compatible:
        created_at = existing_cache.get('created_at', started_at)
    document = {
        'schema_version': _metadata_status_schema_version,
        'kind': 'phone_binary_probe_metadata_status',
        'phone_inventory': inventory,
        'last_requested_collar': collar,
        'created_at': created_at,
        'updated_at': started_at,
        'models': (
            dict(existing_cache.get('models', {})) if compatible else {}),
        'errors': [],
    }
    if compatible:
        for key in ('phone_labels', 'phones_per_label'):
            if key in existing_cache:
                document[key] = existing_cache[key]
    return document


def _summarize_metadata_preflight(models):
    layers = [
        layer
        for model in models
        for layer in model['layers']
    ]
    counts = {
        status: sum(layer['status'] == status for layer in layers)
        for status in ('complete', 'incomplete', 'failed')
    }
    return {
        'n_models': len(models),
        'n_layers': len(layers),
        'complete_models': sum(
            model['status'] == 'complete' for model in models),
        'incomplete_models': sum(
            model['status'] == 'incomplete' for model in models),
        'failed_models': sum(
            model['status'] == 'failed' for model in models),
        'complete_layers': counts['complete'],
        'incomplete_layers': counts['incomplete'],
        'failed_layers': counts['failed'],
        'cached_layers': sum(
            layer['cache_status'] == 'cached' for layer in layers),
        'checked_layers': sum(
            layer['cache_status'] == 'checked' for layer in layers),
    }


def _write_task_status(path, status):
    if path is not None:
        _write_json_atomic(path, status)


def _task_identity(phone, model_name, layer):
    return {
        'phone': phone,
        'model_name': model_name,
        'layer': layer,
    }


def _selected_run_selector(
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
):
    return {
        'representation': 'embedding',
        'target_phoneme': phone,
        'feature_parameters': {
            'model_name': model_name,
            'layer': layer,
            'collar': collar,
            'frame': 'middle',
        },
        'n_samples': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'classifier': probe_utils.classifier_manifest(standardize),
    }


def _selected_run_selector_id(selector):
    serialized = json.dumps(
        selector,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()[:16]


def _selected_run_pointer_path(probe_save_dir, selector):
    features = selector['feature_parameters']
    directory = (
        Path(probe_save_dir) / features['model_name']
        / selector['target_phoneme'] / f'layer{features["layer"]:02d}'
        / f'collar{features["collar"]}ms'
    )
    selector_id = _selected_run_selector_id(selector)
    return directory / f'{_selected_run_pointer_prefix}{selector_id}.json'


def _write_selected_run_pointer(
    *,
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    probe_save_dir,
    run_id,
    worker_status,
):
    selector = _selected_run_selector(
        phone,
        model_name,
        layer,
        collar,
        n_embeds,
        n_splits,
        random_state,
        standardize,
    )
    selector_id = _selected_run_selector_id(selector)
    path = _selected_run_pointer_path(probe_save_dir, selector)
    pointer = {
        'schema_version': _selected_run_pointer_schema_version,
        'kind': 'phone_binary_probe_selected_run',
        'selector_id': selector_id,
        'selector': selector,
        'run_id': run_id,
        'worker_status': worker_status,
        'updated_at': _utc_timestamp(),
    }
    _write_json_atomic(path, pointer)
    return path


def _task_settings(
    *,
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    model_store_path,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    save_probes,
    probe_save_dir,
    save_predictions,
    results_dir,
    overwrite,
):
    return {
        'metadata_path': str(metadata_path),
        'sentence_path': str(sentence_path),
        'phraser_key_path': str(phraser_key_path),
        'duplicate_replacement_phraser_key_path': (
            None
            if duplicate_replacement_phraser_key_path is None
            else str(duplicate_replacement_phraser_key_path)
        ),
        'model_store_path': str(model_store_path),
        'collar': collar,
        'n_embeds': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'standardize': standardize,
        'save_probes': save_probes,
        'probe_save_dir': str(probe_save_dir),
        'save_predictions': save_predictions,
        'results_dir': str(results_dir),
        'overwrite': overwrite,
    }


def _base_task_status(task, settings, started_at):
    return {
        'schema_version': _task_status_schema_version,
        'status': 'running',
        'task': task,
        'settings': settings,
        'process_id': os.getpid(),
        'started_at': started_at,
        'updated_at': started_at,
    }


def _compact_probe_metrics(result):
    return {
        'run_id': result['run_id'],
        'cache_status': result['cache_status'],
        'accuracies': [float(value) for value in result['accuracies']],
        'mean_accuracy': float(result['mean_accuracy']),
        'std_accuracy': float(result['std_accuracy']),
        'n_samples': (
            None
            if result.get('n_samples') is None
            else int(result['n_samples'])
        ),
        'n_missing': (
            None
            if result.get('n_missing') is None
            else int(result['n_missing'])
        ),
    }


def _compact_error(error):
    message = str(error)
    if len(message) > _maximum_error_message_length:
        message = message[:_maximum_error_message_length - 3] + '...'
    return {
        'type': type(error).__name__,
        'message': message,
    }


def _finish_task_status(base, status, started, **fields):
    finished_at = _utc_timestamp()
    output = dict(base)
    output.update({
        'status': status,
        'updated_at': finished_at,
        'finished_at': finished_at,
        'elapsed_seconds': round(time.perf_counter() - started, 6),
        **fields,
    })
    return output


def _close_phones_store(phones):
    store = getattr(phones, '_store', None)
    if store is not None:
        store.close()


def _validate_task_identity(phone, model_name, layer):
    if not isinstance(phone, str) or not phone:
        raise TypeError('phone must be a non-empty string')
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError('model_name must be a non-empty string')
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError('layer must be a positive integer')
    if layer <= 0:
        raise ValueError('layer must be a positive integer')


def _validate_metadata_preflight_arguments(collar, batch_size):
    if isinstance(collar, bool) or not isinstance(collar, int):
        raise TypeError('collar must be a non-negative integer')
    if collar < 0:
        raise ValueError('collar must be a non-negative integer')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('batch_size must be a positive integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')


def _warn_metadata_failure(message):
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _persist_metadata_model(
    cache_document, status_path, model_name, store_path, records, status,
):
    updated_at = _utc_timestamp()
    resolved_store_path = _resolved_path(store_path)
    previous_model = cache_document['models'].get(model_name, {})
    if previous_model.get('store_path') == resolved_store_path:
        layers = dict(previous_model.get('layers', {}))
    else:
        layers = {}
    for record in records:
        layer_key = str(record['layer'])
        collars = dict(layers.get(layer_key, {}))
        collars[str(record['collar'])] = {
            key: value
            for key, value in record.items()
            if key != 'cache_status'
        }
        layers[layer_key] = collars
    cache_document['models'][model_name] = {
        'model_name': model_name,
        'store_path': resolved_store_path,
        'last_status': status,
        'last_collar': records[0]['collar'] if records else None,
        'updated_at': updated_at,
        'layers': layers,
    }
    cache_document['updated_at'] = updated_at
    _write_json_atomic(status_path, cache_document)


def check_phone_binary_probe_metadata(
    *,
    metadata_path=metadata.metadata_file,
    sentence_path=metadata.sentence_file,
    phraser_key_path=metadata.phraser_key_file,
    duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file),
    model_stores_root=default_model_stores_root,
    collar=2000,
    batch_size=_default_metadata_batch_size,
    force_metadata_check=False,
    verbose=True,
):
    '''Check and cache every required checkpoint embedding inventory.

    Only matching complete layer records are reused. Incomplete, failed,
    stale, and explicitly forced records are checked again. Embedding arrays
    are never loaded.
    '''
    _validate_metadata_preflight_arguments(collar, batch_size)
    model_stores_root = Path(model_stores_root).expanduser().resolve()
    status_path = model_stores_root / _metadata_status_filename
    started_at = _utc_timestamp()
    inventory = _phone_inventory_fingerprint(
        phraser_key_path, duplicate_replacement_phraser_key_path)
    inventory_fingerprint = inventory['fingerprint']
    existing_cache = _load_metadata_status(status_path)
    cache_document = _metadata_cache_document(
        inventory, collar, started_at, existing_cache=existing_cache)
    report = {
        'schema_version': _metadata_status_schema_version,
        'status_path': str(status_path),
        'model_stores_root': str(model_stores_root),
        'phone_inventory': inventory,
        'collar': collar,
        'batch_size': batch_size,
        'force_metadata_check': bool(force_metadata_check),
        'started_at': started_at,
        'models': [],
        'errors': [],
    }

    try:
        checkpoint_stores = discover_wav2vec2_checkpoint_stores(
            model_stores_root)
    except Exception as error:
        message = (
            f'Could not discover checkpoint stores below '
            f'{model_stores_root}: {type(error).__name__}: {error}')
        _warn_metadata_failure(message)
        failure = {'stage': 'discovery', 'error': _compact_error(error)}
        report['errors'].append(failure)
        cache_document['errors'].append(failure)
        cache_document['updated_at'] = _utc_timestamp()
        _write_json_atomic(status_path, cache_document)
        report['finished_at'] = cache_document['updated_at']
        report['status'] = 'failed'
        report['summary'] = _summarize_metadata_preflight([])
        return report

    if not checkpoint_stores:
        error = RuntimeError(
            f'no supported checkpoint stores found below {model_stores_root}')
        _warn_metadata_failure(str(error))
        failure = {'stage': 'discovery', 'error': _compact_error(error)}
        report['errors'].append(failure)
        cache_document['errors'] = [failure]
        cache_document['models'] = {}
        cache_document['updated_at'] = _utc_timestamp()
        _write_json_atomic(status_path, cache_document)
        report['finished_at'] = cache_document['updated_at']
        report['status'] = 'failed'
        report['summary'] = _summarize_metadata_preflight([])
        return report

    phones = None
    overall_bar = None
    if verbose:
        overall_bar = _metadata_progress_bar(
            len(checkpoint_stores), 'metadata stores')

    def get_phones():
        nonlocal phones
        if phones is None:
            phones = metadata.Phones(
                path=metadata_path,
                sentence_path=sentence_path,
                phraser_key_path=phraser_key_path,
                duplicate_replacement_phraser_key_path=(
                    duplicate_replacement_phraser_key_path),
            )
        return phones

    discovered_names = set()
    try:
        for model_index, (model_name, store_path) in enumerate(
            checkpoint_stores, start=1,
        ):
            discovered_names.add(model_name)
            layers = checkpoint_probe_layers(model_name)
            cached_layers = {}
            if not force_metadata_check:
                for layer in layers:
                    record = _matching_cached_layer(
                        existing_cache,
                        model_name,
                        store_path,
                        layer,
                        collar,
                        inventory_fingerprint,
                    )
                    if record is not None:
                        cached_layers[layer] = record

            n_total = None
            if len(cached_layers) == len(layers):
                n_total = next(iter(cached_layers.values()))['n_total']
            else:
                try:
                    n_total = len(get_phones().phraser_phones)
                except Exception as error:
                    message = (
                        'Could not load the phone inventory while checking '
                        f'{model_name}: {type(error).__name__}: {error}')
                    _warn_metadata_failure(message)
                    n_total = next(iter(cached_layers.values()), {}).get(
                        'n_total')
                    records = []
                    for layer in layers:
                        if layer in cached_layers:
                            records.append(dict(cached_layers[layer]))
                        else:
                            records.append(_failed_metadata_layer_record(
                                model_name,
                                layer,
                                collar,
                                inventory_fingerprint,
                                error,
                                n_total=n_total,
                            ))
                    model_status = _metadata_model_status(records)
                    _persist_metadata_model(
                        cache_document,
                        status_path,
                        model_name,
                        store_path,
                        records,
                        model_status,
                    )
                    report_layers = []
                    for record in records:
                        output = dict(record)
                        output['cache_status'] = (
                            'cached'
                            if record['layer'] in cached_layers
                            else 'checked')
                        report_layers.append(output)
                    report['models'].append({
                        'model_name': model_name,
                        'store_path': _resolved_path(store_path),
                        'status': model_status,
                        'cache_status': (
                            'partial' if cached_layers else 'checked'),
                        'layers': report_layers,
                    })
                    _update_metadata_progress(
                        overall_bar, model_index,
                        f'metadata stores ({model_name} failed)')
                    continue

            store_work = n_total * len(layers)
            store_bar = (
                _metadata_progress_bar(
                    store_work, f'{model_name} preparing')
                if verbose else None
            )
            completed_work = 0
            records = []
            store = None
            store_error = None
            if len(cached_layers) != len(layers):
                try:
                    store = echoframe.Store(str(store_path))
                except Exception as error:
                    store_error = error
                    _warn_metadata_failure(
                        f'Could not open checkpoint store for {model_name}: '
                        f'{type(error).__name__}: {error}')

            try:
                for layer in layers:
                    label = f'{model_name} layer {layer}'
                    if layer in cached_layers:
                        record = dict(cached_layers[layer])
                        cache_status = 'cached'
                        completed_work += n_total
                        _update_metadata_progress(
                            store_bar, completed_work, f'{label} cached')
                    elif store_error is not None:
                        record = _failed_metadata_layer_record(
                            model_name,
                            layer,
                            collar,
                            inventory_fingerprint,
                            store_error,
                            n_total=n_total,
                        )
                        cache_status = 'checked'
                        completed_work += n_total
                        _update_metadata_progress(
                            store_bar, completed_work, f'{label} failed')
                    else:
                        layer_start = completed_work

                        def update(checked):
                            nonlocal completed_work
                            completed_work += checked
                            _update_metadata_progress(
                                store_bar, completed_work, label)

                        progress_store = _MetadataProgressStore(store, update)
                        try:
                            checked = check_embedding_inventory(
                                get_phones(),
                                progress_store,
                                model_name,
                                layer,
                                collar=collar,
                                batch_size=batch_size,
                                verbose=False,
                            )
                            record = _metadata_layer_record(
                                model_name,
                                layer,
                                collar,
                                inventory_fingerprint,
                                checked,
                            )
                            if not record['complete']:
                                _warn_metadata_failure(
                                    f'{model_name} layer {layer} has '
                                    f'{record["n_missing"]:,} missing '
                                    'embedding metadata records')
                        except Exception as error:
                            _warn_metadata_failure(
                                f'Embedding metadata check failed for '
                                f'{model_name} layer {layer}: '
                                f'{type(error).__name__}: {error}')
                            record = _failed_metadata_layer_record(
                                model_name,
                                layer,
                                collar,
                                inventory_fingerprint,
                                error,
                                n_total=n_total,
                            )
                        cache_status = 'checked'
                        completed_work = layer_start + n_total
                        _update_metadata_progress(
                            store_bar,
                            completed_work,
                            f'{label} {record["status"]}',
                        )

                    records.append(record)
                    _persist_metadata_model(
                        cache_document,
                        status_path,
                        model_name,
                        store_path,
                        records,
                        'checking',
                    )
                    record['cache_status'] = cache_status
            finally:
                if store is not None:
                    try:
                        store.close()
                    except Exception as error:
                        message = (
                            f'Could not close checkpoint store for '
                            f'{model_name}: {type(error).__name__}: {error}')
                        _warn_metadata_failure(message)
                        report['errors'].append({
                            'model_name': model_name,
                            'stage': 'close',
                            'error': _compact_error(error),
                        })

            persistent_records = []
            for record in records:
                persistent = dict(record)
                persistent.pop('cache_status', None)
                persistent_records.append(persistent)
            model_status = _metadata_model_status(persistent_records)
            _persist_metadata_model(
                cache_document,
                status_path,
                model_name,
                store_path,
                persistent_records,
                model_status,
            )
            model_cache_status = (
                'cached'
                if len(cached_layers) == len(layers)
                else 'checked'
                if not cached_layers
                else 'partial'
            )
            report['models'].append({
                'model_name': model_name,
                'store_path': _resolved_path(store_path),
                'status': model_status,
                'cache_status': model_cache_status,
                'layers': records,
            })
            _finish_metadata_progress(
                store_bar, f'{model_name} {model_cache_status}')
            _update_metadata_progress(
                overall_bar,
                model_index,
                f'metadata stores ({model_name} {model_status})',
            )
    finally:
        cached_labels = cache_document.get('phone_labels')
        cached_count = cache_document.get('phones_per_label')
        labels_are_valid = False
        if cached_labels is not None:
            try:
                _validate_cached_phone_labels(
                    cached_labels, cached_count, n_embeds=None)
                labels_are_valid = True
            except (TypeError, ValueError):
                pass
        if not labels_are_valid:
            try:
                labels, phones_per_label = _validated_phone_label_inventory(
                    get_phones())
                cache_document['phone_labels'] = labels
                cache_document['phones_per_label'] = phones_per_label
            except Exception as error:
                failure = {
                    'stage': 'phone_labels',
                    'error': _compact_error(error),
                }
                report['errors'].append(failure)
                cache_document.pop('phone_labels', None)
                cache_document.pop('phones_per_label', None)
        report['phone_labels'] = cache_document.get('phone_labels')
        report['phones_per_label'] = cache_document.get('phones_per_label')
        report['phraser_store_opened'] = bool(
            phones is not None and getattr(phones, '_store', None) is not None)
        if phones is not None:
            _close_phones_store(phones)
        _finish_metadata_progress(overall_bar, 'metadata stores complete')

    cache_document['models'] = {
        model_name: value
        for model_name, value in cache_document['models'].items()
        if model_name in discovered_names
    }
    cache_document['updated_at'] = _utc_timestamp()
    cache_document['errors'] = list(report['errors'])
    _write_json_atomic(status_path, cache_document)
    report['finished_at'] = cache_document['updated_at']
    report['summary'] = _summarize_metadata_preflight(report['models'])
    if report['errors'] or report['summary']['failed_layers']:
        report['status'] = 'failed'
    elif report['summary']['incomplete_layers']:
        report['status'] = 'incomplete'
    else:
        report['status'] = 'complete'
    if verbose:
        summary = report['summary']
        print(
            'Metadata preflight complete: '
            f'{summary["complete_layers"]} complete, '
            f'{summary["incomplete_layers"]} incomplete, '
            f'{summary["failed_layers"]} failed '
            f'({summary["cached_layers"]} cached)',
            flush=True,
        )
    return report


def _validate_sweep_arguments(jobs, n_embeds, n_splits, standardize):
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError('jobs must be a positive integer')
    if jobs <= 0:
        raise ValueError('jobs must be a positive integer')
    if n_embeds is not None:
        if isinstance(n_embeds, bool) or not isinstance(n_embeds, int):
            raise TypeError('n_embeds must be a positive integer or None')
        if n_embeds <= 0:
            raise ValueError('n_embeds must be a positive integer or None')
    probe_utils.validate_probe_arguments(n_splits, standardize)


def _load_sweep_phone_labels(
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    n_embeds,
):
    phones = metadata.Phones(
        path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
    )
    try:
        labels, _ = _validated_phone_label_inventory(
            phones, n_embeds=n_embeds)
    finally:
        _close_phones_store(phones)
    return labels


def _validated_phone_label_inventory(phones, n_embeds=None):
    labels = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes=None, n_samples=n_embeds)
    if len(labels) != _expected_phone_label_count:
        raise ValueError(
            f'expected {_expected_phone_label_count} Netherlandic phone '
            f'labels, found {len(labels)}')
    counts = {
        len(items) for items in phones.label_to_phraser_phone.values()}
    if len(counts) != 1:
        raise ValueError('phone inventory is not balanced')
    return labels, next(iter(counts))


def _validate_cached_phone_labels(labels, phones_per_label, n_embeds):
    labels = _report_phone_labels(labels)
    if len(labels) != _expected_phone_label_count:
        raise ValueError(
            f'expected {_expected_phone_label_count} Netherlandic phone '
            f'labels, found {len(labels)}')
    if (
        isinstance(phones_per_label, bool)
        or not isinstance(phones_per_label, int)
        or phones_per_label <= 0
    ):
        raise ValueError('cached phones_per_label must be a positive integer')
    requested = phones_per_label if n_embeds is None else n_embeds
    if requested > phones_per_label:
        raise ValueError(
            f'n_embeds={requested} exceeds the balanced inventory of '
            f'{phones_per_label} items per label')
    if requested // (len(labels) - 1) == 0:
        raise ValueError(
            f'n_embeds={requested} is too small to split across '
            f'{len(labels) - 1} other phone labels')
    return labels


def _sweep_phone_labels_from_preflight(
    metadata_report,
    *,
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    n_embeds,
):
    labels = metadata_report.get('phone_labels')
    if labels is not None:
        return _validate_cached_phone_labels(
            labels, metadata_report.get('phones_per_label'), n_embeds)
    if metadata_report.get('phraser_store_opened'):
        raise ValueError(
            'metadata preflight opened the Phraser store but could not '
            'produce a valid phone-label inventory; refusing to open a '
            'second Phraser store in this process')
    return _load_sweep_phone_labels(
        metadata_path,
        sentence_path,
        phraser_key_path,
        duplicate_replacement_phraser_key_path,
        n_embeds,
    )


def _build_sweep_task_lists(metadata_report, phone_labels):
    tasks = []
    skipped = []
    for model in metadata_report['models']:
        model_name = model['model_name']
        store_path = model['store_path']
        for layer in model['layers']:
            if layer['status'] == 'complete' and layer['complete']:
                tasks.extend({
                    'phone': phone,
                    'model_name': model_name,
                    'layer': layer['layer'],
                    'model_store_path': store_path,
                } for phone in phone_labels)
                continue
            skipped.append({
                'model_name': model_name,
                'model_store_path': store_path,
                'layer': layer['layer'],
                'metadata_status': layer['status'],
                'n_total': layer.get('n_total'),
                'n_available': layer.get('n_available'),
                'n_missing': layer.get('n_missing'),
                'n_phone_tasks': len(phone_labels),
                'reason': 'embedding metadata inventory is not complete',
                'error': layer.get('error'),
            })
    return tasks, skipped


def _append_boolean_cli_argument(command, name, value):
    command.append(f'--{name}' if value else f'--no-{name}')


def _build_train_subprocess_command(
    task,
    *,
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    model_stores_root,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    save_probes,
    probe_save_dir,
    save_predictions,
    results_dir,
    overwrite,
    verbose,
    task_status_path,
):
    command = [
        sys.executable,
        '-u',
        '-m',
        'probing.phone_binary_probe',
        'train',
        '--phone',
        task['phone'],
        '--model-name',
        task['model_name'],
        '--layer',
        str(task['layer']),
        '--metadata-path',
        str(metadata_path),
        '--sentence-path',
        str(sentence_path),
        '--phraser-key-path',
        str(phraser_key_path),
    ]
    if duplicate_replacement_phraser_key_path is None:
        command.append('--no-duplicate-replacement-phraser-key')
    else:
        command.extend([
            '--duplicate-replacement-phraser-key-path',
            str(duplicate_replacement_phraser_key_path),
        ])
    command.extend([
        '--model-store-path',
        str(task['model_store_path']),
        '--model-stores-root',
        str(model_stores_root),
        '--collar',
        str(collar),
    ])
    if n_embeds is not None:
        command.extend(['--n-embeds', str(n_embeds)])
    command.extend([
        '--n-splits',
        str(n_splits),
        '--random-state',
        str(random_state),
    ])
    _append_boolean_cli_argument(command, 'standardize', standardize)
    _append_boolean_cli_argument(command, 'save-probes', save_probes)
    command.extend(['--probe-save-dir', str(probe_save_dir)])
    _append_boolean_cli_argument(
        command, 'save-predictions', save_predictions)
    command.extend(['--results-dir', str(results_dir)])
    _append_boolean_cli_argument(command, 'overwrite', overwrite)
    _append_boolean_cli_argument(command, 'verbose', verbose)
    command.extend(['--task-status-path', str(task_status_path)])
    return command


def _read_task_status(path):
    try:
        with Path(path).open(encoding='utf-8') as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise TypeError('task status must be a JSON object')
        return value, None
    except (OSError, ValueError, TypeError) as error:
        return None, _compact_error(error)


def _read_log_tail(path, n_lines=_failure_log_tail_lines):
    try:
        with Path(path).open('rb') as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 64 * 1024))
            text = file.read().decode('utf-8', errors='replace')
    except OSError as error:
        return f'Could not read worker log: {type(error).__name__}: {error}'
    return '\n'.join(text.splitlines()[-n_lines:])


def _worker_outcome(worker):
    worker['log_file'].close()
    status, status_error = _read_task_status(worker['status_path'])
    expected_identity = _task_identity(
        worker['task']['phone'],
        worker['task']['model_name'],
        worker['task']['layer'],
    )
    valid_status = (
        status is not None
        and status.get('task') == expected_identity
        and status.get('status') in {'completed', 'already_complete'}
    )
    succeeded = worker['process'].returncode == 0 and valid_status
    outcome = {
        'task_index': worker['task_index'],
        'task': expected_identity,
        'model_store_path': str(worker['task']['model_store_path']),
        'status': status['status'] if succeeded else 'failed',
        'returncode': worker['process'].returncode,
        'elapsed_seconds': round(
            time.perf_counter() - worker['started'], 6),
        'worker_status': status,
        'command': worker['command'],
    }
    if not succeeded:
        if status is not None and status.get('error') is not None:
            outcome['error'] = status['error']
        elif status_error is not None:
            outcome['error'] = status_error
        else:
            outcome['error'] = {
                'type': 'WorkerProcessError',
                'message': (
                    f'worker exited with status '
                    f'{worker["process"].returncode} without a valid '
                    'completed task status'),
            }
        outcome['log_tail'] = _read_log_tail(worker['log_path'])
    return outcome


def _failed_launch_outcome(task, task_index, command, log_path, error):
    return {
        'task_index': task_index,
        'task': _task_identity(
            task['phone'], task['model_name'], task['layer']),
        'model_store_path': str(task['model_store_path']),
        'status': 'failed',
        'returncode': None,
        'elapsed_seconds': 0.0,
        'worker_status': None,
        'command': command,
        'error': _compact_error(error),
        'log_tail': _read_log_tail(log_path),
    }


def _interrupted_worker_outcome(worker):
    worker['log_file'].close()
    status, _ = _read_task_status(worker['status_path'])
    return {
        'task_index': worker['task_index'],
        'task': _task_identity(
            worker['task']['phone'],
            worker['task']['model_name'],
            worker['task']['layer'],
        ),
        'model_store_path': str(worker['task']['model_store_path']),
        'status': 'interrupted',
        'returncode': worker['process'].returncode,
        'elapsed_seconds': round(
            time.perf_counter() - worker['started'], 6),
        'worker_status': status,
        'command': worker['command'],
        'log_tail': _read_log_tail(worker['log_path']),
    }


def _format_clock(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _sweep_counts(outcomes):
    return {
        status: sum(outcome['status'] == status for outcome in outcomes)
        for status in (
            'completed', 'already_complete', 'failed', 'interrupted')
    }


class _SweepProgress:
    def __init__(self, total, started, verbose):
        self.total = total
        self.started = started
        self.verbose = verbose
        self.is_terminal = sys.stdout.isatty()
        self.last_length = 0
        self.last_finished = None
        self.last_printed = 0.0

    def update(self, active, outcomes, force=False):
        if not self.verbose:
            return
        counts = _sweep_counts(outcomes)
        finished = sum(counts.values())
        now = time.perf_counter()
        interval = 1.0 if self.is_terminal else 30.0
        changed = finished != self.last_finished
        if not force and not changed and now - self.last_printed < interval:
            return
        elapsed = now - self.started
        eta = (
            elapsed / finished * (self.total - finished)
            if finished else None
        )
        line = (
            f'Phone probes: {finished}/{self.total} finished | '
            f'{active} active | {counts["completed"]} trained | '
            f'{counts["already_complete"]} already complete | '
            f'{counts["failed"]} failed | '
            f'elapsed {_format_clock(elapsed)} | ETA {_format_clock(eta)}'
        )
        if self.is_terminal:
            print(
                '\r' + line.ljust(self.last_length),
                end='',
                flush=True,
            )
            self.last_length = max(self.last_length, len(line))
        else:
            print(line, flush=True)
        self.last_finished = finished
        self.last_printed = now

    def finish(self, active, outcomes):
        self.update(active, outcomes, force=True)
        if self.verbose and self.is_terminal:
            print(flush=True)


def _print_worker_failure(outcome):
    task = outcome['task']
    print(
        f'Phone probe worker failed: {task["model_name"]} layer '
        f'{task["layer"]} phone {task["phone"]!r}',
        file=sys.stderr,
        flush=True,
    )
    log_tail = outcome.get('log_tail')
    if log_tail:
        print('--- worker log tail ---', file=sys.stderr)
        print(log_tail, file=sys.stderr)
        print('--- end worker log tail ---', file=sys.stderr, flush=True)


def _stop_active_workers(active):
    for worker in active:
        if worker['process'].poll() is None:
            try:
                worker['process'].terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    while any(worker['process'].poll() is None for worker in active):
        if time.monotonic() >= deadline:
            break
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            break
    for worker in active:
        if worker['process'].poll() is None:
            try:
                worker['process'].kill()
            except ProcessLookupError:
                pass
    for worker in active:
        try:
            worker['process'].wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _run_sweep_subprocesses(
    tasks,
    *,
    jobs,
    temporary_directory,
    command_options,
    poll_interval,
    verbose,
):
    tasks = list(tasks)
    active = []
    outcomes = []
    next_task_index = 0
    started = time.perf_counter()
    progress = _SweepProgress(len(tasks), started, verbose)
    interrupted = False
    repository_root = Path(__file__).resolve().parent.parent

    try:
        while next_task_index < len(tasks) or active:
            while next_task_index < len(tasks) and len(active) < jobs:
                task_index = next_task_index
                task = tasks[task_index]
                next_task_index += 1
                status_path = (
                    Path(temporary_directory)
                    / f'task-{task_index:06d}.status.json')
                log_path = (
                    Path(temporary_directory)
                    / f'task-{task_index:06d}.log')
                command = _build_train_subprocess_command(
                    task,
                    task_status_path=status_path,
                    **command_options,
                )
                log_file = log_path.open('wb')
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=repository_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                except Exception as error:
                    log_file.close()
                    outcome = _failed_launch_outcome(
                        task, task_index, command, log_path, error)
                    outcomes.append(outcome)
                    _print_worker_failure(outcome)
                    continue
                active.append({
                    'task_index': task_index,
                    'task': task,
                    'command': command,
                    'process': process,
                    'status_path': status_path,
                    'log_path': log_path,
                    'log_file': log_file,
                    'started': time.perf_counter(),
                })

            completed_workers = [
                worker
                for worker in active
                if worker['process'].poll() is not None
            ]
            for worker in completed_workers:
                active.remove(worker)
                outcome = _worker_outcome(worker)
                outcomes.append(outcome)
                if outcome['status'] == 'failed':
                    _print_worker_failure(outcome)
            progress.update(len(active), outcomes)
            if active and not completed_workers:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        interrupted = True
        _stop_active_workers(active)
        outcomes.extend(
            _interrupted_worker_outcome(worker) for worker in active)
        active.clear()
    except BaseException:
        _stop_active_workers(active)
        for worker in active:
            worker['log_file'].close()
        raise
    finally:
        progress.finish(len(active), outcomes)

    outcomes.sort(key=lambda outcome: outcome['task_index'])
    return {
        'outcomes': outcomes,
        'interrupted': interrupted,
        'n_not_started': len(tasks) - next_task_index,
        'elapsed_seconds': round(time.perf_counter() - started, 6),
    }


def _sweep_status(metadata_report, counts, skipped, n_not_started):
    if counts['interrupted'] or n_not_started:
        return 'interrupted'
    has_metadata_issues = bool(skipped) or metadata_report['status'] != 'complete'
    if counts['failed'] or has_metadata_issues:
        return 'completed_with_issues'
    return 'complete'


def _read_report_json(path, description):
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        with path.open(encoding='utf-8') as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise TypeError('top-level value must be a JSON object')
        return value, None
    except (OSError, ValueError, TypeError) as error:
        return None, {
            'path': str(path),
            'description': description,
            'error': _compact_error(error),
        }


def _path_modified_at(path):
    try:
        timestamp = Path(path).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(
        timestamp, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def _report_phone_labels(phone_labels):
    labels = list(phone_labels)
    if not labels:
        raise ValueError('phone_labels must not be empty')
    for label in labels:
        probe_utils.validate_target_phoneme(label)
    if len(set(labels)) != len(labels):
        raise ValueError('phone_labels contains duplicate labels')
    return labels


def _metadata_record_for_report(
    cache,
    cache_is_current,
    inventory_fingerprint,
    model_name,
    store_path,
    layer,
    collar,
    current_check_records,
):
    current_check = current_check_records.get((model_name, layer, collar))
    base = {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'store_path': str(store_path),
        'cache_status': (
            current_check.get('cache_status')
            if current_check is not None else None),
    }
    if cache is None:
        return {**base, 'status': 'missing', 'complete': False}
    if not cache_is_current:
        return {**base, 'status': 'stale', 'complete': False}

    model = cache.get('models', {}).get(model_name)
    if not isinstance(model, dict):
        return {**base, 'status': 'missing', 'complete': False}
    if model.get('store_path') != str(store_path):
        return {**base, 'status': 'stale', 'complete': False}
    record = model.get('layers', {}).get(
        str(layer), {}).get(str(collar))
    if not isinstance(record, dict):
        return {**base, 'status': 'missing', 'complete': False}

    expected = {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'phone_inventory_fingerprint': inventory_fingerprint,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return {**base, 'status': 'stale', 'complete': False}
    output = {**base, **record}
    output['complete'] = bool(
        output.get('status') == 'complete'
        and output.get('complete') is True
        and isinstance(output.get('n_total'), int)
        and output.get('n_available') == output['n_total']
        and output.get('n_missing') == 0
    )
    if output.get('status') == 'complete' and not output['complete']:
        output['status'] = 'failed'
        output['error'] = {
            'type': 'InvalidMetadataStatus',
            'message': 'complete metadata record has inconsistent counts',
        }
    return output


def _metadata_report_from_cache(
    *,
    model_stores_root,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    collar,
    current_metadata_report=None,
):
    model_stores_root = Path(model_stores_root).expanduser().resolve()
    status_path = model_stores_root / _metadata_status_filename
    cache, cache_error = _read_report_json(
        status_path, 'metadata status cache')
    inventory = _phone_inventory_fingerprint(
        phraser_key_path, duplicate_replacement_phraser_key_path)
    inventory_fingerprint = inventory['fingerprint']
    cache_is_current = bool(
        cache is not None
        and cache.get('schema_version') == _metadata_status_schema_version
        and cache.get('phone_inventory', {}).get('fingerprint')
        == inventory_fingerprint
    )
    errors = [] if cache_error is None else [cache_error]

    discovered = {}
    try:
        discovered = {
            model_name: Path(store_path).expanduser().resolve()
            for model_name, store_path
            in discover_wav2vec2_checkpoint_stores(model_stores_root)
        }
    except (OSError, ValueError, TypeError) as error:
        errors.append({
            'path': str(model_stores_root),
            'description': 'model-store discovery',
            'error': _compact_error(error),
        })

    cached_models = cache.get('models', {}) if isinstance(cache, dict) else {}
    if not discovered:
        for model_name, model in cached_models.items():
            try:
                checkpoint_probe_layers(model_name)
            except (TypeError, ValueError):
                continue
            if isinstance(model, dict) and model.get('store_path'):
                discovered[model_name] = Path(model['store_path'])

    current_check_records = {}
    if isinstance(current_metadata_report, dict):
        for model in current_metadata_report.get('models', []):
            for record in model.get('layers', []):
                current_check_records[(
                    model.get('model_name'),
                    record.get('layer'),
                    record.get('collar'),
                )] = record

    models = []
    for model_name, store_path in discovered.items():
        layers = [
            _metadata_record_for_report(
                cache,
                cache_is_current,
                inventory_fingerprint,
                model_name,
                store_path,
                layer,
                collar,
                current_check_records,
            )
            for layer in checkpoint_probe_layers(model_name)
        ]
        statuses = {record['status'] for record in layers}
        if 'failed' in statuses:
            status = 'failed'
        elif 'incomplete' in statuses:
            status = 'incomplete'
        elif 'stale' in statuses:
            status = 'stale'
        elif 'missing' in statuses:
            status = 'missing'
        else:
            status = 'complete'
        models.append({
            'model_name': model_name,
            'store_path': str(store_path),
            'status': status,
            'layers': layers,
        })

    layer_records = [
        layer for model in models for layer in model['layers']]
    counts = {
        status: sum(record['status'] == status for record in layer_records)
        for status in ('complete', 'incomplete', 'failed', 'missing', 'stale')
    }
    if errors or not models:
        status = 'failed'
    elif counts['failed']:
        status = 'failed'
    elif any(counts[name] for name in ('incomplete', 'missing', 'stale')):
        status = 'incomplete'
    else:
        status = 'complete'
    return {
        'schema_version': _metadata_status_schema_version,
        'status': status,
        'status_path': str(status_path),
        'model_stores_root': str(model_stores_root),
        'phone_inventory': inventory,
        'phone_labels': cache.get('phone_labels') if cache else None,
        'phones_per_label': cache.get('phones_per_label') if cache else None,
        'cache_updated_at': cache.get('updated_at') if cache else None,
        'models': models,
        'errors': errors,
        'summary': {
            'n_models': len(models),
            'n_layers': len(layer_records),
            'complete_layers': counts['complete'],
            'incomplete_layers': counts['incomplete'],
            'failed_layers': counts['failed'],
            'missing_layers': counts['missing'],
            'stale_layers': counts['stale'],
        },
    }


def _manifest_matches_report_task(
    manifest,
    *,
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
):
    if not isinstance(manifest, dict):
        return False
    return all((
        manifest.get('representation') == 'embedding',
        manifest.get('target_phoneme') == phone,
        manifest.get('n_samples') == n_embeds,
        manifest.get('n_splits') == n_splits,
        manifest.get('random_state') == random_state,
        manifest.get('classifier')
        == probe_utils.classifier_manifest(standardize),
        manifest.get('feature_parameters') == {
            'model_name': model_name,
            'layer': layer,
            'collar': collar,
            'frame': 'middle',
        },
    ))


def _report_run_directories(
    root, model_name, phone, layer, collar,
):
    base = (
        Path(root) / model_name / phone / f'layer{layer:02d}'
        / f'collar{collar}ms'
    )
    if not base.is_dir():
        return base, {}
    return base, {
        path.name: path
        for path in base.iterdir()
        if path.is_dir()
    }


def _report_error(error_type, message, path=None):
    output = {'type': error_type, 'message': message}
    if path is not None:
        output['path'] = str(path)
    return output


def _validate_report_fold(
    *,
    probe_run_directory,
    predictions_run_directory,
    run_id,
    fold_index,
    save_probes,
    save_predictions,
    verify_checksums,
):
    probe_path, predictions_path, completion_path = probe_utils.fold_paths(
        probe_run_directory, predictions_run_directory, fold_index)
    output = {
        'fold': fold_index + 1,
        'status': 'missing',
        'accuracy': None,
        'probe_path': str(probe_path),
        'predictions_path': str(predictions_path),
        'completion_path': str(completion_path),
        'errors': [],
    }
    marker, marker_error = _read_report_json(
        completion_path, f'fold {fold_index + 1} completion marker')
    required_paths = []
    if save_probes:
        required_paths.append(('probe', probe_path))
    if save_predictions:
        required_paths.append(('predictions', predictions_path))

    if marker_error is not None:
        output['status'] = 'failed'
        output['errors'].append(marker_error)
        return output
    if marker is None:
        if any(path.is_file() for _, path in required_paths):
            output['status'] = 'partial'
        return output

    expected = {'run_id': run_id, 'fold': fold_index + 1}
    for key, value in expected.items():
        if marker.get(key) != value:
            output['errors'].append(_report_error(
                'InvalidFoldMarker',
                f'expected {key}={value!r}, found {marker.get(key)!r}',
                completion_path,
            ))
    accuracy = marker.get('accuracy')
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
    ):
        output['errors'].append(_report_error(
            'InvalidFoldMarker', 'accuracy must be a finite number',
            completion_path))
    else:
        output['accuracy'] = float(accuracy)
    n_predictions = marker.get('n_predictions')
    if (
        isinstance(n_predictions, bool)
        or not isinstance(n_predictions, int)
        or n_predictions < 0
    ):
        output['errors'].append(_report_error(
            'InvalidFoldMarker',
            'n_predictions must be a non-negative integer',
            completion_path,
        ))
    else:
        output['n_predictions'] = n_predictions

    for label, path in required_paths:
        if not path.is_file():
            output['errors'].append(_report_error(
                'MissingArtifact', f'{label} artifact is missing', path))
            continue
        checksum_key = f'{label}_sha256'
        expected_checksum = marker.get(checksum_key)
        if (
            not isinstance(expected_checksum, str)
            or len(expected_checksum) != 64
        ):
            output['errors'].append(_report_error(
                'InvalidFoldMarker',
                f'{checksum_key} is not a SHA-256 digest',
                completion_path,
            ))
            continue
        if verify_checksums:
            try:
                actual_checksum = _file_fingerprint(path)['sha256']
            except OSError as error:
                output['errors'].append({
                    'path': str(path),
                    'description': f'{label} checksum',
                    'error': _compact_error(error),
                })
                continue
            if actual_checksum != expected_checksum:
                output['errors'].append(_report_error(
                    'ChecksumMismatch',
                    f'{label} SHA-256 does not match its completion marker',
                    path,
                ))

    if output['errors']:
        output['status'] = 'failed'
    elif save_probes and save_predictions:
        output['status'] = 'complete'
        output['checksums_verified'] = bool(verify_checksums)
    else:
        output['status'] = 'partial'
        output['errors'].append(_report_error(
            'UnverifiableFold',
            'both probe and prediction persistence are required for '
            'rebuildable fold completion',
        ))
    return output


def _evaluate_report_run(
    *,
    run_id,
    probe_run_directory,
    predictions_run_directory,
    task_settings,
    save_probes,
    save_predictions,
    verify_checksums,
):
    probe_manifest_path = probe_run_directory / 'run.json'
    predictions_manifest_path = predictions_run_directory / 'run.json'
    probe_manifest, probe_error = _read_report_json(
        probe_manifest_path, 'probe run manifest')
    predictions_manifest, predictions_error = _read_report_json(
        predictions_manifest_path, 'prediction run manifest')
    probe_matches = _manifest_matches_report_task(
        probe_manifest, **task_settings)
    predictions_match = _manifest_matches_report_task(
        predictions_manifest, **task_settings)
    diagnostics = [
        error for error in (probe_error, predictions_error)
        if error is not None
    ]
    if not probe_matches and not predictions_match:
        return None, diagnostics

    errors = list(diagnostics)
    matching_manifest = (
        probe_manifest if probe_matches else predictions_manifest)
    if probe_manifest is None:
        errors.append(_report_error(
            'MissingManifest', 'probe run manifest is missing',
            probe_manifest_path))
    elif not probe_matches:
        errors.append(_report_error(
            'ManifestMismatch',
            'probe run manifest does not match the requested settings',
            probe_manifest_path,
        ))
    if predictions_manifest is None:
        errors.append(_report_error(
            'MissingManifest', 'prediction run manifest is missing',
            predictions_manifest_path))
    elif not predictions_match:
        errors.append(_report_error(
            'ManifestMismatch',
            'prediction run manifest does not match the requested settings',
            predictions_manifest_path,
        ))
    if (
        probe_manifest is not None
        and predictions_manifest is not None
        and probe_manifest != predictions_manifest
    ):
        errors.append(_report_error(
            'ManifestMismatch',
            'probe and prediction run manifests are different',
        ))

    computed_run_id = probe_utils.hash_run_manifest(matching_manifest)
    if computed_run_id != run_id:
        errors.append(_report_error(
            'RunIdMismatch',
            f'run directory is {run_id!r}, but the manifest hashes to '
            f'{computed_run_id!r}',
        ))

    folds = [
        _validate_report_fold(
            probe_run_directory=probe_run_directory,
            predictions_run_directory=predictions_run_directory,
            run_id=run_id,
            fold_index=fold_index,
            save_probes=save_probes,
            save_predictions=save_predictions,
            verify_checksums=verify_checksums,
        )
        for fold_index in range(task_settings['n_splits'])
    ]
    fold_statuses = [fold['status'] for fold in folds]
    if errors or 'failed' in fold_statuses:
        status = 'failed'
    elif all(status == 'complete' for status in fold_statuses):
        status = 'complete'
    elif any(status in {'complete', 'partial'} for status in fold_statuses):
        status = 'partial'
    else:
        status = 'missing'
    accuracies = [
        fold['accuracy']
        for fold in folds
        if fold['status'] == 'complete' and fold['accuracy'] is not None
    ]
    timestamps = [
        value for value in (
            _path_modified_at(probe_manifest_path),
            _path_modified_at(predictions_manifest_path),
            *(
                _path_modified_at(fold['completion_path'])
                for fold in folds
            ),
        )
        if value is not None
    ]
    return {
        'run_id': run_id,
        'status': status,
        'probe_run_directory': str(probe_run_directory),
        'predictions_run_directory': str(predictions_run_directory),
        'manifest': matching_manifest,
        'manifest_hash': computed_run_id,
        'updated_at': max(timestamps, default=None),
        'folds': folds,
        'accuracies': accuracies,
        'mean_accuracy': (
            statistics.fmean(accuracies) if accuracies else None),
        'std_accuracy': (
            statistics.pstdev(accuracies) if accuracies else None),
        'errors': errors,
    }, []


def _matching_report_runs(
    *,
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    save_probes,
    probe_save_dir,
    save_predictions,
    results_dir,
    verify_checksums,
):
    probe_base, probe_directories = _report_run_directories(
        probe_save_dir, model_name, phone, layer, collar)
    predictions_base, predictions_directories = _report_run_directories(
        results_dir, model_name, phone, layer, collar)
    run_ids = sorted(set(probe_directories) | set(predictions_directories))
    task_settings = {
        'phone': phone,
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'n_embeds': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'standardize': standardize,
    }
    matching_runs = []
    diagnostics = []
    for run_id in run_ids:
        run, run_diagnostics = _evaluate_report_run(
            run_id=run_id,
            probe_run_directory=(
                probe_directories.get(run_id, probe_base / run_id)),
            predictions_run_directory=(
                predictions_directories.get(
                    run_id, predictions_base / run_id)),
            task_settings=task_settings,
            save_probes=save_probes,
            save_predictions=save_predictions,
            verify_checksums=verify_checksums,
        )
        diagnostics.extend(run_diagnostics)
        if run is not None:
            matching_runs.append(run)
    return matching_runs, diagnostics


def _read_selected_run_pointer(
    *,
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    probe_save_dir,
):
    selector = _selected_run_selector(
        phone,
        model_name,
        layer,
        collar,
        n_embeds,
        n_splits,
        random_state,
        standardize,
    )
    selector_id = _selected_run_selector_id(selector)
    path = _selected_run_pointer_path(probe_save_dir, selector)
    pointer, read_error = _read_report_json(
        path, 'selected-run pointer')
    if read_error is not None:
        return path, None, read_error
    if pointer is None:
        return path, None, None
    expected = {
        'schema_version': _selected_run_pointer_schema_version,
        'kind': 'phone_binary_probe_selected_run',
        'selector_id': selector_id,
        'selector': selector,
    }
    mismatched = [
        key for key, value in expected.items()
        if pointer.get(key) != value
    ]
    run_id = pointer.get('run_id')
    if mismatched or not isinstance(run_id, str) or not run_id:
        details = (
            f'mismatched fields: {mismatched}' if mismatched
            else 'run_id must be a non-empty string')
        return path, None, {
            'path': str(path),
            'description': 'selected-run pointer',
            'error': _report_error('InvalidSelectedRunPointer', details),
        }
    return path, pointer, None


def _worker_outcome_map(current_worker_outcomes):
    output = {}
    for outcome in current_worker_outcomes or []:
        if not isinstance(outcome, dict):
            continue
        task = outcome.get('task', {})
        identity = (
            task.get('phone'), task.get('model_name'), task.get('layer'))
        if all(value is not None for value in identity):
            output[identity] = outcome
    return output


def _compact_current_worker_outcome(outcome):
    if outcome is None:
        return None
    return {
        key: outcome[key]
        for key in (
            'task_index', 'task', 'status', 'returncode', 'elapsed_seconds',
            'worker_status', 'error', 'log_tail',
        )
        if key in outcome
    }


def _task_metrics_from_worker(outcome):
    if not isinstance(outcome, dict):
        return None
    worker_status = outcome.get('worker_status')
    if not isinstance(worker_status, dict):
        return None
    metrics = worker_status.get('metrics')
    return metrics if isinstance(metrics, dict) else None


def _metadata_skipped_layers(metadata_report, n_phone_labels):
    output = []
    for model in metadata_report['models']:
        for layer in model['layers']:
            if layer['complete']:
                continue
            output.append({
                'model_name': model['model_name'],
                'model_store_path': model['store_path'],
                'layer': layer['layer'],
                'metadata_status': layer['status'],
                'n_total': layer.get('n_total'),
                'n_available': layer.get('n_available'),
                'n_missing': layer.get('n_missing'),
                'n_phone_tasks': n_phone_labels,
                'reason': 'embedding metadata inventory is not complete',
                'error': layer.get('error'),
            })
    return output


def _report_task(
    *,
    phone,
    metadata_layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    save_probes,
    probe_save_dir,
    save_predictions,
    results_dir,
    verify_checksums,
    current_outcome,
):
    model_name = metadata_layer['model_name']
    layer = metadata_layer['layer']
    runs, diagnostics = _matching_report_runs(
        phone=phone,
        model_name=model_name,
        layer=layer,
        collar=collar,
        n_embeds=n_embeds,
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        save_probes=save_probes,
        probe_save_dir=probe_save_dir,
        save_predictions=save_predictions,
        results_dir=results_dir,
        verify_checksums=verify_checksums,
    )
    worker_metrics = _task_metrics_from_worker(current_outcome)
    pointer_path, pointer, pointer_error = _read_selected_run_pointer(
        phone=phone,
        model_name=model_name,
        layer=layer,
        collar=collar,
        n_embeds=n_embeds,
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        probe_save_dir=probe_save_dir,
    )
    selected = None
    errors = []
    if pointer_error is not None:
        errors.append(pointer_error)
    elif pointer is not None:
        pointer_run_id = pointer['run_id']
        selected = next(
            (run for run in runs if run['run_id'] == pointer_run_id), None)
        if selected is None:
            errors.append(_report_error(
                'MissingSelectedRun',
                f'selected-run pointer names {pointer_run_id!r}, but that '
                'run does not match the requested persisted artifacts',
                pointer_path,
            ))
    elif len(runs) == 1:
        selected = runs[0]
    elif len(runs) > 1:
        errors.append(_report_error(
            'AmbiguousMatchingRuns',
            'multiple persisted run IDs match the requested settings and '
            'no selected-run pointer identifies the active run',
        ))
    if selected is None:
        errors.extend(diagnostics)

    current_status = (
        current_outcome.get('status')
        if isinstance(current_outcome, dict) else None)
    if current_status in {'failed', 'interrupted'}:
        status = 'failed'
        if current_outcome.get('error') is not None:
            errors.append(current_outcome['error'])
    elif errors:
        status = 'failed'
    elif selected is not None:
        status = selected['status']
    elif (
        current_status in {'completed', 'already_complete'}
        and not (save_probes and save_predictions)
        and worker_metrics is not None
    ):
        status = 'complete'
    else:
        status = 'missing'

    if selected is not None:
        folds = selected['folds']
        accuracies = selected['accuracies']
        mean_accuracy = selected['mean_accuracy']
        std_accuracy = selected['std_accuracy']
        run_id = selected['run_id']
        artifact = selected
    elif status == 'complete' and worker_metrics is not None:
        accuracies = [
            float(value) for value in worker_metrics.get('accuracies', [])]
        folds = [
            {
                'fold': index,
                'status': 'complete',
                'accuracy': accuracy,
                'source': 'current_worker',
            }
            for index, accuracy in enumerate(accuracies, start=1)
        ]
        mean_accuracy = worker_metrics.get('mean_accuracy')
        std_accuracy = worker_metrics.get('std_accuracy')
        run_id = worker_metrics.get('run_id')
        artifact = None
    else:
        folds = selected['folds'] if selected is not None else []
        accuracies = selected['accuracies'] if selected is not None else []
        mean_accuracy = (
            selected['mean_accuracy'] if selected is not None else None)
        std_accuracy = (
            selected['std_accuracy'] if selected is not None else None)
        run_id = (
            selected['run_id'] if selected is not None
            else pointer.get('run_id') if pointer is not None
            else None
        )
        artifact = selected

    return {
        'task': _task_identity(phone, model_name, layer),
        'status': status,
        'metadata': metadata_layer,
        'run_id': run_id,
        'artifact': artifact,
        'selected_run_pointer_path': str(pointer_path),
        'selected_run_pointer': pointer,
        'matching_run_ids': [run['run_id'] for run in runs],
        'ignored_artifact_diagnostics': (
            diagnostics if selected is not None else []),
        'folds': folds,
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'cache_outcome': current_status,
        'cache_status': (
            worker_metrics.get('cache_status')
            if isinstance(worker_metrics, dict) else None),
        'elapsed_seconds': (
            current_outcome.get('elapsed_seconds')
            if isinstance(current_outcome, dict) else None),
        'current_worker': _compact_current_worker_outcome(current_outcome),
        'errors': errors,
    }


def _report_summary(
    tasks, metadata_report, metadata_skipped, current_worker_outcomes,
    n_not_started, n_phone_labels,
):
    task_counts = {
        status: sum(task['status'] == status for task in tasks)
        for status in ('complete', 'partial', 'missing', 'failed')
    }
    current_counts = _sweep_counts(current_worker_outcomes or [])
    complete_accuracies = [
        task['mean_accuracy']
        for task in tasks
        if task['status'] == 'complete'
        and task['mean_accuracy'] is not None
    ]
    return {
        'n_phone_labels': n_phone_labels,
        'n_models': metadata_report['summary']['n_models'],
        'n_model_layers': metadata_report['summary']['n_layers'],
        'n_expected_tasks': len(tasks),
        'n_complete': task_counts['complete'],
        'n_partial': task_counts['partial'],
        'n_missing': task_counts['missing'],
        'n_failed': task_counts['failed'],
        'n_metadata_skipped_model_layers': len(metadata_skipped),
        'n_metadata_skipped_tasks': sum(
            layer['n_phone_tasks'] for layer in metadata_skipped),
        'n_scheduled_tasks': (
            len(current_worker_outcomes or []) + n_not_started),
        'n_finished_tasks': sum(current_counts.values()),
        'n_trained': current_counts['completed'],
        'n_already_complete': current_counts['already_complete'],
        'n_current_failed': current_counts['failed'],
        'n_interrupted': current_counts['interrupted'],
        'n_not_started': n_not_started,
        'mean_task_accuracy': (
            statistics.fmean(complete_accuracies)
            if complete_accuracies else None),
        'std_task_accuracy': (
            statistics.pstdev(complete_accuracies)
            if complete_accuracies else None),
    }


def build_phone_binary_probe_report(
    *,
    metadata_path=metadata.metadata_file,
    sentence_path=metadata.sentence_file,
    phraser_key_path=metadata.phraser_key_file,
    duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file),
    model_stores_root=default_model_stores_root,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    phone_labels=None,
    current_worker_outcomes=None,
    current_sweep=None,
    current_metadata_report=None,
    verify_checksums=True,
    verbose=True,
):
    '''Build and persist a report from metadata and probe artifacts only.

    This function never trains, opens an Echoframe store, deserializes a
    probe, or loads an embedding array. ``phone_labels`` avoids reloading the
    path-based Netherlandic metadata when a sweep already has the labels.
    '''
    probe_utils.validate_probe_arguments(n_splits, standardize)
    _validate_metadata_preflight_arguments(collar, 1)
    if not isinstance(save_probes, bool):
        raise TypeError('save_probes must be a boolean')
    if not isinstance(save_predictions, bool):
        raise TypeError('save_predictions must be a boolean')
    if not isinstance(verify_checksums, bool):
        raise TypeError('verify_checksums must be a boolean')
    if n_embeds is not None:
        if isinstance(n_embeds, bool) or not isinstance(n_embeds, int):
            raise TypeError('n_embeds must be a positive integer or None')
        if n_embeds <= 0:
            raise ValueError('n_embeds must be a positive integer or None')

    metadata_report = _metadata_report_from_cache(
        model_stores_root=model_stores_root,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        collar=collar,
        current_metadata_report=current_metadata_report,
    )
    if phone_labels is None:
        phone_labels = metadata_report.get('phone_labels')
        if phone_labels is None:
            raise ValueError(
                'metadata status cache does not contain phone_labels; run '
                'the metadata preflight once before rebuilding a report')
        phone_labels = _validate_cached_phone_labels(
            phone_labels,
            metadata_report.get('phones_per_label'),
            n_embeds,
        )
    else:
        phone_labels = _report_phone_labels(phone_labels)
    outcome_by_task = _worker_outcome_map(current_worker_outcomes)
    tasks = []
    expected_identities = set()
    for model in metadata_report['models']:
        for metadata_layer in model['layers']:
            for phone in phone_labels:
                identity = (
                    phone, model['model_name'], metadata_layer['layer'])
                expected_identities.add(identity)
                tasks.append(_report_task(
                    phone=phone,
                    metadata_layer=metadata_layer,
                    collar=collar,
                    n_embeds=n_embeds,
                    n_splits=n_splits,
                    random_state=random_state,
                    standardize=standardize,
                    save_probes=save_probes,
                    probe_save_dir=probe_save_dir,
                    save_predictions=save_predictions,
                    results_dir=results_dir,
                    verify_checksums=verify_checksums,
                    current_outcome=outcome_by_task.get(identity),
                ))

    unexpected_outcomes = [
        _compact_current_worker_outcome(outcome)
        for identity, outcome in outcome_by_task.items()
        if identity not in expected_identities
    ]
    metadata_skipped = _metadata_skipped_layers(
        metadata_report, len(phone_labels))
    current_sweep = dict(current_sweep or {})
    n_not_started = int(current_sweep.get('n_not_started', 0))
    summary = _report_summary(
        tasks,
        metadata_report,
        metadata_skipped,
        current_worker_outcomes,
        n_not_started,
        len(phone_labels),
    )
    interrupted = bool(
        current_sweep.get('interrupted') or summary['n_interrupted'])
    if interrupted:
        status = 'interrupted'
    elif (
        metadata_report['status'] != 'complete'
        or summary['n_partial']
        or summary['n_missing']
        or summary['n_failed']
        or unexpected_outcomes
    ):
        status = 'completed_with_issues'
    else:
        status = 'complete'

    probe_save_dir = Path(probe_save_dir).expanduser().resolve()
    report_path = probe_save_dir / _probe_report_filename
    generated_at = _utc_timestamp()
    report = {
        'schema_version': _probe_report_schema_version,
        'kind': 'phone_binary_probe_report',
        'run_id': current_sweep.get('run_id'),
        'status': status,
        'generated_at': generated_at,
        'started_at': current_sweep.get('started_at'),
        'finished_at': current_sweep.get('finished_at', generated_at),
        'elapsed_seconds': current_sweep.get('elapsed_seconds'),
        'report_path': str(report_path),
        'paths': {
            'metadata_path': _resolved_path(metadata_path),
            'sentence_path': _resolved_path(sentence_path),
            'phraser_key_path': _resolved_path(phraser_key_path),
            'duplicate_replacement_phraser_key_path': (
                None
                if duplicate_replacement_phraser_key_path is None
                else _resolved_path(duplicate_replacement_phraser_key_path)
            ),
            'model_stores_root': _resolved_path(model_stores_root),
            'probe_save_dir': str(probe_save_dir),
            'results_dir': _resolved_path(results_dir),
        },
        'settings': {
            'collar': collar,
            'n_embeds': n_embeds,
            'n_splits': n_splits,
            'random_state': random_state,
            'standardize': standardize,
            'save_probes': save_probes,
            'save_predictions': save_predictions,
            'verify_checksums': verify_checksums,
            **current_sweep.get('settings', {}),
        },
        'phone_labels': phone_labels,
        'metadata_preflight': metadata_report,
        'metadata_skipped_model_layers': metadata_skipped,
        'tasks': tasks,
        'unexpected_current_worker_outcomes': unexpected_outcomes,
        'summary': summary,
    }
    _write_json_atomic(report_path, report)
    if verbose:
        print(
            f'Phone probe report: {summary["n_complete"]} complete, '
            f'{summary["n_partial"]} partial, '
            f'{summary["n_missing"]} missing, '
            f'{summary["n_failed"]} failed; written to {report_path}',
            flush=True,
        )
    return report


def run_phone_binary_probe_sweep(
    *,
    metadata_path=metadata.metadata_file,
    sentence_path=metadata.sentence_file,
    phraser_key_path=metadata.phraser_key_file,
    duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file),
    model_stores_root=default_model_stores_root,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    jobs=_default_sweep_jobs,
    metadata_batch_size=_default_metadata_batch_size,
    force_metadata_check=False,
    verbose=True,
    poll_interval=_default_sweep_poll_interval,
):
    '''Run all complete checkpoint phone probes in bounded subprocesses.

    The metadata preflight always finishes before task construction. Each
    child runs this module's ``train`` command for one phone/model/layer and
    handles every fold for that task. Worker output and atomic status files
    live only in a unique temporary directory and are incorporated into the
    returned structure before that directory is removed.
    '''
    _validate_sweep_arguments(jobs, n_embeds, n_splits, standardize)
    _validate_metadata_preflight_arguments(collar, metadata_batch_size)
    if isinstance(poll_interval, bool) or not isinstance(
        poll_interval, (int, float),
    ):
        raise TypeError('poll_interval must be a positive number')
    if poll_interval <= 0:
        raise ValueError('poll_interval must be a positive number')

    sweep_started = time.perf_counter()
    started_at = _utc_timestamp()
    metadata_report = check_phone_binary_probe_metadata(
        metadata_path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        model_stores_root=model_stores_root,
        collar=collar,
        batch_size=metadata_batch_size,
        force_metadata_check=force_metadata_check,
        verbose=verbose,
    )
    phone_labels = _sweep_phone_labels_from_preflight(
        metadata_report,
        metadata_path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        n_embeds=n_embeds,
    )
    tasks, metadata_skipped = _build_sweep_task_lists(
        metadata_report, phone_labels)
    command_options = {
        'metadata_path': metadata_path,
        'sentence_path': sentence_path,
        'phraser_key_path': phraser_key_path,
        'duplicate_replacement_phraser_key_path': (
            duplicate_replacement_phraser_key_path),
        'model_stores_root': model_stores_root,
        'collar': collar,
        'n_embeds': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'standardize': standardize,
        'save_probes': save_probes,
        'probe_save_dir': probe_save_dir,
        'save_predictions': save_predictions,
        'results_dir': results_dir,
        'overwrite': overwrite,
        'verbose': verbose,
    }

    with tempfile.TemporaryDirectory(
        prefix='diphone-phone-probes-', dir='/tmp',
    ) as temporary_directory:
        run_id = Path(temporary_directory).name.removeprefix(
            'diphone-phone-probes-')
        subprocess_result = _run_sweep_subprocesses(
            tasks,
            jobs=jobs,
            temporary_directory=temporary_directory,
            command_options=command_options,
            poll_interval=float(poll_interval),
            verbose=verbose,
        )
        outcomes = subprocess_result['outcomes']
        counts = _sweep_counts(outcomes)
        status = _sweep_status(
            metadata_report,
            counts,
            metadata_skipped,
            subprocess_result['n_not_started'],
        )
        finished_at = _utc_timestamp()
        elapsed_seconds = round(time.perf_counter() - sweep_started, 6)
        report_settings = {
            'metadata_path': str(metadata_path),
            'sentence_path': str(sentence_path),
            'phraser_key_path': str(phraser_key_path),
            'duplicate_replacement_phraser_key_path': (
                None
                if duplicate_replacement_phraser_key_path is None
                else str(duplicate_replacement_phraser_key_path)
            ),
            'model_stores_root': str(model_stores_root),
            'collar': collar,
            'n_embeds': n_embeds,
            'n_splits': n_splits,
            'random_state': random_state,
            'standardize': standardize,
            'save_probes': save_probes,
            'probe_save_dir': str(probe_save_dir),
            'save_predictions': save_predictions,
            'results_dir': str(results_dir),
            'overwrite': overwrite,
            'verbose': verbose,
            'jobs': jobs,
            'metadata_batch_size': metadata_batch_size,
            'force_metadata_check': bool(force_metadata_check),
        }
        current_sweep = {
            'run_id': run_id,
            'status': status,
            'started_at': started_at,
            'finished_at': finished_at,
            'elapsed_seconds': elapsed_seconds,
            'settings': report_settings,
            'interrupted': subprocess_result['interrupted'],
            'n_not_started': subprocess_result['n_not_started'],
        }
        report = build_phone_binary_probe_report(
            metadata_path=metadata_path,
            sentence_path=sentence_path,
            phraser_key_path=phraser_key_path,
            duplicate_replacement_phraser_key_path=(
                duplicate_replacement_phraser_key_path),
            model_stores_root=model_stores_root,
            collar=collar,
            n_embeds=n_embeds,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            save_probes=save_probes,
            probe_save_dir=probe_save_dir,
            save_predictions=save_predictions,
            results_dir=results_dir,
            phone_labels=phone_labels,
            current_worker_outcomes=outcomes,
            current_sweep=current_sweep,
            current_metadata_report=metadata_report,
            verbose=False,
        )

    if verbose:
        summary = report['summary']
        completion = (
            'interrupted'
            if report['status'] == 'interrupted'
            else 'complete'
        )
        print(
            f'Phone probe sweep {completion}: '
            f'{summary["n_trained"]} trained, '
            f'{summary["n_already_complete"]} already complete, '
            f'{summary["n_current_failed"]} failed, '
            f'{summary["n_metadata_skipped_tasks"]} metadata-skipped',
            flush=True,
        )
    if subprocess_result['interrupted']:
        raise PhoneBinaryProbeSweepInterrupted(report)
    return report


def train_phone_binary_probe(
    phone,
    model_name,
    layer,
    *,
    metadata_path=metadata.metadata_file,
    sentence_path=metadata.sentence_file,
    phraser_key_path=metadata.phraser_key_file,
    duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file),
    model_store_path=None,
    model_stores_root=default_model_stores_root,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    verbose=True,
    task_status_path=None,
):
    '''Train every fold for one phone, model, and layer.

    All resources are opened from paths and closed before returning. A model
    store path supplied by the sweep takes precedence over
    ``model_stores_root``; otherwise the model's dedicated directory below
    that root is used. Existing fold markers are honored by the lower-level
    trainer unless ``overwrite`` is true.
    '''
    _validate_task_identity(phone, model_name, layer)
    if model_store_path is None:
        model_store_path = model_store_directory(
            model_name, stores_root=model_stores_root)
    model_store_path = Path(model_store_path)

    task = _task_identity(phone, model_name, layer)
    settings = _task_settings(
        metadata_path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        model_store_path=model_store_path,
        collar=collar,
        n_embeds=n_embeds,
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        save_probes=save_probes,
        probe_save_dir=probe_save_dir,
        save_predictions=save_predictions,
        results_dir=results_dir,
        overwrite=overwrite,
    )
    started = time.perf_counter()
    started_at = _utc_timestamp()
    status = _base_task_status(task, settings, started_at)

    try:
        _write_task_status(task_status_path, status)
        phones = metadata.Phones(
            path=metadata_path,
            sentence_path=sentence_path,
            phraser_key_path=phraser_key_path,
            duplicate_replacement_phraser_key_path=(
                duplicate_replacement_phraser_key_path),
        )
        with ExitStack() as resources:
            resources.callback(_close_phones_store, phones)
            store = echoframe.Store(str(model_store_path))
            resources.callback(store.close)
            store.attach_phraser_store(
                default_phraser_source_id, phones.store)
            result = train_binary_embedding_probe(
                phones,
                phone,
                store=store,
                model_name=model_name,
                layer=layer,
                collar=collar,
                n_embeds=n_embeds,
                n_splits=n_splits,
                random_state=random_state,
                standardize=standardize,
                save_probes=save_probes,
                probe_save_dir=probe_save_dir,
                save_predictions=save_predictions,
                results_dir=results_dir,
                overwrite=overwrite,
                verbose=verbose,
            )
        outcome = (
            'already_complete' if result.get('skipped') else 'completed')
        selected_run_pointer_path = _write_selected_run_pointer(
            phone=phone,
            model_name=model_name,
            layer=layer,
            collar=collar,
            n_embeds=n_embeds,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            probe_save_dir=probe_save_dir,
            run_id=result['run_id'],
            worker_status=outcome,
        )
    except BaseException as error:
        failed = _finish_task_status(
            status, 'failed', started, error=_compact_error(error))
        try:
            _write_task_status(task_status_path, failed)
        except Exception as status_error:
            if hasattr(error, 'add_note'):
                error.add_note(
                    'Could not write failed task status: '
                    f'{type(status_error).__name__}: {status_error}')
        raise

    finished = _finish_task_status(
        status,
        outcome,
        started,
        metrics=_compact_probe_metrics(result),
        selected_run_pointer_path=str(selected_run_pointer_path),
    )
    _write_task_status(task_status_path, finished)
    return result


def _positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _add_train_arguments(parser):
    parser.add_argument('--phone', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--layer', required=True, type=_positive_integer)
    parser.add_argument(
        '--metadata-path', type=Path, default=metadata.metadata_file)
    parser.add_argument(
        '--sentence-path', type=Path, default=metadata.sentence_file)
    parser.add_argument(
        '--phraser-key-path', type=Path, default=metadata.phraser_key_file)
    replacement_group = parser.add_mutually_exclusive_group()
    replacement_group.add_argument(
        '--duplicate-replacement-phraser-key-path',
        type=Path,
        dest='duplicate_replacement_phraser_key_path',
    )
    replacement_group.add_argument(
        '--no-duplicate-replacement-phraser-key',
        dest='duplicate_replacement_phraser_key_path',
        action='store_const',
        const=None,
    )
    parser.set_defaults(duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file))
    parser.add_argument('--model-store-path', type=Path)
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument('--n-embeds', type=_positive_integer)
    parser.add_argument('--n-splits', type=_positive_integer, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--standardize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--save-probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--probe-save-dir', type=Path, default=default_probe_save_dir)
    parser.add_argument(
        '--save-predictions',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--results-dir', type=Path, default=default_results_dir)
    parser.add_argument(
        '--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--task-status-path', type=Path)


def _add_check_metadata_arguments(parser):
    parser.add_argument(
        '--metadata-path', type=Path, default=metadata.metadata_file)
    parser.add_argument(
        '--sentence-path', type=Path, default=metadata.sentence_file)
    parser.add_argument(
        '--phraser-key-path', type=Path, default=metadata.phraser_key_file)
    parser.add_argument(
        '--duplicate-replacement-phraser-key-path',
        type=Path,
        default=metadata.duplicate_replacement_phraser_key_file,
    )
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument(
        '--batch-size', type=_positive_integer,
        default=_default_metadata_batch_size)
    parser.add_argument('--force-metadata-check', action='store_true')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True)


def _add_sweep_arguments(parser):
    parser.add_argument(
        '--metadata-path', type=Path, default=metadata.metadata_file)
    parser.add_argument(
        '--sentence-path', type=Path, default=metadata.sentence_file)
    parser.add_argument(
        '--phraser-key-path', type=Path, default=metadata.phraser_key_file)
    replacement_group = parser.add_mutually_exclusive_group()
    replacement_group.add_argument(
        '--duplicate-replacement-phraser-key-path',
        type=Path,
        dest='duplicate_replacement_phraser_key_path',
    )
    replacement_group.add_argument(
        '--no-duplicate-replacement-phraser-key',
        dest='duplicate_replacement_phraser_key_path',
        action='store_const',
        const=None,
    )
    parser.set_defaults(duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file))
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument('--n-embeds', type=_positive_integer)
    parser.add_argument('--n-splits', type=_positive_integer, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--standardize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--save-probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--probe-save-dir', type=Path, default=default_probe_save_dir)
    parser.add_argument(
        '--save-predictions',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--results-dir', type=Path, default=default_results_dir)
    parser.add_argument(
        '--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--jobs', type=_positive_integer, default=_default_sweep_jobs)
    parser.add_argument(
        '--metadata-batch-size',
        type=_positive_integer,
        default=_default_metadata_batch_size,
    )
    parser.add_argument('--force-metadata-check', action='store_true')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True)


def _add_report_arguments(parser):
    parser.add_argument(
        '--metadata-path', type=Path, default=metadata.metadata_file)
    parser.add_argument(
        '--sentence-path', type=Path, default=metadata.sentence_file)
    parser.add_argument(
        '--phraser-key-path', type=Path, default=metadata.phraser_key_file)
    replacement_group = parser.add_mutually_exclusive_group()
    replacement_group.add_argument(
        '--duplicate-replacement-phraser-key-path',
        type=Path,
        dest='duplicate_replacement_phraser_key_path',
    )
    replacement_group.add_argument(
        '--no-duplicate-replacement-phraser-key',
        dest='duplicate_replacement_phraser_key_path',
        action='store_const',
        const=None,
    )
    parser.set_defaults(duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file))
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument('--n-embeds', type=_positive_integer)
    parser.add_argument('--n-splits', type=_positive_integer, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--standardize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--save-probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--probe-save-dir', type=Path, default=default_probe_save_dir)
    parser.add_argument(
        '--save-predictions',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--results-dir', type=Path, default=default_results_dir)
    parser.add_argument(
        '--verify-checksums',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Train and inspect path-based binary phone probes.')
    commands = parser.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser(
        'train', help='train every fold for one phone/model/layer task')
    _add_train_arguments(train_parser)
    train_parser.set_defaults(command_handler=_run_train_command)
    metadata_parser = commands.add_parser(
        'check-metadata',
        help='check and cache all checkpoint embedding inventories',
    )
    _add_check_metadata_arguments(metadata_parser)
    metadata_parser.set_defaults(command_handler=_run_check_metadata_command)
    sweep_parser = commands.add_parser(
        'sweep', help='train all complete checkpoint probes in subprocesses')
    _add_sweep_arguments(sweep_parser)
    sweep_parser.set_defaults(command_handler=_run_sweep_command)
    report_parser = commands.add_parser(
        'report', help='rebuild the report from persisted probe artifacts')
    _add_report_arguments(report_parser)
    report_parser.set_defaults(command_handler=_run_report_command)
    return parser


def _run_train_command(arguments):
    train_phone_binary_probe(
        arguments.phone,
        arguments.model_name,
        arguments.layer,
        metadata_path=arguments.metadata_path,
        sentence_path=arguments.sentence_path,
        phraser_key_path=arguments.phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            arguments.duplicate_replacement_phraser_key_path),
        model_store_path=arguments.model_store_path,
        model_stores_root=arguments.model_stores_root,
        collar=arguments.collar,
        n_embeds=arguments.n_embeds,
        n_splits=arguments.n_splits,
        random_state=arguments.random_state,
        standardize=arguments.standardize,
        save_probes=arguments.save_probes,
        probe_save_dir=arguments.probe_save_dir,
        save_predictions=arguments.save_predictions,
        results_dir=arguments.results_dir,
        overwrite=arguments.overwrite,
        verbose=arguments.verbose,
        task_status_path=arguments.task_status_path,
    )
    return 0


def _run_check_metadata_command(arguments):
    report = check_phone_binary_probe_metadata(
        metadata_path=arguments.metadata_path,
        sentence_path=arguments.sentence_path,
        phraser_key_path=arguments.phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            arguments.duplicate_replacement_phraser_key_path),
        model_stores_root=arguments.model_stores_root,
        collar=arguments.collar,
        batch_size=arguments.batch_size,
        force_metadata_check=arguments.force_metadata_check,
        verbose=arguments.verbose,
    )
    return 0 if report['status'] == 'complete' else 1


def _run_sweep_command(arguments):
    report = run_phone_binary_probe_sweep(
        metadata_path=arguments.metadata_path,
        sentence_path=arguments.sentence_path,
        phraser_key_path=arguments.phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            arguments.duplicate_replacement_phraser_key_path),
        model_stores_root=arguments.model_stores_root,
        collar=arguments.collar,
        n_embeds=arguments.n_embeds,
        n_splits=arguments.n_splits,
        random_state=arguments.random_state,
        standardize=arguments.standardize,
        save_probes=arguments.save_probes,
        probe_save_dir=arguments.probe_save_dir,
        save_predictions=arguments.save_predictions,
        results_dir=arguments.results_dir,
        overwrite=arguments.overwrite,
        jobs=arguments.jobs,
        metadata_batch_size=arguments.metadata_batch_size,
        force_metadata_check=arguments.force_metadata_check,
        verbose=arguments.verbose,
    )
    return 0 if report['status'] == 'complete' else 1


def _run_report_command(arguments):
    report = build_phone_binary_probe_report(
        metadata_path=arguments.metadata_path,
        sentence_path=arguments.sentence_path,
        phraser_key_path=arguments.phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            arguments.duplicate_replacement_phraser_key_path),
        model_stores_root=arguments.model_stores_root,
        collar=arguments.collar,
        n_embeds=arguments.n_embeds,
        n_splits=arguments.n_splits,
        random_state=arguments.random_state,
        standardize=arguments.standardize,
        save_probes=arguments.save_probes,
        probe_save_dir=arguments.probe_save_dir,
        save_predictions=arguments.save_predictions,
        results_dir=arguments.results_dir,
        verify_checksums=arguments.verify_checksums,
        verbose=arguments.verbose,
    )
    return 0 if report['status'] == 'complete' else 1


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    try:
        return arguments.command_handler(arguments)
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
