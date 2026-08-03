'''Persistent embedding-inventory checks for binary phone probes.'''

import json
import warnings
from pathlib import Path

import echoframe
from progressbar import (
    Bar, ETA, Percentage, ProgressBar, SimpleProgress, Variable,
)

from probing import metadata
from probing.extract_embeddings import default_model_stores_root
from probing.phone_probe_common import (
    _close_phones_store,
    _compact_error,
    _phone_inventory_fingerprint,
    _resolved_path,
    _utc_timestamp,
    _validate_cached_phone_labels,
    _validated_phone_label_inventory,
    _write_json_atomic,
)
from probing.train_binary_embedding_probe import (
    check_embedding_inventory,
    checkpoint_probe_layers,
    discover_wav2vec2_checkpoint_stores,
)


_metadata_status_schema_version = 2
_metadata_status_filename = 'phone_binary_probe_metadata_status.json'
_default_metadata_batch_size = 1_000


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
