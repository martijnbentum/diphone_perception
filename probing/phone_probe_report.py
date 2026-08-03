'''Artifact-only report reconstruction for binary phone probes.'''

import statistics
from pathlib import Path

from probing import metadata, probe_utils
from probing.extract_embeddings import default_model_stores_root
from probing.phone_probe_artifacts import (
    _matching_report_runs,
    _read_report_json,
    _read_selected_run_pointer,
    _report_error,
)
from probing.phone_probe_common import (
    _compact_error,
    _phone_inventory_fingerprint,
    _report_phone_labels,
    _resolved_path,
    _sweep_counts,
    _task_identity,
    _utc_timestamp,
    _validate_cached_phone_labels,
    _write_json_atomic,
)
from probing.phone_probe_metadata import (
    _metadata_status_filename,
    _metadata_status_schema_version,
    _validate_metadata_preflight_arguments,
)
from probing.probe_utils import default_probe_save_dir, default_results_dir
from probing.train_binary_embedding_probe import (
    checkpoint_probe_layers,
    discover_wav2vec2_checkpoint_stores,
)


_probe_report_schema_version = 1
_probe_report_filename = 'phone_binary_probe_report.json'


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
