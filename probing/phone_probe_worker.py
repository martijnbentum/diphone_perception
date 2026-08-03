'''Single phone/model/layer binary-probe worker.'''

import os
import time
from contextlib import ExitStack
from pathlib import Path

import echoframe

from probing import metadata
from probing.extract_embeddings import (
    default_model_stores_root,
    default_phraser_source_id,
    model_store_path as model_store_directory,
)
from probing.phone_probe_common import (
    _close_phones_store,
    _compact_error,
    _task_identity,
    _utc_timestamp,
    _write_json_atomic,
    _write_selected_run_pointer,
)
from probing.probe_utils import default_probe_save_dir, default_results_dir
from probing.train_binary_embedding_probe import train_binary_embedding_probe


_task_status_schema_version = 1


def _write_task_status(path, status):
    if path is not None:
        _write_json_atomic(path, status)


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


def _validate_task_identity(phone, model_name, layer):
    if not isinstance(phone, str) or not phone:
        raise TypeError('phone must be a non-empty string')
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError('model_name must be a non-empty string')
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError('layer must be a positive integer')
    if layer <= 0:
        raise ValueError('layer must be a positive integer')


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
