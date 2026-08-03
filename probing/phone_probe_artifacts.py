'''Persisted probe-artifact discovery and validation.'''

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from probing import probe_utils
from probing.phone_probe_common import (
    _compact_error,
    _file_fingerprint,
    _selected_run_pointer_path,
    _selected_run_pointer_schema_version,
    _selected_run_selector,
    _selected_run_selector_id,
)


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
