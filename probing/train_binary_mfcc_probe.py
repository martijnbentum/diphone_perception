from datetime import datetime, timezone
from pathlib import Path

import echoframe
import numpy as np

from probing import probe_utils
from probing.extract_mfcc import default_store_root

default_probe_save_dir = probe_utils.default_probe_save_dir
default_results_dir = probe_utils.default_results_dir
default_mfcc_store_root = default_store_root
_frame_modes = {'center', 'mean', 'first', 'last'}
_mfcc_results_schema_version = 1
_mfcc_results_filename = 'mfcc_probe_results.json'
_run_results_filename = 'results.json'


def _validate_frame(frame):
    if frame not in _frame_modes:
        raise ValueError(
            f'frame must be one of {sorted(_frame_modes)}, got {frame!r}')


def _mfcc_echoframe_keys(store, selected):
    return [
        store.make_echoframe_key(
            'acoustic_feature', feature_name='mfcc',
            phraser_key=phraser_phone.key)
        for _, phraser_phone, _ in selected
    ]


def _load_mfcc_vectors(store, selected, frame='center'):
    '''Batch-load one frame reduction from each stored phone MFCC matrix.'''
    _validate_frame(frame)
    keys = _mfcc_echoframe_keys(store, selected)
    vectors = store.load_many_frames(
        keys, frame=frame, keep_missing=True)

    X, y, true_labels, missing = [], [], [], []
    pairs = zip(selected, vectors, strict=True)
    for (phone, _, binary_label), vector in pairs:
        if vector is None:
            missing.append(phone)
            continue
        X.append(np.asarray(vector))
        y.append(binary_label)
        true_labels.append(phone.phoneme_ipa)
    return np.array(X), np.array(y), np.array(true_labels), missing


def _run_directory(root, target_phoneme, frame, run_id):
    return (
        Path(root) / 'mfcc' / target_phoneme / f'frame-{frame}' / run_id
    )


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _compact_mfcc_probe_result(target_phoneme, result):
    if not isinstance(result, dict):
        raise TypeError(
            f'MFCC result for {target_phoneme!r} must be a dictionary')
    result_target = result.get('target_phoneme')
    if result_target != target_phoneme:
        raise ValueError(
            f'MFCC result key {target_phoneme!r} does not match '
            f'target_phoneme {result_target!r}')
    accuracies = [float(value) for value in result['accuracies']]
    return {
        'representation': result.get('representation', 'mfcc'),
        'target_phoneme': target_phoneme,
        'feature_name': result.get('feature_name', 'mfcc'),
        'frame': result['frame'],
        'run_id': result['run_id'],
        'cache_status': result['cache_status'],
        'standardize': bool(result['standardize']),
        'accuracies': accuracies,
        'mean_accuracy': float(result['mean_accuracy']),
        'std_accuracy': float(result['std_accuracy']),
        'n_samples': (
            None if result['n_samples'] is None else int(result['n_samples'])),
        'n_missing': (
            None if result['n_missing'] is None else int(result['n_missing'])),
        'skipped': bool(result['skipped']),
    }


def save_mfcc_probe_results(
    results,
    *,
    output_path=None,
    results_dir=default_results_dir,
    frame='center',
    n_samples=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    overwrite=False,
    verbose=True,
):
    '''Atomically save JSON-safe metrics from one or more MFCC probe runs.

    Fitted estimators are intentionally omitted because they are persisted by
    the fold cache. When ``output_path`` is omitted, the latest consolidated
    result is written below ``results_dir/mfcc``.
    '''
    if not isinstance(results, dict) or not results:
        raise ValueError('results must be a non-empty dictionary')
    for name, value in (
        ('standardize', standardize),
        ('save_probes', save_probes),
        ('save_predictions', save_predictions),
        ('overwrite', overwrite),
        ('verbose', verbose),
    ):
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a boolean')
    _validate_frame(frame)
    probe_utils.validate_probe_arguments(n_splits, standardize)
    compact_results = {
        target: _compact_mfcc_probe_result(target, result)
        for target, result in results.items()
    }
    mismatched_frames = sorted({
        result['frame']
        for result in compact_results.values()
        if result['frame'] != frame
    })
    if mismatched_frames:
        raise ValueError(
            f'result frames {mismatched_frames} do not match frame={frame!r}')

    results_dir = Path(results_dir).expanduser().resolve()
    if output_path is None:
        output_path = results_dir / 'mfcc' / _mfcc_results_filename
    output_path = Path(output_path).expanduser().resolve()
    report = {
        'schema_version': _mfcc_results_schema_version,
        'kind': 'binary_mfcc_probe_results',
        'generated_at': _utc_timestamp(),
        'report_path': str(output_path),
        'settings': {
            'frame': frame,
            'n_samples': n_samples,
            'n_splits': n_splits,
            'random_state': random_state,
            'standardize': standardize,
            'save_probes': save_probes,
            'probe_save_dir': str(
                Path(probe_save_dir).expanduser().resolve()),
            'save_predictions': save_predictions,
            'results_dir': str(results_dir),
            'overwrite': overwrite,
        },
        'target_phonemes': list(compact_results),
        'results': compact_results,
    }
    probe_utils._write_json(output_path, report)
    if verbose:
        print(
            f'MFCC probe results: {len(compact_results)} target(s); '
            f'written to {output_path}',
            flush=True,
        )
    return report


def train_binary_mfcc_probe(
    phones,
    target_phoneme,
    store=None,
    store_root=default_mfcc_store_root,
    frame='center',
    n_samples=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    save_results=True,
    overwrite=False,
    verbose=True,
):
    '''Train a binary target-vs-other probe on stored phone MFCCs.

    The primary representation is the center row of each `(frames, 39)`
    matrix: 13 MFCC coefficients, 13 deltas, and 13 delta-deltas.
    `standardize=False` preserves the legacy raw-feature probe. When True,
    StandardScaler is fitted independently inside every cross-validation
    training fold through an sklearn Pipeline.
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    probe_utils.validate_probe_arguments(n_splits, standardize)
    if not isinstance(save_results, bool):
        raise TypeError('save_results must be a boolean')
    _validate_frame(frame)
    probe_utils.validate_unique_phraser_keys(phones)
    selected = probe_utils.select_phones(
        phones, target_phoneme, n_samples, seed=random_state)

    if store is None:
        store = echoframe.Store(str(store_root))
    echoframe_keys = _mfcc_echoframe_keys(store, selected)
    feature_parameters = {
        'feature_name': 'mfcc',
        'frame': frame,
        'dimensions': '13 static + 13 delta + 13 delta-delta',
    }
    manifest = probe_utils.build_probe_run_manifest(
        store, selected, echoframe_keys, 'mfcc', feature_parameters,
        target_phoneme, n_samples, n_splits, random_state, standardize)
    run_id = probe_utils.hash_run_manifest(manifest)
    probe_run_directory = _run_directory(
        probe_save_dir, target_phoneme, frame, run_id)
    predictions_run_directory = _run_directory(
        results_dir, target_phoneme, frame, run_id)

    def load_vectors():
        return _load_mfcc_vectors(store, selected, frame=frame)

    result_fields = {
        'representation': 'mfcc',
        'target_phoneme': target_phoneme,
        'feature_name': 'mfcc',
        'frame': frame,
    }
    result = probe_utils.run_binary_probe(
        load_vectors=load_vectors,
        manifest=manifest,
        probe_run_directory=probe_run_directory,
        predictions_run_directory=predictions_run_directory,
        result_fields=result_fields,
        display_name=f'{target_phoneme} MFCC {frame} frame',
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        save_probes=save_probes,
        save_predictions=save_predictions,
        overwrite=overwrite,
        verbose=verbose,
    )
    if save_results:
        output_path = predictions_run_directory / _run_results_filename
        report = save_mfcc_probe_results(
            {target_phoneme: result},
            output_path=output_path,
            results_dir=results_dir,
            frame=frame,
            n_samples=n_samples,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            save_probes=save_probes,
            probe_save_dir=probe_save_dir,
            save_predictions=save_predictions,
            overwrite=overwrite,
            verbose=verbose,
        )
        result['results_path'] = report['report_path']
    return result


def train_binary_mfcc_probes(
    phones,
    target_phonemes=None,
    store=None,
    store_root=default_mfcc_store_root,
    frame='center',
    n_samples=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    save_results=True,
    results_path=None,
    overwrite=False,
    verbose=True,
):
    '''Train one binary MFCC probe run for each target phoneme.

    When target_phonemes is None, all labels in
    phones.label_to_phraser_phone are used. The Phraser label inventory
    must contain exactly the same number of items for every label.
    '''
    probe_utils.validate_probe_arguments(n_splits, standardize)
    if not isinstance(save_results, bool):
        raise TypeError('save_results must be a boolean')
    _validate_frame(frame)
    targets = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes, n_samples=n_samples)
    probe_utils.validate_unique_phraser_keys(phones)

    owns_store = store is None
    if store is None:
        store = echoframe.Store(str(store_root))

    def train_one(target_phoneme):
        return train_binary_mfcc_probe(
            phones,
            target_phoneme,
            store=store,
            frame=frame,
            n_samples=n_samples,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            save_probes=save_probes,
            probe_save_dir=probe_save_dir,
            save_predictions=save_predictions,
            results_dir=results_dir,
            save_results=False,
            overwrite=overwrite,
            verbose=verbose,
        )

    try:
        results = probe_utils.run_probe_sweep(
            targets, train_one, 'MFCC', verbose=verbose)
        if save_results:
            report = save_mfcc_probe_results(
                results,
                output_path=results_path,
                results_dir=results_dir,
                frame=frame,
                n_samples=n_samples,
                n_splits=n_splits,
                random_state=random_state,
                standardize=standardize,
                save_probes=save_probes,
                probe_save_dir=probe_save_dir,
                save_predictions=save_predictions,
                overwrite=overwrite,
                verbose=verbose,
            )
            for result in results.values():
                result['results_path'] = report['report_path']
        return results
    finally:
        if owns_store:
            store.close()
