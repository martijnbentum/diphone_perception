import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import echoframe

import locations
from probing import probe_data, probe_run, probe_utils
from probing import result as probe_result

_frame_modes = {'center', 'mean', 'first', 'last'}
_mfcc_results_schema_version = 1
_mfcc_results_filename = 'mfcc_probe_results.json'
_run_results_filename = 'results.json'
_pool_probe_matrix = None


def _validate_frame(frame):
    if frame not in _frame_modes:
        raise ValueError(
            f'frame must be one of {sorted(_frame_modes)}, got {frame!r}')


def _run_directory(root, target_phoneme, frame):
    return Path(root) / 'mfcc' / target_phoneme / f'frame-{frame}'


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
        'accuracies': accuracies,
        'mean_accuracy': float(result['mean_accuracy']),
        'std_accuracy': float(result['std_accuracy']),
        'n_samples': (
            None if result['n_samples'] is None else int(result['n_samples'])),
        'n_missing': (
            None if result['n_missing'] is None else int(result['n_missing'])),
    }


def _phone_report(target_phoneme, frame, results_dir):
    '''Reconstruct one label's accuracy metrics from persisted results.'''
    phone_result = probe_result.PhoneResult.mfcc(target_phoneme, frame,
        root=results_dir)
    run = phone_result.run or {}
    return {'representation': 'mfcc', 'target_phoneme': target_phoneme,
        'feature_name': 'mfcc', 'frame': frame,
        'accuracies': phone_result.accuracies,
        'mean_accuracy': phone_result.mean_accuracy,
        'std_accuracy': phone_result.std_accuracy,
        'n_samples': run.get('actual_n_samples'),
        'n_missing': run.get('actual_n_missing')}


def save_mfcc_probe_results(
    results,
    *,
    output_path=None,
    results_dir=locations.probe_results,
    frame='center',
    probe_save_dir=locations.phone_probes,
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
        ('overwrite', overwrite),
        ('verbose', verbose),
    ):
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a boolean')
    _validate_frame(frame)
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
            'probe_save_dir': str(
                Path(probe_save_dir).expanduser().resolve()),
            'results_dir': str(results_dir),
            'overwrite': overwrite,
        },
        'target_phonemes': list(compact_results),
        'results': compact_results,
    }
    probe_run.write_json(output_path, report)
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
    store_root=locations.echoframe_mfcc_store,
    frame='center',
    expected_target_count=13500,
    probe_matrix=None,
    probe_save_dir=locations.phone_probes,
    results_dir=locations.probe_results,
    save_results=True,
    overwrite=False,
    verbose=True,
):
    '''Train and persist a binary target-vs-other probe on stored phone
    MFCCs.

    The primary representation is the center row of each `(frames, 39)`
    matrix: 13 MFCC coefficients, 13 deltas, and 13 delta-deltas. Every
    available target-phoneme phone is used. Fitted probes and fold
    predictions are always saved. Nothing is returned; read results back
    afterward through probing.result.PhoneResult.

    A true cache hit (all folds already stored, not overwriting) touches
    neither the store nor probe_matrix at all.

    phones:                phone inventory paired with Phraser phones
    target_phoneme:        label used as the positive class
    store:                 optional open Echoframe store
    frame:                 MFCC frame reduction
    expected_target_count:  every phone label must have exactly this many
                            loaded tokens; raises otherwise
    probe_matrix:           optional preloaded probe_data.ProbeMatrix,
                            shared across labels by train_binary_mfcc_probes
    save_results:          whether to write a consolidated JSON report for
                           this one label after training
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    if not isinstance(save_results, bool):
        raise TypeError('save_results must be a boolean')
    _validate_frame(frame)

    phone_result = probe_result.PhoneResult.mfcc(target_phoneme, frame,
        root=results_dir)
    existing_fold_count = len(phone_result.folds)
    complete_before = phone_result.complete
    cache_status = probe_run.classify_cache_status(True, complete_before,
        overwrite, existing_fold_count)

    if cache_status == 'hit':
        if verbose:
            print(f'{target_phoneme} MFCC {frame} frame: cache status: hit',
                flush=True)
    else:
        if probe_matrix is None:
            if store is None:
                store = echoframe.Store(str(store_root))
            probe_matrix = probe_data.build_mfcc_probe_matrix(phones, store,
                frame=frame, expected_target_count=expected_target_count)

        probe_run_directory = _run_directory(probe_save_dir, target_phoneme,
            frame)

        def load_vectors():
            return probe_data.select_balanced_vectors(probe_matrix,
                target_phoneme)

        outcome = probe_run.run(load_vectors=load_vectors,
            probe_run_directory=probe_run_directory, phone_result=phone_result,
            display_name=f'{target_phoneme} MFCC {frame} frame',
            save_probes=True, save_predictions=True,
            overwrite=overwrite, verbose=verbose)

        description = probe_data.describe_probe_run(probe_matrix.phone_labels,
            target_phoneme, 'mfcc', expected_target_count)
        description['actual_n_samples'] = outcome.n_samples
        description['actual_n_missing'] = outcome.n_missing
        phone_result.save_run(description)

        if verbose:
            print(f'{target_phoneme} MFCC {frame} frame: cache status: '
                f'{cache_status}', flush=True)

    if save_results:
        report = _phone_report(target_phoneme, frame, results_dir)
        output_path = phone_result.path / _run_results_filename
        save_mfcc_probe_results(
            {target_phoneme: report},
            output_path=output_path,
            results_dir=results_dir,
            frame=frame,
            probe_save_dir=probe_save_dir,
            overwrite=overwrite,
            verbose=verbose,
        )


def train_binary_mfcc_probes(
    phones,
    target_phonemes=None,
    store=None,
    store_root=locations.echoframe_mfcc_store,
    frame='center',
    expected_target_count=13500,
    max_workers=None,
    probe_save_dir=locations.phone_probes,
    results_dir=locations.probe_results,
    save_results=True,
    results_path=None,
    overwrite=False,
    verbose=True,
    report=False,
):
    '''Train one binary MFCC probe run for each target phoneme.

    Loads one MFCC frame reduction for the whole label inventory once
    (probe_data.build_mfcc_probe_matrix) and reuses that matrix for every
    label, instead of each label re-fetching its own selected subset. All
    labels are used when target_phonemes is omitted. The Phraser label
    inventory must contain exactly the same number of items for every
    label.

    phones:                phone inventory paired with Phraser phones
    target_phonemes:       optional ordered subset of labels to probe
    store:                 optional open Echoframe store shared across
                           labels
    frame:                 MFCC frame reduction
    expected_target_count:  passed to probe_data.build_mfcc_probe_matrix
    max_workers:            trains that many labels concurrently in
                            worker processes, each reusing the one
                            preloaded probe_matrix (sent once per worker
                            process, not once per label). Defaults to
                            min(number of targets, os.cpu_count()), so a
                            small target_phonemes subset or a small
                            machine never over-spawns. Pass 1 to force a
                            single worker process (still out-of-process,
                            just not concurrent). A failure in any label
                            cancels the remaining queued labels and
                            re-raises immediately.
    save_results:          whether to write a consolidated JSON report
                           after training
    report:                when True, reconstruct each label's accuracy
                           metrics from disk after training and return
                           them keyed by target phoneme; when False,
                           return None
    '''
    if not isinstance(save_results, bool):
        raise TypeError('save_results must be a boolean')
    _validate_frame(frame)
    targets = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes)

    if max_workers is None:
        max_workers = max(1, min(len(targets), os.cpu_count() or 1))

    owns_store = store is None
    if store is None:
        store = echoframe.Store(str(store_root))

    probe_matrix = probe_data.build_mfcc_probe_matrix(phones, store,
        frame=frame, expected_target_count=expected_target_count)

    try:
        results = _train_labels_in_pool(targets, probe_matrix, frame,
            expected_target_count, probe_save_dir, results_dir, overwrite,
            verbose, max_workers)
        if save_results:
            output = save_mfcc_probe_results(
                results,
                output_path=results_path,
                results_dir=results_dir,
                frame=frame,
                probe_save_dir=probe_save_dir,
                overwrite=overwrite,
                verbose=verbose,
            )
            for result in results.values():
                result['results_path'] = output['report_path']
        return results if report else None
    finally:
        if owns_store:
            store.close()


def _init_pool_worker(probe_matrix):
    '''Store the shared ProbeMatrix once per worker process, sent through
    ProcessPoolExecutor's initializer rather than as a per-task argument.
    '''
    global _pool_probe_matrix
    _pool_probe_matrix = probe_matrix


def _train_one_label_in_pool(target_phoneme, frame, expected_target_count,
    probe_save_dir, results_dir, overwrite):
    '''Run inside a worker process. Reads the matrix _init_pool_worker
    already stored in this process instead of receiving it as an
    argument. Never touches phones or a store.
    '''
    train_binary_mfcc_probe(None, target_phoneme, frame=frame,
        expected_target_count=expected_target_count,
        probe_matrix=_pool_probe_matrix, probe_save_dir=probe_save_dir,
        results_dir=results_dir, save_results=False, overwrite=overwrite,
        verbose=False)
    return target_phoneme, _phone_report(target_phoneme, frame, results_dir)


def _train_labels_in_pool(targets, probe_matrix, frame, expected_target_count,
    probe_save_dir, results_dir, overwrite, verbose, max_workers):
    '''Train every target phoneme concurrently, sharing one preloaded
    ProbeMatrix once per worker process via an initializer.

    Progress is reported in completion order, but the returned dict is
    reordered to match `targets` - completion order depends on OS
    scheduling and isn't reproducible run to run.
    '''
    context = multiprocessing.get_context('spawn')
    completed_reports = {}
    total = len(targets)
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context,
        initializer=_init_pool_worker, initargs=(probe_matrix,)) as executor:
        futures = {
            executor.submit(_train_one_label_in_pool, target_phoneme, frame,
                expected_target_count, probe_save_dir, results_dir,
                overwrite): target_phoneme
            for target_phoneme in targets}
        try:
            for future in as_completed(futures):
                target_phoneme, label_report = future.result()
                completed_reports[target_phoneme] = label_report
                completed += 1
                if verbose:
                    print(f'[mfcc pool] {completed}/{total} completed '
                        f'{target_phoneme!r}', flush=True)
        except BaseException:
            executor.shutdown(cancel_futures=True)
            raise
    return {target_phoneme: completed_reports[target_phoneme]
        for target_phoneme in targets}
