import multiprocessing
import os
import re
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import echoframe
import numpy as np

import locations
from probing import probe_data, probe_run, probe_utils
from probing import result as probe_result
from probing.extract_embeddings import default_model_name

_pool_probe_matrix = None


def train_binary_embedding_probe(phones, target_phoneme, store=None,
    store_root=locations.echoframe_store, model_name=default_model_name,
    layer=9, collar=2000, expected_target_count=13500, probe_matrix=None,
    probe_save_dir=locations.phone_probes,
    results_dir=locations.probe_results, overwrite=False, verbose=True):
    '''Train and persist a binary target-vs-other probe on middle-frame
    model features.

    Every available target-phoneme phone is used. Fitted probes and fold
    predictions are always saved. Nothing is returned; read results back
    afterward through probing.result.PhoneResult.

    A true cache hit (all folds already stored, not overwriting) touches
    neither the store nor probe_matrix at all.

    phones:                phone inventory paired with Phraser phones
    target_phoneme:        label used as the positive class
    store:                 optional open Echoframe store
    model_name:            model identifier stored with each embedding
    layer:                 hidden-state layer index or 'cnn'
    expected_target_count:  passed to probe_data.build_probe_matrix when
                            probe_matrix is not supplied
    probe_matrix:           optional preloaded probe_data.ProbeMatrix,
                            shared across labels by
                            train_binary_embedding_probes
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    representation = probe_utils.representation_for_layer(layer)

    phone_result = probe_result.PhoneResult.model_feature(target_phoneme,
        model_name, layer, collar, root=results_dir)
    existing_fold_count = len(phone_result.folds)
    complete_before = phone_result.complete
    cache_status = probe_run.classify_cache_status(True, complete_before,
        overwrite, existing_fold_count)

    if cache_status == 'hit':
        if verbose:
            print(f'{target_phoneme} layer {layer}: cache status: hit',
                flush=True)
        return

    if probe_matrix is None:
        if store is None:
            store_root = str(store_root)
            store = echoframe.Store(store_root)
        probe_matrix = probe_data.build_probe_matrix(phones, store,
            model_name, layer, collar=collar,
            expected_target_count=expected_target_count)

    probe_run_directory = _run_directory(probe_save_dir, model_name,
        target_phoneme, layer, collar)

    def load_vectors():
        return probe_data.select_balanced_vectors(probe_matrix,
            target_phoneme)

    outcome = probe_run.run(load_vectors=load_vectors,
        probe_run_directory=probe_run_directory, phone_result=phone_result,
        display_name=f'{target_phoneme} layer {layer}',
        save_probes=True, save_predictions=True,
        overwrite=overwrite, verbose=verbose)

    description = probe_data.describe_probe_run(probe_matrix.phone_labels,
        target_phoneme, representation, expected_target_count)
    description['actual_n_samples'] = outcome.n_samples
    description['actual_n_missing'] = outcome.n_missing
    phone_result.save_run(description)

    if verbose:
        print(f'{target_phoneme} layer {layer}: cache status: '
            f'{cache_status}', flush=True)


def train_binary_embedding_probes(phones, target_phonemes=None, store=None,
    store_root=locations.echoframe_store, model_name=default_model_name,
    layer=9, collar=2000, expected_target_count=13500, max_workers=None,
    probe_save_dir=locations.phone_probes,
    results_dir=locations.probe_results, overwrite=False, verbose=True,
    report=False):
    '''Train one binary model-feature probe run for each target phoneme.

    Loads embeddings for the whole label inventory once
    (probe_data.build_probe_matrix) and reuses that matrix for every
    label, instead of each label re-fetching its own selected subset.
    All labels are used when target_phonemes is omitted.

    phones:                phone inventory paired with Phraser phones
    target_phonemes:       optional ordered subset of labels to probe
    store:                 optional open Echoframe store shared across
                           labels
    model_name:            model identifier stored with each feature
    layer:                 hidden-state layer index or 'cnn'
    expected_target_count:  passed to probe_data.build_probe_matrix
    max_workers:            trains that many labels concurrently in
                            worker processes, each reusing the one
                            preloaded probe_matrix (sent once per worker
                            process, not once per label). Defaults to
                            min(number of targets, os.cpu_count()), so a
                            small target_phonemes subset or a small
                            machine never over-spawns. Pass 1 to force a
                            single worker process (still out-of-process,
                            just not concurrent); building your own
                            probe_matrix and calling
                            train_binary_embedding_probe per label with
                            probe_utils.run_probe_sweep is the fully
                            in-process alternative. A failure in any
                            label cancels the remaining queued labels and
                            re-raises immediately.
    report:                when True, reconstruct each label's accuracy
                           metrics from disk after training and return
                           them keyed by target phoneme; when False,
                           return None
    '''
    representation = probe_utils.representation_for_layer(layer)
    if target_phonemes is None:
        targets = sorted(phones.label_to_phraser_phone)
    else:
        targets = list(target_phonemes)

    if max_workers is None:
        max_workers = max(1, min(len(targets), os.cpu_count() or 1))

    owns_store = store is None
    if store is None:
        store_root = str(store_root)
        store = echoframe.Store(store_root)

    probe_matrix = probe_data.build_probe_matrix(phones, store, model_name,
        layer, collar=collar, expected_target_count=expected_target_count)

    try:
        results = _train_labels_in_pool(targets, probe_matrix,
            model_name, layer, collar, expected_target_count,
            probe_save_dir, results_dir, overwrite, verbose, max_workers,
            representation)
        return results if report else None
    finally:
        if owns_store: store.close()


def _init_pool_worker(probe_matrix):
    '''Store the shared ProbeMatrix once per worker process, sent through
    ProcessPoolExecutor's initializer rather than as a per-task argument.
    '''
    global _pool_probe_matrix
    _pool_probe_matrix = probe_matrix


def _train_one_label_in_pool(target_phoneme, model_name, layer, collar,
    expected_target_count, probe_save_dir, results_dir, overwrite):
    '''Run inside a worker process. Reads the matrix _init_pool_worker
    already stored in this process instead of receiving it as an
    argument. Never touches phones or a store.
    '''
    train_binary_embedding_probe(None, target_phoneme,
        model_name=model_name, layer=layer, collar=collar,
        expected_target_count=expected_target_count,
        probe_matrix=_pool_probe_matrix, probe_save_dir=probe_save_dir,
        results_dir=results_dir, overwrite=overwrite, verbose=False)
    return target_phoneme, _phone_report(target_phoneme, model_name, layer,
        collar, results_dir)


def _train_labels_in_pool(targets, probe_matrix, model_name, layer, collar,
    expected_target_count, probe_save_dir, results_dir, overwrite, verbose,
    max_workers, representation):
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
            executor.submit(_train_one_label_in_pool, target_phoneme,
                model_name, layer, collar, expected_target_count,
                probe_save_dir, results_dir, overwrite): target_phoneme
            for target_phoneme in targets}
        try:
            for future in as_completed(futures):
                target_phoneme, label_report = future.result()
                completed_reports[target_phoneme] = label_report
                completed += 1
                if verbose:
                    print(f'[{representation} pool] {completed}/{total} '
                        f'completed {target_phoneme!r}', flush=True)
        except BaseException:
            executor.shutdown(cancel_futures=True)
            raise
    return {target_phoneme: completed_reports[target_phoneme]
        for target_phoneme in targets}


def train_binary_embedding_probe_checkpoint_sweep(phones,
    store_root=locations.echoframe_model_stores, collar=2000,
    expected_target_count=13500, max_workers=None,
    probe_save_dir=locations.phone_probes,
    results_dir=locations.probe_results, overwrite=False, verbose=True):
    '''Train layer-specific all-label probes across wav2vec2 checkpoints.

    Existing fold results are checked before opening a checkpoint store.
    Layers with complete results are skipped; every incomplete layer loads its
    model features directly and validates the loaded per-label inventory before
    training. Failures are reported without stopping later runs. The returned
    report contains compact metrics rather than fitted classifiers.

    phones:                phone inventory paired with Phraser phones
    store_root:            directory containing checkpoint Echoframe stores
    expected_target_count:  passed to probe_data.build_probe_matrix
    max_workers:            passed to train_binary_embedding_probes; when
                            None (default), each checkpoint/layer's labels
                            train concurrently with a worker count that
                            adapts to the label count and machine (see
                            train_binary_embedding_probes)
    verbose:               whether to print progress and the final report
    '''
    targets = probe_utils.prepare_balanced_probe_targets(phones)
    expected_n_samples = _expected_probe_sample_count(phones)
    n_total = len(phones.phraser_phones)
    store_root = Path(store_root)
    report = _empty_checkpoint_sweep_report(store_root)
    probe_options = {'collar': collar,
        'expected_target_count': expected_target_count,
        'max_workers': max_workers,
        'probe_save_dir': probe_save_dir, 'results_dir': results_dir,
        'overwrite': overwrite, 'verbose': verbose, 'report': True}

    try:
        checkpoint_stores = discover_wav2vec2_checkpoint_stores(store_root)
    except Exception as error:
        _record_discovery_failure(report, store_root, error)
        if verbose: _print_checkpoint_sweep_report(report)
        return report

    for model_name, store_path in checkpoint_stores:
        layers = checkpoint_probe_layers(model_name)
        if overwrite:
            layers_to_run = layers
        else:
            missing = probe_result.find_missing_checkpoint_results(
                targets, model_name, layers, collar=collar,
                root=results_dir)
            layers_to_run = []
            for layer in layers:
                if layer in missing:
                    layers_to_run.append(layer)
                    continue
                report['runs'].append({'model_name': model_name,
                    'layer': layer, 'status': 'skipped',
                    'n_total': n_total, 'n_available': None,
                    'n_missing': None,
                    'reason': 'all fold results already stored'})
            layers_to_run = tuple(layers_to_run)
        if not layers_to_run: continue
        _probe_checkpoint_store(report, phones, model_name, store_path,
            layers_to_run, n_total, expected_n_samples, probe_options,
            verbose)

    status_counts = Counter()
    for run in report['runs']:
        status_counts[run['status']] += 1
    report['status_counts'].update(status_counts)
    if verbose: _print_checkpoint_sweep_report(report)
    return report


def discover_wav2vec2_checkpoint_stores(
    store_root=locations.echoframe_model_stores):
    '''Return supported checkpoint store pairs in numeric order.

    Only the exact random checkpoint and directories named
    wav2vec2_nl1_checkpoint-<integer> are included. Unrelated files and
    directories are ignored.
    '''
    store_root = Path(store_root)
    checkpoints = []
    for store_path in store_root.iterdir():
        if not store_path.is_dir(): continue
        model_name = store_path.name
        checkpoint_number = _checkpoint_number(model_name)
        if checkpoint_number is None: continue
        random_name = locations.wav2vec2_random_checkpoint_name
        random_first = 0 if model_name == random_name else 1
        item = checkpoint_number, random_first, model_name, store_path
        checkpoints.append(item)
    checkpoints.sort(key=lambda item: item[:3])
    stores = []
    for _, _, model_name, store_path in checkpoints:
        stores.append((model_name, store_path))
    return stores


def checkpoint_probe_layers(model_name):
    '''Return model-feature layers for one supported wav2vec2 checkpoint.'''
    if _checkpoint_number(model_name) is None:
        message = f'unsupported checkpoint model name: {model_name!r}'
        raise ValueError(message)
    if model_name in locations.wav2vec2_all_layer_checkpoint_names:
        layers = locations.wav2vec2_all_probe_layers
    else:
        layers = (9,)
    return (*layers, 'cnn')


def find_missing_checkpoint_probe_results(phones,
    store_root=locations.echoframe_model_stores, collar=2000,
    results_dir=locations.probe_results):
    '''Return incomplete phone results grouped by checkpoint and layer.

    phones:       phone inventory defining the target labels
    store_root:   directory containing checkpoint Echoframe stores
    collar:       model context in milliseconds
    results_dir:  root directory containing probe results

    Completeness follows PhoneResult.complete. A result is complete when all
    configured fold-prediction files exist. Classifier artifacts below
    locations.phone_probes are intentionally not inspected.

    Each missing-result record identifies the target phoneme, absent fold
    numbers, and whether its descriptive run manifest is also absent. A
    missing manifest is informational and does not make an otherwise complete
    fold set incomplete.
    '''
    targets = sorted(phones.label_to_phraser_phone)
    missing = {}
    checkpoint_stores = discover_wav2vec2_checkpoint_stores(store_root)
    for model_name, _ in checkpoint_stores:
        layers = checkpoint_probe_layers(model_name)
        checkpoint_missing = probe_result.find_missing_checkpoint_results(
            targets, model_name, layers, collar=collar, root=results_dir)
        for layer, missing_targets in checkpoint_missing.items():
            missing[(model_name, layer)] = missing_targets
    return missing


def _phone_report(target_phoneme, model_name, layer, collar, results_dir):
    '''Reconstruct one label's accuracy metrics from persisted results.'''
    phone_result = probe_result.PhoneResult.model_feature(target_phoneme,
        model_name, layer, collar, root=results_dir)
    run = phone_result.run or {}
    return {'mean_accuracy': phone_result.mean_accuracy,
        'std_accuracy': phone_result.std_accuracy,
        'n_samples': run.get('actual_n_samples'),
        'n_missing': run.get('actual_n_missing')}


def _run_directory(root, model_name, target_phoneme, layer, collar):
    layer_directory = probe_result.layer_directory_name(layer)
    return Path(root) / model_name / target_phoneme / layer_directory / (
        f'collar{collar}ms')


def _checkpoint_number(model_name):
    random_name = locations.wav2vec2_random_checkpoint_name
    if model_name == random_name: return 0
    pattern = locations.wav2vec2_nl1_checkpoint_pattern
    match = re.fullmatch(pattern, model_name)
    if match is None: return None
    checkpoint = match.group(1)
    return int(checkpoint)


def _compact_probe_results(results, expected_n_samples):
    '''Retain probe metrics while allowing large per-label payloads to be
    freed.'''
    compact = {}
    for target_phoneme, result in results.items():
        n_samples = result.get('n_samples')
        if n_samples is None: n_samples = expected_n_samples
        n_missing = result.get('n_missing')
        if n_missing is None: n_missing = 0
        compact[target_phoneme] = {'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'], 'n_samples': n_samples,
            'n_missing': n_missing}
    return compact


def _expected_probe_sample_count(phones):
    grouped = phones.label_to_phraser_phone
    n_labels = len(grouped)
    grouped_values = iter(grouped.values())
    first_group = next(grouped_values)
    items_per_label = len(first_group)
    other_count = items_per_label // (n_labels - 1) * (n_labels - 1)
    return items_per_label + other_count


def _empty_checkpoint_sweep_report(store_root):
    return {'store_root': str(store_root), 'runs': [],
        'status_counts': {'completed': 0, 'skipped': 0, 'failed': 0},
        'errors': []}


def _record_discovery_failure(report, store_root, error):
    error_summary = f'{type(error).__name__}: {error}'
    message = f'Could not discover checkpoint stores below {store_root}: '
    message += error_summary
    _warn_checkpoint_failure(message)
    report['errors'].append(
        {'stage': 'discovery', 'error': error_summary})


def _probe_checkpoint_store(report, phones, model_name, store_path, layers,
    n_total, expected_n_samples, probe_options, verbose):
    if verbose:
        print(f'[{model_name}] opening checkpoint store {store_path}',
            flush=True)
    try:
        store_path_string = str(store_path)
        store = echoframe.Store(store_path_string)
    except Exception as error:
        _record_store_failure(report, model_name, layers, error, n_total)
        return

    try:
        for layer in layers:
            run = _probe_checkpoint_layer(phones, store, model_name, layer,
                n_total, expected_n_samples, probe_options, verbose)
            report['runs'].append(run)
    finally:
        _close_checkpoint_store(report, store, model_name)


def _record_store_failure(report, model_name, layers, error, n_total):
    error_summary = f'{type(error).__name__}: {error}'
    message = f'Could not open checkpoint store for {model_name}: '
    message += error_summary
    _warn_checkpoint_failure(message)
    details = dict(model_name=model_name, stage='store',
        error=error_summary)
    report['errors'].append(details)
    for layer in layers:
        failed_run = _failed_run(model_name, layer, 'store', error,
            n_total=n_total)
        report['runs'].append(failed_run)


def _probe_checkpoint_layer(phones, store, model_name, layer, n_total,
    expected_n_samples, probe_options, verbose):
    run = {'model_name': model_name, 'layer': layer, 'n_total': n_total,
        'n_available': None, 'n_missing': None}
    if verbose:
        print(f'[{model_name} layer {layer}] loading features and training '
            'probes for all phone labels', flush=True)
    try:
        results = train_binary_embedding_probes(phones,
            target_phonemes=None, store=store, model_name=model_name,
            layer=layer, **probe_options)
        labels = _compact_probe_results(results, expected_n_samples)
        accuracies = []
        for summary in labels.values():
            accuracies.append(summary['mean_accuracy'])
        mean_label_accuracy = float(np.mean(accuracies))
        run.update({'status': 'completed', 'n_available': n_total,
            'n_missing': 0, 'n_labels': len(labels),
            'mean_label_accuracy': mean_label_accuracy, 'labels': labels})
        del results
    except Exception as error:
        error_summary = f'{type(error).__name__}: {error}'
        message = f'Probe training failed for {model_name} layer {layer}: '
        message += error_summary
        _warn_checkpoint_failure(message)
        run['status'] = 'failed'
        run['failure_stage'] = 'training'
        run['error'] = error_summary
    return run


def _failed_run(model_name, layer, stage, error, n_total=None):
    return {'model_name': model_name, 'layer': layer, 'status': 'failed',
        'n_total': n_total, 'n_available': None, 'n_missing': None,
        'failure_stage': stage,
        'error': f'{type(error).__name__}: {error}'}


def _close_checkpoint_store(report, store, model_name):
    try:
        store.close()
    except Exception as error:
        error_summary = f'{type(error).__name__}: {error}'
        message = f'Could not close checkpoint store for {model_name}: '
        message += error_summary
        _warn_checkpoint_failure(message)
        details = dict(model_name=model_name, stage='close',
            error=error_summary)
        report['errors'].append(details)


def _warn_checkpoint_failure(message):
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _print_checkpoint_sweep_report(report):
    counts = report['status_counts']
    completed = counts['completed']
    skipped = counts['skipped']
    failed = counts['failed']
    print(f'Checkpoint probe sweep complete: {completed} completed, '
        f'{skipped} skipped, {failed} failed', flush=True)
    for run in report['runs']:
        if run['status'] == 'completed':
            n_labels = run['n_labels']
            mean_accuracy = run['mean_label_accuracy']
            detail = f'{n_labels} labels, mean label accuracy '
            detail += f'{mean_accuracy:.4f}'
        else:
            detail = run.get('reason') or run.get('error') or ''
        model_name = run['model_name']
        layer = run['layer']
        status = run['status']
        print(f'  {model_name} layer {layer}: {status} {detail}', flush=True)
