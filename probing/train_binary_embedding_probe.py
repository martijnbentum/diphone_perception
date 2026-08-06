import re
import warnings
from collections import Counter
from pathlib import Path

import echoframe
import numpy as np

import locations
from probing import probe_run, probe_training, probe_utils
from probing import result as probe_result
from probing.extract_embeddings import default_model_name

_default_inventory_batch_size = 1_000


def train_binary_embedding_probe(phones, target_phoneme, store=None,
    store_root=locations.echoframe_store, model_name=default_model_name,
    layer=9, collar=2000, n_embeds=None, n_splits=5, random_state=42,
    standardize=False, save_probes=True,
    probe_save_dir=locations.phone_probes, save_predictions=True,
    results_dir=locations.probe_results, overwrite=False, verbose=True):
    '''Train a binary target-vs-other probe on middle-frame embeddings.

    Standardization is fitted independently inside each cross-validation
    training fold. Raw features are retained by default.

    phones:          phone inventory paired with Phraser phones
    target_phoneme:  label used as the positive class
    store:           optional open Echoframe store
    model_name:      model identifier stored with each embedding
    layer:           hidden-state layer to probe
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    probe_training.validate_training_options(n_splits, standardize)
    probe_utils.validate_unique_phraser_keys(phones)
    selected = probe_utils.select_phones(phones, target_phoneme, n_embeds,
        seed=random_state)

    if store is None:
        store_root = str(store_root)
        store = echoframe.Store(store_root)
    echoframe_keys = _embedding_echoframe_keys(store, selected, model_name,
        layer, collar)
    feature_parameters = {'model_name': model_name, 'layer': layer,
        'collar': collar, 'frame': 'middle'}
    manifest = probe_run.build_probe_run_manifest(store, selected,
        echoframe_keys, 'embedding', feature_parameters, target_phoneme,
        n_embeds, n_splits, random_state, standardize)
    run_id = probe_run.hash_run_manifest(manifest)
    probe_run_directory = _run_directory(probe_save_dir, model_name,
        target_phoneme, layer, collar, run_id)
    phone_result = probe_result.PhoneResult.embedding(target_phoneme,
        model_name, layer, collar, n_samples=n_embeds, n_splits=n_splits,
        random_state=random_state, standardize=standardize,
        root=results_dir)
    existing_fold_count = len(phone_result.folds) if save_predictions else 0
    complete_before = save_predictions and phone_result.complete

    def load_vectors():
        return _load_middle_frame_vectors(store, selected, model_name, layer,
            collar)

    outcome = probe_run.run(load_vectors=load_vectors,
        manifest=manifest, probe_run_directory=probe_run_directory,
        phone_result=phone_result,
        display_name=f'{target_phoneme} layer {layer}', n_splits=n_splits,
        random_state=random_state, standardize=standardize,
        save_probes=save_probes, save_predictions=save_predictions,
        overwrite=overwrite, verbose=verbose)

    cache_status = probe_run.classify_cache_status(save_predictions,
        complete_before, overwrite, existing_fold_count)
    result_fields = {'representation': 'embedding',
        'target_phoneme': target_phoneme, 'model_name': model_name,
        'layer': layer, 'collar': collar, 'run_id': run_id,
        'cache_status': cache_status, 'standardize': standardize}
    if outcome is None:
        result_fields.update({'accuracies': phone_result.accuracies,
            'mean_accuracy': phone_result.mean_accuracy,
            'std_accuracy': phone_result.std_accuracy,
            'n_samples': None, 'n_missing': None, 'skipped': True})
        return result_fields
    result_fields.update({'accuracies': outcome.accuracies,
        'mean_accuracy': outcome.mean_accuracy,
        'std_accuracy': outcome.std_accuracy,
        'n_samples': outcome.n_samples, 'n_missing': outcome.n_missing,
        'skipped': False})
    return result_fields


def train_binary_embedding_probes(phones, target_phonemes=None, store=None,
    store_root=locations.echoframe_store, model_name=default_model_name,
    layer=9, collar=2000, n_embeds=None, n_splits=5, random_state=42,
    standardize=False, save_probes=True,
    probe_save_dir=locations.phone_probes, save_predictions=True,
    results_dir=locations.probe_results, overwrite=False, verbose=True):
    '''Train one binary embedding probe run for each target phoneme.

    The Phraser label inventory must contain the same number of items for
    every label. All labels are used when target_phonemes is omitted.

    phones:            phone inventory paired with Phraser phones
    target_phonemes:   optional ordered subset of labels to probe
    store:             optional open Echoframe store shared across labels
    model_name:        model identifier stored with each embedding
    n_embeds:          maximum positive examples selected per label
    '''
    probe_training.validate_training_options(n_splits, standardize)
    targets = probe_utils.prepare_balanced_probe_targets(phones,
        target_phonemes, n_samples=n_embeds)
    probe_utils.validate_unique_phraser_keys(phones)

    owns_store = store is None
    if store is None:
        store_root = str(store_root)
        store = echoframe.Store(store_root)

    def train_one(target_phoneme):
        return train_binary_embedding_probe(phones, target_phoneme,
            store=store, model_name=model_name, layer=layer, collar=collar,
            n_embeds=n_embeds, n_splits=n_splits, random_state=random_state,
            standardize=standardize, save_probes=save_probes,
            probe_save_dir=probe_save_dir, save_predictions=save_predictions,
            results_dir=results_dir, overwrite=overwrite, verbose=verbose)

    try:
        return probe_utils.run_probe_sweep(targets, train_one, 'embedding',
            verbose=verbose)
    finally:
        if owns_store: store.close()


def train_binary_embedding_probe_checkpoint_sweep(phones,
    store_root=locations.echoframe_model_stores, collar=2000, n_embeds=None,
    n_splits=5, random_state=42, standardize=False, save_probes=True,
    probe_save_dir=locations.phone_probes, save_predictions=True,
    results_dir=locations.probe_results, overwrite=False,
    metadata_batch_size=_default_inventory_batch_size, verbose=True):
    '''Train layer-specific all-label probes across wav2vec2 checkpoints.

    Every checkpoint and layer is checked for a complete embedding inventory
    before training. Failures are reported without stopping later runs. The
    returned report contains compact metrics rather than fitted classifiers.

    phones:               phone inventory paired with Phraser phones
    store_root:           directory containing checkpoint Echoframe stores
    n_embeds:             maximum positive examples selected per label
    metadata_batch_size:  number of metadata keys checked per request
    verbose:              whether to print progress and the final report
    '''
    _validate_inventory_batch_size(metadata_batch_size)
    probe_training.validate_training_options(n_splits, standardize)
    probe_utils.prepare_balanced_probe_targets(phones, target_phonemes=None,
        n_samples=n_embeds)
    expected_n_samples = _expected_probe_sample_count(phones, n_embeds)
    n_total = len(phones.phraser_phones)
    store_root = Path(store_root)
    report = _empty_checkpoint_sweep_report(store_root)
    probe_options = {'collar': collar, 'n_embeds': n_embeds,
        'n_splits': n_splits, 'random_state': random_state,
        'standardize': standardize, 'save_probes': save_probes,
        'probe_save_dir': probe_save_dir,
        'save_predictions': save_predictions, 'results_dir': results_dir,
        'overwrite': overwrite, 'verbose': verbose}

    try:
        checkpoint_stores = discover_wav2vec2_checkpoint_stores(store_root)
    except Exception as error:
        _record_discovery_failure(report, store_root, error)
        if verbose: _print_checkpoint_sweep_report(report)
        return report

    for model_name, store_path in checkpoint_stores:
        _probe_checkpoint_store(report, phones, model_name, store_path,
            n_total, expected_n_samples, probe_options, metadata_batch_size,
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
    '''Return layers to probe for one supported wav2vec2 checkpoint.'''
    if _checkpoint_number(model_name) is None:
        message = f'unsupported checkpoint model name: {model_name!r}'
        raise ValueError(message)
    if model_name in locations.wav2vec2_all_layer_checkpoint_names:
        return locations.wav2vec2_all_probe_layers
    return (9,)


def check_embedding_inventory(phones, store, model_name, layer, collar=2000,
    batch_size=_default_inventory_batch_size, verbose=True):
    '''Check metadata availability for every phone without loading arrays.

    Metadata is requested with keep_missing enabled in bounded batches.

    phones:      phone inventory paired with Phraser phones
    store:       open Echoframe checkpoint store
    model_name:  model identifier stored with each embedding
    layer:       hidden-state layer to inspect
    batch_size:  number of metadata keys checked per request
    '''
    _validate_inventory_batch_size(batch_size)
    phraser_phones = phones.phraser_phones
    n_total = len(phraser_phones)
    n_available = 0
    for start in range(0, n_total, batch_size):
        batch = phraser_phones[start:start + batch_size]
        keys = []
        for phraser_phone in batch:
            key = store.make_echoframe_key('hidden_state',
                model_name=model_name, phraser_key=phraser_phone.key,
                layer=layer, collar=collar)
            keys.append(key)
        metadatas = store.load_many_metadata(keys, keep_missing=True)
        if len(metadatas) != len(keys):
            message = 'load_many_metadata returned an unexpected number of '
            message += f'records: expected {len(keys)}, '
            message += f'received {len(metadatas)}'
            raise ValueError(message)
        available = sum(metadata is not None for metadata in metadatas)
        n_available += available
        if verbose:
            batch_length = len(batch)
            checked = min(start + batch_length, n_total)
            print(f'[{model_name} layer {layer}] checked '
                f'{checked:,}/{n_total:,} embedding metadata records',
                flush=True)
    n_missing = n_total - n_available
    return {'n_total': n_total, 'n_available': n_available,
        'n_missing': n_missing, 'complete': n_missing == 0}


def _embedding_echoframe_keys(store, selected, model_name, layer, collar):
    keys = []
    for _, phraser_phone, _ in selected:
        key = store.make_echoframe_key('hidden_state', model_name=model_name,
            phraser_key=phraser_phone.key, layer=layer, collar=collar)
        keys.append(key)
    return keys


def _load_middle_frame_vectors(store, selected, model_name, layer, collar):
    '''Load embeddings and reduce each stored phone to its middle frame.'''
    phraser_keys = [phone.key for _, phone, _ in selected]
    embeddings = store.phraser_keys_to_embeddings(phraser_keys, model_name,
        layer, collar=collar)
    by_key = {}
    for embedding in embeddings.embeddings:
        by_key[embedding.phraser_key] = embedding

    X, y, true_labels, missing = [], [], [], []
    for phone, phraser_phone, binary_label in selected:
        embedding = by_key.get(phraser_phone.key)
        if embedding is None:
            missing.append(phone)
            continue
        middle_frame = embedding.middle_frame_segment(phraser_phone)
        X.append(middle_frame)
        y.append(binary_label)
        true_labels.append(phone.phoneme_ipa)
    X = np.array(X)
    y = np.array(y)
    true_labels = np.array(true_labels)
    return X, y, true_labels, missing


def _run_directory(root, model_name, target_phoneme, layer, collar, run_id):
    return Path(root) / model_name / target_phoneme / f'layer{layer:02d}' / (
        f'collar{collar}ms') / run_id


def _checkpoint_number(model_name):
    random_name = locations.wav2vec2_random_checkpoint_name
    if model_name == random_name: return 0
    pattern = locations.wav2vec2_nl1_checkpoint_pattern
    match = re.fullmatch(pattern, model_name)
    if match is None: return None
    checkpoint = match.group(1)
    return int(checkpoint)


def _validate_inventory_batch_size(batch_size):
    message = 'batch_size must be a positive integer'
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(message)
    if batch_size <= 0: raise ValueError(message)


def _compact_probe_results(results, expected_n_samples):
    '''Retain probe metrics while allowing fitted classifiers to be freed.'''
    compact = {}
    for target_phoneme, result in results.items():
        n_samples = result.get('n_samples')
        if n_samples is None: n_samples = expected_n_samples
        n_missing = result.get('n_missing')
        if n_missing is None: n_missing = 0
        compact[target_phoneme] = {'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'], 'n_samples': n_samples,
            'n_missing': n_missing, 'skipped': result['skipped'],
            'cache_status': result['cache_status']}
    return compact


def _expected_probe_sample_count(phones, n_embeds):
    grouped = phones.label_to_phraser_phone
    n_labels = len(grouped)
    grouped_values = iter(grouped.values())
    first_group = next(grouped_values)
    items_per_label = len(first_group)
    target_count = items_per_label if n_embeds is None else n_embeds
    other_count = target_count // (n_labels - 1) * (n_labels - 1)
    return target_count + other_count


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


def _probe_checkpoint_store(report, phones, model_name, store_path, n_total,
    expected_n_samples, probe_options, metadata_batch_size, verbose):
    layers = checkpoint_probe_layers(model_name)
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
                n_total, expected_n_samples, probe_options,
                metadata_batch_size, verbose)
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
    expected_n_samples, probe_options, metadata_batch_size, verbose):
    if verbose:
        print(f'[{model_name} layer {layer}] checking complete embedding '
            f'inventory for {n_total:,} phones', flush=True)
    try:
        inventory = check_embedding_inventory(phones, store, model_name, layer,
            collar=probe_options['collar'],
            batch_size=metadata_batch_size, verbose=verbose)
    except Exception as error:
        error_summary = f'{type(error).__name__}: {error}'
        message = f'Embedding preflight failed for {model_name} layer '
        message += f'{layer}: {error_summary}'
        _warn_checkpoint_failure(message)
        return _failed_run(model_name, layer, 'preflight', error,
            n_total=n_total)

    run = {'model_name': model_name, 'layer': layer,
        'n_total': inventory['n_total'],
        'n_available': inventory['n_available'],
        'n_missing': inventory['n_missing']}
    if not inventory['complete']:
        run['status'] = 'skipped'
        run['reason'] = 'incomplete embedding inventory'
        missing = inventory['n_missing']
        total = inventory['n_total']
        message = f'Skipping {model_name} layer {layer}: '
        message += f'{missing:,} of {total:,} embeddings are missing'
        _warn_checkpoint_failure(message)
        return run

    if verbose:
        print(f'[{model_name} layer {layer}] inventory complete; '
            'training probes for all phone labels', flush=True)
    try:
        results = train_binary_embedding_probes(phones,
            target_phonemes=None, store=store, model_name=model_name,
            layer=layer, **probe_options)
        labels = _compact_probe_results(results, expected_n_samples)
        accuracies = []
        for summary in labels.values():
            accuracies.append(summary['mean_accuracy'])
        mean_label_accuracy = float(np.mean(accuracies))
        run.update({'status': 'completed', 'n_labels': len(labels),
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
