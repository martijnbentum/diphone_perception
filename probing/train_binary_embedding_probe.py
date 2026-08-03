import re
import warnings
from collections import Counter
from pathlib import Path

import echoframe
import numpy as np

from probing import probe_utils
from probing.extract_embeddings import (
    default_model_name,
    default_model_stores_root,
    default_store_root,
)

default_probe_save_dir = probe_utils.default_probe_save_dir
default_results_dir = probe_utils.default_results_dir

_random_checkpoint_name = 'wav2vec2_checkpoint-0'
_nl1_checkpoint_pattern = re.compile(
    r'^wav2vec2_nl1_checkpoint-(\d+)$')
_all_layer_checkpoint_names = {
    _random_checkpoint_name,
    'wav2vec2_nl1_checkpoint-200000',
}
_all_probe_layers = tuple(range(1, 13))
_default_inventory_batch_size = 1_000


def _embedding_echoframe_keys(
    store, selected, model_name, layer, collar,
):
    return [
        store.make_echoframe_key(
            'hidden_state', model_name=model_name,
            phraser_key=phraser_phone.key, layer=layer, collar=collar)
        for _, phraser_phone, _ in selected
    ]


def _checkpoint_number(model_name):
    if model_name == _random_checkpoint_name:
        return 0
    match = _nl1_checkpoint_pattern.fullmatch(model_name)
    if match is None:
        return None
    return int(match.group(1))


def discover_wav2vec2_checkpoint_stores(
    store_root=default_model_stores_root,
):
    '''Return supported ``(model_name, store_path)`` pairs in numeric order.

    Only the exact random checkpoint ``wav2vec2_checkpoint-0`` and directories
    named ``wav2vec2_nl1_checkpoint-<integer>`` are included. Unrelated files
    and directories are ignored.
    '''
    store_root = Path(store_root)
    checkpoints = []
    for store_path in store_root.iterdir():
        if not store_path.is_dir():
            continue
        model_name = store_path.name
        checkpoint_number = _checkpoint_number(model_name)
        if checkpoint_number is None:
            continue
        random_first = 0 if model_name == _random_checkpoint_name else 1
        checkpoints.append((
            checkpoint_number, random_first, model_name, store_path))
    checkpoints.sort(key=lambda item: item[:3])
    return [
        (model_name, store_path)
        for _, _, model_name, store_path in checkpoints
    ]


def checkpoint_probe_layers(model_name):
    '''Return layers to probe for one supported wav2vec2 checkpoint.'''
    if _checkpoint_number(model_name) is None:
        raise ValueError(f'unsupported checkpoint model name: {model_name!r}')
    if model_name in _all_layer_checkpoint_names:
        return _all_probe_layers
    return (9,)


def _validate_inventory_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('batch_size must be a positive integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')


def check_embedding_inventory(
    phones,
    store,
    model_name,
    layer,
    collar=2000,
    batch_size=_default_inventory_batch_size,
    verbose=True,
):
    '''Check metadata availability for every phone without loading arrays.

    Metadata is requested with ``keep_missing=True`` in bounded batches. The
    returned dictionary contains ``n_total``, ``n_available``, ``n_missing``,
    and ``complete``.
    '''
    _validate_inventory_batch_size(batch_size)
    phraser_phones = phones.phraser_phones
    n_total = len(phraser_phones)
    n_available = 0
    for start in range(0, n_total, batch_size):
        batch = phraser_phones[start:start + batch_size]
        keys = [
            store.make_echoframe_key(
                'hidden_state',
                model_name=model_name,
                phraser_key=phraser_phone.key,
                layer=layer,
                collar=collar,
            )
            for phraser_phone in batch
        ]
        metadatas = store.load_many_metadata(keys, keep_missing=True)
        if len(metadatas) != len(keys):
            raise ValueError(
                'load_many_metadata returned an unexpected number of '
                f'records: expected {len(keys)}, received {len(metadatas)}')
        n_available += sum(metadata is not None for metadata in metadatas)
        if verbose:
            checked = min(start + len(batch), n_total)
            print(
                f'[{model_name} layer {layer}] checked '
                f'{checked:,}/{n_total:,} embedding metadata records',
                flush=True,
            )
    n_missing = n_total - n_available
    return {
        'n_total': n_total,
        'n_available': n_available,
        'n_missing': n_missing,
        'complete': n_missing == 0,
    }


def _load_middle_frame_vectors(store, selected, model_name, layer, collar):
    '''Load embeddings and reduce each stored phone to its middle frame.'''
    phraser_keys = [
        phraser_phone.key for _, phraser_phone, _ in selected]
    embeddings = store.phraser_keys_to_embeddings(
        phraser_keys, model_name, layer, collar=collar)
    by_key = {
        embedding.phraser_key: embedding
        for embedding in embeddings.embeddings
    }

    X, y, true_labels, missing = [], [], [], []
    for phone, phraser_phone, binary_label in selected:
        embedding = by_key.get(phraser_phone.key)
        if embedding is None:
            missing.append(phone)
            continue
        X.append(embedding.middle_frame_segment(phraser_phone))
        y.append(binary_label)
        true_labels.append(phone.phoneme_ipa)
    return np.array(X), np.array(y), np.array(true_labels), missing


def _run_directory(
    root, model_name, target_phoneme, layer, collar, run_id,
):
    return (
        Path(root) / model_name / target_phoneme / f'layer{layer:02d}'
        / f'collar{collar}ms' / run_id
    )


def train_binary_embedding_probe(
    phones,
    target_phoneme,
    store=None,
    store_root=default_store_root,
    model_name=default_model_name,
    layer=9,
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
):
    '''Train a binary target-vs-other probe on middle-frame embeddings.

    `standardize=False` preserves the legacy raw-feature probe. When True,
    StandardScaler is fitted independently inside every cross-validation
    training fold through an sklearn Pipeline.
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    probe_utils.validate_probe_arguments(n_splits, standardize)
    probe_utils.validate_unique_phraser_keys(phones)
    selected = probe_utils.select_phones(
        phones, target_phoneme, n_embeds, seed=random_state)

    if store is None:
        store = echoframe.Store(str(store_root))
    echoframe_keys = _embedding_echoframe_keys(
        store, selected, model_name, layer, collar)
    feature_parameters = {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'frame': 'middle',
    }
    manifest = probe_utils.build_probe_run_manifest(
        store, selected, echoframe_keys, 'embedding', feature_parameters,
        target_phoneme, n_embeds, n_splits, random_state, standardize)
    run_id = probe_utils.hash_run_manifest(manifest)
    probe_run_directory = _run_directory(
        probe_save_dir, model_name, target_phoneme, layer, collar, run_id)
    predictions_run_directory = _run_directory(
        results_dir, model_name, target_phoneme, layer, collar, run_id)

    def load_vectors():
        return _load_middle_frame_vectors(
            store, selected, model_name, layer, collar)

    result_fields = {
        'representation': 'embedding',
        'target_phoneme': target_phoneme,
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
    }
    return probe_utils.run_binary_probe(
        load_vectors=load_vectors,
        manifest=manifest,
        probe_run_directory=probe_run_directory,
        predictions_run_directory=predictions_run_directory,
        result_fields=result_fields,
        display_name=f'{target_phoneme} layer {layer}',
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        save_probes=save_probes,
        save_predictions=save_predictions,
        overwrite=overwrite,
        verbose=verbose,
    )


def train_binary_embedding_probes(
    phones,
    target_phonemes=None,
    store=None,
    store_root=default_store_root,
    model_name=default_model_name,
    layer=9,
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
):
    '''Train one binary embedding probe run for each target phoneme.

    When target_phonemes is None, all labels in
    phones.label_to_phraser_phone are used. The Phraser label inventory
    must contain exactly the same number of items for every label.
    '''
    probe_utils.validate_probe_arguments(n_splits, standardize)
    targets = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes, n_samples=n_embeds)
    probe_utils.validate_unique_phraser_keys(phones)

    owns_store = store is None
    if store is None:
        store = echoframe.Store(str(store_root))

    def train_one(target_phoneme):
        return train_binary_embedding_probe(
            phones,
            target_phoneme,
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

    try:
        return probe_utils.run_probe_sweep(
            targets, train_one, 'embedding', verbose=verbose)
    finally:
        if owns_store:
            store.close()


def _compact_probe_results(results, expected_n_samples):
    '''Retain probe metrics while allowing fitted classifiers to be freed.'''
    compact = {}
    for target_phoneme, result in results.items():
        n_samples = result.get('n_samples')
        if n_samples is None:
            n_samples = expected_n_samples
        n_missing = result.get('n_missing')
        if n_missing is None:
            n_missing = 0
        compact[target_phoneme] = {
            'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'],
            'n_samples': n_samples,
            'n_missing': n_missing,
            'skipped': result['skipped'],
            'cache_status': result['cache_status'],
        }
    return compact


def _expected_probe_sample_count(phones, n_embeds):
    grouped = phones.label_to_phraser_phone
    n_labels = len(grouped)
    items_per_label = len(next(iter(grouped.values())))
    target_count = items_per_label if n_embeds is None else n_embeds
    other_count = target_count // (n_labels - 1) * (n_labels - 1)
    return target_count + other_count


def _failed_run(model_name, layer, stage, error, n_total=None):
    return {
        'model_name': model_name,
        'layer': layer,
        'status': 'failed',
        'n_total': n_total,
        'n_available': None,
        'n_missing': None,
        'failure_stage': stage,
        'error': f'{type(error).__name__}: {error}',
    }


def _warn_checkpoint_failure(message):
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _print_checkpoint_sweep_report(report):
    counts = report['status_counts']
    print(
        'Checkpoint probe sweep complete: '
        f'{counts["completed"]} completed, '
        f'{counts["skipped"]} skipped, '
        f'{counts["failed"]} failed',
        flush=True,
    )
    for run in report['runs']:
        if run['status'] == 'completed':
            detail = (
                f'{run["n_labels"]} labels, mean label accuracy '
                f'{run["mean_label_accuracy"]:.4f}'
            )
        else:
            detail = run.get('reason') or run.get('error') or ''
        print(
            f'  {run["model_name"]} layer {run["layer"]}: '
            f'{run["status"]} {detail}',
            flush=True,
        )


def train_binary_embedding_probe_checkpoint_sweep(
    phones,
    store_root=default_model_stores_root,
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
    metadata_batch_size=_default_inventory_batch_size,
    verbose=True,
):
    '''Train layer-specific all-label probes across wav2vec2 checkpoints.

    Supported stores are discovered immediately below ``store_root``. Every
    planned checkpoint/layer is first checked for embedding metadata for every
    item in ``phones.phraser_phones``. Incomplete inventories are skipped;
    store, preflight, and training failures are warned about and recorded so
    later runs can continue.

    The returned report contains compact per-label metrics and cache status,
    never the fitted classifiers returned by the lower-level trainer. Probe
    and prediction persistence still uses its existing model/layer-specific
    directories and cache behavior.
    '''
    _validate_inventory_batch_size(metadata_batch_size)
    probe_utils.validate_probe_arguments(n_splits, standardize)
    probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes=None, n_samples=n_embeds)
    expected_n_samples = _expected_probe_sample_count(phones, n_embeds)
    n_total = len(phones.phraser_phones)
    store_root = Path(store_root)
    report = {
        'store_root': str(store_root),
        'runs': [],
        'status_counts': {
            'completed': 0,
            'skipped': 0,
            'failed': 0,
        },
        'errors': [],
    }

    try:
        checkpoint_stores = discover_wav2vec2_checkpoint_stores(store_root)
    except Exception as error:
        message = (
            f'Could not discover checkpoint stores below {store_root}: '
            f'{type(error).__name__}: {error}')
        _warn_checkpoint_failure(message)
        report['errors'].append({
            'stage': 'discovery',
            'error': f'{type(error).__name__}: {error}',
        })
        if verbose:
            _print_checkpoint_sweep_report(report)
        return report

    for model_name, store_path in checkpoint_stores:
        layers = checkpoint_probe_layers(model_name)
        if verbose:
            print(
                f'[{model_name}] opening checkpoint store {store_path}',
                flush=True,
            )
        try:
            store = echoframe.Store(str(store_path))
        except Exception as error:
            message = (
                f'Could not open checkpoint store for {model_name}: '
                f'{type(error).__name__}: {error}')
            _warn_checkpoint_failure(message)
            report['errors'].append({
                'model_name': model_name,
                'stage': 'store',
                'error': f'{type(error).__name__}: {error}',
            })
            for layer in layers:
                report['runs'].append(_failed_run(
                    model_name, layer, 'store', error, n_total=n_total))
            continue

        try:
            for layer in layers:
                if verbose:
                    print(
                        f'[{model_name} layer {layer}] checking complete '
                        f'embedding inventory for {n_total:,} phones',
                        flush=True,
                    )
                try:
                    inventory = check_embedding_inventory(
                        phones,
                        store,
                        model_name,
                        layer,
                        collar=collar,
                        batch_size=metadata_batch_size,
                        verbose=verbose,
                    )
                except Exception as error:
                    message = (
                        f'Embedding preflight failed for {model_name} layer '
                        f'{layer}: {type(error).__name__}: {error}')
                    _warn_checkpoint_failure(message)
                    report['runs'].append(_failed_run(
                        model_name,
                        layer,
                        'preflight',
                        error,
                        n_total=n_total,
                    ))
                    continue

                run = {
                    'model_name': model_name,
                    'layer': layer,
                    'n_total': inventory['n_total'],
                    'n_available': inventory['n_available'],
                    'n_missing': inventory['n_missing'],
                }
                if not inventory['complete']:
                    run.update({
                        'status': 'skipped',
                        'reason': 'incomplete embedding inventory',
                    })
                    _warn_checkpoint_failure(
                        f'Skipping {model_name} layer {layer}: '
                        f'{inventory["n_missing"]:,} of '
                        f'{inventory["n_total"]:,} embeddings are missing')
                    report['runs'].append(run)
                    continue

                if verbose:
                    print(
                        f'[{model_name} layer {layer}] inventory complete; '
                        'training probes for all phone labels',
                        flush=True,
                    )
                try:
                    results = train_binary_embedding_probes(
                        phones,
                        target_phonemes=None,
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
                    labels = _compact_probe_results(
                        results, expected_n_samples)
                    run.update({
                        'status': 'completed',
                        'n_labels': len(labels),
                        'mean_label_accuracy': float(np.mean([
                            summary['mean_accuracy']
                            for summary in labels.values()
                        ])),
                        'labels': labels,
                    })
                    del results
                except Exception as error:
                    message = (
                        f'Probe training failed for {model_name} layer '
                        f'{layer}: {type(error).__name__}: {error}')
                    _warn_checkpoint_failure(message)
                    run.update({
                        'status': 'failed',
                        'failure_stage': 'training',
                        'error': f'{type(error).__name__}: {error}',
                    })
                report['runs'].append(run)
        finally:
            try:
                store.close()
            except Exception as error:
                message = (
                    f'Could not close checkpoint store for {model_name}: '
                    f'{type(error).__name__}: {error}')
                _warn_checkpoint_failure(message)
                report['errors'].append({
                    'model_name': model_name,
                    'stage': 'close',
                    'error': f'{type(error).__name__}: {error}',
                })

    report['status_counts'].update(Counter(
        run['status'] for run in report['runs']))
    if verbose:
        _print_checkpoint_sweep_report(report)
    return report
