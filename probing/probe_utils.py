import hashlib
import json
import random
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import echoframe
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probing.extract_embeddings import (
    default_model_name,
    default_store_root as default_embedding_store_root,
)
from probing.extract_mfcc import default_store_root as default_mfcc_store_root
from probing.metadata import _data_dir

default_probe_save_dir = _data_dir / 'phone_probes'
default_results_dir = _data_dir / 'probe_results'
_cache_schema_version = 2
_trainer_version = 2
_probe_parameters = {
    'solver': 'liblinear',
    'max_iter': 1000,
}


def validate_target_phoneme(target_phoneme):
    if not isinstance(target_phoneme, str) or not target_phoneme:
        raise TypeError('target_phoneme must be a non-empty string')


def validate_probe_arguments(n_splits, standardize):
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError('n_splits must be an integer')
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if not isinstance(standardize, bool):
        raise TypeError('standardize must be a boolean')


def _hashable_key(value):
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return value


def validate_unique_phraser_keys(phones, example_count=5):
    '''Raise when two metadata phones refer to the same Phraser key.'''
    first_indices = {}
    duplicate_count = 0
    examples = []
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for index, (phone, phraser_phone) in enumerate(pairs):
        key = _hashable_key(phraser_phone.key)
        first_index = first_indices.get(key)
        if first_index is None:
            first_indices[key] = index
            continue
        duplicate_count += 1
        if len(examples) < example_count:
            examples.append({
                'first_index': first_index,
                'duplicate_index': index,
                'key': _stable_key(key),
                'phoneme': str(phone.phoneme_ipa),
            })
    if duplicate_count:
        raise ValueError(
            f'{duplicate_count} duplicate Phraser key occurrences found; '
            f'probe training requires unique keys. Examples: {examples}'
        )


def prepare_balanced_probe_targets(
    phones, target_phonemes=None, n_samples=None,
):
    '''Validate the Phraser label inventory and return requested targets.'''
    grouped = phones.label_to_phraser_phone
    if not grouped:
        raise ValueError('phones.label_to_phraser_phone is empty')

    label_counts = {
        label: len(items) for label, items in grouped.items()
    }
    invalid_labels = [
        label for label in label_counts
        if not isinstance(label, str) or not label
    ]
    if invalid_labels:
        raise TypeError(
            'label_to_phraser_phone keys must be non-empty strings; '
            f'invalid labels: {invalid_labels}')

    unique_counts = set(label_counts.values())
    if len(unique_counts) != 1:
        counts = ', '.join(
            f'{label!r}: {count}'
            for label, count in sorted(label_counts.items())
        )
        raise ValueError(
            'label_to_phraser_phone is not balanced; every label must have '
            f'the same number of items. Counts: {counts}')
    items_per_label = next(iter(unique_counts))
    if items_per_label == 0:
        raise ValueError(
            'label_to_phraser_phone contains no items per label')

    if target_phonemes is None:
        targets = sorted(label_counts)
    else:
        if isinstance(target_phonemes, str):
            raise TypeError(
                'target_phonemes must be an iterable of phoneme strings')
        targets = list(target_phonemes)
        for target in targets:
            validate_target_phoneme(target)
        if len(set(targets)) != len(targets):
            raise ValueError('target_phonemes contains duplicate labels')
        missing = sorted(set(targets) - set(label_counts))
        if missing:
            raise ValueError(
                f'target phonemes not found in label_to_phraser_phone: '
                f'{missing}')
        if not targets:
            raise ValueError('target_phonemes must not be empty')

    requested = items_per_label if n_samples is None else n_samples
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError('n_samples must be a positive integer or None')
    if requested <= 0:
        raise ValueError('n_samples must be a positive integer or None')
    if requested > items_per_label:
        raise ValueError(
            f'n_samples={requested} exceeds the balanced inventory of '
            f'{items_per_label} items per label')
    other_label_count = len(label_counts) - 1
    if other_label_count == 0:
        raise ValueError('at least two phoneme labels are required')
    if requested // other_label_count == 0:
        raise ValueError(
            f'n_samples={requested} is too small to split across '
            f'{other_label_count} other phoneme labels')
    return targets


def _format_duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def run_probe_sweep(target_phonemes, train_one, representation, verbose=True):
    '''Train one binary probe run per target and report elapsed time and ETA.'''
    targets = list(target_phonemes)
    started_at = time.monotonic()
    results = {}
    total = len(targets)
    for index, target in enumerate(targets, start=1):
        if verbose:
            print(
                f'[{representation} probes] {index}/{total} starting '
                f'{target!r}',
                flush=True,
            )
        results[target] = train_one(target)
        elapsed = time.monotonic() - started_at
        eta = elapsed / index * (total - index)
        if verbose:
            print(
                f'[{representation} probes] {index}/{total} completed '
                f'{target!r}; elapsed {_format_duration(elapsed)}; '
                f'ETA {_format_duration(eta)}',
                flush=True,
            )
    return results


def select_phones(phones, target_phoneme, n_samples=None, seed=42):
    '''Select a balanced target-vs-other sample deterministically.'''
    validate_target_phoneme(target_phoneme)
    by_label = defaultdict(list)
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for phone, phraser_phone in pairs:
        by_label[phone.phoneme_ipa].append((phone, phraser_phone))

    if target_phoneme not in by_label:
        raise ValueError(
            f'target_phoneme {target_phoneme!r} not found among phones')

    if n_samples is None:
        n_samples = len(by_label[target_phoneme])
    elif isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError('n_samples must be a positive integer or None')
    elif n_samples <= 0:
        raise ValueError('n_samples must be a positive integer or None')

    other_labels = [label for label in by_label if label != target_phoneme]
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, _ = divmod(n_samples, len(other_labels))
    if n_per_other == 0:
        raise ValueError(
            f'n_samples={n_samples} is too small to split across '
            f'{len(other_labels)} other phoneme classes')

    rng = random.Random(seed)

    def _take(label, quota):
        pool = list(by_label[label])
        if len(pool) < quota:
            raise ValueError(
                f'phoneme {label!r} has only {len(pool)} phones available, '
                f'need {quota}')
        rng.shuffle(pool)
        return pool[:quota]

    selected = [
        (phone, phraser_phone, 'target')
        for phone, phraser_phone in _take(target_phoneme, n_samples)
    ]
    for label in other_labels:
        selected += [
            (phone, phraser_phone, 'other')
            for phone, phraser_phone in _take(label, n_per_other)
        ]

    rng.shuffle(selected)
    return selected


def make_probe(standardize=False):
    if not isinstance(standardize, bool):
        raise TypeError('standardize must be a boolean')
    classifier = LogisticRegression(**_probe_parameters)
    if not standardize:
        return classifier
    return make_pipeline(StandardScaler(), classifier)


def classifier_manifest(standardize=False):
    if not isinstance(standardize, bool):
        raise TypeError('standardize must be a boolean')
    manifest = {
        'class': 'sklearn.linear_model.LogisticRegression',
        **_probe_parameters,
        'standardize': standardize,
    }
    if standardize:
        manifest['preprocessor'] = 'sklearn.preprocessing.StandardScaler'
    else:
        manifest['preprocessor'] = None
    return manifest


def _stable_key(value):
    '''Return a deterministic text representation for a stored key.'''
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f'bytes:{bytes(value).hex()}'
    return f'{type(value).__name__}:{value}'


def _hash_json(value):
    data = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def build_probe_run_manifest(
    store,
    selected,
    echoframe_keys,
    representation,
    feature_parameters,
    target_phoneme,
    n_samples,
    n_splits,
    random_state,
    standardize,
):
    '''Build the canonical identity for an embedding or MFCC probe run.'''
    sample_records = [
        {
            'phraser_key': _stable_key(phraser_phone.key),
            'phoneme': str(phone.phoneme_ipa),
            'binary_label': binary_label,
        }
        for phone, phraser_phone, binary_label in selected
    ]
    metadatas = store.load_many_metadata(
        echoframe_keys, keep_missing=True)
    feature_records = []
    for key, metadata in zip(echoframe_keys, metadatas, strict=True):
        record = {'key': _stable_key(key), 'present': metadata is not None}
        if metadata is not None:
            shape = getattr(metadata, 'shape', None)
            record.update({
                'created_at': str(getattr(metadata, 'created_at', None)),
                'dataset_path': getattr(metadata, 'dataset_path', None),
                'shape': list(shape) if shape is not None else None,
                'shard_id': getattr(metadata, 'shard_id', None),
            })
        feature_records.append(record)
    return {
        'cache_schema_version': _cache_schema_version,
        'trainer_version': _trainer_version,
        'representation': representation,
        'feature_parameters': dict(feature_parameters),
        'target_phoneme': target_phoneme,
        'n_samples': n_samples,
        'n_splits': n_splits,
        'random_state': random_state,
        'classifier': classifier_manifest(standardize),
        'selected_sample_count': len(sample_records),
        'selected_samples_hash': _hash_json(sample_records),
        'feature_set_hash': _hash_json(feature_records),
    }


def hash_run_manifest(manifest):
    return _hash_json(manifest)[:16]


def fold_paths(probe_run_directory, predictions_run_directory, fold_idx):
    number = fold_idx + 1
    return (
        Path(probe_run_directory) / f'fold{number:02d}.joblib',
        Path(predictions_run_directory) / f'fold{number:02d}_predictions.tsv',
        Path(probe_run_directory) / f'fold{number:02d}_complete.json',
    )


@contextmanager
def _atomic_target(path):
    '''Yield a sibling temporary path and atomically replace the target.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f'.{path.name}.', suffix=path.suffix,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        yield temporary_path
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path, value):
    text = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    with _atomic_target(path) as temporary_path:
        temporary_path.write_text(text, encoding='utf-8')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _manifest_matches(run_directory, manifest):
    path = Path(run_directory) / 'run.json'
    try:
        return _read_json(path) == manifest
    except (OSError, ValueError, TypeError):
        return False


def _write_run_manifest(run_directory, manifest):
    _write_json(Path(run_directory) / 'run.json', manifest)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_probe(probe, path):
    with _atomic_target(path) as temporary_path:
        joblib.dump(probe, temporary_path)


def _save_predictions(predictions, path):
    lines = ['true_phoneme\tbinary_true\tbinary_pred\tcorrect\n']
    for true_full, true_bin, pred_bin in predictions:
        correct = int(true_bin == pred_bin)
        lines.append(f'{true_full}\t{true_bin}\t{pred_bin}\t{correct}\n')
    with _atomic_target(path) as temporary_path:
        temporary_path.write_text(''.join(lines), encoding='utf-8')


def _save_cached_fold(
    probe, predictions, accuracy, paths, run_id, fold_idx,
):
    probe_path, predictions_path, completion_path = paths
    completion_path = Path(completion_path)
    completion_path.unlink(missing_ok=True)
    _save_probe(probe, probe_path)
    _save_predictions(predictions, predictions_path)
    marker = {
        'run_id': run_id,
        'fold': fold_idx + 1,
        'accuracy': float(accuracy),
        'n_predictions': len(predictions),
        'probe_sha256': _sha256_file(probe_path),
        'predictions_sha256': _sha256_file(predictions_path),
    }
    _write_json(completion_path, marker)


def _load_cached_fold(paths, run_id, fold_idx):
    probe_path, predictions_path, completion_path = paths
    try:
        marker = _read_json(completion_path)
        if marker['run_id'] != run_id or marker['fold'] != fold_idx + 1:
            return None
        if marker['probe_sha256'] != _sha256_file(probe_path):
            return None
        if marker['predictions_sha256'] != _sha256_file(predictions_path):
            return None
        probe = joblib.load(probe_path)
        return probe, float(marker['accuracy'])
    except Exception:
        return None


def run_binary_probe(
    *,
    load_vectors,
    manifest,
    probe_run_directory,
    predictions_run_directory,
    result_fields,
    display_name,
    n_splits,
    random_state,
    standardize,
    save_probes,
    save_predictions,
    overwrite,
    verbose,
):
    '''Run shared cache, cross-validation, fitting, and persistence logic.'''
    validate_probe_arguments(n_splits, standardize)
    run_id = hash_run_manifest(manifest)
    fold_path_list = [
        fold_paths(probe_run_directory, predictions_run_directory, index)
        for index in range(n_splits)
    ]

    check_existing = save_probes and save_predictions and not overwrite
    cached_folds = {}
    manifests_match = (
        _manifest_matches(probe_run_directory, manifest)
        and _manifest_matches(predictions_run_directory, manifest)
    )
    if check_existing and manifests_match:
        for fold_idx, paths in enumerate(fold_path_list):
            cached = _load_cached_fold(paths, run_id, fold_idx)
            if cached is not None:
                cached_folds[fold_idx] = cached

    if len(cached_folds) == n_splits:
        if verbose:
            print(f'{display_name}: all {n_splits} folds already trained '
                f'under {probe_run_directory} - skipping '
                '(pass overwrite=True to retrain)')
        probes = [cached_folds[index][0] for index in range(n_splits)]
        accuracies = [
            cached_folds[index][1] for index in range(n_splits)]
        result = dict(result_fields)
        result.update({
            'run_id': run_id,
            'cache_status': 'hit',
            'standardize': standardize,
            'accuracies': accuracies,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'probes': probes,
            'n_samples': None,
            'n_missing': None,
            'skipped': True,
        })
        return result

    if save_probes:
        _write_run_manifest(probe_run_directory, manifest)
    if save_predictions:
        _write_run_manifest(predictions_run_directory, manifest)

    X, y, true_labels, missing = load_vectors()
    if verbose:
        print(f'{len(X)} feature vectors loaded ({len(missing)} missing)')
        print(Counter(y))

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    probe_template = make_probe(standardize)
    accuracies, probes = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        probe_path, prediction_path, _ = fold_path_list[fold_idx]

        if fold_idx in cached_folds:
            probe, accuracy = cached_folds[fold_idx]
            if verbose:
                print(f'fold {fold_idx + 1}: already trained '
                    f'(accuracy={accuracy:.4f}), skipping')
        else:
            probe = clone(probe_template)
            probe.fit(X[train_idx], y[train_idx])
            predictions_binary = probe.predict(X[test_idx])
            accuracy = accuracy_score(y[test_idx], predictions_binary)
            predictions = list(zip(
                true_labels[test_idx], y[test_idx], predictions_binary))

            if verbose:
                print(f'fold {fold_idx + 1}: accuracy={accuracy:.4f}')
            if save_probes and save_predictions:
                _save_cached_fold(
                    probe, predictions, accuracy, fold_path_list[fold_idx],
                    run_id, fold_idx)
            elif save_probes:
                _save_probe(probe, probe_path)
            elif save_predictions:
                _save_predictions(predictions, prediction_path)

        accuracies.append(accuracy)
        probes.append(probe)

    mean_accuracy = float(np.mean(accuracies))
    std_accuracy = float(np.std(accuracies))
    if verbose:
        print(
            f'{display_name}: mean={mean_accuracy:.4f} '
            f'std={std_accuracy:.4f}'
        )

    if not save_probes or not save_predictions:
        cache_status = 'disabled'
    elif overwrite:
        cache_status = 'refresh'
    elif cached_folds:
        cache_status = 'partial'
    else:
        cache_status = 'miss'

    result = dict(result_fields)
    result.update({
        'run_id': run_id,
        'cache_status': cache_status,
        'standardize': standardize,
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'probes': probes,
        'n_samples': len(X),
        'n_missing': len(missing),
        'skipped': False,
    })
    return result


def _validate_scale_arguments(sample_size, std_ratio_threshold):
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError('sample_size must be a positive integer')
    if sample_size <= 0:
        raise ValueError('sample_size must be a positive integer')
    if std_ratio_threshold <= 1:
        raise ValueError('std_ratio_threshold must be greater than 1')


def _feature_scale_summary(matrix, std_ratio_threshold):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError('feature-scale input must be a 2D matrix')
    if not np.isfinite(matrix).all():
        raise ValueError('feature-scale input contains non-finite values')
    dimension_means = matrix.mean(axis=0)
    dimension_stds = matrix.std(axis=0, ddof=1)
    nonzero_stds = dimension_stds[dimension_stds > 0]
    if len(nonzero_stds) >= 2:
        std_ratio = float(nonzero_stds.max() / nonzero_stds.min())
    else:
        std_ratio = 1.0
    quantiles = np.quantile(dimension_stds, [0, .25, .5, .75, 1])
    return {
        'n_items': matrix.shape[0],
        'n_dimensions': matrix.shape[1],
        'dimension_means': dimension_means,
        'dimension_stds': dimension_stds,
        'std_quantiles': {
            'min': float(quantiles[0]),
            'q25': float(quantiles[1]),
            'median': float(quantiles[2]),
            'q75': float(quantiles[3]),
            'max': float(quantiles[4]),
        },
        'std_ratio': std_ratio,
        'zero_variance_dimensions': int(
            np.count_nonzero(dimension_stds == 0)),
        'max_abs_mean': float(np.max(np.abs(dimension_means))),
        'recommend_standardize': (
            len(nonzero_stds) >= 2
            and std_ratio >= std_ratio_threshold
        ),
    }


def _print_feature_scale(name, summary, std_ratio_threshold):
    quantiles = summary['std_quantiles']
    recommendation = (
        'consider standardizing' if summary['recommend_standardize']
        else 'scale spread is below the heuristic threshold'
    )
    print(
        f'{name}: {summary["n_items"]} items × '
        f'{summary["n_dimensions"]} dimensions'
    )
    print(
        f'  dimension std min/median/max: {quantiles["min"]:.6g} / '
        f'{quantiles["median"]:.6g} / {quantiles["max"]:.6g}'
    )
    print(
        f'  nonzero std ratio: {summary["std_ratio"]:.3g}; '
        f'threshold: {std_ratio_threshold:g} — {recommendation}'
    )
    print(
        f'  zero-variance dimensions: '
        f'{summary["zero_variance_dimensions"]}'
    )


def inspect_feature_scale(
    phones,
    embedding_store=None,
    mfcc_store=None,
    embedding_store_root=default_embedding_store_root,
    mfcc_store_root=default_mfcc_store_root,
    model_name=default_model_name,
    layer=9,
    collar=2000,
    sample_size=1000,
    random_state=42,
    std_ratio_threshold=10.0,
    verbose=True,
):
    '''Inspect per-dimension scale on paired embedding and MFCC center frames.

    This diagnostic is independent of probe training. It samples the same
    phone indices for both representations, keeps only pairs present in both
    stores, and reports per-dimension standard-deviation spread. The
    recommendation is a heuristic, not an automatic pipeline decision.
    '''
    _validate_scale_arguments(sample_size, std_ratio_threshold)
    embedding_store_owned = embedding_store is None
    mfcc_store_owned = mfcc_store is None
    if embedding_store is None:
        embedding_store = echoframe.Store(str(embedding_store_root))
    if mfcc_store is None:
        mfcc_store = echoframe.Store(str(mfcc_store_root))

    try:
        segments = phones.phraser_phones
        if not segments:
            raise ValueError('phones.phraser_phones is empty')
        selected_count = min(sample_size, len(segments))
        rng = random.Random(random_state)
        selected_indices = rng.sample(range(len(segments)), selected_count)
        selected = [segments[index] for index in selected_indices]

        embedding_keys = [
            embedding_store.make_echoframe_key(
                'hidden_state', model_name=model_name,
                phraser_key=segment.key, layer=layer, collar=collar)
            for segment in selected
        ]
        mfcc_keys = [
            mfcc_store.make_echoframe_key(
                'acoustic_feature', feature_name='mfcc',
                phraser_key=segment.key)
            for segment in selected
        ]
        embedding_vectors = embedding_store.load_many_frames(
            embedding_keys, frame='center', keep_missing=True)
        mfcc_vectors = mfcc_store.load_many_frames(
            mfcc_keys, frame='center', keep_missing=True)

        paired_embeddings, paired_mfccs = [], []
        for embedding, mfcc in zip(
            embedding_vectors, mfcc_vectors, strict=True,
        ):
            if embedding is None or mfcc is None:
                continue
            paired_embeddings.append(np.asarray(embedding))
            paired_mfccs.append(np.asarray(mfcc))
        if len(paired_embeddings) < 2:
            raise ValueError(
                'fewer than two sampled phones have both embedding and MFCC '
                'features; increase sample_size or complete extraction')

        embedding_matrix = np.stack(paired_embeddings)
        mfcc_matrix = np.stack(paired_mfccs)
        embedding_summary = _feature_scale_summary(
            embedding_matrix, std_ratio_threshold)
        mfcc_summary = _feature_scale_summary(
            mfcc_matrix, std_ratio_threshold)
        report = {
            'n_requested': selected_count,
            'n_paired': len(paired_embeddings),
            'n_missing_embedding': sum(
                vector is None for vector in embedding_vectors),
            'n_missing_mfcc': sum(
                vector is None for vector in mfcc_vectors),
            'model_name': model_name,
            'layer': layer,
            'collar': collar,
            'frame': 'center',
            'std_ratio_threshold': std_ratio_threshold,
            'embedding': embedding_summary,
            'mfcc': mfcc_summary,
        }
        if verbose:
            print(
                f'paired feature-scale sample: {report["n_paired"]}/'
                f'{report["n_requested"]} requested phones'
            )
            _print_feature_scale(
                'embedding', embedding_summary, std_ratio_threshold)
            _print_feature_scale(
                'mfcc', mfcc_summary, std_ratio_threshold)
        return report
    finally:
        if embedding_store_owned:
            embedding_store.close()
        if mfcc_store_owned:
            mfcc_store.close()
