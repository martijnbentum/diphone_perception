import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import echoframe
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from probing.extract_embeddings import default_model_name, default_store_root
from probing.metadata import _data_dir

default_probe_save_dir = _data_dir / 'phone_probes'
default_results_dir = _data_dir / 'probe_results'
_cache_schema_version = 1
_trainer_version = 1
_probe_parameters = {'solver': 'liblinear', 'max_iter': 1000}


def _select_phones(phones, target_phoneme, n_embeds=None, seed=42):
    '''Deterministically sample up to n_embeds target-phoneme phones, plus an
    even split of n_embeds across every other phoneme class - mirrors the
    balanced-sampling scheme from the old train_probes_binary.py.

    n_embeds=None (the default) uses every available target-phoneme phone,
    rather than an arbitrary cap - pass a smaller number for a quick trial.

    Returns a list of (phone, phraser_phone, binary_label) tuples, where
    binary_label is 'target' or 'other'.
    '''
    by_label = defaultdict(list)
    for phone, phraser_phone in zip(phones.phones, phones.phraser_phones):
        by_label[phone.phoneme_ipa].append((phone, phraser_phone))

    if target_phoneme not in by_label:
        raise ValueError(
            f'target_phoneme {target_phoneme!r} not found among phones')

    if n_embeds is None:
        n_embeds = len(by_label[target_phoneme])

    other_labels = [label for label in by_label if label != target_phoneme]
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, remainder = divmod(n_embeds, len(other_labels))
    if n_per_other == 0:
        raise ValueError(
            f'n_embeds={n_embeds} is too small to split across '
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

    selected = [(p, pp, 'target') for p, pp in _take(target_phoneme, n_embeds)]
    for label in other_labels:
        selected += [(p, pp, 'other') for p, pp in _take(label, n_per_other)]

    rng.shuffle(selected)
    return selected


def _load_middle_frame_vectors(store, selected, model_name, layer, collar):
    '''Batch-load stored embeddings for `selected` and reduce each to its
    middle frame. Phones missing from the store are dropped, not backfilled
    - their count is returned so callers can tell if a class came up short.
    '''
    phraser_keys = [phraser_phone.key for _, phraser_phone, _ in selected]
    embeddings = store.phraser_keys_to_embeddings(
        phraser_keys, model_name, layer, collar=collar)
    by_key = {e.phraser_key: e for e in embeddings.embeddings}

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


def _stable_key(value):
    '''Return a deterministic text representation for a phraser/echoframe key.'''
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f'bytes:{bytes(value).hex()}'
    return f'{type(value).__name__}:{value}'


def _hash_json(value):
    data = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _build_probe_run_manifest(
    store,
    selected,
    model_name,
    target_phoneme,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
):
    '''Build the canonical, JSON-serializable identity for one probe run.'''
    sample_records = [
        {
            'phraser_key': _stable_key(phraser_phone.key),
            'phoneme': str(phone.phoneme_ipa),
            'binary_label': binary_label,
        }
        for phone, phraser_phone, binary_label in selected
    ]
    echoframe_keys = [
        store.make_echoframe_key(
            'hidden_state', model_name=model_name,
            phraser_key=phraser_phone.key, layer=layer, collar=collar)
        for _, phraser_phone, _ in selected
    ]
    metadatas = store.load_many_metadata(echoframe_keys, keep_missing=True)
    embedding_records = []
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
        embedding_records.append(record)
    return {
        'cache_schema_version': _cache_schema_version,
        'trainer_version': _trainer_version,
        'model_name': model_name,
        'target_phoneme': target_phoneme,
        'layer': layer,
        'collar': collar,
        'n_embeds': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'classifier': {
            'class': 'sklearn.linear_model.LogisticRegression',
            **_probe_parameters,
        },
        'selected_sample_count': len(sample_records),
        'selected_samples_hash': _hash_json(sample_records),
        'embedding_set_hash': _hash_json(embedding_records),
    }


def _hash_run_manifest(manifest):
    return _hash_json(manifest)[:16]


def _run_directory(
    root, model_name, target_phoneme, layer, collar, run_id,
):
    return (
        Path(root) / model_name / target_phoneme / f'layer{layer:02d}'
        / f'collar{collar}ms' / run_id
    )


def _fold_paths(probe_run_directory, predictions_run_directory, fold_idx):
    number = fold_idx + 1
    return (
        Path(probe_run_directory) / f'fold{number:02d}.joblib',
        Path(predictions_run_directory) / f'fold{number:02d}_predictions.tsv',
        Path(probe_run_directory) / f'fold{number:02d}_complete.json',
    )


@contextmanager
def _atomic_target(path):
    '''Yield a temporary sibling and replace path only after a successful write.'''
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


def train_binary_probe(
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
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    verbose=True,
):
    '''Train/test a binary (target-phoneme-vs-other) logistic regression
    probe on middle-frame wav2vec2 embeddings stored in echoframe, with
    5-fold StratifiedKFold(shuffle=True, random_state=random_state).

    phones:            probing.metadata.Phones (phraser_phones must be
                       complete)
    target_phoneme:    IPA label to classify against all other phones
    store:             existing echoframe.Store to read from; opened at
                       store_root if None
    model_name, layer, collar:  identify which stored embeddings to read -
                       must match what extract_phone_embeddings wrote;
                       collar defaults to 2000ms on each side
    n_embeds:          total target-class samples; each other phoneme class
                       gets n_embeds // (number of other phoneme classes).
                       None (default) uses every available target-phoneme
                       phone rather than an arbitrary cap.
    save_probes:       dump each fold's fitted probe under probe_save_dir
                       (default True)
    save_predictions:  dump each fold's per-example predictions under
                       results_dir (default True)
    overwrite:         if False (default), reuse complete folds from the
                       matching run cache. The run identity covers the model,
                       target, layer, collar, sampling/split settings,
                       selected phones, available embedding metadata, and
                       classifier settings. A fold is complete only when its
                       marker and checksums are valid. If either save flag is
                       False, or overwrite is True, every fold is trained.

    Returns a dict with per-fold accuracies, their mean/std, the fitted
    probes, missing/sample counts, run/cache identity, and whether every
    fold was loaded from disk without training anything ('skipped').
    '''
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if store is None:
        store = echoframe.Store(str(store_root))

    selected = _select_phones(
        phones, target_phoneme, n_embeds, seed=random_state)
    manifest = _build_probe_run_manifest(
        store, selected, model_name, target_phoneme, layer, collar, n_embeds,
        n_splits, random_state)
    run_id = _hash_run_manifest(manifest)
    probe_run_directory = _run_directory(
        probe_save_dir, model_name, target_phoneme, layer, collar, run_id)
    predictions_run_directory = _run_directory(
        results_dir, model_name, target_phoneme, layer, collar, run_id)

    check_existing = save_probes and save_predictions and not overwrite
    fold_paths = [
        _fold_paths(probe_run_directory, predictions_run_directory, i)
        for i in range(n_splits)
    ]

    cached_folds = {}
    manifests_match = (
        _manifest_matches(probe_run_directory, manifest)
        and _manifest_matches(predictions_run_directory, manifest)
    )
    if check_existing and manifests_match:
        for fold_idx, paths in enumerate(fold_paths):
            cached = _load_cached_fold(paths, run_id, fold_idx)
            if cached is not None:
                cached_folds[fold_idx] = cached

    if len(cached_folds) == n_splits:
        if verbose:
            print(f'{target_phoneme} layer {layer}: all {n_splits} folds '
                f'already trained under {probe_run_directory} - skipping '
                '(pass overwrite=True to retrain)')
        probes = [cached_folds[i][0] for i in range(n_splits)]
        accuracies = [cached_folds[i][1] for i in range(n_splits)]
        mean_acc, std_acc = float(np.mean(accuracies)), float(np.std(accuracies))
        return {
            'target_phoneme': target_phoneme,
            'layer': layer,
            'collar': collar,
            'run_id': run_id,
            'cache_status': 'hit',
            'accuracies': accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'probes': probes,
            'n_samples': None,
            'n_missing': None,
            'skipped': True,
        }

    if save_probes:
        _write_run_manifest(probe_run_directory, manifest)
    if save_predictions:
        _write_run_manifest(predictions_run_directory, manifest)

    X, y, true_labels, missing = _load_middle_frame_vectors(
        store, selected, model_name, layer, collar)

    if verbose:
        print(f'{len(X)} embeddings loaded ({len(missing)} missing)')
        print(Counter(y))

    kf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies, probes = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        probe_path, pred_path, completion_path = fold_paths[fold_idx]

        if fold_idx in cached_folds:
            probe, acc = cached_folds[fold_idx]
            if verbose:
                print(f'fold {fold_idx + 1}: already trained '
                    f'(accuracy={acc:.4f}), skipping')
        else:
            probe = LogisticRegression(**_probe_parameters)
            probe.fit(X[train_idx], y[train_idx])
            y_pred = probe.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
            predictions = list(
                zip(true_labels[test_idx], y[test_idx], y_pred))

            if verbose:
                print(f'fold {fold_idx + 1}: accuracy={acc:.4f}')
            if save_probes and save_predictions:
                _save_cached_fold(
                    probe, predictions, acc, fold_paths[fold_idx], run_id,
                    fold_idx)
            elif save_probes:
                _save_probe(probe, probe_path)
            elif save_predictions:
                _save_predictions(predictions, pred_path)

        accuracies.append(acc)
        probes.append(probe)

    mean_acc, std_acc = float(np.mean(accuracies)), float(np.std(accuracies))
    if verbose:
        print(f'{target_phoneme} layer {layer}: '
            f'mean={mean_acc:.4f} std={std_acc:.4f}')

    if not save_probes or not save_predictions:
        cache_status = 'disabled'
    elif overwrite:
        cache_status = 'refresh'
    elif cached_folds:
        cache_status = 'partial'
    else:
        cache_status = 'miss'

    return {
        'target_phoneme': target_phoneme,
        'layer': layer,
        'collar': collar,
        'run_id': run_id,
        'cache_status': cache_status,
        'accuracies': accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'probes': probes,
        'n_samples': len(X),
        'n_missing': len(missing),
        'skipped': False,
    }
