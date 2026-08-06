import hashlib
import json
import tempfile
from collections import Counter, namedtuple
from contextlib import contextmanager
from pathlib import Path

import joblib

from probing import probe_training
from probing import result as probe_result

_cache_schema_version = 2
_trainer_version = 2

TrainingOutcome = namedtuple('TrainingOutcome',
    ['accuracies', 'mean_accuracy', 'std_accuracy', 'n_samples',
        'n_missing'])


def run(*, load_vectors, manifest, probe_run_directory,
    phone_result, display_name, n_splits, random_state,
    standardize, save_probes, save_predictions, overwrite, verbose):
    '''Train and persist fold classifiers and predictions.

    load_vectors:               callback returning features and labels
    manifest:                   canonical identity of the probe run
    probe_run_directory:        directory containing fitted fold probes
    phone_result:               identity and storage for fold predictions
    n_splits:                   number of cross-validation folds

    Returns a TrainingOutcome describing what was trained, or None when
    every fold result was already stored and training was skipped.
    '''
    probe_training.validate_training_options(n_splits, standardize)
    if save_predictions: phone_result.check_manifest(manifest)
    if save_predictions and phone_result.complete and not overwrite:
        if verbose:
            print(f'{display_name}: all {n_splits} results already stored '
                f'under {phone_result.path} - skipping '
                '(pass overwrite=True to retrain)')
        return None

    run_id = hash_run_manifest(manifest)
    if save_probes: _write_run_manifest(probe_run_directory, manifest)

    X, y, true_labels, missing = load_vectors()
    if verbose:
        print(f'{len(X)} feature vectors loaded ({len(missing)} missing)')
        label_counts = Counter(y)
        print(label_counts)

    probes = probe_training.Probes(X, y, n_splits, standardize=standardize,
        random_state=random_state)
    probes.run(show_progress=verbose)
    for probe in probes.probes:
        fold_idx = probe.fold_index
        fold_number = fold_idx + 1
        probe_path, _, completion_path = fold_paths(probe_run_directory,
            phone_result.path, fold_idx)
        prediction_rows = zip(true_labels[probe.test_indices],
            probes.y[probe.test_indices], probe.predictions)
        predictions = list(prediction_rows)

        if verbose:
            print(f'fold {fold_number}: accuracy={probe.accuracy:.4f}')
        if save_probes:
            n_predictions = len(predictions)
            _save_cached_probe(probe.classifier, probe.accuracy,
                n_predictions, probe_path, completion_path, run_id,
                fold_idx)
        if save_predictions:
            fold = probe_result.Fold(phone_result, fold_number)
            fold.save_results(predictions)

    if save_predictions:
        manifest_with_counts = dict(manifest)
        manifest_with_counts['actual_n_samples'] = len(X)
        manifest_with_counts['actual_n_missing'] = len(missing)
        phone_result.save_run(manifest_with_counts)

    if verbose:
        print(f'{display_name}: mean={probes.mean_accuracy:.4f} '
            f'std={probes.std_accuracy:.4f}')

    return TrainingOutcome(probes.accuracies, probes.mean_accuracy,
        probes.std_accuracy, len(X), len(missing))


def classify_cache_status(save_predictions, complete_before, overwrite,
    existing_fold_count):
    '''Classify a probe_run.run() call relative to previously stored results.

    Callers snapshot complete_before/existing_fold_count from phone_result
    before calling run(), since run() mutates that state as a side effect.
    '''
    if not save_predictions: return 'disabled'
    if complete_before and not overwrite: return 'hit'
    if overwrite: return 'refresh'
    if existing_fold_count: return 'partial'
    return 'miss'


def build_probe_run_manifest(store, selected, echoframe_keys, representation,
    feature_parameters, target_phoneme, n_samples, n_splits, random_state,
    standardize):
    '''Build the canonical identity for an embedding or MFCC probe run.

    store:               Echoframe feature store
    selected:            selected phone and label triples
    echoframe_keys:      feature keys matching the selected phones
    representation:     representation name stored in the manifest
    feature_parameters:  representation-specific settings
    '''
    sample_records = []
    for phone, phraser_phone, binary_label in selected:
        phraser_key = _stable_key(phraser_phone.key)
        phoneme = str(phone.phoneme_ipa)
        record = {'phraser_key': phraser_key}
        record['phoneme'] = phoneme
        record['binary_label'] = binary_label
        sample_records.append(record)

    metadatas = store.load_many_metadata(echoframe_keys, keep_missing=True)
    feature_records = []
    metadata_pairs = zip(echoframe_keys, metadatas, strict=True)
    for key, metadata in metadata_pairs:
        stable_key = _stable_key(key)
        record = {'key': stable_key, 'present': metadata is not None}
        if metadata is not None:
            shape = getattr(metadata, 'shape', None)
            created_at = getattr(metadata, 'created_at', None)
            dataset_path = getattr(metadata, 'dataset_path', None)
            shard_id = getattr(metadata, 'shard_id', None)
            record.update({'created_at': str(created_at),
                'dataset_path': dataset_path,
                'shape': list(shape) if shape is not None else None,
                'shard_id': shard_id})
        feature_records.append(record)

    classifier = probe_training.configuration(standardize)
    selected_sample_count = len(sample_records)
    selected_samples_hash = _hash_json(sample_records)
    feature_set_hash = _hash_json(feature_records)
    return {'cache_schema_version': _cache_schema_version,
        'trainer_version': _trainer_version,
        'representation': representation,
        'feature_parameters': dict(feature_parameters),
        'target_phoneme': target_phoneme, 'n_samples': n_samples,
        'n_splits': n_splits, 'random_state': random_state,
        'classifier': classifier,
        'selected_sample_count': selected_sample_count,
        'selected_samples_hash': selected_samples_hash,
        'feature_set_hash': feature_set_hash}


def hash_run_manifest(manifest):
    '''Return the short stable identifier for a run manifest.'''
    return _hash_json(manifest)[:16]


def fold_paths(probe_run_directory, predictions_run_directory, fold_idx):
    '''Return probe, prediction, and completion paths for one fold.

    probe_run_directory:        directory containing fitted probes
    predictions_run_directory:  directory containing predictions
    fold_idx:                   zero-based fold index
    '''
    number = fold_idx + 1
    probe_directory = Path(probe_run_directory)
    predictions_directory = Path(predictions_run_directory)
    probe_path = probe_directory / f'fold{number:02d}.joblib'
    prediction_path = (
        predictions_directory / f'fold{number:02d}_predictions.tsv')
    completion_path = probe_directory / f'fold{number:02d}_complete.json'
    return probe_path, prediction_path, completion_path


def write_json(path, value):
    '''Atomically write a JSON-serializable value.

    path:   destination JSON path
    value:  JSON-serializable value
    '''
    text = json.dumps(value, sort_keys=True, indent=2,
        ensure_ascii=False) + '\n'
    with _atomic_target(path) as temporary_path:
        temporary_path.write_text(text, encoding='utf-8')


def read_json(path):
    '''Read and return a JSON value.'''
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    return json.loads(text)


def _stable_key(value):
    '''Return a deterministic text representation for a stored key.'''
    if isinstance(value, (bytes, bytearray, memoryview)):
        byte_value = bytes(value)
        return f'bytes:{byte_value.hex()}'
    return f'{type(value).__name__}:{value}'


def _hash_json(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False)
    encoded = data.encode('utf-8')
    digest = hashlib.sha256(encoded)
    return digest.hexdigest()


@contextmanager
def _atomic_target(path):
    '''Yield a sibling temporary path and atomically replace the target.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,
        prefix=f'.{path.name}.', suffix=path.suffix,
        delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        yield temporary_path
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_run_manifest(run_directory, manifest):
    path = Path(run_directory) / 'run.json'
    write_json(path, manifest)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        chunks = iter(lambda: handle.read(1024 * 1024), b'')
        for chunk in chunks:
            digest.update(chunk)
    return digest.hexdigest()


def _save_probe(classifier, path):
    with _atomic_target(path) as temporary_path:
        joblib.dump(classifier, temporary_path)


def _save_cached_probe(classifier, accuracy, n_predictions, probe_path,
    completion_path, run_id, fold_idx):
    completion_path = Path(completion_path)
    completion_path.unlink(missing_ok=True)
    _save_probe(classifier, probe_path)
    accuracy = float(accuracy)
    probe_sha256 = _sha256_file(probe_path)
    marker = {'run_id': run_id, 'fold': fold_idx + 1,
        'accuracy': accuracy, 'n_predictions': n_predictions,
        'probe_sha256': probe_sha256}
    write_json(completion_path, marker)
