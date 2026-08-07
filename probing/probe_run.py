import hashlib
import json
import tempfile
from collections import Counter, namedtuple
from contextlib import contextmanager
from pathlib import Path

import joblib

from probing import probe_training
from probing import result as probe_result

TrainingOutcome = namedtuple('TrainingOutcome',
    ['accuracies', 'mean_accuracy', 'std_accuracy', 'n_samples',
        'n_missing'])


def run(*, load_vectors, probe_run_directory,
    phone_result, display_name,
    save_probes, save_predictions, overwrite, verbose):
    '''Train and persist fold classifiers and predictions.

    load_vectors:               callback returning features and labels
    probe_run_directory:        directory containing fitted fold probes
    phone_result:                identity and storage for fold predictions

    Returns a TrainingOutcome describing what was trained, or None when
    every fold result was already stored and training was skipped. The
    caller builds and saves its own descriptive identity record (see
    probe_data.describe_probe_run) after this returns.
    '''
    if save_predictions and phone_result.complete and not overwrite:
        if verbose:
            print(f'{display_name}: all {phone_result.n_splits} results '
                f'already stored under {phone_result.path} - skipping '
                '(pass overwrite=True to retrain)')
        return None

    X, y, true_labels, missing = load_vectors()
    if verbose:
        print(f'{len(X)} feature vectors loaded ({len(missing)} missing)')
        label_counts = Counter(y)
        print(label_counts)

    probes = probe_training.Probes(X, y)
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
                n_predictions, probe_path, completion_path, fold_idx)
        if save_predictions:
            fold = probe_result.Fold(phone_result, fold_number)
            fold.save_results(predictions)

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
    completion_path, fold_idx):
    completion_path = Path(completion_path)
    completion_path.unlink(missing_ok=True)
    _save_probe(classifier, probe_path)
    accuracy = float(accuracy)
    probe_sha256 = _sha256_file(probe_path)
    marker = {'fold': fold_idx + 1,
        'accuracy': accuracy, 'n_predictions': n_predictions,
        'probe_sha256': probe_sha256}
    write_json(completion_path, marker)
