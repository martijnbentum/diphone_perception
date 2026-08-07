'''Value objects describing binary phone-probe results.'''

import csv
import json
import statistics
import tempfile
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import locations
from probing.extract_embeddings import default_model_name

_n_splits = 5

_identity_manifest_fields = (
    'representation', 'feature_parameters', 'target_phoneme',
    'classifier', 'feature_set_hash')


def manifests_match(a, b):
    '''Whether two run manifests agree on every experiment-identity field.

    feature_set_hash is included because it tracks embedding availability
    per key, so a run that becomes possible once missing embeddings arrive
    is still treated as a different run. Fields such as the selected-sample
    hash/count and code-version markers are excluded, since those don't
    change without one of the checked fields also changing.
    '''
    return all(a.get(field) == b.get(field)
        for field in _identity_manifest_fields)


def embedding_result_path(target_phoneme, model_name=default_model_name,
    layer=9, collar=2000, *, root=locations.probe_results):
    '''Return the result directory for one embedding probe.

    target_phoneme:  phoneme label used as the positive class
    model_name:      embedding model identifier
    layer:           hidden-state layer
    collar:          embedding context in milliseconds
    root:            root directory containing probe results
    '''
    return Path(root) / model_name / target_phoneme / f'layer{layer:02d}' / (
        f'collar{collar}')


def mfcc_result_path(target_phoneme, frame='center', *,
    root=locations.probe_results):
    '''Return the result directory for one MFCC probe.

    target_phoneme:  phoneme label used as the positive class
    frame:           MFCC frame reduction
    root:            root directory containing probe results
    '''
    return Path(root) / 'mfcc' / target_phoneme / f'frame-{frame}'


class PhoneResult:
    '''Result identity and computed folds for one target phoneme.'''

    def __init__(self, target_phoneme, *, representation, model_name, layer,
        collar, frame, root=locations.probe_results):
        '''Create a validated phone-result identity.

        target_phoneme:  phoneme label used as the positive class
        representation:  feature family, either embedding or MFCC
        model_name:      embedding model identifier, otherwise None
        layer:           embedding hidden-state layer, otherwise None
        collar:          embedding context in milliseconds, otherwise None
        frame:           frame reduction used to create feature vectors
        root:            root directory containing probe results
        '''
        self.target_phoneme = target_phoneme
        self.representation = representation
        self.model_name = model_name
        self.layer = layer
        self.collar = collar
        self.frame = frame
        self.n_splits = _n_splits
        self.root = Path(root)

    def __repr__(self):
        identity = f'{self.target_phoneme!r}, {self.representation!r}'
        if self.representation == 'embedding':
            identity += f', {self.model_name!r}, layer={self.layer}'
        else:
            identity += f', frame={self.frame!r}'
        return f'{type(self).__name__}({identity})'

    def __eq__(self, other):
        if not isinstance(other, PhoneResult): return NotImplemented
        return self.identity == other.identity

    @classmethod
    def embedding(cls, target_phoneme, model_name=default_model_name,
        layer=9, collar=2000, root=locations.probe_results):
        '''Create an embedding result using middle-frame hidden states.

        target_phoneme:  phoneme label used as the positive class
        model_name:      embedding model identifier
        layer:           hidden-state layer
        collar:          embedding context in milliseconds
        root:            root directory containing probe results
        '''
        return cls(target_phoneme, representation='embedding',
            model_name=model_name, layer=layer, collar=collar,
            frame='middle', root=root)

    @classmethod
    def mfcc(cls, target_phoneme, frame='center',
        root=locations.probe_results):
        '''Create an MFCC result using the requested frame reduction.

        target_phoneme:  phoneme label used as the positive class
        frame:           MFCC frame reduction
        root:            root directory containing probe results
        '''
        return cls(target_phoneme, representation='mfcc', model_name=None,
            layer=None, collar=None, frame=frame, root=root)

    def load_run(self):
        '''Load run.json, returning None when it does not exist.'''
        if not self.run_path.is_file(): return None
        with self.run_path.open(encoding='utf-8') as handle:
            return json.load(handle)

    def save_run(self, manifest):
        '''Atomically store a run manifest in this result directory.'''
        _write_json(self.run_path, manifest)
        self.run = manifest

    def check_manifest(self, manifest):
        '''Store a new manifest or raise when the stored manifest differs.'''
        if self.run is not None:
            if not manifests_match(self.run, manifest):
                raise ValueError(f'manifest does not match {self.run_path}')
            return
        fold_paths = self.path.glob('fold*_predictions.tsv')
        if any(fold_paths):
            message = f'fold results exist without {self.run_path}'
            raise ValueError(message)
        self.save_run(manifest)

    def load_folds(self):
        '''Load Fold objects for prediction files present on disk.'''
        folds = []
        for number in range(1, self.n_splits + 1):
            fold = Fold(self, number)
            if fold.path.is_file(): folds.append(fold)
        return folds

    @property
    def label(self):
        '''Alias for target_phoneme.'''
        return self.target_phoneme

    @property
    def path(self):
        '''Directory containing this result's manifest and fold files.'''
        if self.representation == 'embedding':
            return embedding_result_path(self.target_phoneme,
                self.model_name, self.layer, self.collar, root=self.root)
        return mfcc_result_path(self.target_phoneme, self.frame,
            root=self.root)

    @property
    def run_path(self):
        '''Path to this result's run manifest.'''
        return self.path / 'run.json'

    @cached_property
    def run(self):
        '''Run manifest loaded from disk, or None when absent.'''
        return self.load_run()

    @property
    def identity(self):
        '''Tuple containing every public experiment-identity parameter.'''
        return (self.target_phoneme, self.representation, self.model_name,
            self.layer, self.collar, self.frame)

    @cached_property
    def folds(self):
        '''Fold results present on disk, loaded once in number order.'''
        return self.load_folds()

    @property
    def accuracies(self):
        '''Fold accuracies in fold-number order.'''
        return [fold.accuracy for fold in self.folds]

    @property
    def mean_accuracy(self):
        '''Mean fold accuracy, or None when no folds are attached.'''
        if not self.folds: return None
        accuracy_mean = statistics.mean(self.accuracies)
        return float(accuracy_mean)

    @property
    def std_accuracy(self):
        '''Population standard deviation, or None with no folds.'''
        if not self.folds: return None
        accuracy_std = statistics.pstdev(self.accuracies)
        return float(accuracy_std)

    @property
    def complete(self):
        '''Whether all configured fold results are attached.'''
        return len(self.folds) == self.n_splits

    @property
    def missing_fold_numbers(self):
        '''One-based fold numbers without prediction files.'''
        present = {fold.number for fold in self.folds}
        expected = range(1, self.n_splits + 1)
        return [number for number in expected if number not in present]


class Fold:
    '''Result and artifact locations for one cross-validation fold.'''

    def __init__(self, parent, number):
        '''Create a fold linked to its PhoneResult.

        parent:  PhoneResult that owns this fold
        number:  one-based fold number
        '''
        self.parent = parent
        self.number = number
        filename = f'fold{number:02d}_predictions.tsv'
        self.path = parent.path / filename

    def __repr__(self):
        name = type(self).__name__
        return f'{name}(number={self.number}, path={str(self.path)!r})'

    def load_tsv(self):
        '''Load and return prediction lines from this fold's TSV file.'''
        with self.path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            lines = []
            for index, row in enumerate(reader):
                correct = {'0': False, '1': True}[row['correct']]
                line = Line(row['true_phoneme'], row['binary_true'],
                    row['binary_pred'], correct, index, self)
                lines.append(line)
        return lines

    def save_results(self, prediction_rows):
        '''Atomically save true-phoneme, ground-truth, prediction triples.'''
        with _atomic_target(self.path) as temporary_path:
            _write_prediction_tsv(temporary_path, prediction_rows)

    @cached_property
    def results(self):
        '''Prediction lines loaded from this fold's TSV file.'''
        return self.load_tsv()

    @cached_property
    def accuracy(self):
        '''Fraction of correct predictions in this fold.'''
        if not self.results:
            raise ValueError(f'cannot compute accuracy from empty {self.path}')
        correct = [line.correct for line in self.results]
        return float(statistics.mean(correct))


class Line:
    '''One parsed prediction line from a fold TSV file.'''

    def __init__(self, phoneme, gt, pred, correct, index, parent=None):
        '''Create one prediction line.

        phoneme:  true phoneme label
        gt:       ground-truth binary label
        pred:     predicted binary label
        correct:  whether the binary prediction is correct
        index:    zero-based data-row index, excluding the header
        parent:   Fold containing this line, when available
        '''
        self.phoneme = phoneme
        self.gt = gt
        self.pred = pred
        self.correct = correct
        self.index = index
        self.parent = parent


def _write_json(path, value):
    text = json.dumps(value, sort_keys=True, indent=2,
        ensure_ascii=False) + '\n'
    with _atomic_target(path) as temporary_path:
        temporary_path.write_text(text, encoding='utf-8')


def _write_prediction_tsv(path, prediction_rows):
    header = ['true_phoneme', 'binary_true', 'binary_pred', 'correct']
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        for phoneme, gt, pred in prediction_rows:
            correct = int(gt == pred)
            writer.writerow([phoneme, gt, pred, correct])


@contextmanager
def _atomic_target(path):
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
