import numpy as np
from progressbar import progressbar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_classifier_parameters = {'solver': 'liblinear', 'max_iter': 1000}


class Probe:
    '''Fit and evaluate a classifier for one cross-validation fold.'''

    def __init__(self, X, y, train_indices, test_indices, fold_index,
        standardize=False):
        '''Create an uncomputed probe for one fold.

        X:              two-dimensional feature matrix
        y:              one-dimensional class labels
        train_indices:  sample indices used to fit the classifier
        test_indices:   sample indices used for prediction and evaluation
        fold_index:     index of this cross-validation fold
        standardize:    whether to standardize features before classification
        '''
        self.X = X
        self.y = y
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.fold_index = fold_index
        self.standardize = standardize
        self.classifier = None
        self.predictions = None
        self.accuracy = None

    def run(self):
        '''Fit the classifier and compute test predictions and accuracy.'''
        train_indices = self.train_indices
        test_indices = self.test_indices
        classifier = fit(self.X[train_indices], self.y[train_indices],
            standardize=self.standardize)
        predictions = classifier.predict(self.X[test_indices])
        score = accuracy_score(self.y[test_indices], predictions)
        self.classifier = classifier
        self.predictions = predictions
        self.accuracy = float(score)


class Probes:
    '''Create and run probes for stratified cross-validation folds.'''

    def __init__(self, X, y, n_folds, standardize=False, random_state=42):
        '''Create an uncomputed collection of probes.

        X:             two-dimensional feature matrix
        y:             one-dimensional class labels
        n_folds:       number of stratified folds
        standardize:   whether to standardize features before classification
        random_state:  random seed used when shuffling samples
        '''
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.standardize = standardize
        self.random_state = random_state
        self.probes = []

    def run(self, show_progress=True):
        '''Create and run the cross-validation probes.

        show_progress:  whether to display progress while fitting probes
        '''
        self.X, self.y = _arrays(self.X, self.y)
        validate_training_options(self.n_folds, self.standardize)
        folds = make_folds(self.X, self.y, self.n_folds,
            random_state=self.random_state)
        self.probes = []
        if show_progress:
            folds = progressbar(folds, prefix='Fitting probes: ')
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            probe = Probe(self.X, self.y, train_indices, test_indices,
                fold_index, standardize=self.standardize)
            probe.run()
            self.probes.append(probe)

    @property
    def classifiers(self):
        '''Return the fitted classifiers in fold order.'''
        classifiers = []
        for probe in self.probes: classifiers.append(probe.classifier)
        return classifiers

    @property
    def accuracies(self):
        '''Return fold accuracies in fold order.'''
        accuracies = []
        for probe in self.probes: accuracies.append(probe.accuracy)
        return accuracies

    @property
    def mean_accuracy(self):
        '''Return mean accuracy across computed folds.'''
        if not self.probes: return None
        return float(np.mean(self.accuracies))

    @property
    def std_accuracy(self):
        '''Return accuracy standard deviation across computed folds.'''
        if not self.probes: return None
        return float(np.std(self.accuracies))


def fit(X, y, standardize=False):
    '''Fit and return the default probe classifier.

    X:            two-dimensional feature matrix
    y:            one-dimensional class labels
    standardize:  whether to standardize features before classification
    '''
    X, y = _arrays(X, y)
    classifier = make_binary_classifier(standardize)
    classifier.fit(X, y)
    return classifier


def accuracy(X, y, classifier):
    '''Return classifier accuracy for the supplied features and labels.

    X:           two-dimensional feature matrix
    y:           one-dimensional class labels
    classifier:  fitted classifier implementing predict
    '''
    X, y = _arrays(X, y)
    predictions = classifier.predict(X)
    score = accuracy_score(y, predictions)
    return float(score)


def make_folds(X, y, n_folds, random_state=42):
    '''Return stratified train and test index pairs.

    X:             two-dimensional feature matrix
    y:             one-dimensional class labels
    n_folds:       number of stratified folds
    random_state:  random seed used when shuffling samples
    '''
    X, y = _arrays(X, y)
    _validate_n_folds(n_folds)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
        random_state=random_state)
    folds = []
    for train_indices, test_indices in splitter.split(X, y):
        folds.append((train_indices, test_indices))
    return folds


def cross_validation(X, y, n_folds, standardize=False, random_state=42,
    show_progress=True):
    '''Fit and evaluate one independent classifier per stratified fold.

    X:             two-dimensional feature matrix
    y:             one-dimensional class labels
    n_folds:       number of stratified folds
    standardize:   whether to standardize independently within each fold
    random_state:  random seed used when shuffling samples
    show_progress:  whether to display progress while fitting probes
    '''
    probes = Probes(X, y, n_folds, standardize=standardize,
        random_state=random_state)
    probes.run(show_progress=show_progress)
    return probes


def make_binary_classifier(standardize):
    '''Return the configured binary probe classifier.'''
    _validate_standardize(standardize)
    classifier = LogisticRegression(**_classifier_parameters)
    if not standardize: return classifier
    scaler = StandardScaler()
    return make_pipeline(scaler, classifier)


def configuration(standardize=False):
    '''Return the stable classifier configuration used in run identities.'''
    _validate_standardize(standardize)
    settings = {'class': 'sklearn.linear_model.LogisticRegression',
        **_classifier_parameters, 'standardize': standardize}
    if standardize:
        settings['preprocessor'] = 'sklearn.preprocessing.StandardScaler'
    else:
        settings['preprocessor'] = None
    return settings


def validate_training_options(n_folds, standardize):
    '''Validate fold count and standardization settings.

    n_folds:      number of cross-validation folds
    standardize:  whether features should be standardized
    '''
    _validate_n_folds(n_folds)
    _validate_standardize(standardize)


def _arrays(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2: raise ValueError('X must be a two-dimensional matrix')
    if y.ndim != 1: raise ValueError('y must be a one-dimensional array')
    if len(X) != len(y): raise ValueError('X and y must contain equal samples')
    if len(X) == 0: raise ValueError('X and y must not be empty')
    return X, y


def _validate_n_folds(n_folds):
    if isinstance(n_folds, bool) or not isinstance(n_folds, int):
        raise TypeError('n_folds must be an integer')
    if n_folds < 2: raise ValueError('n_folds must be at least 2')


def _validate_standardize(standardize):
    if not isinstance(standardize, bool):
        raise TypeError('standardize must be a boolean')
