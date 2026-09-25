'''Fit and evaluate centered SVD on sample-by-dimension matrices.'''

from pathlib import Path
import warnings

import numpy as np


def fit_svd(X):
    '''Fit centered SVD and return a dictionary of reusable arrays.

    X:  fitting matrix with at least two rows and one dimension

    Rows have equal weight. No normalization or coordinate scaling is applied.
    directions has shape (dimensions, components); its columns are orthonormal.
    Retain min(n_rows - 1, n_dimensions) components, including zero modes.
    Eigenvalues are singular_values squared divided by n_rows - 1.
    '''
    X = _as_matrix(X, min_rows=2)
    mean = X.mean(axis=0)
    centered = X - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(X.shape[0] - 1, X.shape[1])
    singular_values = singular_values[:n_components]
    return {
        'mean': mean,
        'directions': vt[:n_components].T,
        'singular_values': singular_values,
        'eigenvalues': singular_values ** 2 / (X.shape[0] - 1),
        'n_rows': X.shape[0],
    }


def transform(X, decomposition, n_components=None):
    '''Return component scores using the fitting mean and directions.

    X:              matrix with the same dimensions as the fitting data
    decomposition:  dictionary returned by fit_svd or load_svd
    n_components:   number of leading directions; None uses all directions
    '''
    X = _as_matrix(X)
    directions = decomposition['directions']
    if X.shape[1] != directions.shape[0]:
        raise ValueError('X dimensions must match the fitted decomposition')
    if n_components is None: n_components = directions.shape[1]
    if not isinstance(n_components, (int, np.integer)):
        raise ValueError('n_components must be an integer or None')
    if not 1 <= n_components <= directions.shape[1]:
        raise ValueError('n_components is outside the fitted component range')
    return (X - decomposition['mean']) @ directions[:, :n_components]


def summarize_spectrum(decomposition):
    '''Summarize fitting variance, cumulative fractions, and effective size.

    decomposition:  dictionary returned by fit_svd or load_svd

    A zero-variance matrix has zero fractions, dimension counts, and
    participation ratio. rank_ceiling is a bound, not a numerical rank.
    '''
    eigenvalues = decomposition['eigenvalues']
    total = eigenvalues.sum()
    fractions = np.zeros_like(eigenvalues)
    if total > 0: fractions = eigenvalues / total
    cumulative = np.cumsum(fractions)
    n_90, n_95, participation_ratio = 0, 0, 0.0
    if total > 0:
        n_90 = int(np.searchsorted(cumulative, 0.90)) + 1
        n_95 = int(np.searchsorted(cumulative, 0.95)) + 1
        participation_ratio = float(1 / np.sum(fractions ** 2))
    n_dimensions = len(decomposition['mean'])
    rank_ceiling = min(decomposition['n_rows'] - 1, n_dimensions)
    return {
        'total_variance': float(total),
        'variance_fractions': fractions,
        'cumulative_variance': cumulative,
        'n_components_90': n_90,
        'n_components_95': n_95,
        'participation_ratio': participation_ratio,
        'n_rows': decomposition['n_rows'],
        'rank_ceiling': rank_ceiling,
    }


def evaluate_svd(X, decomposition):
    '''Measure held-out variance and mean shifts in the fitted basis.

    X:              evaluation matrix with at least two rows
    decomposition:  dictionary fitted on separate fitting data

    Scores use the fitting mean. Variances use ddof=1 around evaluation
    means, separating spread from mean shift. Fractions divide by total
    evaluation variance and can sum below one for an incomplete fitted basis.
    Zero evaluation variance gives zero fractions.
    '''
    X = _as_matrix(X, min_rows=2)
    scores = transform(X, decomposition)
    variances = scores.var(axis=0, ddof=1)
    total = X.var(axis=0, ddof=1).sum()
    fractions = np.zeros_like(variances)
    if total > 0: fractions = variances / total
    return {
        'n_rows': X.shape[0],
        'component_variances': variances,
        'total_variance': float(total),
        'variance_fractions': fractions,
        'mean_shift': X.mean(axis=0) - decomposition['mean'],
        'score_mean_shift': scores.mean(axis=0),
    }


def evaluate_covariance(X, decomposition):
    '''Compare held-out score covariance with fitted component variances.

    X:              held-out matrix with at least two rows
    decomposition:  dictionary returned by fit_svd or load_svd

    Covariance is centered on held-out score means, so mean shift is separate.
    A fitted basis smaller than the embedding dimension covers only its span.
    Undefined ratios and pairwise correlations are NaN.
    '''
    X = _as_matrix(X, min_rows=2)
    scores = transform(X, decomposition)
    n_components = scores.shape[1]
    if n_components < X.shape[1]:
        message = 'fitted directions do not span all embedding dimensions'
        warnings.warn(message, UserWarning, stacklevel=2)
    if n_components == 1:
        message = 'one component has no off-diagonal correlation pairs'
        warnings.warn(message, UserWarning, stacklevel=2)

    covariance = np.cov(scores, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    variances = np.diag(covariance)
    standard_deviations = np.sqrt(variances)
    correlation_denominator = np.outer(
        standard_deviations, standard_deviations)
    correlation = np.full(covariance.shape, np.nan)
    np.divide(covariance, correlation_denominator, out=correlation,
        where=correlation_denominator > 0)

    off_diagonal = covariance - np.diag(variances)
    covariance_norm = np.linalg.norm(covariance, ord='fro')
    off_diagonal_ratio = np.nan
    if covariance_norm > 0:
        off_diagonal_ratio = np.linalg.norm(off_diagonal, ord='fro')
        off_diagonal_ratio /= covariance_norm

    fitted_covariance = np.diag(decomposition['eigenvalues'])
    fitted_norm = np.linalg.norm(fitted_covariance, ord='fro')
    relative_error = np.nan
    if fitted_norm > 0:
        difference = covariance - fitted_covariance
        relative_error = np.linalg.norm(difference, ord='fro') / fitted_norm

    upper_triangle = correlation[np.triu_indices(n_components, k=1)]
    valid_pairs = upper_triangle[np.isfinite(upper_triangle)]
    rms_correlation = np.nan
    max_correlation = np.nan
    if len(valid_pairs):
        rms_correlation = np.sqrt(np.mean(valid_pairs ** 2))
        max_correlation = np.max(np.abs(valid_pairs))
    return {
        'score_covariance': covariance,
        'score_correlation': correlation,
        'off_diagonal_ratio': float(off_diagonal_ratio),
        'relative_covariance_error': float(relative_error),
        'rms_off_diagonal_correlation': float(rms_correlation),
        'max_abs_off_diagonal_correlation': float(max_correlation),
    }


def save_svd(decomposition, filename):
    '''Save a fitted dictionary as compressed NumPy data without overwriting.

    decomposition:  dictionary returned by fit_svd or load_svd
    filename:       output path; parent directories are created as needed
    '''
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('xb') as handle:
        np.savez_compressed(handle, **decomposition)


def load_svd(filename):
    '''Reload a fitted dictionary saved by save_svd, without allowing pickle.

    filename:  path to the saved NumPy archive
    '''
    keys = ('mean', 'directions', 'singular_values', 'eigenvalues', 'n_rows')
    with np.load(filename, allow_pickle=False) as data:
        decomposition = {key: data[key] for key in keys}
    decomposition['n_rows'] = int(decomposition['n_rows'])
    return decomposition


def _as_matrix(X, min_rows=1):
    '''Convert a finite real matrix to float64 without changing its values.

    X:         sample-by-dimension array or nested sequence
    min_rows:  minimum number of rows required by the calling operation
    '''
    if np.iscomplexobj(X): raise ValueError('X must contain real values')
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < min_rows or X.shape[1] == 0:
        message = f'X must be a 2D matrix with at least {min_rows} '
        raise ValueError(message + 'rows and one dimension')
    if not np.isfinite(X).all():
        raise ValueError('X must contain only finite values')
    return X
