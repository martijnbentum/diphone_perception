'''Compare held-out SVD mode variance with a random-direction reference.'''

import numpy as np

from .svd import _as_matrix


def significant_modes(X, decomposition, alpha=0.01, n_random=100_000,
    n_components=None, rng=None):
    '''Find fitted modes with more held-out variance than random directions.

    X:              evaluation matrix with at least two rows
    decomposition:  dictionary returned by fit_svd or load_svd
    alpha:          target false discovery rate for BH correction
    n_random:       positive number of random directions for the reference
    n_components:   number of leading modes to test; None tests all modes
    rng:            NumPy Generator; None creates a fresh generator

    Return significant modes in descending variance-fraction order. Each result
    contains its zero-based mode_index, fraction, raw p-value, and adjusted q.
    The Monte Carlo p-value cannot be smaller than 1 / (n_random + 1).
    X must contain embeddings that were not used to fit decomposition.
    '''
    directions = decomposition['directions']
    covariance = _evaluation_covariance(X, directions.shape[0])
    total = np.trace(covariance)
    fractions = np.zeros(directions.shape[1])
    if total > 0:
        projected = covariance @ directions
        fractions = np.sum(directions * projected, axis=0) / total
    if n_components is not None:
        if not isinstance(n_components, (int, np.integer)):
            raise ValueError('n_components must be an integer or None')
        if not 1 <= n_components <= len(fractions):
            message = 'n_components is outside the fitted component range'
            raise ValueError(message)
        fractions = fractions[:n_components]

    null_samples = _random_direction_fractions(covariance, n_random, rng)
    p_values = empirical_p_values(fractions, null_samples)
    rejected, adjusted_p = benjamini_hochberg(p_values, alpha)

    indices = np.where(rejected)[0]
    order = indices[np.argsort(-fractions[indices])]
    modes = []
    for index in order:
        modes.append({'mode_index': int(index),
            'fraction': float(fractions[index]), 'p': float(p_values[index]),
            'q': float(adjusted_p[index])})
    return modes


def compare_speech_non_speech(X_speech, X_non_speech, decomposition):
    '''Compare held-out speech and non-speech scores along each fitted mode.

    X_speech:      speech matrix with at least two rows
    X_non_speech:  non-speech matrix with at least two rows
    decomposition:  dictionary returned by fit_svd or load_svd

    Both matrices must contain embeddings that were not used to fit
    decomposition. Return arrays in fitted mode order. mean_difference is the
    non-speech mean score minus the speech mean score. effect_size divides that
    difference by the square root of the mean of the two sample variances.
    It is NaN where both groups have zero variance. These are descriptive
    differences, not significance tests.
    '''
    directions = decomposition['directions']
    X_speech = _as_matrix(X_speech, min_rows=2)
    X_non_speech = _as_matrix(X_non_speech, min_rows=2)
    dimensions = directions.shape[0]
    if X_speech.shape[1] != dimensions or X_non_speech.shape[1] != dimensions:
        message = 'matrix dimensions must match the fitted decomposition'
        raise ValueError(message)

    mean_difference = (X_non_speech.mean(axis=0) -
        X_speech.mean(axis=0)) @ directions
    speech_covariance = _evaluation_covariance(X_speech, dimensions)
    non_speech_covariance = _evaluation_covariance(X_non_speech, dimensions)
    speech_projection = speech_covariance @ directions
    non_speech_projection = non_speech_covariance @ directions
    speech_variance = np.sum(directions * speech_projection, axis=0)
    speech_variance = np.maximum(speech_variance, 0)
    non_speech_variance = np.sum(directions * non_speech_projection, axis=0)
    non_speech_variance = np.maximum(non_speech_variance, 0)
    pooled_sd = np.sqrt((speech_variance + non_speech_variance) / 2)
    effect_size = np.full_like(mean_difference, np.nan)
    np.divide(mean_difference, pooled_sd, out=effect_size, where=pooled_sd > 0)
    return {'mean_difference': mean_difference, 'effect_size': effect_size,
        'speech_variance': speech_variance,
        'non_speech_variance': non_speech_variance,
        'n_speech': len(X_speech), 'n_non_speech': len(X_non_speech)}


def format_report(modes):
    '''Format significant_modes results as printable lines.

    modes:  list of result dictionaries from significant_modes
    '''
    lines = []
    for mode in modes:
        line = f"mode {mode['mode_index']}: fraction={mode['fraction']:.4f} "
        line += f"p={mode['p']:.4g} q={mode['q']:.4g}"
        lines.append(line)
    return lines


def random_direction_fractions(X, decomposition, n_random=100_000, rng=None):
    '''Draw held-out variance fractions for random unit directions.

    X:              evaluation matrix with at least two rows
    decomposition:  fitted decomposition; its ambient dimension is used
    n_random:       positive number of random directions to draw
    rng:            NumPy Generator; None creates a fresh generator

    Fractions use the same total-variance convention as evaluate_svd.
    Covariance eigenvalues give the exact random-direction distribution without
    constructing a sample-by-direction projection matrix. Draws are batched.
    Zero evaluation variance gives zero fractions.
    '''
    dimensions = decomposition['directions'].shape[0]
    covariance = _evaluation_covariance(X, dimensions)
    return _random_direction_fractions(covariance, n_random, rng)


def empirical_p_values(observed, null_samples):
    '''Compute one-sided p-values for observations against a shared reference.

    observed:      one-dimensional array of statistics to test
    null_samples:  nonempty one-dimensional array of reference draws

    Count draws at least as large as each observation, with a +1 correction.
    '''
    observed = np.asarray(observed)
    null_samples = np.asarray(null_samples)
    if observed.ndim != 1 or null_samples.ndim != 1 or not len(null_samples):
        message = 'observed and null_samples must be 1D; '
        message += 'null_samples must be nonempty'
        raise ValueError(message)
    sorted_null = np.sort(null_samples)
    first_matches = np.searchsorted(sorted_null, observed, side='left')
    counts = len(sorted_null) - first_matches
    return (counts + 1) / (len(null_samples) + 1)


def benjamini_hochberg(p_values, alpha=0.01):
    '''Adjust p-values using Benjamini-Hochberg false discovery control.

    p_values:  one-dimensional array of raw p-values, one per mode
    alpha:     target false discovery rate, strictly between zero and one

    Return (rejected, adjusted_p) arrays in the input order.
    '''
    p_values = np.asarray(p_values)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError('p_values must be a finite 1D array')
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError('p_values must be between zero and one')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    n = len(p_values)
    if n == 0: return np.empty(0, dtype=bool), np.empty(0)

    order = np.argsort(p_values)
    ranked = p_values[order]
    ranks = np.arange(1, n + 1)

    adjusted_sorted = ranked * n / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    below_threshold = ranked <= (ranks / n) * alpha
    rejected_sorted = np.zeros(n, dtype=bool)
    if below_threshold.any():
        cutoff = np.max(np.where(below_threshold)[0])
        rejected_sorted[:cutoff + 1] = True

    rejected = np.empty(n, dtype=bool)
    adjusted_p = np.empty(n)
    rejected[order] = rejected_sorted
    adjusted_p[order] = adjusted_sorted
    return rejected, adjusted_p


def _evaluation_covariance(X, dimensions, batch_size=2048):
    '''Accumulate the held-out covariance without copying the full matrix.'''
    X = _as_matrix(X, min_rows=2)
    if X.shape[1] != dimensions:
        raise ValueError('X dimensions must match the fitted decomposition')
    mean = X.mean(axis=0)
    covariance = np.zeros((dimensions, dimensions))
    for start in range(0, len(X), batch_size):
        centered = X[start:start + batch_size] - mean
        covariance += centered.T @ centered
    return covariance / (len(X) - 1)


def _random_direction_fractions(covariance, n_random, rng, batch_size=2048):
    '''Sample unit directions in the covariance eigenbasis in batches.'''
    if not isinstance(n_random, (int, np.integer)) or n_random < 1:
        raise ValueError('n_random must be a positive integer')
    fractions = np.zeros(n_random)
    total = np.trace(covariance)
    if total == 0: return fractions
    if rng is None: rng = np.random.default_rng()
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    dimensions = len(eigenvalues)
    for start in range(0, n_random, batch_size):
        count = min(batch_size, n_random - start)
        normal_squared = rng.normal(size=(count, dimensions))
        normal_squared *= normal_squared
        numerator = normal_squared @ eigenvalues
        denominator = normal_squared.sum(axis=1) * total
        fractions[start:start + count] = numerator / denominator
    return fractions
