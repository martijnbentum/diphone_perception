'''Frequency distance computation for F0 trajectories.'''

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .metrics import (accumulated_adjacent_cosine_scale,
    cosine_distance_matrix, frequency_scale)


def f0_adjacent_distances(cnn, frequencies_hz):
    '''Compute cosine distances between frequency-sorted neighbor samples.
    cnn:              Samples by CNN features.
    frequencies_hz:   One positive, unique frequency per sample.
    Returns a dict with frequencies_hz (sorted), frequency_edges_hz (each
    row the Hz pair spanned by one distance), and distances.
    '''
    cnn = _as_2d_array(cnn, 'cnn')
    frequencies = _validated_frequencies(frequencies_hz, cnn.shape[0])
    scale = accumulated_adjacent_cosine_scale(cnn, frequencies)
    ordered_frequencies = scale['frequencies_hz']
    frequency_edges = np.column_stack((ordered_frequencies[:-1],
        ordered_frequencies[1:]))
    result = {'frequencies_hz': ordered_frequencies,
        'frequency_edges_hz': frequency_edges,
        'distances': scale['adjacent_distances']}
    return result


def f0_pairwise_distances(cnn, frequencies_hz):
    '''Compute cosine distances between every pair of samples.
    cnn:              Samples by CNN features.
    frequencies_hz:   One positive, unique frequency per sample.
    Returns a dict with frequencies_hz (sorted) and distances, an n by n
    matrix of pairwise cosine distances in frequency-sorted order.
    '''
    cnn = _as_2d_array(cnn, 'cnn')
    frequencies = _validated_frequencies(frequencies_hz, cnn.shape[0])
    order = np.argsort(frequencies, kind='stable')
    distances = cosine_distance_matrix(cnn[order])
    result = {'frequencies_hz': frequencies[order], 'distances': distances}
    return result


def f0_pairwise_frequency_correlation(cnn, frequencies_hz, scale='hz',
        max_frequency_distance=8000):
    '''Spearman-correlate pairwise cosine distance with frequency distance.
    cnn:                     Samples by CNN features.
    frequencies_hz:          One positive, unique frequency per sample.
    scale:                   Frequency scale for the target distance: hz,
                             log_hz, mel, or bark.
    max_frequency_distance:  Maximum allowed frequency distance in Hz
                             between a pair; pairs exceeding it (default
                             8000) are excluded, regardless of scale.
    Returns their Spearman rank correlation, then cnn_distances and
    frequency_distances (condensed pairwise vectors, after exclusion).
    '''
    cnn = _as_2d_array(cnn, 'cnn')
    frequencies = _validated_frequencies(frequencies_hz, cnn.shape[0])
    max_frequency_distance = _validated_max_frequency_distance(
        max_frequency_distance)
    scaled_frequencies = frequency_scale(frequencies, scale)
    # Condensed, unsorted form of cosine_distance_matrix; correlation is
    # pairwise and order-invariant, so no sort or full matrix is needed.
    cnn_distances = pdist(cnn, metric='cosine')
    frequency_distances = pdist(scaled_frequencies[:, None])
    # Filtering is always in raw Hz, independent of scale, since a mel/
    # bark/log_hz distance is not comparable to max_frequency_distance.
    hz_distances = pdist(frequencies[:, None])
    mask = hz_distances <= max_frequency_distance
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            'fewer than two pairs remain within max_frequency_distance')
    cnn_distances = cnn_distances[mask]
    frequency_distances = frequency_distances[mask]
    _validated_nonconstant(cnn_distances, 'cnn')
    _validated_nonconstant(frequency_distances, 'frequency')
    result = spearmanr(frequency_distances, cnn_distances)
    correlation = float(result.statistic)
    return correlation, cnn_distances, frequency_distances


def _as_2d_array(values, name):
    '''Validate values as a two-dimensional array.'''
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional array')
    return values


def _validated_nonconstant(distances, name):
    '''Validate that a condensed distance vector is not constant.'''
    if np.ptp(distances) == 0:
        raise ValueError(f'{name} distances are constant')


def _validated_max_frequency_distance(value):
    '''Validate max_frequency_distance as a finite, positive number.'''
    if not np.isfinite(value) or value <= 0:
        raise ValueError('max_frequency_distance must be finite and '
            'positive')
    return value


def _validated_frequencies(values, expected_length):
    '''Validate a per-sample array of finite, unique frequencies.'''
    try:
        frequencies = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('frequencies_hz must be numeric') from error
    if frequencies.shape != (expected_length,):
        raise ValueError('one frequency is required per cnn sample')
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError('frequencies_hz must be finite and positive')
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError('frequencies_hz must be unique')
    return frequencies
