import numpy as np
import pytest

from synthetic_acoustic_probes.f0_distances import (
    f0_adjacent_distances,
    f0_pairwise_distances,
    f0_pairwise_frequency_correlation,
)


def test_f0_adjacent_distances_sorts_and_computes_edges():
    '''Adjacent distances are ordered by frequency with matching edges.'''

    cnn = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    frequencies = np.array([30.0, 10.0, 20.0])

    result = f0_adjacent_distances(cnn, frequencies)

    np.testing.assert_array_equal(
        result['frequencies_hz'], [10.0, 20.0, 30.0]
    )
    np.testing.assert_array_equal(
        result['frequency_edges_hz'], [[10.0, 20.0], [20.0, 30.0]]
    )
    np.testing.assert_allclose(result['distances'], [2.0, 1.0])


def test_f0_pairwise_distances_returns_symmetric_sorted_matrix():
    '''Pairwise distances form a symmetric matrix ordered by frequency.'''

    cnn = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    frequencies = np.array([30.0, 10.0, 20.0])

    result = f0_pairwise_distances(cnn, frequencies)

    np.testing.assert_array_equal(
        result['frequencies_hz'], [10.0, 20.0, 30.0]
    )
    np.testing.assert_allclose(
        result['distances'], [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0],
            [1.0, 2.0, 0.0]]
    )


def test_f0_pairwise_frequency_correlation_computes_expected_value():
    '''Correlation matches a hand-checked value for a small example.'''

    angles = np.array([0.0, 0.3, 0.6, 0.9])
    cnn = np.column_stack((np.cos(angles), np.sin(angles)))
    frequencies = np.array([10.0, 20.0, 40.0, 80.0])

    correlation, cnn_distances, frequency_distances = (
        f0_pairwise_frequency_correlation(cnn, frequencies)
    )

    assert correlation == pytest.approx(0.9276336570439175)
    assert cnn_distances.shape == (6,)
    assert frequency_distances.shape == (6,)


@pytest.mark.parametrize('f0_function', (
    f0_adjacent_distances, f0_pairwise_distances,
    f0_pairwise_frequency_correlation,
))
def test_f0_distance_functions_reject_non_2d_cnn(f0_function):
    '''Every distance function validates cnn through the same helper.'''

    with pytest.raises(ValueError, match='two-dimensional'):
        f0_function([1.0, 2.0], [10.0, 20.0])


def test_f0_pairwise_frequency_correlation_rejects_constant_cnn():
    '''Identical cnn rows give constant distances and fail explicitly.'''

    cnn = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    frequencies = np.array([10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match='cnn distances are constant'):
        f0_pairwise_frequency_correlation(cnn, frequencies)


def test_f0_pairwise_frequency_correlation_excludes_far_pairs():
    '''max_frequency_distance drops pairs beyond the Hz threshold.'''

    angles = np.array([0.0, 0.3, 0.6, 6000.0])
    cnn = np.column_stack((np.cos(angles), np.sin(angles)))
    frequencies = np.array([10.0, 20.0, 40.0, 9000.0])

    correlation, cnn_distances, frequency_distances = (
        f0_pairwise_frequency_correlation(
            cnn, frequencies, max_frequency_distance=100)
    )

    # Only the 3 pairs among the first three (close) samples remain.
    assert cnn_distances.shape == (3,)
    assert frequency_distances.shape == (3,)
    assert np.isfinite(correlation)


def test_f0_pairwise_frequency_correlation_default_matches_unfiltered():
    '''The default max_frequency_distance is a no-op for small Hz spans.'''

    angles = np.array([0.0, 0.3, 0.6, 0.9])
    cnn = np.column_stack((np.cos(angles), np.sin(angles)))
    frequencies = np.array([10.0, 20.0, 40.0, 80.0])

    correlation, cnn_distances, frequency_distances = (
        f0_pairwise_frequency_correlation(cnn, frequencies)
    )

    assert correlation == pytest.approx(0.9276336570439175)
    assert cnn_distances.shape == (6,)
    assert frequency_distances.shape == (6,)


def test_f0_pairwise_frequency_correlation_filters_in_hz_not_scale():
    '''Filtering uses raw Hz distance even when scale is not hz.'''

    cnn = np.array([[1.0, 0.0], [0.9, 0.2], [0.3, 0.7], [-0.8, 0.4]])
    frequencies = np.array([10.0, 20.0, 40.0, 80.0])

    hz_result = f0_pairwise_frequency_correlation(
        cnn, frequencies, scale='hz', max_frequency_distance=25)
    mel_result = f0_pairwise_frequency_correlation(
        cnn, frequencies, scale='mel', max_frequency_distance=25)

    # Mel distances for these frequencies are much smaller than Hz
    # distances, so if filtering used mel distances no pairs would be
    # excluded; the mask (and thus shape) must match the hz case.
    assert mel_result[1].shape == hz_result[1].shape


def test_f0_pairwise_frequency_correlation_rejects_too_few_pairs():
    '''Too tight a max_frequency_distance leaves fewer than two pairs.'''

    cnn = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    frequencies = np.array([10.0, 5000.0, 9000.0])

    with pytest.raises(ValueError, match='fewer than two pairs remain'):
        f0_pairwise_frequency_correlation(
            cnn, frequencies, max_frequency_distance=100)
