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
