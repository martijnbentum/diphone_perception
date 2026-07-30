import numpy as np
import pytest

from synthetic_acoustic_probes.metrics import (
    accumulated_adjacent_cosine_scale,
    compare_frequency_scales,
    conditional_axis_monotonicity,
    cosine_distance_matrix,
    cross_validated_ridge_scores,
    local_neighbor_preservation,
    pairwise_geometry_spearman,
)


def _grid_and_rbf_representation(size=5):
    coordinates = np.array([
        (x, y)
        for x in range(size)
        for y in range(size)
    ], dtype=float)
    squared_distances = np.sum(
        (coordinates[:, None, :] - coordinates[None, :, :]) ** 2,
        axis=2,
    )
    representations = np.exp(-squared_distances / 4)
    return coordinates, representations


def test_ordered_geometry_scores_above_shuffled_control():
    coordinates, representations = _grid_and_rbf_representation()
    rng = np.random.default_rng(4)
    shuffled = representations[rng.permutation(len(representations))]
    ordered_score = pairwise_geometry_spearman(
        representations, coordinates
    )
    shuffled_score = pairwise_geometry_spearman(shuffled, coordinates)
    assert ordered_score > 0.9
    assert ordered_score > shuffled_score

    ordered_neighbors = local_neighbor_preservation(
        representations, coordinates, n_neighbors=4
    )
    shuffled_neighbors = local_neighbor_preservation(
        shuffled, coordinates, n_neighbors=4
    )
    assert ordered_neighbors > 0.8
    assert ordered_neighbors > shuffled_neighbors


def test_ridge_reports_each_axis_separately():
    coordinates, representations = _grid_and_rbf_representation()
    scores = cross_validated_ridge_scores(
        representations,
        coordinates,
        target_names=['f1', 'f2'],
        n_splits=5,
        seed=3,
    )
    assert set(scores) == {'f1', 'f2'}
    assert scores['f1']['mean_r2'] > 0.8
    assert scores['f2']['mean_r2'] > 0.8


def test_conditional_monotonicity_operates_with_other_axis_fixed():
    coordinates, representations = _grid_and_rbf_representation()
    result = conditional_axis_monotonicity(
        representations,
        coordinates,
        axis='f1',
        coordinate_names=['f1', 'f2'],
    )
    assert result['axis'] == 'f1'
    assert result['n_conditions'] == 5
    assert result['mean_spearman'] > 0.9


def test_cosine_distance_rejects_zero_vectors():
    with pytest.raises(ValueError, match='zero vectors'):
        cosine_distance_matrix([[1, 0], [0, 0]])


def test_frequency_scale_comparison_and_accumulation_are_deterministic():
    frequencies = np.array([100, 200, 400, 800, 1600], dtype=float)
    representations = np.column_stack([
        np.cos(np.log(frequencies)),
        np.sin(np.log(frequencies)),
        np.ones_like(frequencies),
    ])
    first = accumulated_adjacent_cosine_scale(
        representations, frequencies
    )
    second = accumulated_adjacent_cosine_scale(
        representations, frequencies
    )
    assert np.array_equal(
        first['normalized_cumulative_distance'],
        second['normalized_cumulative_distance'],
    )
    comparisons = compare_frequency_scales(
        representations, frequencies
    )
    assert set(comparisons['spearman_by_scale']) == {
        'hz', 'log_hz', 'mel', 'bark'
    }
