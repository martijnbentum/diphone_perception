from pathlib import Path

import numpy as np
import pytest

import locations
from synthetic_acoustic_probes.f0_metrics import (
    f0_checkpoint_metrics,
    f0_smoothness_metrics,
    load_f0_checkpoint_metrics,
)


def _write_result(path, model_name, **overrides):
    arrays = {
        'mean_cnn_features': np.array([
            [1.0, 0.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
            [0.0, 1.0],
        ]),
        'frequencies': np.array([10.0, 20.0, 30.0]),
        'random_state': 42,
        'metric': 'cosine',
        'model_name': model_name,
        'aggregation': 'mean',
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)


def test_f0_smoothness_metrics_sorts_and_summarizes_edges():
    '''Smoothness metrics retain ordered edges and summarize their tail.'''

    root_three_over_two = np.sqrt(3) / 2
    representations = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [root_three_over_two, 0.5],
    ])
    frequencies = np.array([30.0, 10.0, 40.0, 20.0])
    thresholds = (0.1, 0.5, 0.9)
    expected_distances = np.array([
        1 - root_three_over_two,
        0.5,
        1.0,
    ])

    result = f0_smoothness_metrics(
        representations,
        frequencies,
        thresholds=thresholds,
    )

    np.testing.assert_array_equal(
        result['frequencies_hz'],
        [10.0, 20.0, 30.0, 40.0],
    )
    np.testing.assert_array_equal(
        result['frequency_edges_hz'],
        [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]],
    )
    np.testing.assert_allclose(
        result['adjacent_distances'],
        expected_distances,
    )
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(expected_distances) / expected_distances.sum(),
    ))
    np.testing.assert_allclose(
        result['normalized_cumulative_distance'],
        cumulative,
    )
    assert result['n_stimuli'] == 4
    assert result['n_edges'] == 3
    assert result['mean'] == pytest.approx(expected_distances.mean())
    assert result['median'] == pytest.approx(np.median(expected_distances))
    assert result['p95'] == pytest.approx(
        np.quantile(expected_distances, 0.95)
    )
    assert result['p99'] == pytest.approx(
        np.quantile(expected_distances, 0.99)
    )
    assert result['maximum'] == pytest.approx(1.0)
    assert result['total_path_length'] == pytest.approx(
        expected_distances.sum()
    )
    np.testing.assert_array_equal(result['thresholds'], thresholds)
    np.testing.assert_allclose(
        result['fractions_above_threshold'],
        [1.0, 1 / 3, 1 / 3],
    )


@pytest.mark.parametrize(
    ('representations', 'frequencies', 'message'),
    (
        ([1.0, 2.0], [10.0, 20.0], 'two-dimensional'),
        ([[1.0, 0.0], [0.0, 1.0]], [10.0], 'one frequency'),
        ([[1.0, 0.0], [0.0, 1.0]], [10.0, np.nan], 'finite'),
        ([[1.0, 0.0], [0.0, 1.0]], [10.0, 10.0], 'unique'),
        ([[1.0, 0.0], [0.0, 0.0]], [10.0, 20.0], 'zero vectors'),
    ),
)
def test_f0_smoothness_metrics_rejects_invalid_inputs(
    representations,
    frequencies,
    message,
):
    '''Invalid representations and frequency grids fail explicitly.'''

    with pytest.raises(ValueError, match=message):
        f0_smoothness_metrics(representations, frequencies)


@pytest.mark.parametrize(
    ('thresholds', 'message'),
    (
        ((), 'non-empty'),
        ((0.1, np.nan), 'finite'),
        ((0.0, 0.1), 'strictly between'),
        ((0.1, 2.0), 'strictly between'),
        ((0.1, 0.1), 'unique'),
        (0.1, 'numeric'),
    ),
)
def test_f0_smoothness_metrics_rejects_invalid_thresholds(
    thresholds,
    message,
):
    '''Thresholds must be a unique finite vector inside cosine range.'''

    with pytest.raises(ValueError, match=message):
        f0_smoothness_metrics(
            [[1.0, 0.0], [0.0, 1.0]],
            [10.0, 20.0],
            thresholds=thresholds,
        )


def test_f0_checkpoint_metrics_loads_arrays_and_metadata(tmp_path):
    '''One checkpoint bundle produces summaries and provenance metadata.'''

    model_name = 'wav2vec2_nl1_checkpoint-200000'
    result_path = tmp_path / f'{model_name}.npz'
    _write_result(result_path, model_name)

    result = f0_checkpoint_metrics(
        result_path,
        thresholds=(0.2, 0.4),
    )

    assert result['model_name'] == model_name
    assert result['checkpoint_step'] == 200000
    assert result['aggregation'] == 'mean'
    assert result['metric'] == 'cosine'
    assert result['random_state'] == 42
    assert result['result_path'] == result_path
    np.testing.assert_array_equal(result['thresholds'], [0.2, 0.4])
    np.testing.assert_allclose(
        result['fractions_above_threshold'],
        [1.0, 0.0],
    )


def test_f0_checkpoint_metrics_rejects_missing_and_mislabeled_results(
    tmp_path,
):
    '''Missing fields and filename/model mismatches fail before reporting.'''

    missing_path = tmp_path / 'wav2vec2_checkpoint-0.npz'
    np.savez_compressed(missing_path, model_name='wav2vec2_checkpoint-0')
    with pytest.raises(ValueError, match='missing fields'):
        f0_checkpoint_metrics(missing_path)

    mismatch_path = tmp_path / 'wrong-name.npz'
    _write_result(mismatch_path, 'wav2vec2_checkpoint-0')
    with pytest.raises(ValueError, match='filename does not match'):
        f0_checkpoint_metrics(mismatch_path)

    with pytest.raises(FileNotFoundError, match='result not found'):
        f0_checkpoint_metrics(tmp_path / 'absent.npz')


def test_load_f0_checkpoint_metrics_returns_checkpoint_order(
    tmp_path,
    monkeypatch,
):
    '''Directory loading sorts model bundles by numeric checkpoint step.'''

    model_names = (
        'wav2vec2_nl1_checkpoint-200000',
        'wav2vec2_checkpoint-0',
        'wav2vec2_nl1_checkpoint-100000',
    )
    for model_name in model_names:
        _write_result(tmp_path / f'{model_name}.npz', model_name)
    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)

    results = load_f0_checkpoint_metrics()

    assert [result['checkpoint_step'] for result in results] == [
        0, 100000, 200000
    ]
    assert [result['model_name'] for result in results] == [
        'wav2vec2_checkpoint-0',
        'wav2vec2_nl1_checkpoint-100000',
        'wav2vec2_nl1_checkpoint-200000',
    ]


def test_load_f0_checkpoint_metrics_requires_results_directory(
    tmp_path,
    monkeypatch,
):
    '''Directory loading distinguishes missing and empty result locations.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='directory not found'):
        load_f0_checkpoint_metrics()

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    with pytest.raises(FileNotFoundError, match='no F0 checkpoint results'):
        load_f0_checkpoint_metrics()
