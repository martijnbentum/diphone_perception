import numpy as np
import pytest

import locations
from synthetic_acoustic_probes.f0_checkpoint import F0Checkpoint, F0Checkpoints


def _write_checkpoint(directory, model_name, **overrides):
    arrays = {
        'mean_cnn_features': np.array([
            [1.0, 0.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
            [0.0, 1.0],
        ]),
        'frequencies': np.array([10.0, 20.0, 30.0]),
        'coordinates': np.zeros((3, 2)),
        'random_state': 42,
        'metric': 'cosine',
        'model_name': model_name,
        'aggregation': 'mean',
    }
    arrays.update(overrides)
    np.savez_compressed(directory / f'{model_name}.npz', **arrays)


def test_f0_checkpoint_loads_fields(tmp_path, monkeypatch):
    '''Constructing from a model name loads and exposes its npz fields.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    model_name = 'wav2vec2_nl1_checkpoint-200000'
    _write_checkpoint(tmp_path, model_name)

    checkpoint = F0Checkpoint(model_name)

    assert checkpoint.model_name == model_name
    assert checkpoint.checkpoint_step == 200000
    assert checkpoint.aggregation == 'mean'
    assert checkpoint.umap_metric == 'cosine'
    assert checkpoint.random_state == 42
    assert checkpoint.result_path == tmp_path / f'{model_name}.npz'
    assert checkpoint.cnn.shape == (3, 2)
    assert repr(checkpoint) == f"F0Checkpoint('{model_name}', 10-30 Hz)"


def test_f0_checkpoint_missing_file_raises(tmp_path, monkeypatch):
    '''A checkpoint with no matching npz file fails explicitly.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    with pytest.raises(FileNotFoundError, match='result not found'):
        F0Checkpoint('does-not-exist')


def test_f0_checkpoint_model_name_mismatch_raises(tmp_path, monkeypatch):
    '''A stored model_name that disagrees with the request is rejected.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    _write_checkpoint(tmp_path, 'wrong-name')
    (tmp_path / 'wrong-name.npz').rename(
        tmp_path / 'wav2vec2_checkpoint-0.npz')

    with pytest.raises(ValueError, match='mismatch'):
        F0Checkpoint('wav2vec2_checkpoint-0')


def test_f0_checkpoint_distance_methods_are_cached(tmp_path, monkeypatch):
    '''Repeated calls reuse the same cached result object.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    model_name = 'wav2vec2_checkpoint-0'
    _write_checkpoint(tmp_path, model_name)
    checkpoint = F0Checkpoint(model_name)

    assert checkpoint.adjacent_distances() is checkpoint.adjacent_distances()
    assert checkpoint.pairwise_distances() is checkpoint.pairwise_distances()
    first = checkpoint.pairwise_frequency_correlation('hz')
    second = checkpoint.pairwise_frequency_correlation('hz')
    assert first is second


def test_f0_checkpoint_correlation_cache_keyed_by_max_frequency_distance(
    tmp_path, monkeypatch,
):
    '''Distinct max_frequency_distance values get distinct cache entries.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    model_name = 'wav2vec2_checkpoint-0'
    _write_checkpoint(tmp_path, model_name,
        mean_cnn_features=np.array([[1.0, 0.0], [0.9, 0.2], [0.3, 0.7],
            [-0.8, 0.4]]),
        frequencies=np.array([10.0, 20.0, 40.0, 80.0]))
    checkpoint = F0Checkpoint(model_name)

    default = checkpoint.pairwise_frequency_correlation('hz')
    narrow = checkpoint.pairwise_frequency_correlation(
        'hz', max_frequency_distance=25)
    default_again = checkpoint.pairwise_frequency_correlation('hz')

    assert default is not narrow
    assert default is default_again
    assert len(narrow[1]) < len(default[1])


def test_f0_checkpoint_sweep_pairwise_frequency_correlation(
    tmp_path, monkeypatch,
):
    '''Sweep returns one correlation per threshold, reusing the cache.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    model_name = 'wav2vec2_checkpoint-0'
    _write_checkpoint(tmp_path, model_name,
        mean_cnn_features=np.array([[1.0, 0.0], [0.9, 0.2], [0.3, 0.7],
            [-0.8, 0.4]]),
        frequencies=np.array([10.0, 20.0, 40.0, 80.0]))
    checkpoint = F0Checkpoint(model_name)

    distances, correlations = checkpoint.sweep_pairwise_frequency_correlation(
        'hz', max_frequency_distances=(25, 100))

    assert distances == (25, 100)
    assert correlations == [
        checkpoint.pairwise_frequency_correlation('hz', 25)[0],
        checkpoint.pairwise_frequency_correlation('hz', 100)[0]]


def test_f0_checkpoint_sweep_accepts_one_shot_iterable():
    '''Sweep materializes thresholds before iterating over them.'''

    checkpoint = object.__new__(F0Checkpoint)

    def correlation(scale, distance):
        return float(distance), None, None

    checkpoint.pairwise_frequency_correlation = correlation
    distances = iter((25, 100))

    returned_distances, correlations = (
        checkpoint.sweep_pairwise_frequency_correlation(
            max_frequency_distances=distances))

    assert returned_distances == (25, 100)
    assert correlations == [25.0, 100.0]


def test_f0_checkpoints_loads_all_in_checkpoint_order(tmp_path, monkeypatch):
    '''Every result under the output directory loads in checkpoint order.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    for model_name in ('wav2vec2_nl1_checkpoint-200000',
        'wav2vec2_checkpoint-0', 'wav2vec2_nl1_checkpoint-100000'):
        _write_checkpoint(tmp_path, model_name)

    checkpoints = F0Checkpoints()

    assert checkpoints.checkpoint_numbers == (0, 100000, 200000)
    assert [c.model_name for c in checkpoints.checkpoints] == [
        'wav2vec2_checkpoint-0', 'wav2vec2_nl1_checkpoint-100000',
        'wav2vec2_nl1_checkpoint-200000']
    assert repr(checkpoints) == 'F0Checkpoints(3 checkpoints, 0-200000)'


def test_f0_checkpoints_requires_results_directory(tmp_path, monkeypatch):
    '''Missing and empty result directories fail with distinct messages.'''

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='directory not found'):
        F0Checkpoints()

    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    with pytest.raises(FileNotFoundError, match='no checkpoint results'):
        F0Checkpoints()


def test_f0_checkpoints_distance_methods_pair_with_checkpoint_numbers(
    tmp_path, monkeypatch,
):
    '''Collection-level distance methods align results with step numbers.'''

    asymmetric_cnn = np.array([[1.0, 0.0], [0.9, 0.2], [0.3, 0.7],
        [-0.8, 0.4]])
    wide_frequencies = np.array([10.0, 20.0, 40.0, 80.0])
    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    _write_checkpoint(tmp_path, 'wav2vec2_checkpoint-0',
        mean_cnn_features=asymmetric_cnn, frequencies=wide_frequencies)
    _write_checkpoint(tmp_path, 'wav2vec2_nl1_checkpoint-100000',
        mean_cnn_features=asymmetric_cnn, frequencies=wide_frequencies)
    checkpoints = F0Checkpoints()

    numbers, adjacent = checkpoints.adjacent_distances()
    assert numbers == checkpoints.checkpoint_numbers
    assert len(adjacent) == 2

    numbers, correlations = checkpoints.pairwise_frequency_correlation('hz')
    assert numbers == checkpoints.checkpoint_numbers
    assert all(isinstance(value, float) for value in correlations)

    numbers, narrow_correlations = checkpoints.pairwise_frequency_correlation(
        'hz', max_frequency_distance=25)
    assert numbers == checkpoints.checkpoint_numbers
    assert all(isinstance(value, float) for value in narrow_correlations)


def test_f0_checkpoints_sweep_pairwise_frequency_correlation(
    tmp_path, monkeypatch,
):
    '''Collection-level sweep aligns step numbers, thresholds, and results.'''

    asymmetric_cnn = np.array([[1.0, 0.0], [0.9, 0.2], [0.3, 0.7],
        [-0.8, 0.4]])
    wide_frequencies = np.array([10.0, 20.0, 40.0, 80.0])
    monkeypatch.setattr(locations, 'f0_output_data', tmp_path)
    _write_checkpoint(tmp_path, 'wav2vec2_checkpoint-0',
        mean_cnn_features=asymmetric_cnn, frequencies=wide_frequencies)
    _write_checkpoint(tmp_path, 'wav2vec2_nl1_checkpoint-100000',
        mean_cnn_features=asymmetric_cnn, frequencies=wide_frequencies)
    checkpoints = F0Checkpoints()

    numbers, distances, results = (
        checkpoints.sweep_pairwise_frequency_correlation(
            'hz', max_frequency_distances=(25, 100))
    )

    assert numbers == checkpoints.checkpoint_numbers
    assert distances == (25, 100)
    assert len(results) == 2
    assert all(len(per_checkpoint) == 2 for per_checkpoint in results)
