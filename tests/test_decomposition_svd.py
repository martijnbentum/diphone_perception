'''Numerical behavior of centered SVD and saved decompositions.'''

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from decomposition import svd


@pytest.mark.parametrize('shape', [(8, 3), (3, 8)])
def test_fit_reconstructs_data_and_preserves_covariance(shape):
    rng = np.random.default_rng(42)
    X = rng.normal(size=shape) + 10
    original = X.copy()
    fitted = svd.fit_svd(X)
    directions = fitted['directions']
    n_components = min(shape[0] - 1, shape[1])
    assert directions.shape == (shape[1], n_components)
    scores = svd.transform(X, fitted)
    assert_allclose(scores @ directions.T + fitted['mean'], X)
    identity = np.eye(n_components)
    assert_allclose(directions.T @ directions, identity, atol=1e-14)
    covariance = directions @ np.diag(fitted['eigenvalues']) @ directions.T
    expected_covariance = np.cov(X, rowvar=False)
    assert_allclose(covariance, expected_covariance, atol=1e-14)
    assert np.all(np.diff(fitted['singular_values']) <= 0)
    assert_array_equal(X, original)


def test_transform_uses_fitting_mean_and_selects_leading_components():
    fitted = svd.fit_svd([[8, 5], [12, 5], [10, 4], [10, 6]])
    X = np.array([[13, 7], [15, 9]])
    scores = svd.transform(X, fitted)
    assert_allclose(scores @ fitted['directions'].T, [[3, 2], [5, 4]])
    n_components = np.int64(1)
    leading = svd.transform(X, fitted, n_components=n_components)
    assert leading.shape == (2, 1)
    assert_allclose(leading, scores[:, :1])


def test_spectrum_has_expected_variance_and_participation_ratio():
    fitted = svd.fit_svd([[-3, 0], [3, 0], [0, -1], [0, 1]])
    summary = svd.summarize_spectrum(fitted)
    assert_allclose(fitted['eigenvalues'], [6, 2 / 3])
    assert summary['total_variance'] == pytest.approx(20 / 3)
    assert_allclose(summary['variance_fractions'], [0.9, 0.1])
    assert_allclose(summary['cumulative_variance'], [0.9, 1])
    assert summary['n_components_90'] == 1
    assert summary['n_components_95'] == 2
    assert summary['participation_ratio'] == pytest.approx(1 / 0.82)
    assert summary['rank_ceiling'] == 2
    assert summary['n_rows'] == 4


def test_constant_data_has_zero_spectrum():
    fitted = svd.fit_svd(np.full((3, 5), 7))
    summary = svd.summarize_spectrum(fitted)
    assert fitted['directions'].shape == (5, 2)
    assert_array_equal(fitted['eigenvalues'], [0, 0])
    assert_array_equal(summary['variance_fractions'], [0, 0])
    assert_array_equal(summary['cumulative_variance'], [0, 0])
    assert summary['total_variance'] == 0
    assert summary['n_components_90'] == summary['n_components_95'] == 0
    assert summary['participation_ratio'] == 0
    assert summary['rank_ceiling'] == 2


def test_evaluation_separates_mean_shift_from_variance():
    fitted = svd.fit_svd([[-2, 0], [2, 0], [0, -1], [0, 1]])
    X = np.array([[-1, -2], [1, 2], [-1, 2], [1, -2]])
    baseline = svd.evaluate_svd(X, fitted)
    shifted = svd.evaluate_svd(X + [10, 20], fitted)
    assert_allclose(shifted['component_variances'], [4 / 3, 16 / 3])
    assert_allclose(shifted['variance_fractions'], [0.2, 0.8])
    assert_allclose(shifted['component_variances'],
        baseline['component_variances'])
    assert shifted['total_variance'] == pytest.approx(20 / 3)
    assert_allclose(shifted['mean_shift'], [10, 20])
    reconstructed = shifted['score_mean_shift'] @ fitted['directions'].T
    assert_allclose(reconstructed, [10, 20])


def test_evaluation_variance_outside_fitted_basis_is_not_captured():
    fitted = svd.fit_svd([[-1, 0], [1, 0]])
    evaluation = svd.evaluate_svd([[0, -2], [0, 2]], fitted)
    assert evaluation['total_variance'] == 8
    assert_allclose(evaluation['component_variances'], [0], atol=1e-14)
    assert_allclose(evaluation['variance_fractions'], [0], atol=1e-14)


def test_constant_evaluation_reports_shift_and_zero_variance():
    fitted = svd.fit_svd([[-1, 0], [1, 0]])
    evaluation = svd.evaluate_svd([[3, 4], [3, 4]], fitted)
    assert evaluation['total_variance'] == 0
    assert_array_equal(evaluation['variance_fractions'], [0])
    assert_allclose(evaluation['mean_shift'], [3, 4])


def test_covariance_evaluation_separates_mean_shift_and_detects_correlation():
    '''Mean shifts leave score covariance and correlation unchanged.'''
    fitted = svd.fit_svd([[-2, 0], [2, 0], [0, -1], [0, 1]])
    scores = np.array([[-2, -1], [0, 0], [2, 1]])
    X = scores @ fitted['directions'].T + fitted['mean']
    baseline = svd.evaluate_covariance(X, fitted)
    shifted = svd.evaluate_covariance(X + [10, 20], fitted)
    expected = np.array([[4, 2], [2, 1]])

    assert_allclose(baseline['score_covariance'], expected)
    assert_allclose(shifted['score_covariance'], expected)
    expected_correlation = np.ones((2, 2))
    assert_allclose(baseline['score_correlation'], expected_correlation)
    assert baseline['off_diagonal_ratio'] == pytest.approx(np.sqrt(8) / 5)
    fitted_covariance = np.diag(fitted['eigenvalues'])
    relative_error = np.linalg.norm(expected - fitted_covariance, ord='fro')
    relative_error /= np.linalg.norm(fitted_covariance, ord='fro')
    observed_error = baseline['relative_covariance_error']
    assert observed_error == pytest.approx(relative_error)
    assert baseline['rms_off_diagonal_correlation'] == pytest.approx(1)
    assert baseline['max_abs_off_diagonal_correlation'] == pytest.approx(1)


def test_covariance_evaluation_marks_zero_variance_correlations_undefined():
    '''A constant component makes its correlation row undefined.'''
    fitted = svd.fit_svd([[-2, 0], [2, 0], [0, -1], [0, 1]])
    scores = np.array([[-1, 0], [0, 0], [1, 0]])
    X = scores @ fitted['directions'].T + fitted['mean']

    result = svd.evaluate_covariance(X, fitted)

    assert_allclose(result['score_covariance'], [[1, 0], [0, 0]])
    assert result['score_correlation'][0, 0] == pytest.approx(1)
    assert np.isnan(result['score_correlation'][0, 1:]).all()
    assert np.isnan(result['score_correlation'][1]).all()
    assert result['off_diagonal_ratio'] == 0
    assert np.isnan(result['rms_off_diagonal_correlation'])
    assert np.isnan(result['max_abs_off_diagonal_correlation'])


def test_covariance_evaluation_handles_zero_denominators():
    '''Zero fitting and evaluation covariances have undefined ratios.'''
    fit_matrix = np.ones((3, 2))
    eval_matrix = np.ones((4, 2))
    fitted = svd.fit_svd(fit_matrix)

    result = svd.evaluate_covariance(eval_matrix, fitted)

    expected = np.zeros((2, 2))
    assert_array_equal(result['score_covariance'], expected)
    assert np.isnan(result['score_correlation']).all()
    assert np.isnan(result['off_diagonal_ratio'])
    assert np.isnan(result['relative_covariance_error'])
    assert np.isnan(result['rms_off_diagonal_correlation'])
    assert np.isnan(result['max_abs_off_diagonal_correlation'])


def test_covariance_evaluation_one_component_retains_matrix_shape():
    '''A one-component result stays 2D and has no pairwise summary.'''
    fitted = svd.fit_svd([[-1], [1]])

    with pytest.warns(UserWarning, match='one component'):
        result = svd.evaluate_covariance([[-1], [1]], fitted)

    assert_array_equal(result['score_covariance'], [[2]])
    assert_allclose(result['score_correlation'], [[1]])
    assert result['off_diagonal_ratio'] == 0
    assert result['relative_covariance_error'] == pytest.approx(0, abs=1e-14)
    assert np.isnan(result['rms_off_diagonal_correlation'])
    assert np.isnan(result['max_abs_off_diagonal_correlation'])


def test_covariance_evaluation_warns_when_fitted_basis_is_incomplete():
    '''Warn when the fitted score space omits embedding directions.'''
    fitted = svd.fit_svd([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    with pytest.warns(UserWarning, match='do not span'):
        result = svd.evaluate_covariance([[2, 0, 0], [0, 2, 0]], fitted)

    assert result['score_covariance'].shape == (2, 2)


@pytest.mark.parametrize('X', [[], [1, 2], [[1, 2]], [[], []],
    [[0], [np.nan]], [[0], [np.inf]], [[0], [1j]]])
def test_fit_rejects_invalid_matrices(X):
    with pytest.raises(ValueError):
        svd.fit_svd(X)


@pytest.mark.parametrize('n_components', [0, -1, 3, 1.5, '1'])
def test_transform_rejects_invalid_component_counts(n_components):
    fitted = svd.fit_svd([[0, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        svd.transform([[1, 2]], fitted, n_components=n_components)


@pytest.mark.parametrize('operation', [svd.transform, svd.evaluate_svd])
@pytest.mark.parametrize('X', [[[1, 2, 3], [4, 5, 6]], [[np.nan, 0], [0, 1]]])
def test_projection_rejects_invalid_evaluation_data(operation, X):
    fitted = svd.fit_svd([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        operation(X, fitted)


def test_evaluation_requires_two_rows():
    fitted = svd.fit_svd([[0], [1]])
    with pytest.raises(ValueError):
        svd.evaluate_svd([[2]], fitted)


def test_save_load_preserves_results_and_refuses_overwrite(tmp_path):
    fitted = svd.fit_svd([[1, 2], [3, 4], [5, 8]])
    filename = tmp_path / 'nested' / 'decomposition.npz'
    svd.save_svd(fitted, filename)
    loaded = svd.load_svd(filename)
    for key in fitted:
        assert_array_equal(loaded[key], fitted[key])
    assert isinstance(loaded['n_rows'], int)
    original_scores = svd.transform([[9, 10]], fitted)
    loaded_scores = svd.transform([[9, 10]], loaded)
    assert_allclose(loaded_scores, original_scores)
    original = filename.read_bytes()
    with pytest.raises(FileExistsError):
        svd.save_svd(fitted, filename)
    assert filename.read_bytes() == original


def test_load_rejects_pickled_arrays(tmp_path):
    filename = tmp_path / 'object.npz'
    mean = np.array([{}], dtype=object)
    np.savez(filename, mean=mean)
    with pytest.raises(ValueError, match='Object arrays'):
        svd.load_svd(filename)
