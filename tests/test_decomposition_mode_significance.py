'''Check held-out SVD mode comparisons and significance reporting.'''

import numpy as np
import pytest

from decomposition import mode_significance


def test_empirical_p_values_include_ties_and_one_draw_correction():
    '''Count null draws at least as large as each observed value.'''
    observed = np.array([1.0, 2.0, 4.0])
    draws = np.array([1.0, 2.0, 2.0, 3.0])

    result = mode_significance.empirical_p_values(observed, draws)

    np.testing.assert_array_equal(result, [1.0, 0.8, 0.2])


def test_bh_adjustment_preserves_input_order():
    '''Reject the two small p-values and return their adjusted values.'''
    p_values = np.array([0.5, 0.001, 0.02])

    rejected, adjusted = mode_significance.benjamini_hochberg(
        p_values, alpha=0.05)

    np.testing.assert_array_equal(rejected, [False, True, True])
    np.testing.assert_allclose(adjusted, [0.5, 0.003, 0.03])


def test_compare_speech_non_speech_reports_direction_and_zero_variance():
    '''Use non-speech minus speech; leave zero-variance effects undefined.'''
    speech = np.array([[0.0, 0.0], [2.0, 0.0]])
    non_speech = np.array([[4.0, 0.0], [6.0, 0.0]])
    decomposition = {'directions': np.eye(2)}

    result = mode_significance.compare_speech_non_speech(
        speech, non_speech, decomposition)

    np.testing.assert_array_equal(result['mean_difference'], [4.0, 0.0])
    np.testing.assert_array_equal(result['speech_variance'], [2.0, 0.0])
    assert result['effect_size'][0] == pytest.approx(4 / np.sqrt(2))
    assert np.isnan(result['effect_size'][1])
    assert result['n_speech'] == result['n_non_speech'] == 2


def test_significant_modes_selects_only_concentrated_held_out_mode():
    '''A rank-one evaluation matrix stands out against random directions.'''
    X = np.array([[-10.0, 0, 0], [10.0, 0, 0], [0.0, 0, 0]])
    decomposition = {'directions': np.eye(3)}

    rng = np.random.default_rng(42)
    modes = mode_significance.significant_modes(X, decomposition,
        n_random=1_000, rng=rng)

    assert len(modes) == 1
    assert modes[0]['mode_index'] == 0
    assert modes[0]['fraction'] == pytest.approx(1.0)
    assert modes[0]['p'] == pytest.approx(1 / 1_001)
    assert modes[0]['q'] < 0.01
