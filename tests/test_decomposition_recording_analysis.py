'''Tests for recording effects in evaluation mode scores.'''

from types import SimpleNamespace

import numpy as np
import pytest

from decomposition.recording_analysis import fit_recording_model
from decomposition.recording_analysis import fit_recording_models
from decomposition.recording_analysis import make_recording_sample


def _row(filename, index, score=0.0, speech=False, select='eval'):
    info = {'filename': filename, 'marker_key': f'{filename}:{index}'.encode(),
        'speech': speech}
    return SimpleNamespace(marker_info=info, scores=np.asarray(score,
        dtype=float), table=SimpleNamespace(select=select))


def test_sample_filters_status_before_minimum_and_caps_reproducibly():
    rows = ([_row('a.wav', index, [index], speech=False)
        for index in range(60)]
        + [_row('b.wav', index, [index], speech=False)
            for index in range(10)]
        + [_row('c.wav', index, [index], speech=False)
            for index in range(9)]
        + [_row('c.wav', index, [index], speech=True)
            for index in range(30)])
    first = make_recording_sample(rows)
    reversed_sample = make_recording_sample(reversed(rows))
    first_keys = [row.marker_info['marker_key'] for row in first['rows']]
    reversed_keys = [row.marker_info['marker_key']
        for row in reversed_sample['rows']]

    assert first_keys == reversed_keys
    assert first['n_input_rows'] == 109
    assert first['n_status_rows'] == 79
    assert first['n_status_recordings'] == 3
    assert first['n_eligible_rows'] == 70
    assert first['n_eligible_recordings'] == 2
    assert first['n_capped_recordings'] == 1
    assert first['n_sampled_rows'] == 60
    assert len([row for row in first['rows']
        if row.marker_info['filename'] == 'a.wav']) == 50
    assert all(not row.marker_info['speech'] for row in first['rows'])
    other_seed = make_recording_sample(rows, seed=43)
    assert first_keys != [row.marker_info['marker_key']
        for row in other_seed['rows']]


def test_sample_rejects_non_eval_rows():
    rows = [_row('a.wav', index, [index], select='fit')
        for index in range(10)]
    with pytest.raises(ValueError, match='eval'):
        make_recording_sample(rows)


def test_sample_can_include_both_speech_statuses():
    rows = ([_row('a.wav', index, [index], speech=index >= 4)
        for index in range(8)]
        + [_row('b.wav', index, [index], speech=index >= 2)
            for index in range(4)]
        + [_row('c.wav', index, [index], speech=index == 1)
            for index in range(2)])
    sample = make_recording_sample(rows, speech=None, min_frames=4,
        max_frames=6)
    reversed_sample = make_recording_sample(reversed(rows), speech=None,
        min_frames=4, max_frames=6)

    assert sample['speech'] is None
    assert sample['n_status_rows'] == 14
    assert sample['n_eligible_rows'] == 12
    assert sample['n_eligible_recordings'] == 2
    assert sample['n_capped_recordings'] == 1
    assert sample['n_sampled_rows'] == 10
    assert {row.marker_info['speech'] for row in sample['rows']} == {
        False, True}
    assert [row.marker_info['marker_key'] for row in sample['rows']] == [
        row.marker_info['marker_key'] for row in reversed_sample['rows']]


def test_balanced_reml_matches_random_intercept_anova():
    # In a balanced one-way design, positive REML components equal the
    # within-group and between-group ANOVA method-of-moments components.
    offsets = [-3.0, -1.0, 2.0, 4.0]
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 2)
    rows = [_row(f'{group}.wav', index, [offset + residual])
        for group, offset in enumerate(offsets)
        for index, residual in enumerate(residuals)]
    fit = fit_recording_model(rows, 0)
    group_size = len(residuals)
    n_groups = len(offsets)
    ms_within = n_groups * sum(residuals ** 2) / (
        n_groups * (group_size - 1))
    ms_between = group_size * sum((np.asarray(offsets)
        - np.mean(offsets)) ** 2) / (n_groups - 1)
    expected_recording_variance = (ms_between - ms_within) / group_size

    random_fit = fit['recording_intercept']
    assert random_fit['residual_variance'] == pytest.approx(ms_within,
        rel=1e-4)
    assert random_fit['recording_variance'] == pytest.approx(
        expected_recording_variance, rel=1e-4)
    assert random_fit['icc'] == pytest.approx(
        expected_recording_variance /
        (expected_recording_variance + ms_within), rel=1e-4)
    assert fit['between_recording_fraction'] > 0.5


def test_equal_group_means_estimate_zero_recording_variance():
    rows = [_row(f'{group}.wav', index, [value])
        for group in range(3)
        for index, value in enumerate([-1.0, 0.0, 1.0] * 4)]
    fit = fit_recording_model(rows, 0)
    assert fit['recording_intercept']['recording_variance'] == 0
    assert fit['recording_intercept']['icc'] == 0
    assert fit['between_recording_fraction'] == pytest.approx(0)


def test_multimode_fit_uses_one_sample():
    rows = [_row(f'{group}.wav', index, [group + index / 10,
        group - index / 10])
        for group in range(3) for index in range(12)]
    result = fit_recording_models(rows, mode_indices=[0, 1])
    assert result['sample']['n_sampled_rows'] == 36
    assert [model['mode_index'] for model in result['models']] == [0, 1]
    assert all(model['n_rows'] == 36 for model in result['models'])
