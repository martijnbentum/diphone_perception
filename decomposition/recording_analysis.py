'''Estimate recording-level variation in evaluation-set mode scores.'''

from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar


def make_recording_sample(rows, speech=False, min_frames=10,
    max_frames=50, seed=42):
    '''Select a reproducible sensitivity sample of evaluation markers.

    rows:        marker_analysis.Row objects from Table(select='eval')
    speech:      False selects non-speech; True selects speech
    min_frames:  minimum selected-status markers in a recording
    max_frames:  maximum markers sampled uniformly per recording
    seed:        non-negative integer seed for the random cap

    Return selected rows and counts before and after filtering. The threshold
    is applied after selecting speech status. Marker keys and filenames make
    selection independent of the input row order.
    '''
    if not isinstance(speech, bool):
        raise ValueError('speech must be True or False')
    for name, value in (('min_frames', min_frames),
            ('max_frames', max_frames)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if max_frames < min_frames:
        raise ValueError('max_frames must be at least min_frames')
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a non-negative integer')

    groups = defaultdict(list)
    n_input_rows = 0
    for row in rows:
        n_input_rows += 1
        select = getattr(getattr(row, 'table', None), 'select', None)
        if select is not None and select != 'eval':
            raise ValueError('rows must come from Table(select=\'eval\')')
        if bool(row.marker_info['speech']) != speech: continue
        filename = str(row.marker_info['filename'])
        groups[filename].append(row)

    n_status_rows = sum(map(len, groups.values()))
    if not groups: raise ValueError('no rows match the selected speech status')
    rng = np.random.default_rng(seed)
    selected = []
    n_eligible_rows = 0
    n_capped_recordings = 0
    n_eligible_recordings = 0
    for filename in sorted(groups):
        group = groups[filename]
        if len(group) < min_frames: continue
        n_eligible_recordings += 1
        n_eligible_rows += len(group)
        group = sorted(group, key=_marker_key)
        keys = [_marker_key(row) for row in group]
        if len(set(keys)) != len(keys):
            raise ValueError(f'duplicate marker key in {filename}')
        if len(group) > max_frames:
            n_capped_recordings += 1
            indices = np.sort(rng.choice(len(group), max_frames,
                replace=False))
            selected.extend(group[index] for index in indices)
        else:
            selected.extend(group)
    if n_eligible_recordings < 2:
        raise ValueError('at least two recordings must meet min_frames')
    return {'rows': tuple(selected), 'speech': speech,
        'n_input_rows': n_input_rows, 'n_status_rows': n_status_rows,
        'n_status_recordings': len(groups),
        'n_eligible_rows': n_eligible_rows,
        'n_eligible_recordings': n_eligible_recordings,
        'n_capped_recordings': n_capped_recordings,
        'n_sampled_rows': len(selected), 'min_frames': min_frames,
        'max_frames': max_frames, 'seed': int(seed)}


def fit_recording_model(rows, mode_index):
    '''Fit intercept-only and Gaussian recording-intercept score models.

    rows:        already selected evaluation marker rows
    mode_index:  zero-based mode whose scores are modeled

    Estimate random-intercept and residual variances by restricted maximum
    likelihood using recording sufficient statistics. Return the recording
    intraclass correlation (ICC) and descriptive in-sample between-recording
    variance fraction. No significance test or confidence interval is implied.
    '''
    if (isinstance(mode_index, bool)
            or not isinstance(mode_index, (int, np.integer))
            or mode_index < 0):
        raise ValueError('mode_index must be a non-negative integer')

    rows = list(rows)
    if not rows: raise ValueError('rows must not be empty')
    filenames = []
    scores = []
    for row in rows:
        select = getattr(getattr(row, 'table', None), 'select', None)
        if select is not None and select != 'eval':
            raise ValueError('rows must come from Table(select=\'eval\')')
        filenames.append(str(row.marker_info['filename']))
        scores.append(float(row.scores[mode_index]))
    y = np.asarray(scores)
    if not np.isfinite(y).all():
        raise ValueError('mode scores must be finite')
    _, group_index = np.unique(filenames, return_inverse=True)
    counts = np.bincount(group_index)
    n_rows, n_recordings = len(y), len(counts)
    if n_recordings < 2 or n_rows <= n_recordings:
        raise ValueError('need two recordings and repeated rows per recording')

    group_means = np.bincount(group_index, weights=y) / counts
    within = y - group_means[group_index]
    within_ss = float(within @ within)
    null_mean = float(y.mean())
    centered = y - null_mean
    total_ss = float(centered @ centered)
    between_ss = max(total_ss - within_ss, 0.0)

    if total_ss == 0:
        residual_variance = recording_variance = 0.0
        fitted_mean = null_mean
        icc = np.nan
    elif within_ss == 0:
        residual_variance = 0.0
        recording_variance = float(group_means.var(ddof=1))
        fitted_mean = float(group_means.mean())
        icc = 1.0
    else:
        ratio = _fit_variance_ratio(counts, group_means, within_ss,
            n_rows)
        weights = counts / (1 + counts * ratio)
        fitted_mean = float(np.average(group_means, weights=weights))
        difference = group_means - fitted_mean
        quadratic = within_ss + float(np.dot(weights, difference ** 2))
        residual_variance = quadratic / (n_rows - 1)
        recording_variance = ratio * residual_variance
        icc = ratio / (1 + ratio)

    return {'mode_index': int(mode_index), 'n_rows': n_rows,
        'n_recordings': n_recordings,
        'intercept_only': {'mean': null_mean,
            'variance': total_ss / (n_rows - 1)},
        'recording_intercept': {'mean': fitted_mean,
            'recording_variance': float(recording_variance),
            'residual_variance': float(residual_variance),
            'icc': float(icc)},
        'between_recording_fraction': (between_ss / total_ss
            if total_ss > 0 else np.nan)}


def fit_recording_models(rows, mode_indices=range(9), speech=False,
    min_frames=10, max_frames=50, seed=42):
    '''Sample evaluation rows once and fit recording models for several modes.

    rows:          marker_analysis.Row objects from Table(select='eval')
    mode_indices:  mode indices to fit; defaults to modes 0 through 8
    speech:        False selects non-speech; True selects speech
    min_frames:    minimum selected-status markers per recording
    max_frames:    maximum sampled markers per recording
    seed:          seed for the random cap

    Return sample diagnostics and one fit per mode. All modes use the same
    selected markers, so mode comparisons do not change the sample.
    '''
    mode_indices = list(mode_indices)
    if not mode_indices: raise ValueError('mode_indices must not be empty')
    sample = make_recording_sample(rows, speech=speech,
        min_frames=min_frames, max_frames=max_frames, seed=seed)
    fits = [fit_recording_model(sample['rows'], index)
        for index in mode_indices]
    diagnostics = {key: value for key, value in sample.items()
        if key != 'rows'}
    return {'sample': diagnostics, 'models': fits}


def _marker_key(row):
    '''Return a stable key for order-independent recording sampling.'''
    return bytes(row.marker_info['marker_key'])


def _fit_variance_ratio(counts, group_means, within_ss, n_rows):
    '''Profile REML over recording variance divided by residual variance.'''
    def objective(ratio):
        weights = counts / (1 + counts * ratio)
        mean = np.average(group_means, weights=weights)
        difference = group_means - mean
        quadratic = within_ss + np.dot(weights, difference ** 2)
        return ((n_rows - 1) * np.log(quadratic / (n_rows - 1))
            + np.log1p(counts * ratio).sum() + np.log(weights.sum()))

    result = minimize_scalar(lambda log_ratio: objective(np.exp(log_ratio)),
        bounds=(-30, 30), method='bounded')
    if not result.success:
        raise RuntimeError('recording variance fit did not converge')
    ratio = float(np.exp(result.x))
    return ratio if result.fun < objective(0.0) else 0.0
