'''F0 trajectory metrics collected across wav2vec2 checkpoints.'''

from pathlib import Path
import re

import numpy as np

import locations

from .metrics import accumulated_adjacent_cosine_scale


DEFAULT_ADJACENT_DISTANCE_THRESHOLDS = (0.1, 0.25, 0.5)

_RESULT_FIELDS = {
    'aggregation',
    'frequencies',
    'mean_cnn_features',
    'metric',
    'model_name',
    'random_state',
}


def f0_smoothness_metrics(
    representations,
    frequencies_hz,
    *,
    thresholds=DEFAULT_ADJACENT_DISTANCE_THRESHOLDS,
):
    '''Summarize consecutive-frequency distances in CNN space.

    representations:  Samples by CNN features.
    frequencies_hz:   One positive, unique frequency per sample.
    thresholds:       Cosine-distance thresholds whose exceedance fractions
                      should be reported.

    Rows are sorted by frequency before adjacent cosine distances are
    calculated. The returned complete edge arrays support later heatmaps and
    alternative summaries without another model extraction.
    '''

    representations = np.asarray(representations)
    if representations.ndim != 2:
        raise ValueError('representations must be a two-dimensional array')
    frequencies = _validated_frequencies(
        frequencies_hz,
        expected_length=representations.shape[0],
    )
    thresholds = _validated_thresholds(thresholds)
    result = accumulated_adjacent_cosine_scale(
        representations,
        frequencies,
    )
    ordered_frequencies = result['frequencies_hz']
    adjacent_distances = result['adjacent_distances']
    frequency_edges = np.column_stack((
        ordered_frequencies[:-1],
        ordered_frequencies[1:],
    ))
    fractions = np.asarray([
        np.mean(adjacent_distances > threshold)
        for threshold in thresholds
    ], dtype=float)

    return {
        'frequencies_hz': ordered_frequencies,
        'frequency_edges_hz': frequency_edges,
        'adjacent_distances': adjacent_distances,
        'normalized_cumulative_distance': (
            result['normalized_cumulative_distance']
        ),
        'n_stimuli': int(ordered_frequencies.size),
        'n_edges': int(adjacent_distances.size),
        'mean': float(np.mean(adjacent_distances)),
        'median': float(np.median(adjacent_distances)),
        'p95': float(np.quantile(adjacent_distances, 0.95)),
        'p99': float(np.quantile(adjacent_distances, 0.99)),
        'maximum': float(np.max(adjacent_distances)),
        'total_path_length': float(np.sum(adjacent_distances)),
        'thresholds': thresholds,
        'fractions_above_threshold': fractions,
    }


def f0_checkpoint_step(model_name):
    '''Return the numeric training step encoded by an F0 model name.'''

    if not isinstance(model_name, str) or not model_name:
        raise ValueError('model_name must be a non-empty string')
    if model_name == locations.wav2vec2_random_checkpoint_name:
        return 0
    match = re.fullmatch(
        locations.wav2vec2_nl1_checkpoint_pattern,
        model_name,
    )
    if match is None:
        raise ValueError(f'unsupported F0 checkpoint model: {model_name!r}')
    return int(match.group(1))


def f0_checkpoint_metrics(
    result_path,
    *,
    thresholds=DEFAULT_ADJACENT_DISTANCE_THRESHOLDS,
):
    '''Load and summarize one model-specific F0 result bundle.'''

    result_path = Path(result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f'F0 checkpoint result not found: {result_path}')
    with np.load(result_path, allow_pickle=False) as result:
        missing = _RESULT_FIELDS - set(result.files)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'F0 result is missing fields: {names}')
        model_name = _scalar_string(result['model_name'], 'model_name')
        if result_path.stem != model_name:
            message = (
                'F0 result filename does not match model_name: '
                f'{result_path.name!r} != {model_name!r}'
            )
            raise ValueError(message)
        metrics = f0_smoothness_metrics(
            result['mean_cnn_features'],
            result['frequencies'],
            thresholds=thresholds,
        )
        metrics.update({
            'model_name': model_name,
            'checkpoint_step': f0_checkpoint_step(model_name),
            'aggregation': _scalar_string(
                result['aggregation'],
                'aggregation',
            ),
            'metric': _scalar_string(result['metric'], 'metric'),
            'random_state': _scalar_integer(
                result['random_state'],
                'random_state',
            ),
            'result_path': result_path,
        })
    return metrics


def load_f0_checkpoint_metrics(
    *,
    thresholds=DEFAULT_ADJACENT_DISTANCE_THRESHOLDS,
):
    '''Load every F0 result bundle and return metrics in checkpoint order.'''

    output_directory = Path(locations.f0_output_data)
    if not output_directory.is_dir():
        message = f'F0 output-data directory not found: {output_directory}'
        raise FileNotFoundError(message)
    result_paths = tuple(output_directory.glob('*.npz'))
    if not result_paths:
        raise FileNotFoundError(
            f'no F0 checkpoint results found in: {output_directory}'
        )
    metrics = [
        f0_checkpoint_metrics(path, thresholds=thresholds)
        for path in result_paths
    ]
    metrics.sort(key=lambda row: (row['checkpoint_step'], row['model_name']))
    return tuple(metrics)


def _validated_frequencies(values, expected_length):
    try:
        frequencies = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('frequencies_hz must be numeric') from error
    if frequencies.shape != (expected_length,):
        raise ValueError('one frequency is required per representation')
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError('frequencies_hz must be finite and positive')
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError('frequencies_hz must be unique')
    return frequencies


def _validated_thresholds(values):
    try:
        thresholds = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('thresholds must be numeric') from error
    if thresholds.ndim != 1 or not thresholds.size:
        raise ValueError('thresholds must be a non-empty vector')
    if not np.all(np.isfinite(thresholds)):
        raise ValueError('thresholds must be finite')
    if np.any(thresholds <= 0) or np.any(thresholds >= 2):
        raise ValueError('thresholds must lie strictly between zero and two')
    if np.unique(thresholds).size != thresholds.size:
        raise ValueError('thresholds must be unique')
    return thresholds


def _scalar_string(value, name):
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in 'SU':
        raise ValueError(f'{name} must be a scalar string')
    output = str(value.item())
    if not output:
        raise ValueError(f'{name} must not be empty')
    return output


def _scalar_integer(value, name):
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a scalar integer')
    return int(value.item())
