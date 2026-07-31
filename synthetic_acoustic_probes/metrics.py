'''Quantitative structure scores computed in representation space.'''

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cosine_distance_matrix(representations):
    '''Return pairwise cosine distances after strict input validation.'''

    representations = _representations(representations)
    return cdist(representations, representations, metric='cosine')


def pairwise_geometry_spearman(
    representations,
    coordinates,
    coordinate_spans=None,
):
    '''Correlate representation distances with normalized target distances.'''

    representations = _representations(representations)
    coordinates = _coordinates(
        coordinates, representations.shape[0], coordinate_spans
    )
    representation_distances = pdist(representations, metric='cosine')
    target_distances = pdist(coordinates, metric='euclidean')
    if np.ptp(representation_distances) == 0:
        raise ValueError('representation distances are constant')
    if np.ptp(target_distances) == 0:
        raise ValueError('target distances are constant')
    result = spearmanr(target_distances, representation_distances)
    return float(result.statistic)


def local_neighbor_preservation(
    representations,
    coordinates,
    n_neighbors=4,
    coordinate_spans=None,
):
    '''Mean recall of target-space neighbors in representation space.'''

    representations = _representations(representations)
    n_samples = representations.shape[0]
    if not 1 <= n_neighbors < n_samples:
        raise ValueError('n_neighbors must be in [1, n_samples)')
    coordinates = _coordinates(coordinates, n_samples, coordinate_spans)
    target_distances = cdist(coordinates, coordinates, metric='euclidean')
    representation_distances = cosine_distance_matrix(representations)
    np.fill_diagonal(target_distances, np.inf)
    np.fill_diagonal(representation_distances, np.inf)
    target_neighbors = np.argsort(
        target_distances, axis=1, kind='stable'
    )[:, :n_neighbors]
    representation_neighbors = np.argsort(
        representation_distances, axis=1, kind='stable'
    )[:, :n_neighbors]
    recalls = [
        len(set(target).intersection(predicted)) / n_neighbors
        for target, predicted in zip(
            target_neighbors, representation_neighbors
        )
    ]
    return float(np.mean(recalls))


def cross_validated_ridge_scores(
    representations,
    targets,
    target_names=None,
    alpha=1.0,
    n_splits=5,
    seed=0,
):
    '''Cross-validated R² for every controlled target dimension.'''

    representations = _representations(representations)
    target_values, names = _named_targets(targets, target_names)
    if target_values.shape[0] != representations.shape[0]:
        raise ValueError('representations and targets have different lengths')
    if n_splits < 2 or n_splits > representations.shape[0] // 2:
        raise ValueError(
            'n_splits must leave at least two samples in every test fold'
        )
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output = {}
    for index, name in enumerate(names):
        target = target_values[:, index]
        if not np.all(np.isfinite(target)):
            raise ValueError(f'target {name!r} contains non-finite values')
        if np.ptp(target) == 0:
            raise ValueError(f'target {name!r} is constant')
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        scores = cross_val_score(
            estimator,
            representations,
            target,
            cv=splitter,
            scoring='r2',
        )
        output[name] = {
            'mean_r2': float(np.mean(scores)),
            'std_r2': float(np.std(scores)),
            'fold_r2': scores.tolist(),
        }
    return output


def conditional_axis_monotonicity(
    representations,
    coordinates,
    axis,
    coordinate_names=None,
):
    '''Distance monotonicity for one axis while all other axes are fixed.'''

    representations = _representations(representations)
    coordinate_values, names = _named_targets(
        coordinates, coordinate_names
    )
    if coordinate_values.shape[0] != representations.shape[0]:
        raise ValueError('representations and coordinates have different lengths')
    axis_index = _axis_index(axis, names)
    other_indices = [
        index for index in range(coordinate_values.shape[1])
        if index != axis_index
    ]
    if not np.all(np.isfinite(coordinate_values)):
        raise ValueError('coordinates contain non-finite values')
    if other_indices:
        groups = {}
        for row, values in enumerate(coordinate_values):
            key = tuple(values[index] for index in other_indices)
            groups.setdefault(key, []).append(row)
    else:
        groups = {(): list(range(len(coordinate_values)))}
    scores = []
    for key, rows in groups.items():
        if len(rows) < 3:
            continue
        target = coordinate_values[rows, axis_index][:, None]
        target_distances = pdist(target, metric='euclidean')
        representation_distances = pdist(
            representations[rows], metric='cosine'
        )
        if np.ptp(target_distances) == 0:
            continue
        if np.ptp(representation_distances) == 0:
            continue
        score = spearmanr(
            target_distances, representation_distances
        ).statistic
        scores.append({
            'condition': _serializable_condition(key),
            'n_samples': len(rows),
            'spearman': float(score),
        })
    if not scores:
        raise ValueError('no condition has enough non-constant samples')
    return {
        'axis': names[axis_index],
        'mean_spearman': float(np.mean([
            score['spearman'] for score in scores
        ])),
        'n_conditions': len(scores),
        'conditions': scores,
    }


def frequency_scale(frequencies_hz, scale):
    '''Map positive frequencies to Hz, log-Hz, Mel, or Bark coordinates.'''

    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or not frequencies.size:
        raise ValueError('frequencies_hz must be a non-empty vector')
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError('frequencies_hz must be finite and positive')
    if scale == 'hz':
        return frequencies
    if scale == 'log_hz':
        return np.log(frequencies)
    if scale == 'mel':
        return 2595 * np.log10(1 + frequencies / 700)
    if scale == 'bark':
        return 6 * np.arcsinh(frequencies / 600)
    raise ValueError('scale must be hz, log_hz, mel, or bark')


def accumulated_adjacent_cosine_scale(representations, frequencies_hz):
    '''Cumulative cosine distance after sorting stimuli by frequency.'''

    representations = _representations(representations)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.shape != (representations.shape[0],):
        raise ValueError('one frequency is required per representation')
    order = np.argsort(frequencies, kind='stable')
    ordered = representations[order]
    adjacent = np.array([
        _cosine_distance(left, right)
        for left, right in zip(ordered[:-1], ordered[1:])
    ])
    cumulative = np.concatenate(([0.0], np.cumsum(adjacent)))
    if cumulative[-1] > 0:
        cumulative /= cumulative[-1]
    return {
        'frequencies_hz': frequencies[order],
        'adjacent_distances': adjacent,
        'normalized_cumulative_distance': cumulative,
    }


def compare_frequency_scales(representations, frequencies_hz):
    '''Compare the learned adjacent-distance scale with common transforms.'''

    result = accumulated_adjacent_cosine_scale(
        representations, frequencies_hz
    )
    cumulative = result['normalized_cumulative_distance']
    comparisons = {}
    for scale in ('hz', 'log_hz', 'mel', 'bark'):
        target = frequency_scale(result['frequencies_hz'], scale)
        target = (target - target[0]) / (target[-1] - target[0])
        comparisons[scale] = float(spearmanr(target, cumulative).statistic)
    result['spearman_by_scale'] = comparisons
    return result


def structure_report(
    representations,
    coordinates,
    coordinate_names=None,
    coordinate_spans=None,
    n_neighbors=4,
    ridge_alpha=1.0,
    n_splits=5,
    seed=0,
):
    '''Compute the complementary primary structure scores.'''

    values, names = _named_targets(coordinates, coordinate_names)
    return {
        'pairwise_geometry_spearman': pairwise_geometry_spearman(
            representations, values, coordinate_spans
        ),
        'local_neighbor_preservation': local_neighbor_preservation(
            representations, values, n_neighbors, coordinate_spans
        ),
        'ridge': cross_validated_ridge_scores(
            representations,
            values,
            names,
            alpha=ridge_alpha,
            n_splits=n_splits,
            seed=seed,
        ),
        'conditional_axes': {
            name: conditional_axis_monotonicity(
                representations, values, name, names
            )
            for name in names
        },
    }


def _representations(values):
    representations = np.asarray(values, dtype=float)
    if representations.ndim != 2:
        raise ValueError('representations must be a two-dimensional array')
    if representations.shape[0] < 2 or representations.shape[1] < 1:
        raise ValueError('representations must contain at least two rows')
    if not np.all(np.isfinite(representations)):
        raise ValueError('representations contain non-finite values')
    if np.any(np.linalg.norm(representations, axis=1) == 0):
        raise ValueError('cosine distance is undefined for zero vectors')
    return representations


def _coordinates(values, n_samples, spans):
    coordinates, _ = _named_targets(values, None)
    if coordinates.shape[0] != n_samples:
        raise ValueError('representations and coordinates have different lengths')
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('coordinates contain non-finite values')
    if spans is None:
        spans = np.ptp(coordinates, axis=0)
    spans = np.asarray(spans, dtype=float)
    if spans.ndim == 0:
        spans = np.repeat(spans, coordinates.shape[1])
    if spans.shape != (coordinates.shape[1],):
        raise ValueError('coordinate_spans has the wrong shape')
    if not np.all(np.isfinite(spans)) or np.any(spans <= 0):
        raise ValueError('coordinate spans must be finite and positive')
    return coordinates / spans


def _named_targets(values, names):
    if isinstance(values, dict):
        inferred_names = list(values)
        target_values = np.column_stack([
            np.asarray(values[name], dtype=float)
            for name in inferred_names
        ])
    else:
        target_values = np.asarray(values, dtype=float)
        if target_values.ndim == 1:
            target_values = target_values[:, None]
        if target_values.ndim != 2:
            raise ValueError('targets must be one- or two-dimensional')
        inferred_names = [
            f'target_{index}' for index in range(target_values.shape[1])
        ]
    if names is not None:
        inferred_names = list(names)
        if len(inferred_names) != target_values.shape[1]:
            raise ValueError('target_names has the wrong length')
    return target_values, inferred_names


def _axis_index(axis, names):
    if isinstance(axis, str):
        try:
            return names.index(axis)
        except ValueError as error:
            raise ValueError(f'unknown axis {axis!r}') from error
    if not isinstance(axis, (int, np.integer)):
        raise ValueError('axis must be a name or integer index')
    if not 0 <= axis < len(names):
        raise ValueError('axis index is out of bounds')
    return int(axis)


def _serializable_condition(key):
    if not isinstance(key, tuple):
        key = (key,)
    return [
        value.item() if isinstance(value, np.generic) else value
        for value in key
    ]


def _cosine_distance(left, right):
    distance = 1 - np.dot(left, right) / (
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    return float(max(0.0, distance))
