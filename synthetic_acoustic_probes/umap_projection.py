'''Reusable UMAP projection for synthetic acoustic representations.'''

import numpy as np


def project_umap(X, *, metric='cosine', random_state=42):
    '''Project a representation matrix into two UMAP dimensions.
    X:             Samples by representation features.
    metric:        UMAP distance metric.
    random_state:  Seed controlling the stochastic projection.
    All UMAP settings other than the metric and seed retain their defaults.
    Returns a samples-by-two NumPy array.
    '''
    representations = _validated_representations(X, metric)
    UMAP = _umap_class()
    reducer = UMAP(
        n_components=2,
        metric=metric,
        random_state=random_state,
    )
    transformed = reducer.fit_transform(representations)
    coordinates = np.asarray(transformed, dtype=float)
    expected_shape = (representations.shape[0], 2)
    shape_message = (
        f'UMAP returned shape {coordinates.shape}, expected {expected_shape}'
    )
    if coordinates.shape != expected_shape: raise ValueError(shape_message)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('UMAP returned non-finite coordinates')
    return coordinates


def _validated_representations(values, metric):
    try:
        representations = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('X must contain numeric representations') from error
    if representations.ndim != 2:
        raise ValueError('X must be a two-dimensional array')
    if representations.shape[0] < 2 or representations.shape[1] < 1:
        raise ValueError('X must contain at least two rows and one feature')
    if not np.all(np.isfinite(representations)):
        raise ValueError('X contains non-finite values')
    has_zero_vector = np.any(np.linalg.norm(representations, axis=1) == 0)
    if metric == 'cosine' and has_zero_vector:
        raise ValueError('cosine distance is undefined for zero vectors')
    return representations


def _umap_class():
    try:
        from umap import UMAP
    except ImportError as error:
        message = 'project_umap requires the umap-learn package'
        raise ImportError(message) from error
    return UMAP
