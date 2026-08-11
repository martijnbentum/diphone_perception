import numpy as np
import pytest

import synthetic_acoustic_probes.umap_projection as umap_projection


class FakeUMAP:
    '''Small deterministic stand-in for the third-party UMAP estimator.'''

    calls = []

    def __init__(self, *, n_components, metric, random_state):
        '''Record constructor parameters.

        n_components:  Requested output dimensionality.
        metric:        Requested distance metric.
        random_state:  Requested random seed.
        '''

        parameters = {}
        parameters['n_components'] = n_components
        parameters['metric'] = metric
        parameters['random_state'] = random_state
        self.parameters = parameters
        self.calls.append(parameters)

    def fit_transform(self, X):
        '''Return deterministic coordinates derived from X and the seed.'''

        seed = self.parameters['random_state']
        offset = np.random.default_rng(seed).uniform()
        return np.column_stack((X[:, 0], X[:, -1])) + offset


@pytest.fixture(autouse=True)
def use_fake_umap(monkeypatch):
    '''Replace UMAP so these unit tests do not compile or fit real models.'''

    FakeUMAP.calls.clear()
    monkeypatch.setattr(umap_projection, '_umap_class', lambda: FakeUMAP)


def test_projection_uses_paper_metric_and_reproducible_seed():
    '''Projection uses cosine distance and seed 42 by default.'''

    X = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)

    first = umap_projection.project_umap(X)
    second = umap_projection.project_umap(X)

    assert first.shape == (3, 2)
    assert np.array_equal(first, second)
    assert FakeUMAP.calls == [
        {'n_components': 2, 'metric': 'cosine', 'random_state': 42},
        {'n_components': 2, 'metric': 'cosine', 'random_state': 42},
    ]


@pytest.mark.parametrize(
    ('X', 'message'),
    (
        ([1, 2], 'two-dimensional'),
        ([[1, 2]], 'at least two rows'),
        ([['low'], ['high']], 'numeric representations'),
        ([[1, np.nan], [2, 3]], 'non-finite'),
        ([[0, 0], [2, 3]], 'zero vectors'),
    ),
)
def test_projection_rejects_invalid_representation_matrices(X, message):
    '''Projection rejects malformed or cosine-incompatible matrices.

    X:        Candidate representation input.
    message:  Expected validation-message fragment.
    '''

    with pytest.raises(ValueError, match=message):
        umap_projection.project_umap(X)


def test_non_cosine_projection_accepts_zero_vectors():
    '''Zero vectors are valid when the selected metric supports them.'''

    result = umap_projection.project_umap(
        [[0, 0], [2, 3]],
        metric='euclidean',
    )

    assert result.shape == (2, 2)
    assert FakeUMAP.calls[0]['metric'] == 'euclidean'


def test_projection_rejects_invalid_estimator_output(monkeypatch):
    '''Projection checks the shape returned by the estimator.'''

    class WrongShapeUMAP(FakeUMAP):
        def fit_transform(self, X):
            return np.ones((len(X), 3))

    monkeypatch.setattr(
        umap_projection,
        '_umap_class',
        lambda: WrongShapeUMAP,
    )

    with pytest.raises(ValueError, match='returned shape'):
        umap_projection.project_umap([[1, 2], [2, 3]])
