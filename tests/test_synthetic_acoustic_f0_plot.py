import inspect

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
import numpy as np
import pytest

import locations
import synthetic_acoustic_probes.f0_plot as f0_plot


@pytest.fixture(autouse=True)
def close_figures():
    '''Close Matplotlib figures after every test.'''

    yield
    pyplot.close('all')


@pytest.fixture
def fixed_projection(monkeypatch):
    '''Replace UMAP with known coordinates and return its call record.'''

    coordinates = np.array([
        [8.0, 80.0],
        [0.1, 1.0],
        [1.0, 10.0],
        [2.0, 20.0],
    ])
    calls = []

    def fake_project(X, *, metric, random_state):
        parameters = {'metric': metric, 'random_state': random_state}
        calls.append((X, parameters))
        return coordinates

    monkeypatch.setattr(f0_plot, 'project_umap', fake_project)
    return coordinates, calls


def test_plot_adds_ordered_path_frequency_colors_and_landmarks(
    fixed_projection,
):
    '''Plot combines ordered path, colors, and paper landmarks.'''

    coordinates, calls = fixed_projection
    X = np.ones((4, 3))
    frequencies = np.array([8000, 10, 1000, 2000])

    figure, axis = f0_plot.plot_f0_umap(
        X,
        frequencies,
        output_path='',
    )

    assert calls == [(X, {'metric': 'cosine', 'random_state': 42})]
    line = axis.lines[0]
    expected_order = np.array([1, 2, 3, 0])
    expected_x = coordinates[expected_order, 0]
    expected_y = coordinates[expected_order, 1]
    plotted_frequencies = axis.collections[0].get_array()
    plotted_x = line.get_xdata()
    plotted_y = line.get_ydata()
    assert np.array_equal(plotted_x, expected_x)
    assert np.array_equal(plotted_y, expected_y)
    assert np.array_equal(plotted_frequencies, frequencies)
    assert [text.get_text() for text in axis.texts] == [
        '10 Hz', '1 kHz', '2 kHz', '8 kHz'
    ]
    assert figure.axes[1].get_ylabel() == 'F0 (Hz)'
    assert axis.get_xlabel() == 'UMAP 1'
    assert axis.get_ylabel() == 'UMAP 2'


def test_plot_can_save_and_return_figure(tmp_path, fixed_projection):
    '''Plot saves to a nested path and returns the rendered objects.

    tmp_path:          Temporary output root supplied by Pytest.
    fixed_projection:  Fixture replacing UMAP with known coordinates.
    '''

    output_path = tmp_path / 'figures' / 'f0.png'
    X = np.ones((4, 2))

    figure, axis = f0_plot.plot_f0_umap(
        X,
        [10, 1000, 2000, 8000],
        output_path=output_path,
        dpi=120,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert axis in figure.axes


def test_default_output_is_pdf_in_f0_experiment_directory():
    '''Default output is the shared F0 experiment PDF path.'''

    parameters = inspect.signature(f0_plot.plot_f0_umap).parameters
    output_path = parameters['output_path'].default

    assert output_path == locations.f0_umap_plot
    assert output_path.parent == locations.f0_experiment
    assert output_path.suffix == '.pdf'


@pytest.mark.parametrize(
    ('X', 'y', 'message'),
    (
        (np.ones(3), [10, 20, 30], 'two-dimensional'),
        (np.ones((2, 3)), [10, 20, 30], 'same number'),
        (np.ones((2, 3)), ['low', 'high'], 'numeric'),
        (np.ones((2, 3)), [10, np.nan], 'non-finite'),
        (np.ones((2, 3)), [0, 10], 'positive'),
    ),
)
def test_plot_rejects_invalid_inputs(X, y, message, monkeypatch):
    '''Plot validates X and y before invoking UMAP.

    X:            Candidate representation input.
    y:            Candidate frequency input.
    message:      Expected validation-message fragment.
    monkeypatch:  Pytest fixture used to prevent projection.
    '''

    def fail_projection(X, *, metric, random_state):
        pytest.fail('projection should not run')

    monkeypatch.setattr(f0_plot, 'project_umap', fail_projection)

    with pytest.raises(ValueError, match=message):
        f0_plot.plot_f0_umap(X, y)
