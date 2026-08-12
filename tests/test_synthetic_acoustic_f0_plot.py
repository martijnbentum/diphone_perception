import inspect

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.markers import MarkerStyle
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
    landmarks = axis.collections[1]
    expected_landmarks = coordinates[[1, 2, 3, 0]]
    np.testing.assert_array_equal(landmarks.get_offsets(), expected_landmarks)
    np.testing.assert_array_equal(
        landmarks.get_facecolors(),
        [[1.0, 0.0, 0.0, 1.0]],
    )
    star = MarkerStyle('*')
    expected_star = star.get_path().transformed(star.get_transform())
    np.testing.assert_allclose(
        landmarks.get_paths()[0].vertices,
        expected_star.vertices,
    )
    assert figure.axes[1].get_ylabel() == 'F0 (Hz)'
    assert axis.get_xlabel() == 'UMAP 1'
    assert axis.get_ylabel() == 'UMAP 2'
    assert axis.get_title() == 'F0 representation space'
    assert line.get_color() == '#D62728'
    assert line.get_linestyle() == '--'
    assert line.get_linewidth() == 0.6
    assert line.get_zorder() > axis.collections[0].get_zorder()


def test_plot_annotates_isolated_frequency_jumps(monkeypatch):
    '''An isolated point far from its frequency neighbors is labeled once.'''

    coordinates = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [50.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ])

    def fake_project(X, *, metric, random_state):
        return coordinates

    monkeypatch.setattr(f0_plot, 'project_umap', fake_project)
    frequencies = np.arange(100, 180, 10)

    figure, axis = f0_plot.plot_f0_umap(
        np.ones((8, 2)),
        frequencies,
        output_path='',
    )

    assert figure.axes[0] is axis
    assert [text.get_text() for text in axis.texts] == ['130 Hz jump']
    assert axis.texts[0].xy == (50.0, 0.0)


def test_large_jump_detection_flags_isolated_point():
    '''Jump detection flags the shared point of two long incident steps.'''

    coordinates = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [30.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    indices = f0_plot._large_jump_indices(coordinates)

    np.testing.assert_array_equal(indices, [2])


def test_jump_label_replaces_duplicate_landmark_label(monkeypatch):
    '''A landmark that is also a jump receives one combined label.'''

    coordinates = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [50.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ])

    def fake_project(X, *, metric, random_state):
        return coordinates

    monkeypatch.setattr(f0_plot, 'project_umap', fake_project)
    frequencies = np.arange(970, 1050, 10)

    figure, axis = f0_plot.plot_f0_umap(
        np.ones((8, 2)),
        frequencies,
        output_path='',
    )

    assert figure.axes[0] is axis
    assert [text.get_text() for text in axis.texts] == ['1 kHz jump']


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


def test_plot_checkpoint_result_uses_stored_coordinates(
    tmp_path,
    monkeypatch,
):
    '''Checkpoint plotting loads its fixed NPZ and writes the plots folder.'''
    model_name = 'wav2vec2_nl1_checkpoint-200000'
    output_data = tmp_path / 'output_data'
    plots = tmp_path / 'plots'
    output_data.mkdir()
    coordinates = np.array([
        [8.0, 80.0],
        [0.1, 1.0],
        [1.0, 10.0],
        [2.0, 20.0],
    ])
    frequencies = np.array([8000, 10, 1000, 2000])
    np.savez_compressed(
        output_data / f'{model_name}.npz',
        coordinates=coordinates,
        frequencies=frequencies,
    )
    monkeypatch.setattr(locations, 'f0_output_data', output_data)
    monkeypatch.setattr(locations, 'f0_plots', plots)

    def fail_projection(X, *, metric, random_state):
        pytest.fail('stored checkpoint plotting should not run UMAP')

    monkeypatch.setattr(f0_plot, 'project_umap', fail_projection)

    figure, axis = f0_plot.plot_f0_checkpoint_result(model_name, dpi=120)

    plot_path = plots / f'{model_name}.pdf'
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    expected_order = np.array([1, 2, 3, 0])
    line = axis.lines[0]
    np.testing.assert_array_equal(
        line.get_xdata(),
        coordinates[expected_order, 0],
    )
    np.testing.assert_array_equal(
        line.get_ydata(),
        coordinates[expected_order, 1],
    )
    plotted_frequencies = axis.collections[0].get_array()
    np.testing.assert_array_equal(plotted_frequencies, frequencies)
    assert axis.get_title() == (
        'F0 representation space\n'
        'wav2vec2_nl1_checkpoint-200000'
    )
    assert axis in figure.axes


def test_default_output_does_not_create_a_shared_plot():
    '''Generic plotting requires an explicit output path.'''

    parameters = inspect.signature(f0_plot.plot_f0_umap).parameters
    output_path = parameters['output_path'].default

    assert output_path is None


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
