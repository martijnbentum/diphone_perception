import inspect

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
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


@pytest.fixture
def checkpoint_metrics():
    '''Return unsorted checkpoint rows with a shared frequency grid.'''

    frequency_edges = np.array([
        [10.0, 20.0],
        [20.0, 30.0],
        [30.0, 40.0],
    ])
    thresholds = np.array([0.1, 0.25, 0.5])
    return (
        {
            'checkpoint_step': 200,
            'mean': 0.12,
            'median': 0.08,
            'p95': 0.25,
            'p99': 0.4,
            'maximum': 0.6,
            'thresholds': thresholds,
            'fractions_above_threshold': np.array([0.3, 0.1, 0.0]),
            'frequency_edges_hz': frequency_edges,
            'adjacent_distances': np.array([0.1, 0.2, 0.3]),
        },
        {
            'checkpoint_step': 0,
            'mean': 0.3,
            'median': 0.2,
            'p95': 0.5,
            'p99': 0.8,
            'maximum': 1.0,
            'thresholds': thresholds,
            'fractions_above_threshold': np.array([0.7, 0.5, 0.2]),
            'frequency_edges_hz': frequency_edges,
            'adjacent_distances': np.array([0.3, 0.4, 0.5]),
        },
        {
            'checkpoint_step': 100,
            'mean': 0.2,
            'median': 0.1,
            'p95': 0.3,
            'p99': 0.5,
            'maximum': 0.7,
            'thresholds': thresholds,
            'fractions_above_threshold': np.array([0.5, 0.2, 0.1]),
            'frequency_edges_hz': frequency_edges,
            'adjacent_distances': np.array([0.2, 0.3, 0.4]),
        },
    )


def test_plot_checkpoint_smoothness_uses_numeric_checkpoint_order(
    checkpoint_metrics,
    tmp_path,
):
    '''Summary panels plot typical, extreme, and threshold measurements.'''

    output_path = tmp_path / 'plots' / 'smoothness.pdf'

    figure, axes = f0_plot.plot_f0_checkpoint_smoothness(
        checkpoint_metrics,
        output_path=output_path,
        dpi=120,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert len(axes) == 3
    expected_steps = np.array([0, 100, 200])
    for axis in axes:
        for line in axis.lines:
            np.testing.assert_array_equal(line.get_xdata(), expected_steps)
    assert [line.get_label() for line in axes[0].lines] == [
        'Median', 'Mean', 'P95'
    ]
    np.testing.assert_allclose(
        axes[0].lines[0].get_ydata(),
        [0.2, 0.1, 0.08],
    )
    assert [line.get_label() for line in axes[1].lines] == [
        'P99', 'Maximum'
    ]
    assert [line.get_label() for line in axes[2].lines] == [
        '> 0.1', '> 0.25', '> 0.5'
    ]
    np.testing.assert_allclose(
        axes[2].lines[1].get_ydata(),
        [0.5, 0.2, 0.1],
    )
    assert axes[2].get_xlabel() == 'Checkpoint step'
    assert axes[2].get_ylabel() == 'Fraction of edges'
    assert figure._suptitle.get_text() == (
        'F0 trajectory smoothness over checkpoints'
    )


def test_plot_checkpoint_distance_heatmap_uses_all_edges(
    checkpoint_metrics,
    tmp_path,
):
    '''Heatmap columns follow checkpoints and rows retain frequency edges.'''

    output_path = tmp_path / 'plots' / 'distance-heatmap.png'

    figure, axis = f0_plot.plot_f0_checkpoint_distance_heatmap(
        checkpoint_metrics,
        output_path=output_path,
        dpi=120,
        vmax=1.0,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    image = axis.collections[0]
    plotted = np.asarray(image.get_array()).reshape(3, 3)
    np.testing.assert_allclose(
        plotted,
        np.array([
            [0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2],
            [0.5, 0.4, 0.3],
        ]),
    )
    expected_positions = np.log10(np.array([0, 100, 200]) + 1)
    expected_boundaries = np.empty(4)
    expected_boundaries[1:-1] = (
        expected_positions[:-1] + expected_positions[1:]
    ) / 2
    expected_boundaries[0] = (
        2 * expected_positions[0] - expected_boundaries[1]
    )
    expected_boundaries[-1] = (
        2 * expected_positions[-1] - expected_boundaries[-2]
    )
    coordinates = image.get_coordinates()
    np.testing.assert_allclose(coordinates[0, :, 0], expected_boundaries)
    np.testing.assert_allclose(axis.get_xticks(), expected_positions)
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        'init', '100', '200'
    ]
    assert axis.get_xlabel() == (
        'Checkpoint step (log₁₀(step + 1) spacing)'
    )
    assert axis.get_ylabel() == 'Frequency edge (Hz)'
    assert figure.axes[1].get_ylabel() == 'Adjacent cosine distance'


@pytest.mark.parametrize(
    ('scale', 'expected_boundaries', 'expected_label'),
    (
        ('linear', [-50.0, 50.0, 150.0, 250.0],
            'Checkpoint step (linear spacing)'),
        ('categorical', [-0.5, 0.5, 1.5, 2.5],
            'Checkpoint (equal column spacing)'),
    ),
)
def test_checkpoint_heatmap_supports_alternative_spacing(
    checkpoint_metrics,
    scale,
    expected_boundaries,
    expected_label,
):
    '''Linear and categorical modes use their stated column geometry.'''

    figure, axis = f0_plot.plot_f0_checkpoint_distance_heatmap(
        checkpoint_metrics,
        checkpoint_scale=scale,
    )

    assert figure.axes[0] is axis
    coordinates = axis.collections[0].get_coordinates()
    np.testing.assert_allclose(
        coordinates[0, :, 0],
        expected_boundaries,
    )
    assert axis.get_xlabel() == expected_label


def test_checkpoint_heatmap_samples_labels_at_true_log_positions(
    checkpoint_metrics,
):
    '''Dense checkpoint sweeps receive sparse labels at column centers.'''

    template = checkpoint_metrics[0]
    steps = np.array([
        0, 100, 500, 1000, 5000, 10_000,
        20_000, 50_000, 100_000, 200_000,
    ])
    rows = []
    for step in steps:
        row = dict(template)
        row['checkpoint_step'] = int(step)
        rows.append(row)

    figure, axis = f0_plot.plot_f0_checkpoint_distance_heatmap(
        rows,
        max_checkpoint_labels=4,
    )

    assert figure.axes[0] is axis
    ticks = axis.get_xticks()
    labels = [label.get_text() for label in axis.get_xticklabels()]
    available_positions = np.log10(steps + 1)
    assert len(ticks) <= 4
    assert labels[0] == 'init'
    assert labels[-1] == '200k'
    assert ticks[0] == pytest.approx(available_positions[0])
    assert ticks[-1] == pytest.approx(available_positions[-1])
    for tick in ticks:
        assert np.min(np.abs(available_positions - tick)) < 1e-12


def test_checkpoint_metric_plots_validate_shared_inputs(checkpoint_metrics):
    '''Plots reject empty, duplicate, and incompatible checkpoint rows.'''

    with pytest.raises(ValueError, match='must not be empty'):
        f0_plot.plot_f0_checkpoint_smoothness(())

    duplicates = [dict(row) for row in checkpoint_metrics]
    duplicates[1]['checkpoint_step'] = 200
    with pytest.raises(ValueError, match='duplicate checkpoint_step'):
        f0_plot.plot_f0_checkpoint_smoothness(duplicates)

    mismatched_thresholds = [dict(row) for row in checkpoint_metrics]
    mismatched_thresholds[1]['thresholds'] = np.array([0.1, 0.2, 0.5])
    with pytest.raises(ValueError, match='thresholds do not match'):
        f0_plot.plot_f0_checkpoint_smoothness(mismatched_thresholds)

    mismatched_edges = [dict(row) for row in checkpoint_metrics]
    mismatched_edges[1]['frequency_edges_hz'] = np.array([
        [10.0, 20.0],
        [20.0, 31.0],
        [31.0, 40.0],
    ])
    with pytest.raises(ValueError, match='frequency edges do not match'):
        f0_plot.plot_f0_checkpoint_distance_heatmap(mismatched_edges)

    with pytest.raises(ValueError, match='vmax'):
        f0_plot.plot_f0_checkpoint_distance_heatmap(
            checkpoint_metrics,
            vmax=0,
        )

    with pytest.raises(ValueError, match='checkpoint_scale'):
        f0_plot.plot_f0_checkpoint_distance_heatmap(
            checkpoint_metrics,
            checkpoint_scale='log',
        )

    with pytest.raises(ValueError, match='max_checkpoint_labels'):
        f0_plot.plot_f0_checkpoint_distance_heatmap(
            checkpoint_metrics,
            max_checkpoint_labels=0,
        )


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


def test_label_placement_avoids_markers_and_trajectory():
    '''Available whitespace wins over positions covering plotted objects.'''

    figure, axis = pyplot.subplots(figsize=(6, 4))
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    annotation = axis.annotate(
        'obstacle-aware label',
        (0.0, 0.0),
        xytext=f0_plot._LABEL_CANDIDATE_OFFSETS[0],
        textcoords='offset points',
        arrowprops={'arrowstyle': '->'},
    )
    figure.tight_layout()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    initial_box = Text.get_window_extent(annotation, renderer=renderer)
    initial_center = np.array([
        (initial_box.x0 + initial_box.x1) / 2,
        (initial_box.y0 + initial_box.y1) / 2,
    ])
    inverse = axis.transData.inverted()
    marker = inverse.transform(initial_center)
    axes_box = axis.get_window_extent(renderer)
    trajectory = inverse.transform(np.array([
        [axes_box.x0, initial_center[1]],
        [axes_box.x1, initial_center[1]],
    ]))
    coordinates = np.vstack(((0.0, 0.0), marker))

    selected = f0_plot._spread_annotation_labels(
        figure,
        axis,
        [annotation],
        coordinates=coordinates,
        trajectory=trajectory,
    )

    assert selected[0]['offset'] != f0_plot._LABEL_CANDIDATE_OFFSETS[0]
    assert selected[0]['anchor_obstructions'] == 0
    assert selected[0]['point_obstructions'] == 0
    assert selected[0]['trajectory_obstructions'] == 0
    assert selected[0]['leader_anchor_obstructions'] == 0
    assert selected[0]['leader_point_obstructions'] == 0


def test_label_placement_avoids_labels_and_leader_lines():
    '''Labels and leaders receive non-crossing positions when available.'''

    figure, axis = pyplot.subplots(figsize=(6, 4))
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    annotations = [
        axis.annotate(
            label,
            anchor,
            xytext=f0_plot._LABEL_CANDIDATE_OFFSETS[0],
            textcoords='offset points',
            arrowprops={'arrowstyle': '->'},
        )
        for label, anchor in (
            ('left jump', (-2.0, 0.0)),
            ('right jump', (2.0, 0.0)),
        )
    ]
    coordinates = np.array([annotation.xy for annotation in annotations])
    trajectory = np.array([[-8.0, -8.0], [8.0, -8.0]])
    figure.tight_layout()

    selected = f0_plot._spread_annotation_labels(
        figure,
        axis,
        annotations,
        coordinates=coordinates,
        trajectory=trajectory,
    )
    interactions = f0_plot._empty_label_interactions()
    f0_plot._add_label_interactions(
        interactions,
        selected[0],
        selected[1],
    )

    assert interactions == {
        'label_overlaps': 0,
        'label_overlap_area': 0.0,
        'leader_label_crossings': 0,
        'leader_crossings': 0,
    }


def test_label_placement_is_deterministic(fixed_projection):
    '''Repeated rendering selects the same offsets for every annotation.'''

    X = np.ones((4, 3))
    frequencies = np.array([8000, 10, 1000, 2000])

    first_figure, first_axis = f0_plot.plot_f0_umap(
        X,
        frequencies,
        output_path='',
    )
    second_figure, second_axis = f0_plot.plot_f0_umap(
        X,
        frequencies,
        output_path='',
    )

    assert first_figure.axes[0] is first_axis
    assert second_figure.axes[0] is second_axis
    assert [text.get_position() for text in first_axis.texts] == [
        text.get_position()
        for text in second_axis.texts
    ]


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


def test_plot_checkpoint_comparison_defaults_to_init_and_final(
    tmp_path,
    monkeypatch,
):
    '''Default comparison loads, titles, and saves two horizontal panels.'''

    model_names = (
        'wav2vec2_checkpoint-0',
        'wav2vec2_nl1_checkpoint-200000',
    )
    output_data = tmp_path / 'output_data'
    plots = tmp_path / 'plots'
    output_data.mkdir()
    frequencies = np.array([8000, 10, 1000, 2000])
    coordinates = (
        np.array([
            [8.0, 80.0],
            [0.1, 1.0],
            [1.0, 10.0],
            [2.0, 20.0],
        ]),
        np.array([
            [-8.0, 8.0],
            [-0.1, 0.1],
            [-1.0, 1.0],
            [-2.0, 2.0],
        ]),
    )
    for model_name, model_coordinates in zip(model_names, coordinates):
        np.savez_compressed(
            output_data / f'{model_name}.npz',
            coordinates=model_coordinates,
            frequencies=frequencies,
        )
    monkeypatch.setattr(locations, 'f0_output_data', output_data)
    monkeypatch.setattr(locations, 'f0_plots', plots)

    def fail_projection(X, *, metric, random_state):
        pytest.fail('stored checkpoint plotting should not run UMAP')

    monkeypatch.setattr(f0_plot, 'project_umap', fail_projection)

    figure, axes = f0_plot.plot_f0_checkpoint_comparison(dpi=120)

    output_path = plots / f'{model_names[0]}_vs_{model_names[1]}.pdf'
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert tuple(axis.get_title() for axis in axes) == model_names
    assert figure._suptitle.get_text() == 'F0 representation space'
    np.testing.assert_allclose(figure.get_size_inches(), [16, 7])
    assert axes[0].get_position().x0 < axes[1].get_position().x0
    assert axes[0].get_position().y0 == pytest.approx(
        axes[1].get_position().y0)
    assert figure.axes[2].get_ylabel() == 'F0 (Hz)'
    left_points = axes[0].collections[0]
    right_points = axes[1].collections[0]
    assert left_points.norm is right_points.norm
    assert left_points.norm.vmin == 10
    assert left_points.norm.vmax == 8000
    for axis, model_coordinates in zip(axes, coordinates):
        line = axis.lines[0]
        expected_order = np.array([1, 2, 3, 0])
        np.testing.assert_array_equal(
            line.get_xdata(),
            model_coordinates[expected_order, 0],
        )


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
