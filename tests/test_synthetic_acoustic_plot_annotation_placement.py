import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.text import Text
import numpy as np
import pytest

import synthetic_acoustic_probes._plot_annotation_placement as annotation_placement


@pytest.fixture(autouse=True)
def close_figures():
    '''Close Matplotlib figures after every test.'''

    yield
    pyplot.close('all')


def test_label_placement_avoids_markers_and_trajectory():
    '''Available whitespace wins over positions covering plotted objects.'''

    figure, axis = pyplot.subplots(figsize=(6, 4))
    axis.set_xlim(-10, 10)
    axis.set_ylim(-10, 10)
    annotation = axis.annotate(
        'obstacle-aware label',
        (0.0, 0.0),
        xytext=annotation_placement._LABEL_CANDIDATE_OFFSETS[0],
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

    selected = annotation_placement._spread_annotation_labels(
        figure,
        axis,
        [annotation],
        coordinates=coordinates,
        trajectory=trajectory,
    )

    assert selected[0]['offset'] != annotation_placement._LABEL_CANDIDATE_OFFSETS[0]
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
            xytext=annotation_placement._LABEL_CANDIDATE_OFFSETS[0],
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

    selected = annotation_placement._spread_annotation_labels(
        figure,
        axis,
        annotations,
        coordinates=coordinates,
        trajectory=trajectory,
    )
    interactions = annotation_placement._empty_label_interactions()
    annotation_placement._add_label_interactions(
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
