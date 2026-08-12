'''Visualization of the pure-tone F0 representation space.'''

from pathlib import Path

import numpy as np

import locations

from .umap_projection import project_umap


_F0_LANDMARKS_HZ = (10, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000)
_JUMP_DISTANCE_FACTOR = 8
_JUMP_NEIGHBORHOOD_SIZE = 5
_LABEL_CANDIDATE_OFFSETS = (
    (8, 8),
    (8, -8),
    (-8, 8),
    (-8, -8),
    (12, 24),
    (12, -24),
    (-12, 24),
    (-12, -24),
    (30, 12),
    (30, -12),
    (-30, 12),
    (-30, -12),
    (24, 36),
    (24, -36),
    (-24, 36),
    (-24, -36),
    (48, 20),
    (48, -20),
    (-48, 20),
    (-48, -20),
    (18, 52),
    (18, -52),
    (-18, 52),
    (-18, -52),
)
_LABEL_PADDING_POINTS = 2


def plot_f0_checkpoint_result(model_name, *, figsize=None, dpi=300):
    '''Plot the stored F0 UMAP result for one checkpoint.

    Loads ``output_data/{model_name}.npz`` and saves the rendered figure as
    ``plots/{model_name}.pdf`` below the F0 experiment directory. Returns the
    Matplotlib figure and primary axis.
    '''
    result_path = locations.f0_output_data / f'{model_name}.npz'
    with np.load(result_path, allow_pickle=False) as result:
        coordinates = np.asarray(result['coordinates'])
        frequencies = _validated_frequencies(result['frequencies'])
    output_path = locations.f0_plots / f'{model_name}.pdf'
    return _plot_f0_coordinates(
        coordinates,
        frequencies,
        output_path=output_path,
        figsize=figsize,
        dpi=dpi,
        title=f'F0 representation space\n{model_name}',
    )


def plot_f0_umap(
    X,
    y,
    *,
    random_state=42,
    output_path=None,
    figsize=None,
    dpi=300,
):
    '''Plot the ordered pure-tone trajectory in UMAP space.

    X:             Samples by CNN features, normally mean-aggregated frames.
    y:             Numeric fundamental frequencies in Hz.
    random_state:  Seed controlling the UMAP projection.
    output_path:   Optional destination for the rendered figure. Omit it to
                   disable saving. Model-specific experiment plots should be
                   written with ``plot_f0_checkpoint_result``.
    figsize:       Optional Matplotlib figure size.
    dpi:            Resolution used when saving.

    Points are colored by frequency and connected in ascending-frequency
    order. Paper landmarks are marked with red stars and annotated when
    present. Isolated points whose distances to both frequency neighbors are
    markedly larger than nearby steps are annotated as jumps. Returns the
    Matplotlib figure and primary axis.
    '''

    frequencies = _validated_frequencies(y)
    if np.ndim(X) != 2: raise ValueError('X must be a two-dimensional array')
    if np.shape(X)[0] != frequencies.size:
        raise ValueError('X and y must contain the same number of samples')

    coordinates = project_umap(
        X,
        metric='cosine',
        random_state=random_state,
    )
    return _plot_f0_coordinates(
        coordinates,
        frequencies,
        output_path=output_path,
        figsize=figsize,
        dpi=dpi,
        title='F0 representation space',
    )


def _plot_f0_coordinates(
    coordinates,
    frequencies,
    *,
    output_path,
    figsize,
    dpi,
    title,
):
    from matplotlib import pyplot

    coordinates = np.asarray(coordinates, dtype=float)
    expected_shape = (frequencies.size, 2)
    if coordinates.shape != expected_shape:
        message = f'coordinates must have shape {expected_shape}'
        raise ValueError(message)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('coordinates contain non-finite values')

    order = np.argsort(frequencies, kind='stable')
    ordered_coordinates = coordinates[order]
    ordered_frequencies = frequencies[order]

    figure, axis = pyplot.subplots(figsize=figsize)
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=frequencies,
        cmap='viridis',
        s=18,
        zorder=1,
    )
    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label('F0 (Hz)')
    jump_indices = _large_jump_indices(ordered_coordinates)
    jump_frequencies = ordered_frequencies[jump_indices]
    annotations = _annotate_landmarks(
        axis,
        coordinates,
        frequencies,
        jump_frequencies=jump_frequencies,
    )
    annotations.extend(_annotate_jumps(
        axis,
        ordered_coordinates,
        ordered_frequencies,
        jump_indices=jump_indices,
    ))

    axis.set_xlabel('UMAP 1')
    axis.set_ylabel('UMAP 2')
    axis.set_title(title)
    figure.tight_layout()
    _spread_annotation_labels(figure, axis, annotations)
    axis.plot(
        ordered_coordinates[:, 0],
        ordered_coordinates[:, 1],
        color='#D62728',
        linestyle='--',
        linewidth=0.6,
        zorder=2,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return figure, axis


def _validated_frequencies(values):
    try:
        frequencies = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('y must contain numeric frequencies') from error
    if frequencies.ndim != 1 or not frequencies.size:
        raise ValueError('y must be a non-empty one-dimensional array')
    if not np.all(np.isfinite(frequencies)):
        raise ValueError('y contains non-finite frequencies')
    positive_message = 'y frequencies must be positive'
    if np.any(frequencies <= 0): raise ValueError(positive_message)
    return frequencies


def _annotate_landmarks(
    axis,
    coordinates,
    frequencies,
    *,
    jump_frequencies=(),
):
    landmark_indices = []
    annotations = []
    for frequency in _F0_LANDMARKS_HZ:
        close_matches = np.isclose(frequencies, frequency)
        matches = np.flatnonzero(close_matches)
        if not matches.size: continue
        index = matches[0]
        landmark_indices.append(index)
        if np.any(np.isclose(jump_frequencies, frequency)): continue
        if frequency < 1000: label = f'{frequency} Hz'
        else: label = f'{frequency // 1000} kHz'
        annotation = axis.annotate(
            label,
            coordinates[index],
            xytext=_LABEL_CANDIDATE_OFFSETS[0],
            textcoords='offset points',
            bbox=_label_background(),
            zorder=4,
        )
        annotations.append(annotation)
    if landmark_indices:
        landmark_coordinates = coordinates[landmark_indices]
        axis.scatter(
            landmark_coordinates[:, 0],
            landmark_coordinates[:, 1],
            color='red',
            edgecolors='white',
            linewidths=0.5,
            marker='*',
            s=80,
            zorder=3,
        )
    return annotations


def _annotate_jumps(axis, coordinates, frequencies, *, jump_indices=None):
    if jump_indices is None: jump_indices = _large_jump_indices(coordinates)
    annotations = []
    for index in jump_indices:
        frequency = frequencies[index]
        x_offset, y_offset = _LABEL_CANDIDATE_OFFSETS[0]
        annotation = axis.annotate(
            f'{_frequency_label(frequency)} jump',
            coordinates[index],
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            color='#D55E00',
            fontsize='small',
            horizontalalignment='left' if x_offset >= 0 else 'right',
            verticalalignment='bottom' if y_offset >= 0 else 'top',
            bbox=_label_background(),
            zorder=4,
            arrowprops={
                'arrowstyle': '->',
                'color': '#D55E00',
                'linewidth': 0.8,
            },
        )
        annotations.append(annotation)
    return annotations


def _spread_annotation_labels(figure, axis, annotations):
    '''Place labels without overlaps while keeping them inside the axes.'''

    if not annotations: return
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = _LABEL_PADDING_POINTS * figure.dpi / 72
    axes_box = _padded_bbox(axis.get_window_extent(renderer), -padding)
    anchor_points = [
        axis.transData.transform(annotation.xy)
        for annotation in annotations
    ]
    occupied_boxes = []

    for annotation, own_anchor in zip(annotations, anchor_points):
        best = None
        for candidate_number, offset in enumerate(_LABEL_CANDIDATE_OFFSETS):
            _position_annotation(annotation, offset)
            label_box = Text.get_window_extent(annotation, renderer=renderer)
            padded_box = _padded_bbox(label_box, padding)
            overlap = sum(
                _overlap_area(padded_box, occupied_box)
                for occupied_box in occupied_boxes
            )
            anchor_overlap = sum(
                padded_box.contains(*anchor)
                for anchor in anchor_points
                if not np.array_equal(anchor, own_anchor)
            )
            overflow = _bbox_overflow(padded_box, axes_box)
            distance = np.hypot(*offset)
            score = (
                overflow * 1_000_000
                + overlap * 10_000
                + anchor_overlap * 100_000
                + distance
                + candidate_number / 100
            )
            if best is None or score < best[0]:
                best = score, offset, padded_box
        _, offset, padded_box = best
        _position_annotation(annotation, offset)
        occupied_boxes.append(padded_box)


def _position_annotation(annotation, offset):
    x_offset, y_offset = offset
    annotation.set_position(offset)
    annotation.set_horizontalalignment(
        'left' if x_offset >= 0 else 'right')
    annotation.set_verticalalignment(
        'bottom' if y_offset >= 0 else 'top')


def _padded_bbox(box, padding):
    from matplotlib.transforms import Bbox

    return Bbox.from_extents(
        box.x0 - padding,
        box.y0 - padding,
        box.x1 + padding,
        box.y1 + padding,
    )


def _overlap_area(first, second):
    width = max(0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def _bbox_overflow(box, bounds):
    return (
        max(0, bounds.x0 - box.x0)
        + max(0, box.x1 - bounds.x1)
        + max(0, bounds.y0 - box.y0)
        + max(0, box.y1 - bounds.y1)
    )


def _label_background():
    return {
        'facecolor': 'white',
        'edgecolor': 'none',
        'alpha': 0.8,
        'pad': 0.5,
    }


def _large_jump_indices(coordinates):
    '''Return isolated points with two unusually long trajectory steps.'''

    if len(coordinates) < 4: return np.empty(0, dtype=int)
    step_distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    jump_indices = []
    for index in range(1, len(coordinates) - 1):
        start = max(0, index - _JUMP_NEIGHBORHOOD_SIZE - 1)
        stop = min(
            len(step_distances),
            index + _JUMP_NEIGHBORHOOD_SIZE + 1,
        )
        nearby_distances = np.concatenate((
            step_distances[start:index - 1],
            step_distances[index + 1:stop],
        ))
        if not nearby_distances.size: continue
        local_distance = np.median(nearby_distances)
        incident_distance = min(
            step_distances[index - 1],
            step_distances[index],
        )
        if local_distance == 0:
            is_jump = incident_distance > 0
        else:
            is_jump = (
                incident_distance
                >= _JUMP_DISTANCE_FACTOR * local_distance
            )
        if is_jump: jump_indices.append(index)
    return np.asarray(jump_indices, dtype=int)


def _frequency_label(frequency):
    if frequency < 1000: return f'{frequency:g} Hz'
    return f'{frequency / 1000:g} kHz'
