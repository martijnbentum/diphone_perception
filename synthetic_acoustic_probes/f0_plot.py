'''Visualization of the pure-tone F0 representation space.'''

from collections.abc import Mapping
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


def plot_f0_checkpoint_smoothness(
    checkpoint_metrics,
    *,
    output_path=None,
    figsize=None,
    dpi=300,
):
    '''Plot adjacent-distance summaries against numeric checkpoint step.

    checkpoint_metrics:  Iterable returned by
                         ``load_f0_checkpoint_metrics``.
    output_path:         Optional destination; omit it to disable saving.
    figsize:             Optional Matplotlib figure size.
    dpi:                 Resolution used when saving.

    Returns the Matplotlib figure and three axes for typical distances,
    extreme distances, and threshold-exceedance fractions.
    '''

    from matplotlib import pyplot

    rows = _validated_checkpoint_metrics(checkpoint_metrics)
    steps = np.asarray([row['checkpoint_step'] for row in rows])
    figure, axes = pyplot.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
    )
    for key, label in (
        ('median', 'Median'),
        ('mean', 'Mean'),
        ('p95', 'P95'),
    ):
        axes[0].plot(
            steps,
            [row[key] for row in rows],
            marker='o',
            label=label,
        )
    for key, label in (('p99', 'P99'), ('maximum', 'Maximum')):
        axes[1].plot(
            steps,
            [row[key] for row in rows],
            marker='o',
            label=label,
        )

    thresholds = rows[0]['thresholds']
    fractions = np.vstack([
        row['fractions_above_threshold']
        for row in rows
    ])
    for index, threshold in enumerate(thresholds):
        axes[2].plot(
            steps,
            fractions[:, index],
            marker='o',
            label=f'> {threshold:g}',
        )

    axes[0].set_ylabel('Adjacent cosine distance')
    axes[1].set_ylabel('Adjacent cosine distance')
    axes[2].set_ylabel('Fraction of edges')
    axes[2].set_xlabel('Checkpoint step')
    axes[2].set_ylim(bottom=0)
    axes[2].ticklabel_format(style='plain', axis='x')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle('F0 trajectory smoothness over checkpoints')
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)
    return figure, axes


def plot_f0_checkpoint_distance_heatmap(
    checkpoint_metrics,
    *,
    output_path=None,
    figsize=None,
    dpi=300,
    cmap='magma',
    vmax=None,
):
    '''Plot every adjacent-frequency distance across checkpoints.

    checkpoint_metrics:  Iterable returned by
                         ``load_f0_checkpoint_metrics``.
    output_path:         Optional destination; omit it to disable saving.
    figsize:             Optional Matplotlib figure size.
    dpi:                 Resolution used when saving.
    cmap:                Matplotlib colormap for cosine distance.
    vmax:                Optional shared upper color limit.

    Checkpoints are equally spaced columns labeled by their numeric training
    steps; frequency edges retain their physical Hz boundaries. Returns the
    Matplotlib figure and primary axis.
    '''

    from matplotlib import pyplot

    rows = _validated_checkpoint_metrics(
        checkpoint_metrics,
        require_edges=True,
    )
    if vmax is not None:
        if not np.isscalar(vmax) or not np.isfinite(vmax) or vmax <= 0:
            raise ValueError('vmax must be finite and positive')

    frequency_edges = rows[0]['frequency_edges_hz']
    frequency_boundaries = np.concatenate((
        frequency_edges[:1, 0],
        frequency_edges[:, 1],
    ))
    distances = np.vstack([
        row['adjacent_distances']
        for row in rows
    ])
    checkpoint_boundaries = np.arange(len(rows) + 1)

    figure, axis = pyplot.subplots(figsize=figsize)
    image = axis.pcolormesh(
        checkpoint_boundaries,
        frequency_boundaries,
        distances.T,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        shading='flat',
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label('Adjacent cosine distance')
    axis.set_xticks(
        np.arange(len(rows)) + 0.5,
        [f"{row['checkpoint_step']:,}" for row in rows],
    )
    axis.set_xlabel('Checkpoint step')
    axis.set_ylabel('Frequency edge (Hz)')
    axis.set_title('F0 adjacent-distance trajectory over checkpoints')
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)
    return figure, axis


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
        _save_figure(figure, output_path, dpi)
    return figure, axis


def _validated_checkpoint_metrics(values, *, require_edges=False):
    if isinstance(values, Mapping):
        raise ValueError('checkpoint_metrics must be an iterable of rows')
    try:
        rows = tuple(values)
    except TypeError as error:
        raise ValueError(
            'checkpoint_metrics must be an iterable of rows'
        ) from error
    if not rows:
        raise ValueError('checkpoint_metrics must not be empty')

    required = {
        'checkpoint_step',
        'fractions_above_threshold',
        'maximum',
        'mean',
        'median',
        'p95',
        'p99',
        'thresholds',
    }
    if require_edges:
        required.update({'adjacent_distances', 'frequency_edges_hz'})

    output = []
    seen_steps = set()
    reference_thresholds = None
    reference_edges = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('each checkpoint metric row must be a mapping')
        missing = required - set(row)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'checkpoint metric row is missing: {names}')
        step = row['checkpoint_step']
        if (
            isinstance(step, (bool, np.bool_))
            or not isinstance(step, (int, np.integer))
            or step < 0
        ):
            raise ValueError('checkpoint_step must be a non-negative integer')
        step = int(step)
        if step in seen_steps:
            raise ValueError(f'duplicate checkpoint_step: {step}')
        seen_steps.add(step)

        normalized = dict(row)
        normalized['checkpoint_step'] = step
        summary = np.asarray([
            _finite_metric(row, name)
            for name in ('mean', 'median', 'p95', 'p99', 'maximum')
        ])
        if np.any(summary < 0) or np.any(summary > 2):
            raise ValueError('cosine-distance summaries must be in [0, 2]')
        if not summary[1] <= summary[2] <= summary[3] <= summary[4]:
            raise ValueError('cosine-distance quantiles are not ordered')

        thresholds = _metric_vector(row['thresholds'], 'thresholds')
        if np.any(thresholds <= 0) or np.any(thresholds >= 2):
            raise ValueError('thresholds must lie strictly between zero and two')
        fractions = _metric_vector(
            row['fractions_above_threshold'],
            'fractions_above_threshold',
        )
        if fractions.shape != thresholds.shape:
            raise ValueError('thresholds and fractions have different lengths')
        if np.any(fractions < 0) or np.any(fractions > 1):
            raise ValueError('threshold fractions must be in [0, 1]')
        if reference_thresholds is None:
            reference_thresholds = thresholds
        elif not np.array_equal(reference_thresholds, thresholds):
            raise ValueError('checkpoint thresholds do not match')
        normalized['thresholds'] = thresholds
        normalized['fractions_above_threshold'] = fractions

        if require_edges:
            adjacent = _metric_vector(
                row['adjacent_distances'],
                'adjacent_distances',
            )
            if np.any(adjacent < 0) or np.any(adjacent > 2):
                raise ValueError('adjacent distances must be in [0, 2]')
            edges = np.asarray(row['frequency_edges_hz'], dtype=float)
            if edges.shape != (adjacent.size, 2):
                raise ValueError(
                    'frequency_edges_hz must contain one pair per distance'
                )
            if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
                raise ValueError('frequency edges must be finite and positive')
            if np.any(edges[:, 0] >= edges[:, 1]):
                raise ValueError('frequency edges must increase')
            if not np.array_equal(edges[:-1, 1], edges[1:, 0]):
                raise ValueError('frequency edges must be contiguous')
            if reference_edges is None:
                reference_edges = edges
            elif not np.array_equal(reference_edges, edges):
                raise ValueError('checkpoint frequency edges do not match')
            normalized['adjacent_distances'] = adjacent
            normalized['frequency_edges_hz'] = edges
        output.append(normalized)

    output.sort(key=lambda row: row['checkpoint_step'])
    return tuple(output)


def _finite_metric(row, name):
    value = row[name]
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a finite scalar')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite scalar') from error
    if not np.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar')
    return value


def _metric_vector(values, name):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a numeric vector') from error
    if vector.ndim != 1 or not vector.size:
        raise ValueError(f'{name} must be a non-empty vector')
    if not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain finite values')
    return vector


def _save_figure(figure, output_path, dpi):
    if not output_path: return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')


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
