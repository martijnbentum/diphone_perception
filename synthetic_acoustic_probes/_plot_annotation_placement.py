'''Global placement of Matplotlib annotation labels away from obstructions.'''

import numpy as np


_LABEL_CANDIDATE_RADII = (12, 24, 38, 54, 72, 92)
_LABEL_CANDIDATE_ANGLES = (45, -45, 135, -135, 20, -20, 160, -160)
_LABEL_PADDING_POINTS = 3
_POINT_CLEARANCE_POINTS = 3
_ANCHOR_CLEARANCE_POINTS = 6
_TRAJECTORY_CLEARANCE_POINTS = 2
_LABEL_OPTIMIZATION_PASSES = 8


def _make_label_candidate_offsets():
    offsets = []
    for radius in _LABEL_CANDIDATE_RADII:
        for angle in _LABEL_CANDIDATE_ANGLES:
            radians = np.deg2rad(angle)
            offset = (
                int(round(radius * np.cos(radians))),
                int(round(radius * np.sin(radians))),
            )
            if offset not in offsets: offsets.append(offset)
    return tuple(offsets)


_LABEL_CANDIDATE_OFFSETS = _make_label_candidate_offsets()


def _spread_annotation_labels(
    figure,
    axis,
    annotations,
    *,
    coordinates,
    trajectory,
):
    '''Globally place labels away from text, markers, and the trajectory.'''
    if not annotations: return ()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72
    padding = _LABEL_PADDING_POINTS * pixels_per_point
    point_clearance = _POINT_CLEARANCE_POINTS * pixels_per_point
    anchor_clearance = _ANCHOR_CLEARANCE_POINTS * pixels_per_point
    trajectory_clearance = _TRAJECTORY_CLEARANCE_POINTS * pixels_per_point
    axes_box = _padded_bbox(axis.get_window_extent(renderer), -padding)
    display_coordinates = axis.transData.transform(coordinates)
    display_trajectory = axis.transData.transform(trajectory)
    anchor_points = np.asarray([
        axis.transData.transform(annotation.xy)
        for annotation in annotations
    ])
    candidate_groups = tuple(
        _annotation_candidates(
            annotation,
            renderer,
            axes_box=axes_box,
            padding=padding,
            coordinates=display_coordinates,
            anchor_points=anchor_points,
            trajectory=display_trajectory,
            point_clearance=point_clearance,
            anchor_clearance=anchor_clearance,
            trajectory_clearance=trajectory_clearance,
        )
        for annotation in annotations
    )
    selected = _optimize_label_candidates(candidate_groups)
    for annotation, candidates, candidate_index in zip(
        annotations,
        candidate_groups,
        selected,
    ):
        _position_annotation(annotation, candidates[candidate_index]['offset'])
    return tuple(
        candidates[candidate_index]
        for candidates, candidate_index in zip(candidate_groups, selected)
    )


def _annotation_candidates(
    annotation,
    renderer,
    *,
    axes_box,
    padding,
    coordinates,
    anchor_points,
    trajectory,
    point_clearance,
    anchor_clearance,
    trajectory_clearance,
):
    from matplotlib.text import Text
    anchor = annotation.axes.transData.transform(annotation.xy)
    candidates = []
    for offset in _LABEL_CANDIDATE_OFFSETS:
        _position_annotation(annotation, offset)
        text_box = Text.get_window_extent(annotation, renderer=renderer)
        box = _padded_bbox(text_box, padding)
        leader = None
        if annotation.arrow_patch is not None:
            leader = _leader_segment(anchor, box)
        candidates.append({
            'offset': offset,
            'box': box,
            'leader': leader,
            'overflow': _bbox_overflow(box, axes_box),
            'anchor_obstructions': _points_in_box(
                anchor_points,
                box,
                anchor_clearance,
            ),
            'point_obstructions': _points_in_box(
                coordinates,
                box,
                point_clearance,
            ),
            'trajectory_obstructions': int(_polyline_intersects_box(
                trajectory,
                box,
                trajectory_clearance,
            )),
            'leader_anchor_obstructions': _leader_point_obstructions(
                leader,
                anchor_points,
                anchor_clearance,
            ),
            'leader_point_obstructions': _leader_point_obstructions(
                leader,
                coordinates,
                point_clearance,
            ),
            'distance': float(np.hypot(*offset)),
        })
    return tuple(candidates)


def _optimize_label_candidates(candidate_groups):
    count = len(candidate_groups)
    natural_order = tuple(range(count))
    orders = (
        natural_order,
        tuple(reversed(natural_order)),
        tuple(sorted(
            natural_order,
            key=lambda index: candidate_groups[index][0]['box'].width,
            reverse=True,
        )),
    )
    best_selection = None
    best_score = None
    for order in orders:
        selection = _greedy_label_selection(candidate_groups, order)
        selection = _refine_label_selection(
            candidate_groups,
            selection,
            order,
        )
        score = _label_layout_score(candidate_groups, selection)
        comparison = score, tuple(selection)
        if best_score is None or comparison < best_score:
            best_score = comparison
            best_selection = selection
    return tuple(best_selection)


def _greedy_label_selection(candidate_groups, order):
    selection = [None] * len(candidate_groups)
    for annotation_index in order:
        choices = (
            (_candidate_layout_score(
                candidate_groups,
                selection,
                annotation_index,
                candidate_index,
            ), candidate_index)
            for candidate_index in range(
                len(candidate_groups[annotation_index]))
        )
        _, selection[annotation_index] = min(choices)
    return selection


def _refine_label_selection(candidate_groups, selection, order):
    selection = list(selection)
    for pass_number in range(_LABEL_OPTIMIZATION_PASSES):
        changed = False
        update_order = order if pass_number % 2 == 0 else tuple(reversed(order))
        for annotation_index in update_order:
            choices = (
                (_candidate_layout_score(
                    candidate_groups,
                    selection,
                    annotation_index,
                    candidate_index,
                ), candidate_index)
                for candidate_index in range(
                    len(candidate_groups[annotation_index]))
            )
            _, candidate_index = min(choices)
            if candidate_index != selection[annotation_index]:
                selection[annotation_index] = candidate_index
                changed = True
        if not changed: break
    return selection


def _candidate_layout_score(
    candidate_groups,
    selection,
    annotation_index,
    candidate_index,
):
    candidate = candidate_groups[annotation_index][candidate_index]
    interactions = _empty_label_interactions()
    for other_index, other_candidate_index in enumerate(selection):
        if other_index == annotation_index or other_candidate_index is None:
            continue
        other = candidate_groups[other_index][other_candidate_index]
        _add_label_interactions(interactions, candidate, other)
    return _combined_label_score(candidate, interactions)


def _label_layout_score(candidate_groups, selection):
    totals = _empty_label_interactions()
    static = {
        name: 0
        for name in (
            'overflow',
            'anchor_obstructions',
            'point_obstructions',
            'trajectory_obstructions',
            'leader_anchor_obstructions',
            'leader_point_obstructions',
            'distance',
        )
    }
    selected = [
        candidates[candidate_index]
        for candidates, candidate_index in zip(candidate_groups, selection)
    ]
    for candidate in selected:
        for name in static: static[name] += candidate[name]
    for index, candidate in enumerate(selected):
        for other in selected[index + 1:]:
            _add_label_interactions(totals, candidate, other)
    return _combined_label_score(static, totals)


def _empty_label_interactions():
    return {
        'label_overlaps': 0,
        'label_overlap_area': 0.0,
        'leader_label_crossings': 0,
        'leader_crossings': 0,
    }


def _add_label_interactions(totals, first, second):
    overlap_area = _overlap_area(first['box'], second['box'])
    if overlap_area:
        totals['label_overlaps'] += 1
        totals['label_overlap_area'] += overlap_area
    first_leader = first['leader']
    second_leader = second['leader']
    if first_leader is not None:
        totals['leader_label_crossings'] += int(
            _segment_intersects_box(first_leader, second['box']))
    if second_leader is not None:
        totals['leader_label_crossings'] += int(
            _segment_intersects_box(second_leader, first['box']))
    if first_leader is not None and second_leader is not None:
        totals['leader_crossings'] += int(
            _segments_intersect(first_leader, second_leader))


def _combined_label_score(candidate, interactions):
    return (
        candidate['overflow'],
        interactions['label_overlaps'],
        interactions['label_overlap_area'],
        candidate['anchor_obstructions'],
        candidate['point_obstructions'],
        candidate['trajectory_obstructions'],
        interactions['leader_label_crossings'],
        candidate['leader_anchor_obstructions'],
        candidate['leader_point_obstructions'],
        interactions['leader_crossings'],
        candidate['distance'],
    )


def _leader_segment(anchor, box):
    endpoint = np.array([
        np.clip(anchor[0], box.x0, box.x1),
        np.clip(anchor[1], box.y0, box.y1),
    ])
    return np.vstack((anchor, endpoint))


def _points_in_box(points, box, clearance):
    if not len(points): return 0
    return int(np.count_nonzero(
        (points[:, 0] >= box.x0 - clearance)
        & (points[:, 0] <= box.x1 + clearance)
        & (points[:, 1] >= box.y0 - clearance)
        & (points[:, 1] <= box.y1 + clearance)
    ))


def _leader_point_obstructions(leader, points, clearance):
    if leader is None or not len(points): return 0
    distances = _point_segment_distances(points, leader)
    own_anchor = np.linalg.norm(points - leader[0], axis=1) <= clearance
    return int(np.count_nonzero((distances <= clearance) & ~own_anchor))


def _point_segment_distances(points, segment):
    delta = segment[1] - segment[0]
    length_squared = np.dot(delta, delta)
    if length_squared == 0:
        return np.linalg.norm(points - segment[0], axis=1)
    fractions = np.clip(
        ((points - segment[0]) @ delta) / length_squared,
        0,
        1,
    )
    projections = segment[0] + fractions[:, np.newaxis] * delta
    return np.linalg.norm(points - projections, axis=1)


def _polyline_intersects_box(points, box, padding=0):
    if len(points) < 2: return False
    starts = points[:-1]
    stops = points[1:]
    return bool(np.any(_segments_intersect_box(
        starts,
        stops,
        _padded_bbox(box, padding),
    )))


def _segment_intersects_box(segment, box):
    return bool(_segments_intersect_box(
        segment[:1],
        segment[1:],
        box,
    )[0])


def _segments_intersect_box(starts, stops, box):
    deltas = stops - starts
    lower = np.zeros(len(starts), dtype=float)
    upper = np.ones(len(starts), dtype=float)
    valid = np.ones(len(starts), dtype=bool)
    boundaries = (
        (-deltas[:, 0], starts[:, 0] - box.x0),
        (deltas[:, 0], box.x1 - starts[:, 0]),
        (-deltas[:, 1], starts[:, 1] - box.y0),
        (deltas[:, 1], box.y1 - starts[:, 1]),
    )
    for denominator, numerator in boundaries:
        parallel = np.isclose(denominator, 0)
        valid &= ~(parallel & (numerator < 0))
        active = ~parallel
        ratios = np.zeros_like(numerator)
        ratios[active] = numerator[active] / denominator[active]
        entering = active & (denominator < 0)
        leaving = active & (denominator > 0)
        lower[entering] = np.maximum(lower[entering], ratios[entering])
        upper[leaving] = np.minimum(upper[leaving], ratios[leaving])
    return valid & (lower <= upper)


def _segments_intersect(first, second):
    first_start, first_stop = first
    second_start, second_stop = second
    first_sides = (
        _cross_product(first_start, first_stop, second_start),
        _cross_product(first_start, first_stop, second_stop),
    )
    second_sides = (
        _cross_product(second_start, second_stop, first_start),
        _cross_product(second_start, second_stop, first_stop),
    )
    return (
        first_sides[0] * first_sides[1] <= 0
        and second_sides[0] * second_sides[1] <= 0
        and _segment_bounds_overlap(first, second)
    )


def _cross_product(start, stop, point):
    direction = stop - start
    relative = point - start
    return direction[0] * relative[1] - direction[1] * relative[0]


def _segment_bounds_overlap(first, second):
    return (
        max(first[:, 0].min(), second[:, 0].min())
        <= min(first[:, 0].max(), second[:, 0].max())
        and max(first[:, 1].min(), second[:, 1].min())
        <= min(first[:, 1].max(), second[:, 1].max())
    )


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
