'''Plots of source-specific Dutch vowel anchors in F1/F2 space.'''

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from vowel_formant_reference.formant_tables import (
    adank_2004_formants,
    load_formant_table,
    pols_1973_formants,
    van_nierop_1973_formants,
    weenink_1985_formants,
)


DEFAULT_VOWEL_PLOT_SOURCE_IDS = (
    'cgn_selected_phones',
    'pols_1973_male',
    'van_nierop_1973_female',
    'weenink_1985',
    'adank_2004_nsd',
    'adank_2004_ssd',
)

DEFAULT_VOWEL_PLOT_GENDERS = ('female', 'male')

_VOWEL_EXAMPLES = (
    ('ɪ', 'pit'),
    ('ɛ', 'pet'),
    ('ɑ', 'pat'),
    ('ɔ', 'pot'),
    ('ʉ', 'put'),
    ('iː', 'biet'),
    ('yː', 'fuut'),
    ('eː', 'beet'),
    ('øː', 'neus'),
    ('aː', 'maat'),
    ('oː', 'boot'),
    ('uː', 'boek'),
    ('ə', 'de'),
)


@dataclass(frozen=True)
class _PlotSource:
    title: str
    population: str


_PLOT_SOURCES = {
    'cgn_selected_phones': _PlotSource(
        title='CGN selected phones',
        population='CGN selected phones',
    ),
    'pols_1973_male': _PlotSource(
        title='Pols et al. 1973',
        population='Dutch adults',
    ),
    'van_nierop_1973_female': _PlotSource(
        title='Van Nierop et al. 1973',
        population='Dutch adults',
    ),
    'weenink_1985': _PlotSource(
        title='Weenink 1985',
        population='Dutch adult',
    ),
    'adank_2004_nsd': _PlotSource(
        title='Adank et al. 2004 — NSD',
        population='Northern Standard Dutch',
    ),
    'adank_2004_ssd': _PlotSource(
        title='Adank et al. 2004 — SSD',
        population='Southern Standard Dutch',
    ),
}

_GENDER_COLORS = {
    'female': '#D55E00',
    'male': '#0072B2',
}

_GENDER_MARKERS = {
    'female': 'o',
    'male': 's',
}


def plot_vowel_formant_space(
    *,
    panel_by,
    source_ids=DEFAULT_VOWEL_PLOT_SOURCE_IDS,
    genders=DEFAULT_VOWEL_PLOT_GENDERS,
    data_root=None,
    output_path='',
    add_examples=False,
    figsize=None,
    dpi=300,
):
    '''Plot Dutch vowel anchors with panels organized by source or gender.

    ``panel_by`` must be either ``'source'`` or ``'gender'``. ``source_ids``
    and ``genders`` explicitly control which data are included. The returned
    axes are a one-dimensional NumPy array in panel order. Set
    ``add_examples=True`` to append an aligned vowel/example-word panel.
    When ``output_path`` is provided, the same figure is also saved there.
    '''

    from matplotlib import pyplot

    source_ids = _validated_source_ids(source_ids)
    genders = _validated_genders(genders)
    if panel_by not in {'source', 'gender'}:
        raise ValueError("panel_by must be either 'source' or 'gender'")
    if not isinstance(add_examples, (bool, np.bool_)):
        raise TypeError('add_examples must be a boolean')
    rows = _selected_rows(source_ids, genders, data_root)
    if not rows:
        raise ValueError('no vowel anchors match the selected sources/genders')

    panel_values = source_ids if panel_by == 'source' else genders
    figure, axes = _figure_and_axes(
        len(panel_values) + int(add_examples),
        panel_by,
        figsize,
        pyplot,
    )
    data_axes = axes[:len(panel_values)]
    if panel_by == 'source':
        _plot_source_panels(data_axes, panel_values, genders, rows)
    else:
        _plot_gender_panels(
            data_axes,
            panel_values,
            source_ids,
            rows,
            pyplot,
        )
    _format_vowel_axes(data_axes, rows)
    if add_examples:
        _plot_examples_panel(axes[-1])
    figure.suptitle('Dutch vowel formant space')
    figure.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return figure, axes


def _selected_rows(source_ids, genders, data_root):
    rows = []
    for source_id in source_ids:
        for row in _source_rows(source_id, data_root):
            if row['gender'] not in genders:
                continue
            _validate_formant_row(source_id, row)
            rows.append({**row, 'plot_source_id': source_id})
    return rows


def _source_rows(source_id, data_root):
    spec = _PLOT_SOURCES[source_id]
    if source_id == 'cgn_selected_phones':
        table = load_formant_table(
            'gender_formants',
            view='native',
            data_root=data_root,
        )
        return [
            {**row, 'population': spec.population}
            for row in table.data
        ]
    if source_id == 'pols_1973_male':
        return pols_1973_formants(data_root=data_root).data
    if source_id == 'van_nierop_1973_female':
        return van_nierop_1973_formants(data_root=data_root).data
    if source_id == 'weenink_1985':
        return weenink_1985_formants(data_root=data_root).data
    return adank_2004_formants(
        population=spec.population,
        data_root=data_root,
    ).data


def _figure_and_axes(n_panels, panel_by, figsize, pyplot):
    if panel_by == 'gender':
        n_columns = n_panels
    else:
        n_columns = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_columns)
    if figsize is None:
        figsize = (5.5 * n_columns, 4.8 * n_rows)
    figure, axes_grid = pyplot.subplots(
        n_rows,
        n_columns,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid.ravel()
    for axis in axes[n_panels:]:
        axis.remove()
    return figure, axes[:n_panels]


def _plot_source_panels(axes, source_ids, genders, rows):
    for axis, source_id in zip(axes, source_ids):
        axis.set_title(_PLOT_SOURCES[source_id].title)
        has_data = False
        for gender in genders:
            selected = [
                row for row in rows
                if row['plot_source_id'] == source_id
                and row['gender'] == gender
            ]
            if not selected:
                continue
            has_data = True
            _plot_series(
                axis,
                selected,
                label=gender.capitalize(),
                color=_GENDER_COLORS[gender],
                marker=_GENDER_MARKERS[gender],
            )
        if has_data:
            axis.legend(title='Gender')
        else:
            _mark_empty_axis(axis)


def _plot_gender_panels(axes, genders, source_ids, rows, pyplot):
    color_map = pyplot.get_cmap('tab10')
    source_colors = {
        source_id: color_map(index % color_map.N)
        for index, source_id in enumerate(source_ids)
    }
    for axis, gender in zip(axes, genders):
        axis.set_title(gender.capitalize())
        has_data = False
        for source_id in source_ids:
            selected = [
                row for row in rows
                if row['plot_source_id'] == source_id
                and row['gender'] == gender
            ]
            if not selected:
                continue
            has_data = True
            _plot_series(
                axis,
                selected,
                label=_source_legend_label(source_id),
                color=source_colors[source_id],
                marker=_GENDER_MARKERS[gender],
            )
        if has_data:
            axis.legend(title='Source', fontsize='small')
        else:
            _mark_empty_axis(axis)


def _plot_series(axis, rows, label, color, marker):
    f2_values = [row['f2_hz'] for row in rows]
    f1_values = [row['f1_hz'] for row in rows]
    axis.scatter(
        f2_values,
        f1_values,
        label=label,
        color=color,
        marker=marker,
        alpha=0.8,
    )
    for row in rows:
        axis.annotate(
            row['ipa'],
            (row['f2_hz'], row['f1_hz']),
            xytext=(4, 4),
            textcoords='offset points',
            color=color,
            fontsize='small',
        )


def _source_legend_label(source_id):
    if source_id == 'cgn_selected_phones':
        return 'CGN selected vowels'
    return _PLOT_SOURCES[source_id].title


def _plot_examples_panel(axis):
    axis.set_axis_off()
    row_positions = np.linspace(0.92, 0.08, len(_VOWEL_EXAMPLES))
    for (ipa, word), position in zip(_VOWEL_EXAMPLES, row_positions):
        axis.text(0.04, position, ipa, transform=axis.transAxes)
        axis.text(0.17, position, word, transform=axis.transAxes)


def _format_vowel_axes(axes, rows):
    f1_values = [row['f1_hz'] for row in rows]
    f2_values = [row['f2_hz'] for row in rows]
    f1_margin = max(25, 0.05 * (max(f1_values) - min(f1_values)))
    f2_margin = max(50, 0.05 * (max(f2_values) - min(f2_values)))
    axes[0].set_xlim(
        max(f2_values) + f2_margin,
        min(f2_values) - f2_margin,
    )
    axes[0].set_ylim(
        max(f1_values) + f1_margin,
        min(f1_values) - f1_margin,
    )
    for axis in axes:
        axis.set_xlabel('F2 (Hz)')
        axis.set_ylabel('F1 (Hz)')
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)


def _mark_empty_axis(axis):
    axis.text(
        0.5,
        0.5,
        'No selected data',
        ha='center',
        va='center',
        transform=axis.transAxes,
    )


def _validated_source_ids(source_ids):
    if isinstance(source_ids, str):
        raise TypeError('source_ids must be an iterable of source IDs')
    source_ids = tuple(source_ids)
    if not source_ids:
        raise ValueError('at least one source_id is required')
    unknown = [
        source_id for source_id in source_ids
        if source_id not in _PLOT_SOURCES
    ]
    if unknown:
        choices = ', '.join(DEFAULT_VOWEL_PLOT_SOURCE_IDS)
        raise ValueError(
            f'unsupported source IDs {unknown!r}; choose from {choices}'
        )
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('source_ids must not contain duplicates')
    return source_ids


def _validated_genders(genders):
    if isinstance(genders, str):
        raise TypeError('genders must be an iterable of gender IDs')
    genders = tuple(genders)
    if not genders:
        raise ValueError('at least one gender is required')
    unknown = set(genders) - set(DEFAULT_VOWEL_PLOT_GENDERS)
    if unknown:
        raise ValueError(f'unsupported genders: {sorted(unknown)!r}')
    if len(set(genders)) != len(genders):
        raise ValueError('genders must not contain duplicates')
    return genders


def _validate_formant_row(source_id, row):
    for name in ('f1_hz', 'f2_hz'):
        value = row.get(name)
        if (
            not isinstance(value, (int, float, np.number))
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f'{source_id!r} contains a non-finite {name} value'
            )
