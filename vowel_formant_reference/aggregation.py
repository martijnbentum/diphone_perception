'''Speaker-balanced aggregation of local formant measurements.'''

from collections.abc import Mapping

import numpy as np


MEASUREMENT_COLUMNS = [
    'f0_hz', 'f1_hz', 'f2_hz', 'f3_hz', 'b1_hz', 'b2_hz', 'b3_hz'
]


def aggregate_speaker_measurements(token_data):
    '''Return one median row per speaker, gender, and monophthong.'''

    data = _successful_records(token_data)
    required = {'speaker_id', 'gender', 'ipa'}
    _require_columns(data, required)
    values = [
        column for column in MEASUREMENT_COLUMNS
        if any(_is_number(row.get(column)) for row in data)
    ]
    groups = _group_records(data, ('speaker_id', 'gender', 'ipa'))
    output = []
    for (speaker_id, gender, ipa), rows in groups.items():
        row = {
            'speaker_id': speaker_id,
            'gender': gender,
            'ipa': ipa,
            'n_tokens': len(rows),
            'source': 'phone_formant_speakers',
            'record_level': 'speaker_vowel_summary',
            'aggregation': 'median across successful tokens',
        }
        for column in values:
            observed = [
                item[column] for item in rows
                if _is_number(item.get(column))
            ]
            row[column] = (
                float(np.median(observed))
                if observed else None
            )
        output.append(row)
    return output


def aggregate_gender_measurements(
    speaker_data,
    n_bootstrap=1000,
    confidence=0.95,
    seed=0,
):
    '''Return median-of-speaker-medians summaries with bootstrap intervals.'''

    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be at least 1')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    data = _records(speaker_data)
    _require_columns(data, {'speaker_id', 'gender', 'ipa'})
    values = [
        column for column in MEASUREMENT_COLUMNS
        if any(_is_number(row.get(column)) for row in data)
    ]
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    rows = []
    for (gender, ipa), group in _group_records(
        data, ('gender', 'ipa')
    ).items():
        row = {
            'source': 'gender_formants',
            'record_level': 'group_summary',
            'gender': gender,
            'ipa': ipa,
            'n_speakers': len({
                item.get('speaker_id') for item in group
                if item.get('speaker_id') is not None
            }),
            'n_tokens': int(sum(
                (
                    item['n_tokens']
                    if _is_number(item.get('n_tokens')) else 1
                )
                for item in group
            )),
            'aggregation': 'median of per-speaker medians',
            'bootstrap_seed': seed,
            'bootstrap_replicates': n_bootstrap,
            'confidence': confidence,
        }
        for column in values:
            observed = np.asarray([
                item[column] for item in group
                if _is_number(item.get(column))
            ], dtype=float)
            if not observed.size:
                row[column] = None
                row[f'{column}_ci_low'] = None
                row[f'{column}_ci_high'] = None
                continue
            row[column] = float(np.median(observed))
            bootstrap = np.median(
                rng.choice(
                    observed,
                    size=(n_bootstrap, observed.size),
                    replace=True,
                ),
                axis=1,
            )
            row[f'{column}_ci_low'] = float(np.quantile(bootstrap, alpha))
            row[f'{column}_ci_high'] = float(
                np.quantile(bootstrap, 1 - alpha)
            )
        rows.append(row)
    return rows


def _successful_records(data):
    return [
        row for row in _records(data)
        if 'success' not in row or _boolean(row['success'])
    ]


def _records(data):
    if isinstance(data, Mapping):
        columns = list(data)
        values = [list(data[column]) for column in columns]
        lengths = {len(column) for column in values}
        if len(lengths) > 1:
            raise ValueError('column values have different lengths')
        return [
            dict(zip(columns, row))
            for row in zip(*values)
        ]
    try:
        return [dict(row) for row in data]
    except (TypeError, ValueError) as error:
        raise TypeError('formant data must contain mapping records') from error


def _group_records(records, columns):
    groups = {}
    for row in records:
        key = tuple(row.get(column) for column in columns)
        groups.setdefault(key, []).append(row)
    return groups


def _require_columns(data, required):
    columns = set().union(*(row.keys() for row in data)) if data else set()
    missing = required.difference(columns)
    if missing:
        names = ', '.join(sorted(missing))
        raise ValueError(f'missing required columns: {names}')


def _is_number(value):
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _boolean(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)
