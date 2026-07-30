'''Speaker-balanced aggregation of local formant measurements.'''

import numpy as np
import pandas as pd


MEASUREMENT_COLUMNS = [
    'f0_hz', 'f1_hz', 'f2_hz', 'f3_hz', 'b1_hz', 'b2_hz', 'b3_hz'
]


def aggregate_speaker_measurements(token_data):
    '''Return one median row per speaker, gender, and monophthong.'''

    data = _successful_frame(token_data)
    required = {'speaker_id', 'gender', 'ipa'}
    _require_columns(data, required)
    values = [
        column for column in MEASUREMENT_COLUMNS
        if column in data and data[column].notna().any()
    ]
    groups = ['speaker_id', 'gender', 'ipa']
    medians = data.groupby(groups, dropna=False)[values].median().reset_index()
    counts = (
        data.groupby(groups, dropna=False)
        .size()
        .rename('n_tokens')
        .reset_index()
    )
    output = medians.merge(counts, on=groups)
    output['source'] = 'selected_phone_speakers'
    output['record_level'] = 'speaker_vowel_summary'
    output['aggregation'] = 'median across successful tokens'
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
    data = pd.DataFrame(speaker_data).copy()
    _require_columns(data, {'speaker_id', 'gender', 'ipa'})
    values = [
        column for column in MEASUREMENT_COLUMNS
        if column in data and data[column].notna().any()
    ]
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    rows = []
    for (gender, ipa), group in data.groupby(['gender', 'ipa'], dropna=False):
        row = {
            'source': 'selected_phone_genders',
            'record_level': 'group_summary',
            'gender': gender,
            'ipa': ipa,
            'n_speakers': group['speaker_id'].nunique(),
            'n_tokens': (
                int(group['n_tokens'].sum())
                if 'n_tokens' in group else len(group)
            ),
            'aggregation': 'median of per-speaker medians',
            'bootstrap_seed': seed,
            'bootstrap_replicates': n_bootstrap,
            'confidence': confidence,
        }
        for column in values:
            observed = group[column].dropna().to_numpy(dtype=float)
            if not observed.size:
                row[column] = np.nan
                row[f'{column}_ci_low'] = np.nan
                row[f'{column}_ci_high'] = np.nan
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
    return pd.DataFrame(rows)


def _successful_frame(data):
    output = pd.DataFrame(data).copy()
    if 'success' in output:
        output = output.loc[output['success'].fillna(False).astype(bool)]
    return output


def _require_columns(data, required):
    missing = required.difference(data.columns)
    if missing:
        names = ', '.join(sorted(missing))
        raise ValueError(f'missing required columns: {names}')
