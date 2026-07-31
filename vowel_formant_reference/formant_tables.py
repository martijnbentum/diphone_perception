'''Source-separated Dutch monophthong formant tables.

The public loader never pools sources. A table can be requested in its native
schema, a standardized schema, or as a within-source summary.
'''

from collections.abc import Mapping
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / 'formants'

LITERATURE_MONOPHTHONGS = (
    'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
    'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
)

_LITERATURE_TABLES = {
    'pols_1973_male',
    'van_nierop_1973_female',
    'weenink_1985',
    'adank_2004_table_1_monophthongs',
}


@dataclass(frozen=True)
class FormantSource:
    '''Provenance and storage information for one physical table.'''

    name: str
    relative_path: str
    reference: str
    url: str | None
    page_range: str | None
    source_page: str | None
    table_number: str | None
    record_level: str
    notes: str
    schema_version: int = 1

    def path(self, data_root=None):
        root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        return root / self.relative_path


@dataclass(frozen=True)
class FormantTable:
    '''One formant table together with source metadata.'''

    source: FormantSource
    data: list[dict]
    view: str

    @property
    def name(self):
        return self.source.name

    def citation_metadata(self):
        return asdict(self.source)


_SOURCES = {
    'pols_1973_male': FormantSource(
        name='pols_1973_male',
        relative_path='literature/pols_1973_male.csv',
        reference=(
            'Pols, L. C. W., Tromp, H. R. C., & Plomp, R. (1973). '
            'Frequency analysis of Dutch vowels from 50 male speakers. '
            'Journal of the Acoustical Society of America, 53, 1093-1101.'
        ),
        url='https://www.fon.hum.uva.nl/praat/manual/Pols_et_al___1973_.html',
        page_range='1093-1101',
        source_page=None,
        table_number=None,
        record_level='speaker_vowel',
        notes=(
            'Male rows extracted from Praat command "Create formant table '
            '(Pols & Van Nierop 1973)".'
        ),
    ),
    'van_nierop_1973_female': FormantSource(
        name='van_nierop_1973_female',
        relative_path='literature/van_nierop_1973_female.csv',
        reference=(
            'Van Nierop, D. J. P. J., Pols, L. C. W., & Plomp, R. '
            '(1973). Frequency analysis of Dutch vowels from 25 female '
            'speakers. Acustica, 29, 110-118.'
        ),
        url=(
            'https://www.fon.hum.uva.nl/praat/manual/'
            'Van_Nierop_et_al___1973_.html'
        ),
        page_range='110-118',
        source_page=None,
        table_number=None,
        record_level='speaker_vowel',
        notes=(
            'Female rows extracted from Praat command "Create formant table '
            '(Pols & Van Nierop 1973)".'
        ),
    ),
    'weenink_1985': FormantSource(
        name='weenink_1985',
        relative_path='literature/weenink_1985.csv',
        reference=(
            'Weenink, D. J. M. (1985). Formant analysis of Dutch vowels '
            'from 10 children. Proceedings of the Institute of Phonetic '
            'Sciences of the University of Amsterdam, 9, 45-52.'
        ),
        url='https://www.fon.hum.uva.nl/praat/manual/Weenink__1985_.html',
        page_range='45-52',
        source_page=None,
        table_number=None,
        record_level='speaker_vowel',
        notes=(
            'Created by Praat command "Create formant table (Weenink 1985)".'
        ),
    ),
    'adank_2004_table_1_monophthongs': FormantSource(
        name='adank_2004_table_1_monophthongs',
        relative_path='literature/adank_2004_table_1_monophthongs.csv',
        reference=(
            'Adank, P., Van Hout, R., & Smits, R. (2004). An acoustic '
            'description of the vowels of Northern and Southern Standard '
            'Dutch. Journal of the Acoustical Society of America, 116(3), '
            '1729-1738.'
        ),
        url='https://doi.org/10.1121/1.1779271',
        page_range='1729-1738',
        source_page='1731',
        table_number='I',
        record_level='group_summary',
        notes=(
            'Monophthong rows only. Group means at the temporal midpoint; '
            'N=20 per sex/region cell. Dynamic Table II is out of scope.'
        ),
    ),
    'phone_formants': FormantSource(
        name='phone_formants',
        relative_path='selected_phones/phone_formants.csv',
        reference='Local selected-phone corpus measurements.',
        url=None,
        page_range=None,
        source_page=None,
        table_number=None,
        record_level='token',
        notes=(
            'Praat measurements from the stable central portion of selected '
            'monophthongs. Includes successful and rejected tokens.'
        ),
    ),
    'gender_formants': FormantSource(
        name='gender_formants',
        relative_path='selected_phones/gender_formants.csv',
        reference='Local selected-phone corpus, speaker-balanced summaries.',
        url=None,
        page_range=None,
        source_page=None,
        table_number=None,
        record_level='group_summary',
        notes=(
            'Median of per-speaker medians with speaker-bootstrap intervals.'
        ),
    ),
}

_PRAAT_IPA_TO_PROJECT = {
    'u': 'uː',
    'a': 'aː',
    'o': 'oː',
    r'\as': 'ɑ',
    r'\o/': 'øː',
    'i': 'iː',
    'y': 'yː',
    'e': 'eː',
    r'\yc': 'ʉ',
    r'\ep': 'ɛ',
    r'\ct': 'ɔ',
    r'\ic': 'ɪ',
}

_ADANK_IPA_TO_PROJECT = {
    'ɑ': 'ɑ',
    'a': 'aː',
    'ɛ': 'ɛ',
    'e': 'eː',
    'ø': 'øː',
    'ɪ': 'ɪ',
    'i': 'iː',
    'ɔ': 'ɔ',
    'u': 'uː',
    'o': 'oː',
    'ʏ': 'ʉ',
    'y': 'yː',
}

_STANDARD_COLUMNS = [
    'source',
    'dataset',
    'record_level',
    'population',
    'speaker_id',
    'speaker_type',
    'gender',
    'age',
    'vowel_label',
    'source_ipa',
    'ipa',
    'f0_hz',
    'f1_hz',
    'f2_hz',
    'f3_hz',
    'l1_db',
    'l2_db',
    'l3_db',
    'b1_hz',
    'b2_hz',
    'b3_hz',
    'duration_seconds',
    'stress',
    'n_speakers',
    'n_tokens',
    'aggregation',
    'ci_low',
    'ci_high',
    'success',
    'rejection_reason',
    'provenance',
]


def registered_formant_tables():
    '''Return every registered table name, whether generated locally or not.'''

    return tuple(_SOURCES)


def available_formant_tables(data_root=None):
    '''Return registered table names whose data files currently exist.'''

    return tuple(
        name for name, source in _SOURCES.items()
        if source.path(data_root).exists()
    )


def formant_source(name):
    try:
        return _SOURCES[name]
    except KeyError as error:
        choices = ', '.join(_SOURCES)
        raise KeyError(
            f'unknown formant table {name!r}; choose from {choices}'
        ) from error


def load_formant_table(name, view='native', data_root=None):
    '''Load one source without combining it with any other source.'''

    if view not in {'native', 'standardized', 'summary'}:
        raise ValueError(
            f'unknown table view {view!r}; use native, standardized, or summary'
        )
    source = formant_source(name)
    path = source.path(data_root)
    if not path.exists():
        raise FileNotFoundError(
            f'formant table {name!r} has not been generated: {path}'
        )
    data = _read_table(path)
    if view == 'standardized':
        data = _standardize(source, data)
    elif view == 'summary':
        data = _within_source_summary(source, _standardize(source, data))
    return FormantTable(source=source, data=data, view=view)


def literature_gender_formants(
    name='weenink_1985',
    population=None,
    data_root=None,
):
    '''Return source-specific adult formant anchors grouped by gender.

    The default Weenink table supplies F0 and F1--F3 for male and female
    adults. Every returned population/gender group must contain all 12 full
    Dutch monophthongs. Schwa is not present in the literature tables and is
    therefore deliberately not estimated by this function.

    No sources are pooled and no formants are inferred with a global
    male-to-female scale factor. A source containing only one gender returns
    only that published gender.
    '''

    if name not in _LITERATURE_TABLES:
        choices = ', '.join(sorted(_LITERATURE_TABLES))
        raise ValueError(
            f'{name!r} is not a literature formant table; choose from '
            f'{choices}'
        )
    table = load_formant_table(name, view='summary', data_root=data_root)
    data = [
        row.copy() for row in table.data
        if row['speaker_type'] in {'man', 'woman'}
        and row['gender'] in {'male', 'female'}
    ]
    if population is not None:
        available = sorted({
            row['population'] for row in data
            if row['population'] is not None
        })
        data = [
            row for row in data
            if row['population'] == population
        ]
        if not data:
            choices = ', '.join(available)
            raise ValueError(
                f'population {population!r} is not available for {name!r}; '
                f'choose from {choices}'
            )
    if not data:
        raise ValueError(f'{name!r} contains no adult male/female summaries')
    expected = set(LITERATURE_MONOPHTHONGS)
    for group, rows in _group_records(data, ('population', 'gender')).items():
        observed = {row['ipa'] for row in rows}
        if observed != expected:
            missing = ', '.join(sorted(expected - observed))
            extra = ', '.join(sorted(observed - expected))
            raise ValueError(
                f'incomplete literature group {group!r}: '
                f'missing [{missing}], unexpected [{extra}]'
            )
    vowel_order = {
        ipa: index for index, ipa in enumerate(LITERATURE_MONOPHTHONGS)
    }
    data.sort(
        key=lambda row: (
            row['population'],
            row['gender'],
            vowel_order[row['ipa']],
        )
    )
    return FormantTable(
        source=table.source,
        data=data,
        view='literature_gender_formants',
    )


def pols_1973_formants(data_root=None):
    '''Return the Pols et al. adult male monophthong anchors.'''

    return literature_gender_formants(
        'pols_1973_male',
        data_root=data_root,
    )


def van_nierop_1973_formants(data_root=None):
    '''Return the Van Nierop et al. adult female monophthong anchors.'''

    return literature_gender_formants(
        'van_nierop_1973_female',
        data_root=data_root,
    )


def weenink_1985_formants(data_root=None):
    '''Return the Weenink adult male and female monophthong anchors.'''

    return literature_gender_formants(
        'weenink_1985',
        data_root=data_root,
    )


def adank_2004_formants(population=None, data_root=None):
    '''Return Adank et al. male and female monophthong anchors.

    Leave ``population`` unset to retain separate Northern and Southern
    Standard Dutch rows, or select one by its readable population label.
    '''

    return literature_gender_formants(
        'adank_2004_table_1_monophthongs',
        population=population,
        data_root=data_root,
    )


def write_formant_table(name, data, data_root=None):
    '''Explicitly write one registered table and its metadata sidecar.'''

    source = formant_source(name)
    path = source.path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(path, data)
    metadata_path = path.with_name(f'{path.stem}_metadata.json')
    metadata = asdict(source)
    metadata['sha256'] = _file_sha256(path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
    )
    return path


def write_manifest(data_root=None):
    '''Write a manifest for all currently available tables.'''

    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in available_formant_tables(root):
        source = formant_source(name)
        entry = asdict(source)
        entry['path'] = str(source.path(root).relative_to(root))
        entry['sha256'] = _file_sha256(source.path(root))
        entries.append(entry)
    path = root / 'manifest.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'generated_by': (
                    'vowel_formant_reference.formant_tables.write_manifest'
                ),
                'software_versions': _software_versions(),
                'tables': entries,
            },
            indent=2,
            ensure_ascii=False,
        ) + '\n'
    )
    return path


def build_literature_tables(data_root=None):
    '''Create all source-separated literature tables.

    This is an explicit data-generation operation. Importing this module never
    invokes Praat and never writes files.
    '''

    combined = _praat_table(
        'Create formant table (Pols & Van Nierop 1973)'
    )
    write_formant_table(
        'pols_1973_male',
        [row for row in combined if row['Sex'] == 'm'],
        data_root,
    )
    write_formant_table(
        'van_nierop_1973_female',
        [row for row in combined if row['Sex'] == 'f'],
        data_root,
    )
    write_formant_table(
        'weenink_1985',
        _praat_table('Create formant table (Weenink 1985)'),
        data_root,
    )
    write_formant_table(
        'adank_2004_table_1_monophthongs',
        _adank_table_1(),
        data_root,
    )
    write_manifest(data_root)


def _read_table(path):
    if path.suffix != '.csv':
        raise ValueError(f'unsupported formant table format: {path.suffix}')
    with path.open(newline='', encoding='utf-8') as stream:
        return [
            {
                column: _csv_value(column, value)
                for column, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def _write_table(path, data):
    if path.suffix != '.csv':
        raise ValueError(f'unsupported formant table format: {path.suffix}')
    records = _records(data)
    if not records:
        raise ValueError('cannot infer CSV columns from an empty table')
    columns = list(dict.fromkeys(
        column for row in records for column in row
    ))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows([
            {
                column: _csv_cell(row.get(column))
                for column in columns
            }
            for row in records
        ])


def _praat_table(command):
    try:
        from parselmouth.praat import call
    except ImportError as error:
        raise ImportError(
            'building Praat literature tables requires praat-parselmouth'
        ) from error

    table = call(command)
    n_rows = int(call(table, 'Get number of rows'))
    n_columns = int(call(table, 'Get number of columns'))
    columns = [
        call(table, 'Get column label', column)
        for column in range(1, n_columns + 1)
    ]
    rows = [
        {
            column: call(table, 'Get value', row, column)
            for column in columns
        }
        for row in range(1, n_rows + 1)
    ]
    numeric = {
        'Speaker', 'F0', 'F1', 'F2', 'F3', 'L1', 'L2', 'L3'
    }
    for row in rows:
        for column in numeric.intersection(row):
            row[column] = _number(row[column])
    return rows


def _standardize(source, data):
    if source.name in {
        'pols_1973_male',
        'van_nierop_1973_female',
    }:
        standardized = _standardize_pols_van_nierop(source, data)
    elif source.name == 'weenink_1985':
        standardized = _standardize_weenink(source, data)
    elif source.name == 'adank_2004_table_1_monophthongs':
        standardized = _standardize_adank(source, data)
    else:
        standardized = [
            {
                **row,
                'source': source.name,
                'record_level': source.record_level,
            }
            for row in data
        ]
    return [
        {
            column: row.get(column)
            for column in _STANDARD_COLUMNS
        }
        for row in standardized
    ]


def _standardize_pols_van_nierop(source, data):
    return [
        {
            'source': source.name,
            'dataset': 'Pols and Van Nierop 1973 Praat table',
            'record_level': source.record_level,
            'population': 'Dutch adults',
            'speaker_id': _identifier(row['Speaker']),
            'speaker_type': {'m': 'man', 'f': 'woman'}[row['Sex']],
            'gender': {'m': 'male', 'f': 'female'}[row['Sex']],
            'vowel_label': row['Vowel'],
            'source_ipa': row['IPA'],
            'ipa': _PRAAT_IPA_TO_PROJECT[row['IPA']],
            'f1_hz': row['F1'],
            'f2_hz': row['F2'],
            'f3_hz': row['F3'],
            'l1_db': row['L1'],
            'l2_db': row['L2'],
            'l3_db': row['L3'],
            'n_speakers': 1,
            'n_tokens': 1,
            'aggregation': 'single production',
            'success': True,
            'provenance': source.reference,
        }
        for row in data
    ]


def _standardize_weenink(source, data):
    populations = {
        'm': 'Dutch adult',
        'w': 'Dutch adult',
        'c': 'Dutch child',
    }
    speaker_types = {'m': 'man', 'w': 'woman', 'c': 'child'}
    return [
        {
            'source': source.name,
            'dataset': 'Weenink 1985 Praat table',
            'record_level': source.record_level,
            'population': populations[row['Type']],
            'speaker_id': _identifier(row['Speaker']),
            'speaker_type': speaker_types[row['Type']],
            'gender': {'m': 'male', 'f': 'female'}[row['Sex']],
            'vowel_label': row['Vowel'],
            'source_ipa': row['IPA'],
            'ipa': _PRAAT_IPA_TO_PROJECT[row['IPA']],
            'f0_hz': row['F0'],
            'f1_hz': row['F1'],
            'f2_hz': row['F2'],
            'f3_hz': row['F3'],
            'n_speakers': 1,
            'n_tokens': 1,
            'aggregation': 'single production',
            'success': True,
            'provenance': source.reference,
        }
        for row in data
    ]


def _standardize_adank(source, data):
    populations = {
        'NSD': 'Northern Standard Dutch',
        'SSD': 'Southern Standard Dutch',
    }
    return [
        {
            'source': source.name,
            'dataset': 'Adank et al. 2004 Table I',
            'record_level': source.record_level,
            'population': populations[row['Region']],
            'speaker_type': {'M': 'man', 'F': 'woman'}[row['Sex']],
            'gender': {'M': 'male', 'F': 'female'}[row['Sex']],
            'vowel_label': row['IPA'],
            'source_ipa': row['IPA'],
            'ipa': _ADANK_IPA_TO_PROJECT[row['IPA']],
            'f0_hz': row['F0'],
            'f1_hz': row['F1'],
            'f2_hz': row['F2'],
            'f3_hz': row['F3'],
            'duration_seconds': row['Duration_ms'] / 1000,
            'n_speakers': row['N'],
            'n_tokens': row['N'] * 2,
            'aggregation': 'published group mean',
            'success': True,
            'provenance': source.reference,
        }
        for row in data
    ]


def _within_source_summary(source, data):
    if source.record_level == 'group_summary':
        return [row.copy() for row in data]
    successful = [
        row for row in data
        if _boolean(row.get('success'), default=True)
    ]
    if not successful:
        return successful
    group_columns = (
        'source', 'dataset', 'population', 'speaker_type', 'gender', 'ipa'
    )
    value_columns = [
        column for column in (
            'f0_hz', 'f1_hz', 'f2_hz', 'f3_hz',
            'l1_db', 'l2_db', 'l3_db',
        )
        if any(_is_number(row.get(column)) for row in successful)
    ]
    summary = []
    for key, rows in _group_records(successful, group_columns).items():
        row = dict(zip(group_columns, key))
        for column in value_columns:
            values = [
                item[column] for item in rows
                if _is_number(item.get(column))
            ]
            row[column] = float(np.median(values))
        row.update({
            'n_speakers': len({
                item['speaker_id'] for item in rows
                if item.get('speaker_id') is not None
            }),
            'n_tokens': len(rows),
            'record_level': 'within_source_group_summary',
            'aggregation': 'median across speaker observations',
            'provenance': source.reference,
        })
        summary.append({
            column: row.get(column)
            for column in _STANDARD_COLUMNS
        })
    return summary


def _adank_table_1():
    vowels = ['ɑ', 'a', 'ɛ', 'e', 'ø', 'ɪ', 'i', 'ɔ', 'u', 'o', 'ʏ', 'y']
    values = {
        ('NSD', 'F'): {
            'Duration_ms': [94, 214, 101, 177, 184, 89, 92, 96, 98, 183, 89, 96],
            'F0': [226, 194, 220, 207, 201, 221, 248, 218, 249, 201, 246, 245],
            'F1': [758, 912, 535, 442, 445, 399, 294, 419, 286, 445, 417, 305],
            'F2': [1280, 1572, 1990, 2343, 1713, 2276, 2524, 918, 938, 964, 1830, 1918],
            'F3': [
                2895, 2852, 2871, 2908, 2550, 2883,
                2911, 3013, 2736, 2417, 2711, 2635,
            ],
        },
        ('NSD', 'M'): {
            'Duration_ms': [96, 203, 95, 181, 184, 82, 94, 90, 98, 184, 88, 93],
            'F0': [149, 134, 154, 131, 142, 154, 157, 152, 164, 139, 154, 162],
            'F1': [578, 670, 475, 400, 375, 361, 278, 402, 259, 412, 366, 259],
            'F2': [1172, 1425, 1739, 1995, 1563, 1919, 2162, 821, 805, 929, 1595, 1734],
            'F3': [
                2435, 2485, 2492, 2583, 2241, 2536,
                2665, 2851, 2253, 2306, 2345, 2205,
            ],
        },
        ('SSD', 'F'): {
            'Duration_ms': [107, 240, 101, 192, 200, 88, 147, 97, 128, 210, 89, 153],
            'F0': [225, 203, 224, 219, 217, 256, 234, 233, 237, 215, 249, 236],
            'F1': [725, 868, 581, 436, 439, 455, 317, 475, 321, 418, 457, 337],
            'F2': [
                1262, 1640, 1932, 2420, 1804, 2115,
                2647, 987, 1019, 968, 1785, 2077,
            ],
            'F3': [
                3041, 3031, 2978, 3021, 2666, 2948,
                3312, 3133, 2871, 2992, 2884, 2634,
            ],
        },
        ('SSD', 'M'): {
            'Duration_ms': [90, 204, 86, 169, 175, 76, 96, 83, 99, 182, 77, 109],
            'F0': [126, 116, 128, 119, 121, 135, 148, 136, 149, 125, 138, 144],
            'F1': [555, 717, 475, 384, 374, 364, 278, 398, 266, 369, 353, 265],
            'F2': [1066, 1429, 1616, 1993, 1539, 1745, 2179, 850, 978, 862, 1492, 1825],
            'F3': [
                2655, 2651, 2572, 2616, 2377, 2566,
                2787, 2665, 2422, 2540, 2514, 2348,
            ],
        },
    }
    rows = []
    for (region, sex), measurements in values.items():
        for index, ipa in enumerate(vowels):
            row = {
                'Region': region,
                'Sex': sex,
                'IPA': ipa,
                'N': 20,
            }
            for name, series in measurements.items():
                row[name] = series[index]
            rows.append(row)
    return rows


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


def _json_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


_CSV_INTEGER_COLUMNS = {
    'Speaker', 'N', 'Duration_ms', 'n_speakers', 'n_tokens',
    'bootstrap_seed', 'bootstrap_replicates',
}

_CSV_FLOAT_COLUMNS = {
    'F0', 'F1', 'F2', 'F3', 'L1', 'L2', 'L3',
    'f0_hz', 'f1_hz', 'f2_hz', 'f3_hz',
    'b1_hz', 'b2_hz', 'b3_hz',
    'duration_seconds', 'confidence',
    'f0_hz_ci_low', 'f0_hz_ci_high',
    'f1_hz_ci_low', 'f1_hz_ci_high',
    'f2_hz_ci_low', 'f2_hz_ci_high',
    'f3_hz_ci_low', 'f3_hz_ci_high',
    'b1_hz_ci_low', 'b1_hz_ci_high',
    'b2_hz_ci_low', 'b2_hz_ci_high',
    'b3_hz_ci_low', 'b3_hz_ci_high',
}


def _csv_value(column, value):
    if value in {None, ''}:
        return None
    if column == 'success':
        return _boolean(value)
    if column in _CSV_INTEGER_COLUMNS:
        return int(float(value))
    if column in _CSV_FLOAT_COLUMNS:
        return float(value)
    return value


def _csv_cell(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ''
    return value


def _number(value):
    number = float(value)
    return int(number) if number.is_integer() else number


def _identifier(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _is_number(value):
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _boolean(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _software_versions():
    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    try:
        import parselmouth
    except ImportError:
        return versions
    versions.update({
        'parselmouth': parselmouth.__version__,
        'praat': parselmouth.PRAAT_VERSION,
    })
    return versions
