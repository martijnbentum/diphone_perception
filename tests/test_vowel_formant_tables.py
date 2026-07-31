import csv
import json

import pytest

from vowel_formant_reference.formant_tables import (
    LITERATURE_MONOPHTHONGS,
    adank_2004_formants,
    available_formant_tables,
    build_literature_tables,
    formant_source,
    literature_gender_formants,
    load_formant_table,
    pols_1973_formants,
    registered_formant_tables,
    van_nierop_1973_formants,
    weenink_1985_formants,
)


LITERATURE_TABLES = {
    'pols_1973_male': 600,
    'van_nierop_1973_female': 300,
    'weenink_1985': 360,
    'adank_2004_table_1_monophthongs': 48,
}


def _values(records, column):
    return [row.get(column) for row in records]


def _group_vowels(records, *columns):
    groups = {}
    for row in records:
        key = tuple(row[column] for column in columns)
        groups.setdefault(key, set()).add(row['ipa'])
    return groups


def test_literature_tables_are_available_and_separate():
    available = set(available_formant_tables())
    assert set(LITERATURE_TABLES) <= available
    assert 'pols_1973_male' in registered_formant_tables()
    assert 'van_nierop_1973_female' in registered_formant_tables()

    pols = load_formant_table('pols_1973_male').data
    van_nierop = load_formant_table('van_nierop_1973_female').data
    assert set(_values(pols, 'Sex')) == {'m'}
    assert set(_values(van_nierop, 'Sex')) == {'f'}


@pytest.mark.parametrize(('name', 'row_count'), LITERATURE_TABLES.items())
def test_native_table_row_counts_and_citation_metadata(name, row_count):
    table = load_formant_table(name)
    assert len(table.data) == row_count
    assert table.source.reference
    assert table.source.url
    assert table.source.page_range
    assert table.source.record_level
    assert table.source.schema_version == 1


def test_adank_table_has_page_and_table_and_only_monophthongs():
    table = load_formant_table('adank_2004_table_1_monophthongs')
    assert table.source.source_page == '1731'
    assert table.source.table_number == 'I'
    assert set(_values(table.data, 'IPA')) == {
        'ɑ', 'a', 'ɛ', 'e', 'ø', 'ɪ', 'i', 'ɔ', 'u', 'o', 'ʏ', 'y'
    }
    assert not {'ɛi', 'œy', 'ɑu'}.intersection(
        _values(table.data, 'IPA')
    )


def test_literature_artifacts_use_csv_with_json_metadata():
    source = formant_source('pols_1973_male')
    with source.path().open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    metadata_path = source.path().with_name(
        f'{source.path().stem}_metadata.json'
    )
    metadata = json.loads(metadata_path.read_text())

    assert source.path().suffix == '.csv'
    assert len(records) == LITERATURE_TABLES[source.name]
    assert metadata['schema_version'] == 1
    assert metadata['name'] == source.name


def test_praat_builder_writes_csv_literature_tables(tmp_path):
    build_literature_tables(tmp_path)

    assert set(available_formant_tables(tmp_path)) == set(LITERATURE_TABLES)
    for name, row_count in LITERATURE_TABLES.items():
        source = formant_source(name)
        with source.path(tmp_path).open(
            newline='', encoding='utf-8'
        ) as stream:
            assert len(list(csv.DictReader(stream))) == row_count
        metadata_path = source.path(tmp_path).with_name(
            f'{source.path(tmp_path).stem}_metadata.json'
        )
        assert metadata_path.exists()

    pols = load_formant_table('pols_1973_male', data_root=tmp_path)
    van_nierop = load_formant_table(
        'van_nierop_1973_female',
        data_root=tmp_path,
    )
    assert set(_values(pols.data, 'Sex')) == {'m'}
    assert set(_values(van_nierop.data, 'Sex')) == {'f'}


def test_praat_builder_records_versions_without_pandas(tmp_path):
    build_literature_tables(tmp_path)

    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    versions = manifest['software_versions']
    assert versions['parselmouth']
    assert versions['praat']
    assert 'pandas' not in versions


@pytest.mark.parametrize('name', LITERATURE_TABLES)
def test_standardized_tables_preserve_all_formant_information(name):
    data = load_formant_table(name, view='standardized').data
    assert set(_values(data, 'ipa')) == {
        'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
        'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
    }
    assert all(
        row[column] is not None
        for row in data
        for column in ('f1_hz', 'f2_hz', 'f3_hz')
    )
    assert all(row['f1_hz'] < row['f2_hz'] for row in data)
    assert all(row['f2_hz'] < row['f3_hz'] for row in data)


def test_pols_native_table_retains_formant_levels():
    native = load_formant_table('pols_1973_male').data
    assert all({'L1', 'L2', 'L3'} <= set(row) for row in native)
    standardized = load_formant_table(
        'pols_1973_male', view='standardized'
    ).data
    assert all(
        row[column] is not None
        for row in standardized
        for column in ('l1_db', 'l2_db', 'l3_db')
    )


def test_adank_is_group_summary_not_fabricated_speakers():
    data = load_formant_table(
        'adank_2004_table_1_monophthongs',
        view='standardized',
    ).data
    assert set(_values(data, 'record_level')) == {'group_summary'}
    assert all(row['speaker_id'] is None for row in data)
    assert set(_values(data, 'n_speakers')) == {20}
    assert set(_values(data, 'aggregation')) == {'published group mean'}


def test_within_source_summary_does_not_pool_sources():
    table = load_formant_table('weenink_1985', view='summary')
    assert set(_values(table.data, 'source')) == {'weenink_1985'}
    assert len(table.data) == 36


def test_default_gender_formants_cover_adults_and_all_full_monophthongs():
    table = literature_gender_formants()
    data = table.data

    assert table.name == 'weenink_1985'
    assert table.view == 'literature_gender_formants'
    assert set(_values(data, 'population')) == {'Dutch adult'}
    assert set(_values(data, 'gender')) == {'male', 'female'}
    assert len(data) == 24
    assert set(_values(data, 'ipa')) == set(LITERATURE_MONOPHTHONGS)
    assert 'ə' not in set(_values(data, 'ipa'))
    assert all(
        row[column] is not None
        for row in data
        for column in ('f0_hz', 'f1_hz', 'f2_hz', 'f3_hz')
    )
    assert {
        key[0]: len(vowels)
        for key, vowels in _group_vowels(data, 'gender').items()
    } == {'female': 12, 'male': 12}


def test_gender_formants_can_select_one_adank_population():
    table = adank_2004_formants(
        population='Northern Standard Dutch',
    )
    data = table.data

    assert len(data) == 24
    assert set(_values(data, 'population')) == {
        'Northern Standard Dutch'
    }
    assert all(
        len(vowels) == 12
        for vowels in _group_vowels(data, 'gender').values()
    )


@pytest.mark.parametrize(
    ('loader', 'source', 'row_count', 'genders'),
    (
        (
            pols_1973_formants,
            'pols_1973_male',
            12,
            {'male'},
        ),
        (
            van_nierop_1973_formants,
            'van_nierop_1973_female',
            12,
            {'female'},
        ),
        (
            weenink_1985_formants,
            'weenink_1985',
            24,
            {'male', 'female'},
        ),
        (
            adank_2004_formants,
            'adank_2004_table_1_monophthongs',
            48,
            {'male', 'female'},
        ),
    ),
)
def test_dataset_functions_return_complete_source_specific_anchors(
    loader,
    source,
    row_count,
    genders,
):
    table = loader()

    assert table.name == source
    assert len(table.data) == row_count
    assert set(_values(table.data, 'gender')) == genders
    assert all(
        len(vowels) == 12
        for vowels in _group_vowels(
            table.data, 'population', 'gender'
        ).values()
    )


def test_gender_formants_reject_local_tables_and_unknown_populations():
    with pytest.raises(ValueError, match='not a literature formant table'):
        literature_gender_formants('gender_formants')
    with pytest.raises(ValueError, match='population.*not available'):
        literature_gender_formants(population='Southern Standard Dutch')


def test_unknown_table_and_view_fail_clearly():
    with pytest.raises(KeyError, match='unknown formant table'):
        formant_source('missing')
    with pytest.raises(ValueError, match='unknown table view'):
        load_formant_table('pols_1973_male', view='combined')
