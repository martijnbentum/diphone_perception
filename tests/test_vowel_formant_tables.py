import pytest

from vowel_formant_reference.formant_tables import (
    LITERATURE_MONOPHTHONGS,
    adank_2004_formants,
    available_formant_tables,
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


def test_literature_tables_are_available_and_separate():
    available = set(available_formant_tables())
    assert set(LITERATURE_TABLES) <= available
    assert 'pols_1973_male' in registered_formant_tables()
    assert 'van_nierop_1973_female' in registered_formant_tables()

    pols = load_formant_table('pols_1973_male').data
    van_nierop = load_formant_table('van_nierop_1973_female').data
    assert set(pols['Sex']) == {'m'}
    assert set(van_nierop['Sex']) == {'f'}


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
    assert set(table.data['IPA']) == {
        'ɑ', 'a', 'ɛ', 'e', 'ø', 'ɪ', 'i', 'ɔ', 'u', 'o', 'ʏ', 'y'
    }
    assert not {'ɛi', 'œy', 'ɑu'}.intersection(table.data['IPA'])


@pytest.mark.parametrize('name', LITERATURE_TABLES)
def test_standardized_tables_preserve_all_formant_information(name):
    data = load_formant_table(name, view='standardized').data
    assert set(data['ipa']) == {
        'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
        'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
    }
    assert data[['f1_hz', 'f2_hz', 'f3_hz']].notna().all().all()
    assert (data['f1_hz'] < data['f2_hz']).all()
    assert (data['f2_hz'] < data['f3_hz']).all()


def test_pols_native_table_retains_formant_levels():
    native = load_formant_table('pols_1973_male').data
    assert {'L1', 'L2', 'L3'} <= set(native)
    standardized = load_formant_table(
        'pols_1973_male', view='standardized'
    ).data
    assert standardized[['l1_db', 'l2_db', 'l3_db']].notna().all().all()


def test_adank_is_group_summary_not_fabricated_speakers():
    data = load_formant_table(
        'adank_2004_table_1_monophthongs',
        view='standardized',
    ).data
    assert set(data['record_level']) == {'group_summary'}
    assert data['speaker_id'].isna().all()
    assert set(data['n_speakers']) == {20}
    assert set(data['aggregation']) == {'published group mean'}


def test_within_source_summary_does_not_pool_sources():
    table = load_formant_table('weenink_1985', view='summary')
    assert set(table.data['source']) == {'weenink_1985'}
    assert len(table.data) == 36


def test_default_gender_formants_cover_adults_and_all_full_monophthongs():
    table = literature_gender_formants()
    data = table.data

    assert table.name == 'weenink_1985'
    assert table.view == 'literature_gender_formants'
    assert set(data['population']) == {'Dutch adult'}
    assert set(data['gender']) == {'male', 'female'}
    assert len(data) == 24
    assert set(data['ipa']) == set(LITERATURE_MONOPHTHONGS)
    assert 'ə' not in set(data['ipa'])
    assert data[['f0_hz', 'f1_hz', 'f2_hz', 'f3_hz']].notna().all().all()
    assert data.groupby('gender')['ipa'].nunique().to_dict() == {
        'female': 12,
        'male': 12,
    }


def test_gender_formants_can_select_one_adank_population():
    table = adank_2004_formants(
        population='Northern Standard Dutch',
    )
    data = table.data

    assert len(data) == 24
    assert set(data['population']) == {'Northern Standard Dutch'}
    assert data.groupby('gender')['ipa'].nunique().eq(12).all()


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
    assert set(table.data['gender']) == genders
    assert table.data.groupby(
        ['population', 'gender']
    )['ipa'].nunique().eq(12).all()


def test_gender_formants_reject_local_tables_and_unknown_populations():
    with pytest.raises(ValueError, match='not a literature formant table'):
        literature_gender_formants('selected_phone_genders')
    with pytest.raises(ValueError, match='population.*not available'):
        literature_gender_formants(population='Southern Standard Dutch')


def test_unknown_table_and_view_fail_clearly():
    with pytest.raises(KeyError, match='unknown formant table'):
        formant_source('missing')
    with pytest.raises(ValueError, match='unknown table view'):
        load_formant_table('pols_1973_male', view='combined')
