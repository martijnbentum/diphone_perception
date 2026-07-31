import inspect

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
import pytest

from synthetic_acoustic_probes import (
    DEFAULT_VOWEL_PLOT_GENDERS,
    DEFAULT_VOWEL_PLOT_SOURCE_IDS,
    plot_vowel_formant_space,
)


SOURCE_COUNTS = {
    'cgn_selected_phones': 22,
    'pols_1973_male': 12,
    'van_nierop_1973_female': 12,
    'weenink_1985': 24,
    'adank_2004_nsd': 24,
    'adank_2004_ssd': 24,
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    pyplot.close('all')


def _point_count(axis):
    return sum(len(collection.get_offsets()) for collection in axis.collections)


def test_source_panels_include_every_available_anchor_and_ipa_label():
    figure, axes = plot_vowel_formant_space(panel_by='source')

    assert DEFAULT_VOWEL_PLOT_SOURCE_IDS == tuple(SOURCE_COUNTS)
    assert len(axes) == 6
    assert [_point_count(axis) for axis in axes] == list(
        SOURCE_COUNTS.values()
    )
    assert sum(_point_count(axis) for axis in axes) == 118
    assert all(
        len(axis.texts) == expected
        for axis, expected in zip(axes, SOURCE_COUNTS.values())
    )
    assert all(axis.xaxis_inverted() for axis in axes)
    assert all(axis.yaxis_inverted() for axis in axes)
    assert all(axis.get_xlabel() == 'F2 (Hz)' for axis in axes)
    assert all(axis.get_ylabel() == 'F1 (Hz)' for axis in axes)
    assert figure.axes == list(axes)


def test_gender_panels_include_all_sources_available_for_each_gender():
    _, axes = plot_vowel_formant_space(panel_by='gender')

    assert DEFAULT_VOWEL_PLOT_GENDERS == ('female', 'male')
    assert [axis.get_title() for axis in axes] == ['Female', 'Male']
    assert [_point_count(axis) for axis in axes] == [59, 59]
    assert [len(axis.texts) for axis in axes] == [59, 59]
    assert [len(axis.collections) for axis in axes] == [5, 5]
    for axis in axes:
        legend_labels = [
            text.get_text() for text in axis.get_legend().get_texts()
        ]
        assert 'CGN selected vowels' in legend_labels
        assert 'CGN selected phones' not in legend_labels


def test_source_and_gender_flags_select_data_without_filling_missing_groups():
    sources = ('pols_1973_male', 'van_nierop_1973_female')
    _, axes = plot_vowel_formant_space(
        panel_by='source',
        source_ids=sources,
        genders=('female',),
    )

    assert len(axes) == 2
    assert [_point_count(axis) for axis in axes] == [0, 12]
    assert [text.get_text() for text in axes[0].texts] == [
        'No selected data'
    ]
    assert len(axes[1].texts) == 12


def test_plot_can_save_and_return_the_same_figure(tmp_path):
    output_path = tmp_path / 'nested' / 'vowel-space.png'

    figure, axes = plot_vowel_formant_space(
        panel_by='gender',
        source_ids=('cgn_selected_phones',),
        output_path=output_path,
        dpi=120,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(axes) == 2
    assert figure.axes == list(axes)


def test_empty_output_path_is_default_and_does_not_save(tmp_path, monkeypatch):
    parameter = inspect.signature(
        plot_vowel_formant_space
    ).parameters['output_path']
    assert parameter.default == ''
    monkeypatch.chdir(tmp_path)

    plot_vowel_formant_space(
        panel_by='gender',
        output_path='',
    )

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ('panel_by', 'data_panel_count'),
    (('source', 6), ('gender', 2)),
)
def test_examples_flag_appends_aligned_vowel_word_panel(
    panel_by,
    data_panel_count,
):
    figure, axes = plot_vowel_formant_space(
        panel_by=panel_by,
        add_examples=True,
    )

    assert len(axes) == data_panel_count + 1
    assert figure.axes == list(axes)
    example_axis = axes[-1]
    assert example_axis.get_title() == ''
    assert not example_axis.axison
    assert [text.get_text() for text in example_axis.texts] == [
        'ɪ', 'pit',
        'ɛ', 'pet',
        'ɑ', 'pat',
        'ɔ', 'pot',
        'ʉ', 'put',
        'iː', 'biet',
        'yː', 'fuut',
        'eː', 'beet',
        'øː', 'neus',
        'aː', 'maat',
        'oː', 'boot',
        'uː', 'boek',
        'ə', 'de',
    ]
    assert {
        text.get_position()[0]
        for text in example_axis.texts[::2]
    } == {0.04}
    assert {
        text.get_position()[0]
        for text in example_axis.texts[1::2]
    } == {0.17}


def test_examples_flag_must_be_boolean():
    with pytest.raises(TypeError, match='add_examples'):
        plot_vowel_formant_space(
            panel_by='gender',
            add_examples='yes',
        )


@pytest.mark.parametrize('panel_by', ('sources', 'genders', None))
def test_unknown_panel_layout_fails(panel_by):
    with pytest.raises(ValueError, match='panel_by'):
        plot_vowel_formant_space(panel_by=panel_by)


def test_invalid_or_empty_selection_fails_clearly():
    with pytest.raises(ValueError, match='unsupported source'):
        plot_vowel_formant_space(
            panel_by='source',
            source_ids=('unknown',),
        )
    with pytest.raises(ValueError, match='at least one gender'):
        plot_vowel_formant_space(
            panel_by='gender',
            genders=(),
        )
    with pytest.raises(ValueError, match='no vowel anchors'):
        plot_vowel_formant_space(
            panel_by='gender',
            source_ids=('pols_1973_male',),
            genders=('female',),
        )
