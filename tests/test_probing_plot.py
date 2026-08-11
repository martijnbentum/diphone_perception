import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import plot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    pyplot.close('all')


def _stub_panel_results(monkeypatch):
    monkeypatch.setattr(plot, '_checkpoint_accuracies',
        lambda phoneme, layer, root: [(0, 0.6), (900, 0.7)])
    monkeypatch.setattr(plot, '_cnn_checkpoint_accuracies',
        lambda phoneme, root: [(0, 0.55), (900, 0.65)])
    monkeypatch.setattr(plot.PhoneResult, 'mfcc',
        lambda phoneme, root: SimpleNamespace(mean_accuracy=0.58))
    monkeypatch.setattr(plot.plt, 'show', lambda: None)


def test_phoneme_panels_add_cnn_with_distinct_small_markers(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phoneme_panels(['p'])

    axis = pyplot.gcf().axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    embedding = lines['embedding (layer 9)']
    cnn = lines['CNN']
    assert embedding.get_marker() == 'o'
    assert cnn.get_marker() == 'x'
    assert embedding.get_markersize() == 3
    assert cnn.get_markersize() == 3
    assert list(cnn.get_xdata()) == [0, 900]
    assert list(cnn.get_ydata()) == [0.55, 0.65]
    assert axis.get_xlim() == pytest.approx((0, 200_000))
    assert axis.get_xscale() == 'symlog'
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        'embedding (layer 9)', 'mfcc baseline', 'CNN']


def test_phoneme_panels_accept_custom_x_axis_limits(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phoneme_panels(['p'], x_start=100, x_end=100_000)

    axis = pyplot.gcf().axes[0]
    assert axis.get_xlim() == pytest.approx((100, 100_000))


def test_phoneme_panels_can_use_linear_x_axis(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phoneme_panels(['p'], x_axis_log_scale=False)

    axis = pyplot.gcf().axes[0]
    assert axis.get_xscale() == 'linear'


def test_standalone_phoneme_plot_remains_cnn_free(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phoneme('p')

    labels = [line.get_label() for line in pyplot.gcf().axes[0].lines]
    assert labels == ['embedding (layer 9)', 'mfcc baseline']


def test_phonemes_accepts_x_axis_limits_and_linear_scale(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phonemes(['p'], x_start=100, x_end=100_000,
        x_axis_log_scale=False)

    axis = pyplot.gcf().axes[0]
    assert axis.get_xlim() == pytest.approx((100, 100_000))
    assert axis.get_xscale() == 'linear'


def test_phonemes_defaults_to_bounded_log_x_axis(monkeypatch):
    _stub_panel_results(monkeypatch)

    plot.plot_phonemes(['p'])

    axis = pyplot.gcf().axes[0]
    assert axis.get_xlim() == pytest.approx((0, 200_000))
    assert axis.get_xscale() == 'symlog'


def _stub_layer_results(monkeypatch, *, incomplete=None):
    def result(representation, phoneme, layer, accuracy):
        identity = (representation, phoneme, layer)
        return SimpleNamespace(representation=representation, layer=layer,
            mean_accuracy=accuracy, complete=identity != incomplete)

    def embedding(phoneme, model_name, layer, root):
        offset = 0 if phoneme == 'p' else 0.1
        return result('embedding', phoneme, layer, 0.6 + layer / 100 + offset)

    def cnn(phoneme, model_name, root):
        offset = 0 if phoneme == 'p' else 0.1
        return result('cnn', phoneme, 'cnn', 0.55 + offset)

    def mfcc(phoneme, root):
        offset = 0 if phoneme == 'p' else 0.1
        return result('mfcc', phoneme, None, 0.5 + offset)

    monkeypatch.setattr(plot.PhoneResult, 'embedding', embedding)
    monkeypatch.setattr(plot.PhoneResult, 'cnn', cnn)
    monkeypatch.setattr(plot.PhoneResult, 'mfcc', mfcc)
    monkeypatch.setattr(plot.plt, 'show', lambda: None)


def test_checkpoint_layers_plots_each_phoneme_and_matching_mfcc(monkeypatch):
    _stub_layer_results(monkeypatch)

    plot.plot_checkpoint_layers(['p', 't'], checkpoint='model-a')

    axis = pyplot.gcf().axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        'CNN', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
        '12']
    assert [line.get_label() for line in axis.lines] == [
        'p', '_child1', 't', '_child3']
    assert list(axis.lines[0].get_ydata()) == pytest.approx(
        [0.55, *(0.6 + layer / 100 for layer in range(1, 13))])
    assert list(axis.lines[1].get_ydata()) == pytest.approx([0.5, 0.5])
    assert axis.lines[0].get_color() == axis.lines[1].get_color()
    assert axis.lines[2].get_color() == axis.lines[3].get_color()
    assert axis.lines[1].get_linestyle() == '--'
    assert axis.lines[3].get_linestyle() == '--'


def test_checkpoint_layers_all_plots_complete_label_means(
    tmp_path, monkeypatch,
):
    _stub_layer_results(monkeypatch)
    for phoneme in ('p', 't'):
        (tmp_path / 'model-a' / phoneme).mkdir(parents=True)

    plot.plot_checkpoint_layers('all', checkpoint='model-a', root=tmp_path)

    axis = pyplot.gcf().axes[0]
    assert [line.get_label() for line in axis.lines] == [
        'mean (2 phonemes)', 'mean MFCC']
    assert list(axis.lines[0].get_ydata()) == pytest.approx(
        [0.6, *(0.65 + layer / 100 for layer in range(1, 13))])
    assert list(axis.lines[1].get_ydata()) == pytest.approx([0.55, 0.55])
    assert axis.lines[0].get_color() == axis.lines[1].get_color()


def test_checkpoint_layers_raises_for_incomplete_result(monkeypatch):
    _stub_layer_results(monkeypatch, incomplete=('embedding', 'p', 5))

    with pytest.raises(ValueError, match="incomplete layer 5.*'p'"):
        plot.plot_checkpoint_layers(['p'], checkpoint='model-a')


def test_checkpoint_layer_panels_defaults_to_random_and_trained():
    parameter = inspect.signature(
        plot.plot_checkpoint_layer_panels).parameters['checkpoints']

    assert parameter.default == [
        'wav2vec2_checkpoint-0', 'wav2vec2_nl1_checkpoint-200000']


def test_checkpoint_layer_panels_share_y_and_only_first_has_legend(
    monkeypatch,
):
    _stub_layer_results(monkeypatch)

    plot.plot_checkpoint_layer_panels(
        ['p'], checkpoints=['model-a', 'model-b'])

    axes = pyplot.gcf().axes
    assert [axis.get_title() for axis in axes] == ['model-a', 'model-b']
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[0].get_legend() is not None
    assert axes[1].get_legend() is None
