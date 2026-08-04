import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import plot


def _task(phone, model_name, layer, mean_accuracy, status='complete'):
    accuracies = [mean_accuracy - .01, mean_accuracy + .01]
    return {
        'task': {
            'phone': phone,
            'model_name': model_name,
            'layer': layer,
        },
        'status': status,
        'run_id': f'{phone}-{model_name}-{layer}',
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': .01,
    }


@pytest.fixture
def phone_report():
    random_model = 'wav2vec2_checkpoint-0'
    checkpoint = 'wav2vec2_nl1_checkpoint-1000'
    return {
        'kind': 'phone_binary_probe_report',
        'tasks': [
            _task('p', random_model, 9, .60),
            _task('t', random_model, 9, .70),
            _task('p', checkpoint, 9, .80),
            _task('t', checkpoint, 9, .90),
            _task('p', checkpoint, 8, .75),
            _task('t', checkpoint, 8, .85),
            _task('x', checkpoint, 9, .50, status='failed'),
        ],
    }


@pytest.fixture
def mfcc_report():
    return {
        'kind': 'binary_mfcc_probe_results',
        'results': {
            'p': {
                'frame': 'center',
                'run_id': 'mfcc-p',
                'accuracies': [.69, .71],
                'mean_accuracy': .70,
                'std_accuracy': .01,
            },
            't': {
                'frame': 'center',
                'run_id': 'mfcc-t',
                'accuracies': [.79, .81],
                'mean_accuracy': .80,
                'std_accuracy': .01,
            },
        },
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    pyplot.close('all')


def test_load_and_collect_checkpoint_report(tmp_path, phone_report):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(phone_report), encoding='utf-8')

    loaded = plot.load_probe_report(path)
    records = plot.collect_probe_accuracies(loaded)
    folds = plot.collect_probe_accuracies(loaded, level='fold')

    assert len(records) == 6
    assert len(folds) == 12
    assert {record['checkpoint_step'] for record in records} == {0, 1000}
    assert folds[0]['fold'] == 1
    assert folds[0]['accuracy'] == pytest.approx(.59)


def test_collect_mfcc_report(mfcc_report):
    records = plot.collect_probe_accuracies(mfcc_report)

    assert [record['phone'] for record in records] == ['p', 't']
    assert all(record['representation'] == 'mfcc' for record in records)
    assert all(record['checkpoint_step'] is None for record in records)


def test_plot_checkpoint_accuracy_aggregates_phones(
    tmp_path,
    phone_report,
):
    output_path = tmp_path / 'checkpoint.png'

    figure, axis = plot.plot_checkpoint_accuracy(
        phone_report, layer=9, output_path=output_path)

    line = axis.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1000])
    np.testing.assert_allclose(line.get_ydata(), [.65, .85])
    assert axis.get_ylim() == (0., 1.)
    assert output_path.exists()
    assert figure.axes[0] is axis


def test_plot_layer_accuracy_can_draw_one_line_per_phone(phone_report):
    _, axis = plot.plot_layer_accuracy(
        phone_report,
        model_name='wav2vec2_nl1_checkpoint-1000',
        aggregate=False,
    )

    assert len(axis.lines) == 2
    assert {line.get_label() for line in axis.lines} == {'p', 't'}
    assert list(axis.get_xticks()) == [8, 9]


def test_plot_can_use_a_caller_supplied_axis(phone_report):
    supplied_figure, supplied_axis = pyplot.subplots()

    figure, axis = plot.plot_checkpoint_accuracy(
        phone_report, layer=9, ax=supplied_axis)

    assert figure is supplied_figure
    assert axis is supplied_axis


def test_plot_phone_checkpoint_heatmap(phone_report):
    figure, axis = plot.plot_phone_checkpoint_heatmap(
        phone_report, layer=9)

    matrix = np.asarray(axis.images[0].get_array())
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, [[.60, .80], [.70, .90]])
    assert len(figure.axes) == 2  # data axis plus colorbar


def test_plot_probe_status_supports_both_report_kinds(
    phone_report,
    mfcc_report,
):
    _, checkpoint_axis = plot.plot_probe_status(phone_report)
    _, mfcc_axis = plot.plot_probe_status(mfcc_report)

    assert [bar.get_height() for bar in checkpoint_axis.patches] == [6, 1]
    assert [bar.get_height() for bar in mfcc_axis.patches] == [2]


@pytest.mark.parametrize('name', ('other-model', 'checkpoint-1000'))
def test_checkpoint_step_returns_none_for_unsupported_names(name):
    assert plot.checkpoint_step(name) is None
