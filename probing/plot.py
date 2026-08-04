'''Collection and plotting helpers for binary phone-probe accuracies.'''

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from probing.probe_utils import default_probe_save_dir, default_results_dir


default_phone_probe_report_path = (
    default_probe_save_dir / 'phone_binary_probe_report.json')
default_mfcc_probe_results_path = (
    default_results_dir / 'mfcc' / 'mfcc_probe_results.json')

_checkpoint_pattern = re.compile(r'wav2vec2_nl1_checkpoint-(\d+)')
_random_checkpoint_name = 'wav2vec2_checkpoint-0'
_report_kinds = {
    'phone_binary_probe_report',
    'binary_mfcc_probe_results',
}


def load_probe_report(path=default_phone_probe_report_path):
    '''Load and minimally validate a phone-probe or MFCC results report.'''
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding='utf-8') as file:
            report = json.load(file)
    except (OSError, ValueError) as error:
        message = f'could not load probe report {path}: {error}'
        raise ValueError(message) from error
    if not isinstance(report, dict):
        raise TypeError('probe report must contain a JSON object')
    kind = report.get('kind')
    if kind not in _report_kinds:
        raise ValueError(
            f'unsupported probe report kind {kind!r}; '
            f'expected one of {sorted(_report_kinds)}')
    return report


def checkpoint_step(model_name):
    '''Return the numeric training step encoded in a supported model name.'''
    if not isinstance(model_name, str):
        raise TypeError('model_name must be a string')
    if model_name == _random_checkpoint_name:
        return 0
    match = _checkpoint_pattern.fullmatch(model_name)
    return None if match is None else int(match.group(1))


def _phone_probe_records(report):
    tasks = report.get('tasks')
    if not isinstance(tasks, list):
        raise ValueError('phone binary-probe report has no tasks list')
    records = []
    for task in tasks:
        identity = task.get('task', {})
        model_name = identity.get('model_name')
        records.append({
            'representation': 'embedding',
            'phone': identity.get('phone'),
            'model_name': model_name,
            'checkpoint_step': checkpoint_step(model_name),
            'layer': identity.get('layer'),
            'frame': 'middle',
            'status': task.get('status'),
            'run_id': task.get('run_id'),
            'accuracies': list(task.get('accuracies') or []),
            'mean_accuracy': task.get('mean_accuracy'),
            'std_accuracy': task.get('std_accuracy'),
        })
    return records


def _mfcc_probe_records(report):
    results = report.get('results')
    if not isinstance(results, dict):
        raise ValueError('MFCC probe report has no results dictionary')
    records = []
    for phone, result in results.items():
        records.append({
            'representation': 'mfcc',
            'phone': phone,
            'model_name': 'mfcc',
            'checkpoint_step': None,
            'layer': None,
            'frame': result.get('frame'),
            'status': 'complete',
            'run_id': result.get('run_id'),
            'accuracies': list(result.get('accuracies') or []),
            'mean_accuracy': result.get('mean_accuracy'),
            'std_accuracy': result.get('std_accuracy'),
        })
    return records


def collect_probe_accuracies(report, level='task', complete_only=True):
    '''Normalize a persisted report into task- or fold-level dictionaries.'''
    if isinstance(report, (str, Path)):
        report = load_probe_report(report)
    if not isinstance(report, dict):
        raise TypeError('report must be a dictionary or path')
    if level not in {'task', 'fold'}:
        raise ValueError("level must be either 'task' or 'fold'")
    if not isinstance(complete_only, bool):
        raise TypeError('complete_only must be a boolean')

    kind = report.get('kind')
    if kind == 'phone_binary_probe_report':
        records = _phone_probe_records(report)
    elif kind == 'binary_mfcc_probe_results':
        records = _mfcc_probe_records(report)
    else:
        raise ValueError(f'unsupported probe report kind {kind!r}')
    if complete_only:
        records = [
            record for record in records
            if record['status'] == 'complete'
            and record['mean_accuracy'] is not None
        ]
    if level == 'task':
        return records

    folds = []
    for record in records:
        for fold, accuracy in enumerate(record['accuracies'], start=1):
            folds.append({
                key: value
                for key, value in record.items()
                if key not in {'accuracies', 'mean_accuracy', 'std_accuracy'}
            } | {
                'fold': fold,
                'accuracy': float(accuracy),
            })
    return folds


def _task_records(records):
    if isinstance(records, dict):
        records = collect_probe_accuracies(records)
    elif isinstance(records, (str, Path)):
        records = collect_probe_accuracies(records)
    else:
        records = list(records)
    if not records:
        raise ValueError('no probe accuracy records were supplied')
    required = {'phone', 'mean_accuracy'}
    for record in records:
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError(
                'plotting requires task-level probe accuracy records')
    return records


def _selected_phones(records, phones):
    if phones is None:
        return records
    if isinstance(phones, str):
        phones = [phones]
    phones = list(phones)
    if not phones:
        raise ValueError('phones must contain at least one phone label')
    return [record for record in records if record['phone'] in phones]


def _figure_axis(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    from matplotlib import pyplot

    return pyplot.subplots(figsize=figsize)


def _save_figure(figure, output_path, dpi):
    if not output_path:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')


def _plot_accuracy_series(axis, records, x_key, aggregate):
    if not isinstance(aggregate, bool):
        raise TypeError('aggregate must be a boolean')
    if aggregate:
        grouped = defaultdict(list)
        for record in records:
            grouped[record[x_key]].append(float(record['mean_accuracy']))
        x_values = sorted(grouped)
        means = np.array([np.mean(grouped[value]) for value in x_values])
        stds = np.array([np.std(grouped[value]) for value in x_values])
        axis.plot(x_values, means, marker='o', label='Mean across phones')
        axis.fill_between(
            x_values,
            np.clip(means - stds, 0, 1),
            np.clip(means + stds, 0, 1),
            alpha=.2,
            label='±1 SD across phones',
        )
    else:
        by_phone = defaultdict(list)
        for record in records:
            by_phone[record['phone']].append(record)
        for phone, phone_records in sorted(by_phone.items()):
            phone_records.sort(key=lambda record: record[x_key])
            axis.plot(
                [record[x_key] for record in phone_records],
                [record['mean_accuracy'] for record in phone_records],
                marker='o',
                label=phone,
            )
    axis.set_ylim(0, 1)
    axis.grid(alpha=.25)
    axis.legend()


def plot_checkpoint_accuracy(
    records,
    *,
    layer=9,
    phones=None,
    aggregate=True,
    ax=None,
    output_path='',
    dpi=300,
):
    '''Plot mean probe accuracy over checkpoint training steps.'''
    records = _selected_phones(_task_records(records), phones)
    records = [
        record for record in records
        if record.get('representation') == 'embedding'
        and record.get('layer') == layer
        and record.get('checkpoint_step') is not None
    ]
    if not records:
        raise ValueError(
            f'no embedding probe records match layer={layer!r} and phones')
    figure, axis = _figure_axis(ax, (10, 5))
    _plot_accuracy_series(axis, records, 'checkpoint_step', aggregate)
    axis.set_xlabel('Checkpoint training step')
    axis.set_ylabel('Cross-validation accuracy')
    axis.set_title(f'Phone probe accuracy over training — layer {layer}')
    _save_figure(figure, output_path, dpi)
    return figure, axis


def plot_layer_accuracy(
    records,
    *,
    model_name,
    phones=None,
    aggregate=True,
    ax=None,
    output_path='',
    dpi=300,
):
    '''Plot mean probe accuracy over representation layers for one model.'''
    records = _selected_phones(_task_records(records), phones)
    records = [
        record for record in records
        if record.get('representation') == 'embedding'
        and record.get('model_name') == model_name
        and record.get('layer') is not None
    ]
    if not records:
        raise ValueError(f'no embedding probe records match {model_name!r}')
    figure, axis = _figure_axis(ax, (9, 5))
    _plot_accuracy_series(axis, records, 'layer', aggregate)
    layers = sorted({record['layer'] for record in records})
    axis.set_xticks(layers)
    axis.set_xlabel('Layer')
    axis.set_ylabel('Cross-validation accuracy')
    axis.set_title(f'Phone probe accuracy by layer — {model_name}')
    _save_figure(figure, output_path, dpi)
    return figure, axis


def plot_phone_checkpoint_heatmap(
    records,
    *,
    layer=9,
    phones=None,
    ax=None,
    output_path='',
    dpi=300,
):
    '''Plot phone-by-checkpoint accuracies for one representation layer.'''
    records = _selected_phones(_task_records(records), phones)
    records = [
        record for record in records
        if record.get('representation') == 'embedding'
        and record.get('layer') == layer
        and record.get('checkpoint_step') is not None
    ]
    if not records:
        raise ValueError(
            f'no embedding probe records match layer={layer!r} and phones')
    phone_labels = sorted({record['phone'] for record in records})
    steps = sorted({record['checkpoint_step'] for record in records})
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['phone'], record['checkpoint_step'])].append(
            float(record['mean_accuracy']))
    matrix = np.full((len(phone_labels), len(steps)), np.nan)
    for row, phone in enumerate(phone_labels):
        for column, step in enumerate(steps):
            values = grouped.get((phone, step))
            if values:
                matrix[row, column] = np.mean(values)

    figure, axis = _figure_axis(
        ax, (max(8, len(steps) * .45), max(5, len(phone_labels) * .3)))
    image = axis.imshow(
        matrix, aspect='auto', interpolation='nearest', vmin=0, vmax=1,
        cmap='viridis')
    axis.set_xticks(range(len(steps)), labels=steps, rotation=45, ha='right')
    axis.set_yticks(range(len(phone_labels)), labels=phone_labels)
    axis.set_xlabel('Checkpoint training step')
    axis.set_ylabel('Target phone')
    axis.set_title(f'Phone probe accuracy heatmap — layer {layer}')
    figure.colorbar(image, ax=axis, label='Cross-validation accuracy')
    _save_figure(figure, output_path, dpi)
    return figure, axis


def plot_probe_status(
    report,
    *,
    ax=None,
    output_path='',
    dpi=300,
):
    '''Plot task status counts from a checkpoint or MFCC report.'''
    if isinstance(report, (str, Path)):
        report = load_probe_report(report)
    if not isinstance(report, dict):
        raise TypeError('report must be a dictionary or path')
    kind = report.get('kind')
    if kind == 'phone_binary_probe_report':
        statuses = [task.get('status') for task in report.get('tasks', [])]
    elif kind == 'binary_mfcc_probe_results':
        statuses = ['complete'] * len(report.get('results', {}))
    else:
        raise ValueError(f'unsupported probe report kind {kind!r}')
    if not statuses:
        raise ValueError('probe report contains no result statuses')
    counts = Counter(statuses)
    order = [
        status for status in ('complete', 'partial', 'missing', 'failed')
        if counts[status]
    ]
    figure, axis = _figure_axis(ax, (7, 4))
    bars = axis.bar(order, [counts[status] for status in order])
    axis.bar_label(bars)
    axis.set_ylabel('Tasks')
    axis.set_title('Probe result status')
    _save_figure(figure, output_path, dpi)
    return figure, axis
