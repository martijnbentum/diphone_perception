'''Plotting helpers for phone-probe results.'''

import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

import locations
from probing.extract_embeddings import default_model_name
from probing.result import PhoneResult

_checkpoint_pattern = re.compile(locations.wav2vec2_nl1_checkpoint_pattern)

_phoneme_colors = (
    '#2a78d6',  # blue
    '#eb6834',  # orange
    '#1baf7a',  # aqua
    '#eda100',  # yellow
    '#e87ba4',  # magenta
    '#008300',  # green
    '#4a3aa7',  # violet
    '#e34948',  # red
)
_phoneme_markers = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')


def plot_phoneme(phoneme, layer=9, root=locations.probe_results, ax=None,
    title=None, legend=True, markersize=None):
    '''Plot embedding-probe accuracy across wav2vec2 checkpoints.

    One point is drawn per checkpoint directory found under root that has
    a complete-enough embedding result for phoneme at layer. The matching
    MFCC probe result is drawn as a dashed baseline.

    phoneme:  target phoneme to plot
    layer:    embedding hidden-state layer probed at each checkpoint
    root:     root directory containing probe results
    ax:       axes to draw on, or None to create and show a new figure
    title:    axes title, or None for the default descriptive title
    legend:   whether to draw the legend on this axes
    markersize: marker size, or None for Matplotlib's default
    '''
    checkpoints = _checkpoint_accuracies(phoneme, layer, root)
    if not checkpoints:
        raise ValueError(f'no checkpoint results found for {phoneme!r}')
    numbers, accuracies = zip(*checkpoints)

    mfcc_result = PhoneResult.mfcc(phoneme, root=root)

    standalone = ax is None
    if standalone: _, ax = plt.subplots(figsize=(10, 5))

    plot_options = {'marker': 'o', 'color': 'blue',
        'label': f'embedding (layer {layer})'}
    if markersize is not None: plot_options['markersize'] = markersize
    ax.plot(numbers, accuracies, **plot_options)
    if mfcc_result.mean_accuracy is not None:
        ax.axhline(mfcc_result.mean_accuracy, color='grey', linestyle='--',
            linewidth=2, label='mfcc baseline')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0.5, 1)
    ax.tick_params(which='both', length=3)
    ax.set_xlabel('checkpoint (training step)')
    ax.set_ylabel('accuracy')
    ax.grid(alpha=0.3)
    if legend: ax.legend()
    if title is None:
        title = f'Probe accuracy across checkpoints: {phoneme!r}'
    ax.set_title(title)
    if standalone: plt.show()


def plot_phoneme_panels(phonemes, layer=9, root=locations.probe_results,
    x_start=0, x_end=200_000, x_axis_log_scale=True):
    '''Plot embedding and CNN checkpoint results in one panel per phoneme.

    Layer embeddings use circle markers and CNN frontend features use cross
    markers. The matching MFCC result remains a dashed baseline. The legend
    is only drawn on the leftmost panel; every panel's title is just its
    phoneme label.

    phonemes:  phonemes to plot, one panel per phoneme
    layer:     embedding hidden-state layer probed at each checkpoint
    root:      root directory containing probe results
    x_start:   lower x-axis limit in training steps
    x_end:     upper x-axis limit in training steps
    x_axis_log_scale: whether to use a symmetric logarithmic x-axis
    '''
    _, axes = plt.subplots(1, len(phonemes), figsize=(5 * len(phonemes), 5),
        sharey=True)
    if len(phonemes) == 1: axes = [axes]
    for index, (phoneme, ax) in enumerate(zip(phonemes, axes)):
        plot_phoneme(phoneme, layer=layer, root=root, ax=ax, title=phoneme,
            legend=False, markersize=3)
        checkpoints = _cnn_checkpoint_accuracies(phoneme, root)
        if not checkpoints:
            raise ValueError(f'no CNN checkpoint results found for '
                f'{phoneme!r}')
        numbers, accuracies = zip(*checkpoints)
        ax.plot(numbers, accuracies, marker='x', markersize=3,
            color='#eb6834', label='CNN')
        ax.set_xlim(x_start, x_end)
        if not x_axis_log_scale: ax.set_xscale('linear')
        if index == 0: ax.legend()
    plt.tight_layout()
    plt.show()


def plot_phonemes(phonemes, layer=9, root=locations.probe_results,
    x_start=0, x_end=200_000, x_axis_log_scale=True):
    '''Plot embedding-probe accuracy across checkpoints for several phonemes
    overlaid in one figure.

    Each phoneme gets its own color and marker shape, fixed by its position
    in phonemes and reused beyond 8 phonemes. Each phoneme's MFCC baseline
    is drawn as a dashed line in that same phoneme's color.

    phonemes:  phonemes to plot, overlaid in one figure
    layer:     embedding hidden-state layer probed at each checkpoint
    root:      root directory containing probe results
    x_start:   lower x-axis limit in training steps
    x_end:     upper x-axis limit in training steps
    x_axis_log_scale: whether to use a symmetric logarithmic x-axis
    '''
    _, ax = plt.subplots(figsize=(10, 5))
    for index, phoneme in enumerate(phonemes):
        color = _phoneme_colors[index % len(_phoneme_colors)]
        marker = _phoneme_markers[index % len(_phoneme_markers)]
        checkpoints = _checkpoint_accuracies(phoneme, layer, root)
        if not checkpoints:
            raise ValueError(f'no checkpoint results found for {phoneme!r}')
        numbers, accuracies = zip(*checkpoints)
        ax.plot(numbers, accuracies, marker=marker, markersize=4,
            color=color, label=phoneme)
        mfcc_result = PhoneResult.mfcc(phoneme, root=root)
        if mfcc_result.mean_accuracy is not None:
            ax.axhline(mfcc_result.mean_accuracy, color=color,
                linestyle='--', linewidth=1.5)
    if x_axis_log_scale:
        ax.set_xscale('symlog', linthresh=1)
    else:
        ax.set_xscale('linear')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0.5, 1)
    ax.tick_params(which='both', length=3)
    ax.set_xlabel('checkpoint (training step)')
    ax.set_ylabel('accuracy')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Probe accuracy across checkpoints')
    plt.show()


def plot_checkpoint_layers(phonemes, checkpoint=default_model_name,
    root=locations.probe_results, ax=None, title=None, legend=True):
    '''Plot CNN and every transformer-layer result for one checkpoint.

    Explicit phonemes are drawn as separate colored marker series, with each
    phoneme's MFCC accuracy as a dashed line in the same color. Pass
    phonemes='all' to average each representation and the MFCC baseline over
    every phoneme stored for checkpoint. Every requested result must contain
    all folds; missing or incomplete results raise ValueError.

    phonemes:   phoneme labels to plot, or 'all' to plot their mean
    checkpoint: model checkpoint whose CNN and layer results are plotted
    root:       root directory containing probe results
    ax:         axes to draw on, or None to create and show a new figure
    title:      axes title, or None for the default descriptive title
    legend:     whether to draw the legend on this axes
    '''
    mean_all = phonemes == 'all'
    if mean_all:
        phonemes = _checkpoint_phonemes(checkpoint, root)
    elif isinstance(phonemes, str):
        raise ValueError("phonemes must be a collection of labels or 'all'")
    else:
        phonemes = list(phonemes)
    if not phonemes: raise ValueError('at least one phoneme is required')

    rows = [_phoneme_layer_accuracies(phoneme, checkpoint, root)
        for phoneme in phonemes]
    if mean_all:
        layer_values = zip(*(accuracies for _, accuracies, _ in rows))
        accuracies = [statistics.fmean(values) for values in layer_values]
        mfcc_accuracy = statistics.fmean(row[2] for row in rows)
        label = f'mean ({len(rows)} phonemes)'
        rows = [(label, accuracies, mfcc_accuracy)]

    labels = ['CNN', *(str(layer)
        for layer in locations.wav2vec2_all_probe_layers)]
    positions = range(len(labels))
    standalone = ax is None
    if standalone: _, ax = plt.subplots(figsize=(10, 5))
    for index, (phoneme, accuracies, mfcc_accuracy) in enumerate(rows):
        color = _phoneme_colors[index % len(_phoneme_colors)]
        marker = _phoneme_markers[index % len(_phoneme_markers)]
        ax.plot(positions, accuracies, marker=marker, markersize=4,
            color=color, label=phoneme)
        baseline_label = 'mean MFCC' if mean_all else None
        ax.axhline(mfcc_accuracy, color=color, linestyle='--', linewidth=1.5,
            label=baseline_label)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.5, 1)
    ax.tick_params(which='both', length=3)
    ax.set_xlabel('representation / transformer layer')
    ax.set_ylabel('accuracy')
    ax.grid(alpha=0.3)
    if legend: ax.legend()
    if title is None: title = f'Probe accuracy by layer: {checkpoint}'
    ax.set_title(title)
    if standalone: plt.show()


def plot_checkpoint_layer_panels(phonemes, checkpoints=[
    locations.wav2vec2_random_checkpoint_name, default_model_name],
    root=locations.probe_results):
    '''Plot one plot_checkpoint_layers panel per checkpoint in a single row.

    Panels share their y-axis. Only the leftmost panel has a legend, and each
    panel title is its checkpoint name.

    phonemes:    phoneme labels to plot, or 'all' to plot their mean
    checkpoints: model checkpoints to plot, one per panel
    root:        root directory containing probe results
    '''
    if isinstance(checkpoints, str):
        raise ValueError('checkpoints must be a collection of model names')
    checkpoints = list(checkpoints)
    if not checkpoints: raise ValueError('at least one checkpoint is required')
    if phonemes != 'all':
        if isinstance(phonemes, str):
            raise ValueError("phonemes must be a collection of labels or "
                "'all'")
        phonemes = list(phonemes)

    _, axes = plt.subplots(1, len(checkpoints),
        figsize=(5 * len(checkpoints), 5), sharey=True)
    if len(checkpoints) == 1: axes = [axes]
    for index, (checkpoint, ax) in enumerate(zip(checkpoints, axes)):
        plot_checkpoint_layers(phonemes, checkpoint=checkpoint, root=root,
            ax=ax, title=checkpoint, legend=index == 0)
    plt.tight_layout()
    plt.show()


def plot_mfcc_random_trained_for_all_phonemes(checkpoint=default_model_name,
    layer=9, root=locations.probe_results):
    '''Plot mfcc, random-init, and trained-checkpoint accuracy per phoneme.

    Phoneme labels are auto-discovered from the mfcc results under root and
    drawn as three marker series, sorted by ascending mfcc accuracy.
    Phonemes missing any of the three results are skipped.

    checkpoint:  trained-checkpoint model name to compare against
    layer:       embedding hidden-state layer probed at checkpoint
    root:        root directory containing probe results
    '''
    random_name = locations.wav2vec2_random_checkpoint_name
    phoneme_dirs = (Path(root) / 'mfcc').iterdir()
    phonemes = sorted(path.name for path in phoneme_dirs if path.is_dir())

    rows = []
    for phoneme in phonemes:
        mfcc_accuracy = PhoneResult.mfcc(phoneme, root=root).mean_accuracy
        random_accuracy = PhoneResult.embedding(phoneme,
            model_name=random_name, layer=layer, root=root).mean_accuracy
        trained_accuracy = PhoneResult.embedding(phoneme,
            model_name=checkpoint, layer=layer, root=root).mean_accuracy
        if None in (mfcc_accuracy, random_accuracy, trained_accuracy):
            continue
        rows.append(
            (phoneme, mfcc_accuracy, random_accuracy, trained_accuracy))
    if not rows:
        message = 'no phoneme has complete mfcc/random/checkpoint results'
        raise ValueError(message)
    rows.sort(key=lambda row: row[1])
    labels, mfcc_values, random_values, trained_values = zip(*rows)

    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(labels)), 5))
    ax.plot(positions, mfcc_values, linestyle='none', marker='s',
        markersize=6, color='#eb6834', label='mfcc')
    ax.plot(positions, random_values, linestyle='none', marker='^',
        markersize=6, color='grey', label='random init')
    ax.plot(positions, trained_values, linestyle='none', marker='o',
        markersize=6, color='#2a78d6', label=f'checkpoint {checkpoint}')
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.5, 1)
    ax.tick_params(which='both', length=3)
    ax.set_xlabel('phoneme')
    ax.set_ylabel('accuracy')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3)
    fig.suptitle('mfcc vs random-init vs trained-checkpoint probe accuracy',
        y=1.03)
    plt.tight_layout()
    plt.show()


def _checkpoint_accuracies(phoneme, layer, root):
    '''Return (checkpoint_number, mean_accuracy) pairs in ascending order.'''
    def result_for_model(model_name):
        return PhoneResult.embedding(phoneme, model_name=model_name,
            layer=layer, root=root)

    return _collect_checkpoint_accuracies(root, result_for_model)


def _checkpoint_phonemes(checkpoint, root):
    '''Return checkpoint phoneme directory names in sorted order.'''
    checkpoint_path = Path(root) / checkpoint
    if not checkpoint_path.is_dir():
        raise ValueError(f'checkpoint results not found: {checkpoint!r}')
    phonemes = sorted(path.name for path in checkpoint_path.iterdir()
        if path.is_dir())
    if not phonemes:
        raise ValueError(f'no phoneme results found for {checkpoint!r}')
    return phonemes


def _phoneme_layer_accuracies(phoneme, checkpoint, root):
    '''Return one phoneme's complete CNN, layer, and MFCC accuracies.'''
    results = [PhoneResult.cnn(phoneme, model_name=checkpoint, root=root)]
    results.extend(PhoneResult.embedding(phoneme, model_name=checkpoint,
        layer=layer, root=root)
        for layer in locations.wav2vec2_all_probe_layers)
    mfcc_result = PhoneResult.mfcc(phoneme, root=root)
    for result in [*results, mfcc_result]:
        if result.complete: continue
        representation = (f'layer {result.layer}'
            if result.representation == 'embedding'
            else result.representation.upper())
        message = (f'incomplete {representation} result for phoneme '
            f'{phoneme!r} at checkpoint {checkpoint!r}')
        raise ValueError(message)
    return (phoneme, [result.mean_accuracy for result in results],
        mfcc_result.mean_accuracy)


def _cnn_checkpoint_accuracies(phoneme, root):
    '''Return CNN checkpoint accuracies in ascending checkpoint order.'''
    def result_for_model(model_name):
        return PhoneResult.cnn(phoneme, model_name=model_name, root=root)

    return _collect_checkpoint_accuracies(root, result_for_model)


def _collect_checkpoint_accuracies(root, result_for_model):
    '''Collect checkpoint accuracies using result_for_model as a factory.'''
    results = []
    for path in Path(root).iterdir():
        if not path.is_dir(): continue
        number = _checkpoint_number(path.name)
        if number is None: continue
        result = result_for_model(path.name)
        if result.mean_accuracy is None: continue
        results.append((number, result.mean_accuracy))
    results.sort(key=lambda item: item[0])
    return results


def _checkpoint_number(model_name):
    '''Training step for a checkpoint model name, or None when unsupported.'''
    if model_name == locations.wav2vec2_random_checkpoint_name: return 0
    match = _checkpoint_pattern.fullmatch(model_name)
    if match is None: return None
    return int(match.group(1))
