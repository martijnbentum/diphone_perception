'''Plotting helpers for phone-probe results.'''

import re
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
    title=None, legend=True):
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
    '''
    checkpoints = _checkpoint_accuracies(phoneme, layer, root)
    if not checkpoints:
        raise ValueError(f'no checkpoint results found for {phoneme!r}')
    numbers, accuracies = zip(*checkpoints)

    mfcc_result = PhoneResult.mfcc(phoneme, root=root)

    standalone = ax is None
    if standalone: _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(numbers, accuracies, marker='o', color='blue',
        label=f'embedding (layer {layer})')
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


def plot_phoneme_panels(phonemes, layer=9, root=locations.probe_results):
    '''Plot one plot_phoneme panel per phoneme, laid out in a single row.

    The legend is only drawn on the leftmost panel; every panel's title is
    just its phoneme label.

    phonemes:  phonemes to plot, one panel per phoneme
    layer:     embedding hidden-state layer probed at each checkpoint
    root:      root directory containing probe results
    '''
    _, axes = plt.subplots(1, len(phonemes), figsize=(5 * len(phonemes), 5),
        sharey=True)
    if len(phonemes) == 1: axes = [axes]
    for index, (phoneme, ax) in enumerate(zip(phonemes, axes)):
        plot_phoneme(phoneme, layer=layer, root=root, ax=ax, title=phoneme,
            legend=index == 0)
    plt.tight_layout()
    plt.show()


def plot_phonemes(phonemes, layer=9, root=locations.probe_results):
    '''Plot embedding-probe accuracy across checkpoints for several phonemes
    overlaid in one figure.

    Each phoneme gets its own color and marker shape, fixed by its position
    in phonemes and reused beyond 8 phonemes. Each phoneme's MFCC baseline
    is drawn as a dashed line in that same phoneme's color.

    phonemes:  phonemes to plot, overlaid in one figure
    layer:     embedding hidden-state layer probed at each checkpoint
    root:      root directory containing probe results
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
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0.5, 1)
    ax.tick_params(which='both', length=3)
    ax.set_xlabel('checkpoint (training step)')
    ax.set_ylabel('accuracy')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Probe accuracy across checkpoints')
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
    results = []
    for path in Path(root).iterdir():
        if not path.is_dir(): continue
        number = _checkpoint_number(path.name)
        if number is None: continue
        result = PhoneResult.embedding(phoneme, model_name=path.name,
            layer=layer, root=root)
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
