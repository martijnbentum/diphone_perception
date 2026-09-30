'''Plot marker-aligned modes and recording-level sample distributions.'''

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_mode_vs_intensity(rows, mode_index, xlim=(-50, 100),
    use_imgcat=False, save=False, output_dir=None):
    '''Plot a mode score against intensity, coloured by speech status.

    rows:        marker_analysis.Row objects, or an iterable of them
    mode_index:  zero-based SVD mode index
    xlim:        intensity axis limits in dB
    use_imgcat:  display in a compatible terminal instead of pyplot
    save:        save a PNG, failing if its filename already exists
    output_dir:  PNG directory; None uses decomposition/plots

    Display through pyplot by default. Return the saved path when save is True,
    otherwise None. Rows with non-finite intensity or score are skipped.
    '''
    if (isinstance(mode_index, bool)
            or not isinstance(mode_index, (int, np.integer))
            or mode_index < 0):
        raise ValueError('mode_index must be a non-negative integer')

    groups = {False: ([], []), True: ([], [])}
    for row in rows:
        intensity = float(row.intensity)
        score = float(row.scores[mode_index])
        if not (np.isfinite(intensity) and np.isfinite(score)):
            continue
        x, y = groups[bool(row.marker_info['speech'])]
        x.append(intensity)
        y.append(score)
    if not any(groups[speech][0] for speech in groups):
        raise ValueError('rows contain no finite intensity and mode scores')

    if use_imgcat:
        from imgcat import imgcat

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        for speech, color, label in (
                (False, 'tab:orange', 'Non-speech'),
                (True, 'tab:blue', 'Speech')):
            x, y = groups[speech]
            ax.scatter(x, y, s=3, alpha=0.12, color=color,
                label=f'{label} (n={len(x):,})', rasterized=True)
        ax.set(xlabel='Intensity (dB)',
            ylabel=f'Mode {mode_index} score')
        ax.set_xlim(*xlim)
        ax.legend(markerscale=4)
        fig.tight_layout()

        path = None
        if save:
            directory = (Path(output_dir) if output_dir is not None
                else Path(__file__).resolve().parent / 'plots')
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'mode_{mode_index}_vs_intensity.png'
            with path.open('xb') as handle:
                fig.savefig(handle, format='png', dpi=200)
        if use_imgcat: imgcat(fig)
        else: plt.show()
        return path
    finally:
        plt.close(fig)


def plot_binned_mode_vs_intensity(rows, mode_index, bin_edges=None,
    min_count=100, use_imgcat=False, save=False, output_dir=None):
    '''Plot median mode scores and interquartile ranges by intensity bin.

    rows:        marker_analysis.Row objects, or an iterable of them
    mode_index:  zero-based SVD mode index
    bin_edges:   intensity bin edges; None uses 20 to 60 dB in 2 dB steps
    min_count:   minimum rows per speech group and bin to show that bin
    use_imgcat:  display in a compatible terminal instead of pyplot
    save:        save a PNG, failing if its filename already exists
    output_dir:  PNG directory; None uses decomposition/plots

    Return the saved path when save is True, otherwise None. Shading shows
    score quartiles, not uncertainty intervals. Non-finite rows are skipped.
    '''
    if (isinstance(mode_index, bool)
            or not isinstance(mode_index, (int, np.integer))
            or mode_index < 0):
        raise ValueError('mode_index must be a non-negative integer')
    if (isinstance(min_count, bool)
            or not isinstance(min_count, (int, np.integer))
            or min_count < 1):
        raise ValueError('min_count must be a positive integer')
    if bin_edges is None: bin_edges = np.arange(20, 62, 2)
    bin_edges = np.asarray(bin_edges, dtype=float)
    if (bin_edges.ndim != 1 or len(bin_edges) < 2
            or not np.isfinite(bin_edges).all()
            or not np.all(np.diff(bin_edges) > 0)):
        raise ValueError('bin_edges must be finite and strictly increasing')

    groups = {False: ([], []), True: ([], [])}
    for row in rows:
        intensity = float(row.intensity)
        score = float(row.scores[mode_index])
        if not (np.isfinite(intensity) and np.isfinite(score)):
            continue
        x, y = groups[bool(row.marker_info['speech'])]
        x.append(intensity)
        y.append(score)

    if use_imgcat:
        from imgcat import imgcat

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        plotted = False
        for speech, color, label in (
                (False, 'tab:orange', 'Non-speech'),
                (True, 'tab:blue', 'Speech')):
            x, y = (np.asarray(values) for values in groups[speech])
            centers, lower, median, upper = [], [], [], []
            for left, right in zip(bin_edges[:-1], bin_edges[1:]):
                selected = (x >= left) & (x < right)
                if np.count_nonzero(selected) < min_count: continue
                q25, q50, q75 = np.quantile(y[selected], [0.25, 0.5, 0.75])
                centers.append((left + right) / 2)
                lower.append(q25)
                median.append(q50)
                upper.append(q75)
            if not centers: continue
            plotted = True
            ax.plot(centers, median, color=color, label=label)
            ax.fill_between(centers, lower, upper, color=color, alpha=0.2)
        if not plotted:
            raise ValueError('no intensity bin meets min_count')
        ax.set(xlabel='Intensity (dB)',
            ylabel=f'Mode {mode_index} score')
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.legend()
        fig.tight_layout()

        path = None
        if save:
            directory = (Path(output_dir) if output_dir is not None
                else Path(__file__).resolve().parent / 'plots')
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'binned_intensity_v_mode-{mode_index}.png'
            with path.open('xb') as handle:
                fig.savefig(handle, format='png', dpi=200)
        if use_imgcat: imgcat(fig)
        else: plt.show()
        return path
    finally:
        plt.close(fig)


def plot_file_distributions(rows, min_frames=10, max_frames=50,
    use_imgcat=False, save=False, output_dir=None):
    '''Plot recording durations and sampled marker counts per recording.

    rows:        marker_analysis.Row objects, or an iterable of them
    min_frames:  reference line for a proposed minimum marker count
    max_frames:  reference line for a proposed maximum marker count
    use_imgcat:  display in a compatible terminal instead of pyplot
    save:        save a PNG, failing if its filename already exists
    output_dir:  PNG directory; None uses decomposition/plots

    Plot all supplied recordings; reference lines do not filter the data.
    Durations are converted from Phraser milliseconds to seconds. Counts are
    sampled markers in rows, not all eligible audio frames. Return the saved
    path when save is True, otherwise None.
    '''
    for name, value in (('min_frames', min_frames),
            ('max_frames', max_frames)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if max_frames < min_frames:
        raise ValueError('max_frames must be at least min_frames')

    counts = Counter()
    durations = {}
    for row in rows:
        filename = str(row.marker_info['filename'])
        duration = float(row.marker.audio.duration) / 1000
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('recording duration must be positive and finite')
        previous = durations.get(filename)
        if previous is not None and not np.isclose(previous, duration):
            raise ValueError(f'inconsistent duration for {filename}')
        durations[filename] = duration
        counts[filename] += 1
    if not counts: raise ValueError('rows must not be empty')

    if use_imgcat:
        from imgcat import imgcat

    duration_values = list(durations.values())
    count_values = list(counts.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        axes[0].hist(duration_values, bins=_positive_log_bins(duration_values))
        axes[0].set(xscale='log', xlabel='Recording duration (seconds)',
            ylabel='Number of recordings')

        axes[1].hist(count_values, bins=_positive_log_bins(count_values))
        axes[1].axvline(min_frames, color='tab:orange', linestyle='--',
            label=f'minimum: {min_frames}')
        axes[1].axvline(max_frames, color='tab:red', linestyle='--',
            label=f'cap: {max_frames}')
        axes[1].set(xscale='log', xlabel='Sampled markers per recording',
            ylabel='Number of recordings')
        axes[1].legend()

        eligible = [count for count in count_values if count >= min_frames]
        retained = sum(min(count, max_frames) for count in eligible)
        title = f'{len(counts):,} recordings; {sum(count_values):,} markers'
        title += f'\nProposed limits: {len(eligible):,} recordings, '
        title += f'{retained:,} markers retained'
        fig.suptitle(title)
        fig.tight_layout()
        path = None
        if save:
            directory = (Path(output_dir) if output_dir is not None
                else Path(__file__).resolve().parent / 'plots')
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'file_duration_and_sampled_frames.png'
            with path.open('xb') as handle:
                fig.savefig(handle, format='png', dpi=200)
        if use_imgcat: imgcat(fig)
        else: plt.show()
        return path
    finally:
        plt.close(fig)


def _positive_log_bins(values, n_bins=40):
    '''Return log-spaced histogram edges for positive observations.'''
    low, high = min(values), max(values)
    if low == high: return np.array([low / 1.1, high * 1.1])
    return np.geomspace(low, high, n_bins + 1)
