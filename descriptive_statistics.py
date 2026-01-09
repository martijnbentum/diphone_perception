import data

def load_labels():
    return data.load_labels()

def phoneme_gate_duration_dict(labels = None, phoneme_position=1, gate = 1):
    if labels is None: labels = load_labels()
    d = {}
    for label in labels.labels:
        phoneme = getattr(label, f'phoneme{phoneme_position}')
        if phoneme not in d: d[phoneme] = {'gate':[],'phoneme':[]}
        td = label.timestamp_dict
        if td is None: continue
        if f'gate_{gate}_timestamp' not in td: continue
        start = td[f'phoneme_{phoneme_position}_start_time']
        phoneme_end = td[f'phoneme_{phoneme_position}_end_time']
        gate_end = td[f'gate_{gate}_timestamp']
        gate_duration = gate_end - start
        phoneme_duration = phoneme_end - start
        d[phoneme]['gate_duration'].append(duration)
    return d


import math
import matplotlib.pyplot as plt

def mean(values):
    return sum(values) / len(values)

def sample_std(values):
    m = mean(values)
    # sample std (ddof=1)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def mean_ci95(values):
    m = mean(values)
    n = len(values)
    if n < 2:
        return m, 0.0  # no CI with <2 points
    s = sample_std(values)
    sem = s / math.sqrt(n)
    ci_halfwidth = 1.96 * sem
    return m, ci_halfwidth

def plot_mean_ci95(data_dict, title=None, ylabel=None):
    # stable order (alphabetical by key); change to sorted(..., key=...) if needed

    stats = []
    for k, values in data_dict.items():
        m, ci = mean_ci95(values)
        stats.append((k, m, ci))

    # sort by mean (descending)
    stats.sort(key=lambda x: x[1], reverse=True)

    keys  = [k for k, _, _ in stats]
    means = [m for _, m, _ in stats]
    ci    = [c for _, _, c in stats]

    x = list(range(len(keys)))
    plt.figure()
    plt.errorbar(x, means, yerr=ci, fmt='o', capsize=4)
    plt.xticks(x, keys)
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Example usage:
# d = {'a': [0.1, 0.2, ...], 'b': [...], ...}
# plot_mean_ci95(d, title='Means with 95% CI', ylabel='value')

        
    
