import data
import math
import matplotlib.pyplot as plt

def load_labels():
    return data.load_labels()

def phoneme_gate_duration_dict(labels = None, phoneme_position=1, gate = 1):
    '''collect gate durations and phoneme durations per phoneme 
    diphone recordings with gate timesteps (gate 1 - 6)
    gate 1 - 3 overlaps with phoneme position 1 
    gate 4 - 6 overlaps with phoneme position 2
    
    labels: data.Labels object; if None, load from data module
    phoneme_position: 1 or 2
    gate: 1-6
        
    '''
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
        d[phoneme]['gate'].append(gate_duration)
        d[phoneme]['phoneme'].append(phoneme_duration)
    return d

def mean(values):
    return sum(values) / len(values)

def sample_std(values):
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def mean_ci95(values):
    n = len(values)
    if n < 2:
        return (mean(values) if n == 1 else float('nan')), 0.0
    m = mean(values)
    sem = sample_std(values) / math.sqrt(n)
    return m, 1.96 * sem  # normal approx

def gate_percent_of_phoneme(gate_list, phoneme_list):
    n = min(len(gate_list), len(phoneme_list))
    perc = []
    for i in range(n):
        p = phoneme_list[i]
        g = gate_list[i]
        if p and p > 0:
            perc.append(100.0 * g / p)
    return perc

def plot_gate_phoneme_with_percent(data, title=None, show_gate = True, 
    show_phoneme = True,show_perc= True):
    # data = {char: {'gate': [...], 'phoneme': [...]}}

    rows = []
    for ch, d in data.items():
        gate = d['gate']
        phon = d['phoneme']

        gate_m, gate_ci = mean_ci95(gate)
        phon_m, phon_ci = mean_ci95(phon)

        perc = gate_percent_of_phoneme(gate, phon)
        perc_m, perc_ci = mean_ci95(perc)

        rows.append({
            'ch': ch,
            'gate_m': gate_m, 'gate_ci': gate_ci,
            'phon_m': phon_m, 'phon_ci': phon_ci,
            'perc_m': perc_m, 'perc_ci': perc_ci,
        })

    # sort chars descending by mean gate duration (default)
    rows.sort(key=lambda r: r['gate_m'], reverse=True)

    labels = [r['ch'] for r in rows]
    x = list(range(len(rows)))

    gate_means = [r['gate_m'] for r in rows]
    gate_cis   = [r['gate_ci'] for r in rows]

    phon_means = [r['phon_m'] for r in rows]
    phon_cis   = [r['phon_ci'] for r in rows]

    perc_means = [r['perc_m'] for r in rows]
    perc_cis   = [r['perc_ci'] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Left axis: durations
    legend_handles = []
    legend_names = []
    if show_gate:
        h1 = ax.errorbar(x, gate_means, yerr=gate_cis, fmt='o', capsize=4, 
            label='gate duration')
        legend_handles.append(h1)
        legend_names.append('gate duration')
            
    if show_phoneme:
        h2 = ax.errorbar(x, phon_means, yerr=phon_cis, fmt='o', capsize=4,
                     color='grey', alpha=0.5, label='phoneme duration')
        legend_handles.append(h2)
        legend_names.append('phoneme duration')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Duration')
    if title:
        ax.set_title(title)

    # Right axis: % gate of phoneme
    if show_perc:
        ax2 = ax.twinx()
        h3 = ax2.errorbar(x, perc_means, yerr=perc_cis, fmt='s', capsize=4,
            linestyle='none', label='gate as % of phoneme', color='red')
        ax2.set_ylabel('Gate (% of phoneme)')
        legend_handles.append(h3)
        legend_names.append('gate as % of phoneme')
        # optional: keep it sane; comment out if you don't want this
        # ax2.set_ylim(0, 120)

    # Combined legend
    ax.legend(legend_handles, legend_names, loc='best')

    fig.tight_layout()
    ax.set_axisbelow(True) 
    ax.grid(True, axis='x', alpha=0.5)  
    ax.grid(True, axis='y', alpha=0.5)
    plt.show()

# usage:
# plot_gate_phoneme_with_percent(data, title='Gate vs phoneme + gate% (sorted by mean gate)')


'''

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

'''
        
    
