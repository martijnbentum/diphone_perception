from collections import Counter
import math
import pretraining_metadata as pm
import re

def load_pretraining_datasets(exclude_not_in_manifest = True):
    '''Load pretraining datasets from pretraining_metadata module.
    '''
    return pm.load_pretraining_datasets(exclude_not_in_manifest)

def make_ipa_phrases(pretraining_datasets = None, remove_word_boundaries = True):
    '''Return list of IPA phrases from pretraining datasets.
    '''
    if not pretraining_datasets: 
        pretraining_datasets = load_pretraining_datasets()
    ipas = []
    for line in pretraining_datasets:
        ipa = line['ipa']
        if remove_word_boundaries: ipa = ipa.replace('   ', ' ')
        ipas.append(ipa)
    return ipas

def phone_frequencies(pretraining_datasets):
    '''Return a Counter mapping phoneme (IPA symbol) to frequency count.
    '''
    ipas = make_ipa_phrases(pretraining_datasets, remove_word_boundaries = True)
    ipas = ' '.join(ipas)
    ipas = re.sub(r'\s{2,}', ' ', ipas)
    return Counter(ipas.split(' '))

def bin_equal_width(counter_dict, n_bins):
    '''bin items into n_bins of approximately equal size.
    '''
    length = len(counter_dict)
    phonemes = [x[0] for x in counter_dict.most_common()]
    n_items = math.ceil(length / n_bins)
    binned_phonemes = []
    for i in range(n_bins):
        start = i * n_items
        end = min((i + 1) * n_items, length)
        binned_phonemes.append(phonemes[start:end])
    if end < (len(phonemes) - 1):
        binned_phonemes[-1].extend(phonemes[end:])
    return binned_phonemes
        

def bin_by_cumulative_mass(counter_dict, n_bins):
    '''
    Split items into bins with approximately equal cumulative count mass.
    High-frequency items concentrate in early bins; tail bins contain many 
    low-frequency items.

    counter_dict: dict mapping item → count
    n_bins: number of bins to create
    '''
    items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in items)
    cut_points = [(i + 1) * total / n_bins for i in range(n_bins - 1)]
    bins = [[] for _ in range(n_bins)]
    cum = 0
    bin_idx = 0

    for k, c in items:
        cum += c
        while bin_idx < n_bins - 1 and cum > cut_points[bin_idx]:
            bin_idx += 1
        bins[bin_idx].append(k)

    return bins

def bin_by_rank_geometric(counter_dict, n_bins):
    '''
    Split items into bins using geometric cut points in rank space.
    Bins are evenly spaced on a log-rank scale rather than by mass.

    counter_dict: dict mapping item → count
    n_bins: number of bins to create
    '''

    items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    n = len(items)

    # geometric cut points in rank space
    cuts = [int(n ** ((i + 1) / n_bins)) for i in range(n_bins - 1)]

    bins = []
    start = 0
    for cut in cuts:
        bins.append([k for k, _ in items[start:cut]])
        start = cut
    bins.append([k for k, _ in items[start:]])

    return bins



