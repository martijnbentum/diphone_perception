from collections import Counter
import pretraining_metadata as pm
import re

def load_pretraining_datasets(exclude_not_in_manifest = True):
    return pm.load_pretraining_datasets(exclude_not_in_manifest)

def make_ipa_phrases(pretraining_datasets = None, remove_word_boundaries = True):
    if not pretraining_datasets: 
        pretraining_datasets = load_pretraining_datasets()
    ipas = []
    for line in pretraining_datasets:
        ipa = line['ipa']
        if remove_word_boundaries: ipa = ipa.replace('   ', ' ')
        ipas.append(ipa)
    return ipas

def phone_frequencies(pretraining_datasets):
    ipas = make_ipa_phrases(pretraining_datasets, remove_word_boundaries = True)
    ipas = ' '.join(ipas)
    ipas = re.sub(r'\s{2,}', ' ', ipas)
    return Counter(ipas.split(' '))
        

