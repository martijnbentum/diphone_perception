import audio
import locations
from progressbar import progressbar

def make_targets_and_contexts(phoneme_infos=None, sentence_infos=None):
    targets = make_targets_from_phoneme_infos(phoneme_infos)
    contexts = make_contexts_from_sentence_infos(sentence_infos)
    targets = audio.Targets(targets, contexts)
    return targets

def make_targets_from_phoneme_infos(phoneme_infos = None):
    if phoneme_infos is None:
        phoneme_infos = load_selected_phonemes()
    targets = []
    for line in phoneme_infos:
        line['label'] = line['phoneme']
        line['context_id'] = line['audio_filename']
        targets.append(audio.Target(**line))
    return targets

def make_contexts_from_sentence_infos(sentence_infos = None):
    if sentence_infos is None:
        sentence_infos = load_sentence_info()
    contexts = []
    for line in sentence_infos:
        contexts.append(audio.Context(**line))
    return contexts

def map_to_list_of_dicts(data, header):
    output = [{k:v for k, v in zip(header, row)} for row in data]
    for line in output:
        for k in line:
            if k in ['start_time', 'end_time', 'duration']:
                line[k] = float(line[k])
            if k in ['overlap']:
                line[k] = bool(line[k])
    return output

def load_selected_phonemes():
    with open(locations.selected_phonemes, 'r') as f:
        t = [x.split(',') for x in f.read().split('\n') if x]
    header, data = t[0], t[1:]
    return map_to_list_of_dicts(data, header)

def load_sentence_info():
    with open(locations.news_books_cgn_sentences, 'r') as f:
        t = [x.split('\t') for x in f.read().split('\n') if x]
    header, data = t[0], t[1:]
    output = map_to_list_of_dicts(data, header)
    cgn_dir = str(locations.cgn) + '/'
    for line in output:
        line['audio_filename'] = cgn_dir +  line['audio_filename']
    return output

        

def make_or_load_selected_sentences(selected_phonemes, sentence_info):
    if locations.selected_sentences.exists():
        with locations.selected_sentences.open('r') as f:
            selected_sentences = json.load(f)
        return selected_sentences, []
    selected_sentences = []
    found, not_found = [], []
    for phoneme in progressbar(selected_phonemes):
        if phoneme['audio_filename'] in found:
            continue
        for sentence in sentence_info:
            if phoneme['audio_filename'] == sentence['identifier']:
                selected_sentences.append(sentence)
                found.append(phoneme['audio_filename'])
                break
        if phoneme['audio_filename'] not in found:
            not_found.append(phoneme['audio_filename'])
        with locations.selected_sentences.open('w') as f:
            json.dump(selected_sentences, f, indent=4)
    return selected_sentences, not_found



    
