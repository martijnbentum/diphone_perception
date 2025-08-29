import json
import locations
import cgn_phonemes
from progressbar import progressbar

def load_manifest(manifest_file=None):
    if manifest_file is None:
        manifest_file = locations.manifest
    with manifest_file.open('r') as f:
        t = f.read().split('\n')
    manifest = [x.split('\t')[0].split('/')[-1] for x in t[1:] if x]
    return manifest

def load_cgn_phrases_file_list():
    with locations.cgn_phrases_file_list.open('r') as f:
        t = f.read().split('\n')
    file_list = [x for x in t if x]
    return file_list


def load_cgn_speakers():
    with locations.cgn_speakers.open('r') as f:
        cgn_speakers = json.load(f)
    return cgn_speakers

def extract_all_cgn_phrases(cgn_speakers=None):
    phrases = []
    for speaker in cgn_speakers:
        for phrase in cgn_speakers[speaker]['phrases']:
            phrases.append(phrase)
    return phrases

def clean_cgn_phrases(phrases):
    bads = []
    outputs = []
    error = []
    d = cgn_phonemes.Sampa().simple_sampa_to_simple_ipa
    d[' '] = ' '
    for phrase in progressbar(phrases):
        sampa = phrase['sampa']
        sampa = sampa.strip('! ').replace('!','').strip(' ')
        if len(sampa) == 0:
            bads.append(phrase)
            continue
        ok = sum([c in d for c in set(sampa)])
        if ok < len(set(sampa)):
            error.append(phrase)
            continue
        ipa = ' '.join([d[c] for c in sampa])
        output = {'ipa': ipa, 
            'orthographic': phrase['orthographic'],
            'speaker_id': phrase['speaker_id'], 
            'gender': phrase['gender'],
            'age': phrase['age'],
            'audio_filename': phrase['audio_filename'],
            'corpus': 'cgn',
            'language': phrase['language'],
        }
        outputs.append(output)
    return outputs, bads, error

def load_common_voice_phonemes():
    with locations.common_voice_phonemes.open('r') as f:
        t = f.read().split('\n')
    output = [x.split('\t') for x in t[1:] if x]
    return output

def load_common_voice_words():
    with locations.common_voice_words.open('r') as f:
        t = f.read().split('\n')
    output = [x.split('\t') for x in t[1:] if x]
    return output

def clean_common_voice_phrases(common_voice_words= None):
    if not common_voice_words:
        common_voice_words = load_common_voice_words()
    d = cgn_phonemes.Ipa().ipa_to_simple_ipa_dict_mauser
    cv_phrases = _common_voice_words_to_phrases(common_voice_words)
    outputs = []
    for filename, words in cv_phrases.items():
        speaker = words[0][5]
        speaker_id, gender, age = _extract_speakerid_gender_age(speaker)
        ipa_phrase, ort_phrase = _common_voice_words_to_ipa_ort_phrase(words, d)
        output = {
            'ipa': ipa_phrase,
            'orthographic': ort_phrase,
            'speaker_id': speaker_id,
            'gender': gender,
            'age': age,
            'audio_filename': filename,
            'corpus': 'common voice',
            'language': 'Netherlandic Dutch',
            }
        outputs.append(output)
    return outputs

def _common_voice_words_to_phrases(common_voice_words= None):
    if not common_voice_words:
        common_voice_words = load_common_voice_words()
    d = {}
    for x in common_voice_words:
        filename = x[0]
        if filename not in d:
            d[filename] = []
        d[filename].append(x)
    return d

def _common_voice_words_to_ipa_ort_phrase(words, ipa_dict):
    ipa_phrase = []
    ort_phrase = []
    for word in words:
        word_ipa_mauser = word[-2].split(' ')
        word_ipa = ' '.join([ipa_dict[x] for x in word_ipa_mauser])
        ipa_phrase.append(word_ipa)
        ort_phrase.append(word[4])
    return '  '.join(ipa_phrase), ' '.join(ort_phrase)

def _extract_speakerid_gender_age(s):
    if not s: return None, None, None
    speaker_id, gender, age= s.split('_')
    if not speaker_id: speaker_id = None
    if not gender: gender = None
    if not age: age = None
    return speaker_id, gender, age
    
