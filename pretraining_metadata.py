import json
import locations
import cgn_phonemes
from progressbar import progressbar

def make_or_load_cgn_phrases_json(phrases = None, speakers = None, 
    overwrite = False):
    if locations.cgn_phrases_json.exists() and not overwrite:
        print('loading existing cgn phrases json')
        with locations.cgn_phrases_json.open('r') as f:
            outputs = json.load(f)
        return outputs
    print('making new cgn phrases json')
    if speakers is None and phrases is None:
        speakers = load_cgn_speakers()
    if phrases is None:
        phrases = extract_all_cgn_phrases(speakers)
    outputs, bads, error = clean_cgn_phrases(phrases)
    with locations.cgn_phrases_json.open('w') as f:
        json.dump(outputs, f, indent=4)
    print(f'Wrote {len(outputs)} phrases to {locations.cgn_phrases_json}')
    print(f'Skipped {len(bads)} empty phrases')
    print(f'Skipped {len(error)} phrases with unknown phonemes')
    return outputs 

def make_or_load_common_voice_phrases_json(common_voice_words= None, 
    overwrite = False):
    if locations.common_voice_phrases_json.exists() and not overwrite:
        print('loading existing common voice phrases json')
        with locations.common_voice_phrases_json.open('r') as f:
            outputs = json.load(f)
        return outputs
    print('making new common voice phrases json')
    outputs = clean_common_voice_phrases(common_voice_words)
    with locations.common_voice_phrases_json.open('w') as f:
        json.dump(outputs, f, indent=4)
    print(f'Wrote {len(outputs)} phrases to {locations.common_voice_phrases_json}')
    return outputs


def make_cgn_phrase_name(speaker_id, file_id, start_time, end_time):
    m = f'{speaker_id}_{file_id}_{start_time:.3f}-{end_time:.3f}'
    m = m.replace('.','__')
    m += '.wav'
    return m

def phrase_to_phrase_name(phrase):
    speaker_id = phrase['speaker_id']
    audio_filename = phrase['audio_filename']
    file_id = audio_filename.split('.')[0].split('/')[-1]
    start_time = phrase['start_time']
    end_time = phrase['end_time']
    return make_cgn_phrase_name(speaker_id, file_id, start_time, end_time)

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

def clean_cgn_phrases(phrases, check_manifest= True):
    bads = []
    outputs = []
    error = []
    d = cgn_phonemes.Sampa().simple_sampa_to_simple_ipa
    d[' '] = ' '
    manifest = load_manifest()
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
        phrase_filename = phrase_to_phrase_name(phrase)
        in_manifest = phrase_filename in manifest if check_manifest else None
        output = {'ipa': ipa, 
            'orthographic': phrase['orthographic'],
            'speaker_id': phrase['speaker_id'], 
            'gender': phrase['gender'],
            'age': phrase['age'],
            'audio_filename': phrase['audio_filename'],
            'corpus': 'cgn',
            'language': phrase['language'],
            'phrase_filename': phrase_filename,
            'in_manifest': in_manifest,
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

def clean_common_voice_phrases(common_voice_words= None, check_manifest= True):
    if not common_voice_words:
        common_voice_words = load_common_voice_words()
    d = cgn_phonemes.Ipa().ipa_to_simple_ipa_dict_mauser
    cv_phrases = _common_voice_words_to_phrases(common_voice_words)
    manifest = load_manifest()
    outputs = []
    for filename, words in progressbar(cv_phrases.items()):
        speaker = words[0][5]
        speaker_id, gender, age = _extract_speakerid_gender_age(speaker)
        ipa_phrase, ort_phrase = _common_voice_words_to_ipa_ort_phrase(words, d)
        in_manifest = filename in manifest if check_manifest else None
        output = {
            'ipa': ipa_phrase,
            'orthographic': ort_phrase,
            'speaker_id': speaker_id,
            'gender': gender,
            'age': age,
            'audio_filename': filename,
            'corpus': 'common voice',
            'language': 'Netherlandic Dutch',
            'in_manifest': in_manifest,
            }
        outputs.append(output)
    return outputs

def _common_voice_words_to_phrases(common_voice_words= None):
    if not common_voice_words:
        print('loading common voice words')
        common_voice_words = load_common_voice_words()
    d = {}
    for x in common_voice_words:
        filename = x[0]
        if filename not in d:
            d[filename] = []
        d[filename].append(x)
    return d

def _mls_words_to_phrases(mls_words= None):
    if not mls_words:
        print('loading mls words')
        mls_words = load_mls_words()
    return _common_voice_words_to_phrases(mls_words)

def _common_voice_words_to_ipa_ort_phrase(words, ipa_dict):
    ipa_phrase = []
    ort_phrase = []
    for word in words:
        word_ipa_mauser = word[-2].split(' ')
        word_ipa = ' '.join([ipa_dict[x] for x in word_ipa_mauser])
        ipa_phrase.append(word_ipa)
        ort_phrase.append(word[4])
    return '  '.join(ipa_phrase), ' '.join(ort_phrase)

def _mls_words_to_ipa_ort_phrase(words, ipa_dict):
    ipa_phrase = []
    ort_phrase = []
    for word in words:
        word_ipa_mauser = word[6].split(' ')
        word_ipa = ' '.join([ipa_dict[x] for x in word_ipa_mauser])
        ipa_phrase.append(word_ipa)
        ort_phrase.append(word[5])
    return '  '.join(ipa_phrase), ' '.join(ort_phrase)

def _extract_speakerid_gender_age(s):
    if not s: return None, None, None
    speaker_id, gender, age= None, None, None
    o =  s.split('_')
    if len(o) > 0: speaker_id = o[0]
    if len(o) > 1: gender = o[1]
    if len(o) > 2: age = o[2]
    if not speaker_id: speaker_id = None
    if not gender: gender = None
    if not age: age = None
    return speaker_id, gender, age

def load_mls_phonemes():
    with locations.mls_phonemes.open('r') as f:
        t = f.read().split('\n')
    output = [x.split('\t') for x in t[1:] if x]
    return output

def load_mls_words():
    with locations.mls_words.open('r') as f:
        t = f.read().split('\n')
    output = [x.split('\t') for x in t[1:] if x]
    return output


def clean_mls_phrases(mls_words= None, check_manifest= True):
    if not mls_words:
        mls_words = load_mls_words()
    d = cgn_phonemes.Ipa().ipa_to_simple_ipa_dict_mauser
    mls_phrases = _mls_words_to_phrases(common_voice_words)
    manifest = load_manifest()
    outputs = []
    error = []
    for filename, words in progressbar(cv_phrases.items()):
        if len(words) == 0:
            error.append(filename)
            continue
        speaker = words[0][-2]
        speaker_id, gender, age = _extract_speakerid_gender_age(speaker)
        ipa_phrase, ort_phrase = _mls_words_to_ipa_ort_phrase(words, d)
        in_manifest = filename in manifest if check_manifest else None
        output = {
            'ipa': ipa_phrase,
            'orthographic': ort_phrase,
            'speaker_id': speaker_id,
            'gender': gender,
            'age': age,
            'audio_filename': filename,
            'corpus': 'mls',
            'language': 'Netherlandic Dutch',
            'in_manifest': in_manifest,
            }
        outputs.append(output)
    print(f'Skipped {len(error)} empty phrases', ' '.join(error))
    return outputs
    
