from pathlib import Path

base = Path(__file__).resolve().parent.parent

original = base / 'original'
gated = base / 'gated'
labels = base / 'labels'
responses = base / 'responses'

matrices = responses / 'matrices'
rawdata = responses / 'rawdata'

matrix_plots = base / 'confusion_matrix_plots'
gate_timestamps = base / 'tone_timestamps.csv'

model_responses = Path('model_responses')

cgn = Path('/vol/bigdata/corpora2/CGN2/')

st_phonetics= Path('/vol/mlusers/mbentum/st_phonetics/')
news_books_cgn_sentences = st_phonetics / 'news_books_sentences_zs.tsv'

csv_logs = Path('csv_logs')
selected_phonemes = csv_logs / 'success.csv'
selected_sentences = Path('selected_sentences.json')


pretraining = Path('/vol/mlusers/mbentum/pretraining_clean/')
pretraining_metadata = pretraining / 'metadata/'
manifest = pretraining / 'train.tsv'
cgn_phrases = pretraining / 'cgn_phrases'
cgn_phrases_file_list = pretraining / 'cgn_phrases_file_list.txt'
cgn_speakers = pretraining_metadata / 'cgn_speakers.json'

common_voice_phonemes = pretraining_metadata / 'dutch_cv_phonemes_zs.tsv'
common_voice_words= pretraining_metadata / 'dutch_cv_words_zs-transcribed.tsv'

cgn_phrases_json = Path('../cgn_phrases.json')
