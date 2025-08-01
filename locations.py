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

