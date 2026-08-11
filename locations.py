from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / 'data'

metadata_file = data / 'metadata.csv'
sentence_file = data / 'news_books_sentences_zs.tsv'
phraser_key_file = data / 'phraser_phone_keys.bin'
duplicate_replacement_phraser_key_file = (
    data / 'duplicate_replacement_phraser_phone_keys.bin')
flemish_phraser_phone_key_file = data / 'flemish_phraser_phone_keys.bin'
duplicate_phone_counts_file = data / 'duplicate_phone_counts.json'
model_paths_file = data / 'model_paths.json'

synthetic_acoustic_probes_data = data / 'synthetic_acoustic_probes'
f0_experiment = synthetic_acoustic_probes_data / 'f0'
f0_umap_plot = f0_experiment / 'f0_umap.pdf'
f0_pure_tone_stimuli = synthetic_acoustic_probes_data / 'f0_pure_tones'
f0_pure_tone_phraser_store = (
    synthetic_acoustic_probes_data / 'f0_pure_tones_phraser_store')
synthetic_acoustic_probes_echoframe_store = (
    synthetic_acoustic_probes_data / 'echoframe_store')

echoframe_store = data / 'echoframe_store'
echoframe_mfcc_store = data / 'echoframe_mfcc_store'
echoframe_model_stores = data / 'echoframe_model_stores'
echoframe_model_flemish_stores = data / 'echoframe_model_flemish_stores'
phone_probes = data / 'phone_probes'
probe_results = data / 'probe_results'

wav2vec2_random_checkpoint_name = 'wav2vec2_checkpoint-0'
wav2vec2_nl1_checkpoint_pattern = r'^wav2vec2_nl1_checkpoint-(\d+)$'
wav2vec2_all_layer_checkpoint_names = {
    wav2vec2_random_checkpoint_name, 'wav2vec2_nl1_checkpoint-200000'}
wav2vec2_all_probe_layers = tuple(range(1, 13))

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
cgn_lmdb = Path('/vol/mlusers/mbentum/phraser/data/cgn_awd_lmdb')

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

mls_phonemes = pretraining_metadata / 'dutch_mls_phonemes_zs.tsv'
mls_words = pretraining_metadata / 'dutch_mls_words_zs.tsv'

cgn_phrases_json = Path('../cgn_phrases.json')
common_voice_phrases_json = Path('../common_voice_phrases.json')
mls_phrases_json = Path('../mls_phrases.json')

textgrid_directory = Path('textgrids/')
