import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse
import torch
import pandas as pd
import numpy as np
import zipfile
import io
import frame
from src.phoneme_mapper import Mapper
from sklearn.utils import shuffle
import logging
import tempfile
import shutil
import torchaudio


def extract_subset(zip_path, extract_to, num_files):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        all_files = zipf.namelist()
        subset_files = all_files[:num_files]
        for file in subset_files:
            zipf.extract(file, extract_to)
    return [os.path.basename(f) for f in subset_files]


def save_error_log(rows, filename):
    if rows:
        pd.DataFrame(rows).to_csv(filename, index=False)


def apply_model(wav_file, model, processor, device):

    audio, sr = torchaudio.load(wav_file)
    input_values = processor(audio,
                             sampling_rate=sr,
                             return_tensors="pt",
                             padding="longest").input_values
    
    input_values = input_values.squeeze(0).to(device)

    with torch.no_grad():
        output_dict = model(input_values, output_hidden_states=True)

    return output_dict.hidden_states


def get_phoneme_embeds(metadata, model, processor, target_phonemes, device, args):

    # remove rows with parsing errors & shuffle metadata
    metadata = metadata[metadata['phoneme'].apply(lambda x: len(x) <= 4)]
    metadata = shuffle(metadata, random_state=42)

    # load duration threshold information (we use Q3 of each phoneme's duration distribution as the max. value)
    duration_thresholds = pd.read_csv(args.phoneme_duration_threshold_file, sep=',', header=0)
    duration_lookup = duration_thresholds.set_index('ipa_phoneme')['max_threshold'].to_dict() #TODO: change back to Q3 if necessary

    # count number of files for component K and O
    for comp in ['k', 'o']:
        subset = metadata[metadata['comp'] == comp]
        num_files = subset['audio_filename'].nunique()
        logging.info(f'Number of unique audio files in metadata for component "{comp}": {num_files}')

    all_embeddings = {}
    global_phoneme_counts = {p: 0 for p in target_phonemes}

    # track various error cases (start frame bigger/equal/one smaller than end frame + too long duration)
    error_bigger, error_equal, error_one, error_duration, success, none = [], [], [], [], [], []

    temp_data_dir = tempfile.mkdtemp()
    audio_cache = {} # cache audio for reuse

    try:

        if args.use_subset:

            NUM_FILES_TO_EXTRACT = 30

            # extract subset of files
            extracted_files_k = extract_subset(args.data_dir1, temp_data_dir, NUM_FILES_TO_EXTRACT)
            extracted_files_o = extract_subset(args.data_dir2, temp_data_dir, NUM_FILES_TO_EXTRACT)
            extracted_files = set(extracted_files_k + extracted_files_o)

            # filter metadata to only include rows for the subset'
            metadata = metadata[metadata['audio_filename'].isin(extracted_files)]

        else:
            # unpack first zip file (comp k)
            with zipfile.ZipFile(args.data_dir1, 'r') as zipf1:
                zipf1.extractall(temp_data_dir)

            # unpack second zip file (comp o)
            with zipfile.ZipFile(args.data_dir2, 'r') as zipf2:
                zipf2.extractall(temp_data_dir)

        logging.info(f'All files extracted to: {temp_data_dir}')

        if args.use_subset:
            # Count number of files for component K and O
            for comp in ['k', 'o']:
                subset = metadata[metadata['comp'] == comp]
                num_files = subset['audio_filename'].nunique()
                logging.info(f'Number of unique audio files in subset metadata for component "{comp}": {num_files}')

        # Group metadata by speaker
        for s in metadata['speaker_id'].unique():
            if not s.startswith('N'):
                continue

            speaker_df = metadata[metadata['speaker_id'] == s]
            phoneme_counts = {p: 0 for p in target_phonemes}
            logging.info(f'Extracting phoneme embeds for speaker {s}...')

            # Group phonemes for this specific speaker by audio file
            grouped = speaker_df.groupby('audio_filename')

            for audio_filename, file_df in grouped:
                file_path = os.path.join(temp_data_dir, 'data/cgn_sentences/split_files', audio_filename)
                if not os.path.exists(file_path):
                    logging.warning(f"{file_path} not found")
                    continue

                # Use audio cache
                if audio_filename in audio_cache:
                    hidden_states = audio_cache[audio_filename]
                else:
                    with open(file_path, 'rb') as f:
                        file_stream = io.BytesIO(f.read())
                        hidden_states = apply_model(file_stream, model, processor, device)

                if hidden_states is None:
                    logging.warning(f"Hidden states is None for {audio_filename}")
                    continue

                # convert to numpy
                hidden_states_np = [layer.squeeze(0).detach().cpu().numpy() for layer in hidden_states]

                # process each row (phoneme) in the file
                for _, row in file_df.iterrows():
                    phoneme = row['ipa_phoneme']
                    if global_phoneme_counts[phoneme] >= args.max_phonemes_total:
                        continue
                    if phoneme_counts[phoneme] >= args.n_phonemes:
                        continue

                    # check if the phoneme duration is not too long
                    q3_value = duration_lookup.get(phoneme)
                    if row['duration'] > q3_value:
                        # log these cases
                        error_duration.append(row)
                        continue

                    # time to frame indices
                    f = frame.Frames(n_frames=hidden_states[0].shape[1])
                    start = f.start_frame(start=row['start_time'], percentage_overlap=100)
                    end = f.end_frame(end=row['end_time'], percentage_overlap=100)

                    # skip cases where less than 2 frames are selected (+ log these cases)
                    if start is None or end is None:
                        none.append(row)
                        continue
                    if start.index > end.index:
                        error_bigger.append(row)
                        continue
                    elif start.index == end.index:
                        error_equal.append(row)
                        continue
                    elif end.index - start.index == 1:
                        error_one.append(row)
                        continue
                    else:
                        success.append(row)

                        # save the frame sequence
                        for layer_idx, layer_states in enumerate(hidden_states_np):
                            phoneme_seq = layer_states[start.index:end.index, :]
                            key = f'{s}/{phoneme}/layer{layer_idx:02d}/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                            all_embeddings[key] = phoneme_seq

                        # count how many examples we have for this phoneme
                        phoneme_counts[phoneme] += 1
                        global_phoneme_counts[phoneme] += 1
                        logging.info(f'global phoneme count {phoneme}: {global_phoneme_counts[phoneme]}')

                maxed_phonemes = [p for p, c in global_phoneme_counts.items() if c >= args.max_phonemes_total]
                
                if maxed_phonemes:
                    logging.info(f"Phonemes that reached max: {len(maxed_phonemes)} - {', '.join(maxed_phonemes)}")

                if all(count >= args.n_phonemes for count in phoneme_counts.values()):
                    logging.info(f"Speaker {s}: enough examples collected.")
                    break

                if all(count >= args.max_phonemes_total for count in global_phoneme_counts.values()):
                    logging.info(f"Global phoneme limit reached for all phonemes.")
                    break

            if all(count >= args.max_phonemes_total for count in global_phoneme_counts.values()):
                logging.info(f"Global phoneme limit reached for all phonemes.")
                break

        # save all embeddings
        output_file = os.path.join(f'{args.output_dir}_{args.model_type}', 'phoneme_embeddings.npz')
        np.savez_compressed(output_file, **all_embeddings)
        logging.info(f'All embeddings saved to: {output_file}')

        # save csv logs with error & success cases
        os.makedirs(f'csv_logs', exist_ok=True)
        os.makedirs(f'csv_logs/{args.model_type}', exist_ok=True)
        save_error_log(error_bigger, f'csv_logs/{args.model_type}/start_bigger_than_end.csv')
        save_error_log(error_equal, f'csv_logs/{args.model_type}/start_equal_to_end.csv')
        save_error_log(error_one, f'csv_logs/{args.model_type}/start_one_smaller_than_end.csv')
        save_error_log(success, f'csv_logs/{args.model_type}/success.csv')
        save_error_log(error_duration, f'csv_logs/{args.model_type}/too_long.csv')
        save_error_log(none, f'csv_logs/{args.model_type}/none_error.csv')

    finally:
        # clean up extracted files
        shutil.rmtree(temp_data_dir)

 

def main(args):

    logging.basicConfig(filename=f'extract_embeds_{args.model_type}.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    # load model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    #processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2Model.from_pretrained(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f'Moved model to {device}')

    # create output dir
    os.makedirs(f'{args.output_dir}_{args.model_type}', exist_ok=True)

    # load metadata with timestamps
    metadata = pd.read_csv(args.metadata_file, sep='\t', header=0)

    target_phonemes = ['p', 'b', 't', 'd', 'k', 'f', 'v', 's', 'z', 'G', 'x', 'm', 'n', 'N', 'h',
                       'I', 'E', 'A', 'O', 'i', 'e', 'a', 'o', 'u', 'y', '@', 'E+', 'Y+', 'A+',
                       'l', 'r', 'j', 'w', 'Y'] # 'S', 'Z', 'g', '2'

    # map cgn to ipa
    mapper = Mapper(language='dutch')
    ipa_target_phonemes = [mapper.cgn_to_ipa[p] for p in target_phonemes if p in mapper.cgn_to_ipa]
    missing = [p for p in target_phonemes if p not in mapper.cgn_to_ipa]
    if missing:
        logging.warning(f"Missing phonemes in CGN->IPA mapping: {missing}")

    # map metadata to IPA
    metadata['ipa_phoneme'] = metadata['phoneme'].map(mapper.cgn_to_ipa)

    # filter metadata
    metadata = metadata[metadata['ipa_phoneme'].isin(ipa_target_phonemes)]

    # logging
    logging.info(str(ipa_target_phonemes))
    logging.info(str(metadata['ipa_phoneme'].dropna().unique()))

    # get phoneme embeddings
    get_phoneme_embeds(metadata, model, processor, list(set(ipa_target_phonemes)), device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/checkpoint_229_100000')
    parser.add_argument('--model_type', type=str, default='pretrained')
    parser.add_argument('--metadata_file', type=str, default='metadata/timestamps/news_books_phonemes_zs.tsv')
    parser.add_argument('--phoneme_duration_threshold_file', type=str, default='metadata/cgn_phoneme_duration_stats/phoneme_duration_thresholds.csv')
    parser.add_argument('--data_dir1', type=str, default='compressed_data/cgn_comp_k.zip')
    parser.add_argument('--data_dir2', type=str, default='compressed_data/cgn_comp_o.zip')
    parser.add_argument('--output_dir', type=str, default='phoneme_embeds_cgn_K+O')
    parser.add_argument('--n_phonemes', type=int, default=10)
    parser.add_argument('--max_phonemes_total', type=int, default=1000)
    parser.add_argument('--use_subset', type=bool, default=False)
    args = parser.parse_args()
    main(args)
