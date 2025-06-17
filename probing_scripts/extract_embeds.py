import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import math
import argparse
import torch
import pandas as pd
import librosa
import numpy as np
import zipfile
import io
import frame
from src.phoneme_mapper import Mapper
from sklearn.utils import shuffle
import logging
import tempfile
import shutil


def apply_model(wav_file, model, processor, device):

    audio, sr = librosa.load(wav_file, sr=16000)
    input_values = processor(audio,
                             sampling_rate=sr,
                             return_tensors="pt",
                             padding="longest").input_values
    
    input_values = input_values.to(device)

    with torch.no_grad():
        output_dict = model(input_values, output_hidden_states=True)

    return output_dict.hidden_states


def get_phoneme_embeds(metadata, model, processor, target_phonemes, device, args):
    # Remove rows with parsing errors
    metadata = metadata[metadata['phoneme'].apply(lambda x: len(x) <= 4)]

    print('n speakers:', len(metadata['speaker_id'].unique()))

    # Shuffle metadata
    metadata = shuffle(metadata, random_state=42)

    # Count number of files for component K and O
    for comp in ['k', 'o']:
        subset = metadata[metadata['comp'] == comp]
        num_files = subset['audio_filename'].nunique()
        logging.info(f'Number of unique audio files in metadata for component "{comp}": {num_files}')

    all_embeddings = {}
    global_phoneme_counts = {p: 0 for p in target_phonemes}

    # track metadata rows with phonemes that are too short
    lst1, lst2, lst3 = [], [], []

    temp_data_dir = tempfile.mkdtemp()
    try:
        # unpack first zip file (comp k)
        with zipfile.ZipFile(args.data_dir1, 'r') as zipf1:
            zipf1.extractall(temp_data_dir)

        # unpack second zip file (comp o)
        if args.data_dir2 != 'none':
            with zipfile.ZipFile(args.data_dir2, 'r') as zipf2:
                zipf2.extractall(temp_data_dir)

        logging.info(f'All files extracted to: {temp_data_dir}')

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

                try:
                    with open(file_path, 'rb') as f:
                        file_stream = io.BytesIO(f.read())
                        hidden_states = apply_model(file_stream, model, processor, device)
                except Exception as e:
                    logging.warning(f"Error processing {audio_filename}: {e}")
                    continue

                if hidden_states is None:
                    logging.warning(f"Hidden states is None for {audio_filename}")
                    continue

                # process each row (phoneme) in the file
                for _, row in file_df.iterrows():
                    phoneme = row['phoneme']
                    if phoneme not in target_phonemes:
                        continue
                    if global_phoneme_counts[phoneme] >= args.max_phonemes_total:
                        continue
                    if phoneme_counts[phoneme] >= args.n_phonemes:
                        continue

                    # time to frame indices
                    f = frame.Frames(n_frames=hidden_states[0].shape[1])
                    start = f.start_frame(start=row['start_time'], percentage_overlap=100)
                    end = f.end_frame(end=row['end_time'], percentage_overlap=100)

                    if start.index > end.index:
                        lst1.append(row)
                        continue
                    elif start.index == end.index:
                        lst2.append(row)
                        continue
                    elif end.index - start.index == 1:
                        lst3.append(row)
                        continue
                    else:
                        logging.info(f'{phoneme} start idx {start}, end idx {end}')

                        for layer_idx, layer_states in enumerate(hidden_states):
                            phoneme_seq = layer_states[:, start.index:end.index, :]
                            phoneme_seq_np = phoneme_seq.squeeze(0).detach().cpu().numpy()

                            key = f'{s}/{phoneme}/layer{layer_idx:02d}/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                            all_embeddings[key] = phoneme_seq_np

                        phoneme_counts[phoneme] += 1
                        global_phoneme_counts[phoneme] += 1

                        #logging.info(f'global phoneme count {phoneme}: {global_phoneme_counts[phoneme]}')

                if all(count >= args.n_phonemes for count in phoneme_counts.values()):
                    logging.info(f"Speaker {s}: enough examples collected.")
                    break

            if all(count >= args.max_phonemes_total for count in global_phoneme_counts.values()):
                logging.info(f"Global phoneme limit reached.")
                break

        # save all embeddings
        output_file = os.path.join(args.output_dir, 'phoneme_embeddings.npz')
        np.savez_compressed(output_file, **all_embeddings)
        logging.info(f'All embeddings saved to: {output_file}')

        # convert to dataframes
        df_1 = pd.DataFrame(lst1)
        df_2 = pd.DataFrame(lst2)
        df_3 = pd.DataFrame(lst3)

        # save to CSV
        df_1.to_csv('start_bigger_than_end.csv', index=False)
        df_2.to_csv('start_equal_to_end.csv', index=False)
        df_3.to_csv('start_one_smaller_than_end.csv', index=False)

    finally:
        # clean up extracted files
        shutil.rmtree(temp_data_dir)

 

def main(args):

    logging.basicConfig(filename='extract_embeds.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    # load model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f'Moved model to {device}')

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load metadata with timestamps
    metadata = pd.read_csv(args.metadata_file, sep='\t', header=0)

    target_phonemes = ['p', 'b', 't', 'd', 'k', 'f', 'v', 's', 'z', 'G', 'x', 'm', 'n', 'N', 'l', 'r', 'j', 'w', 'h',
                       'I', 'E', 'A', 'O', 'Y', 'i', 'e', 'a', 'o', 'u', 'y', '@', 'E+', 'Y+', 'A+'] # 'S', 'Z', 'g', '2'

    # map cgn to ipa
    mapper = Mapper(language='dutch')
    ipa_target_phonemes = [mapper.cgn_to_ipa[p] for p in target_phonemes]
    print(ipa_target_phonemes)

    print('Number of phoneme classes:', len(ipa_target_phonemes))

    # get phoneme embeddings
    get_phoneme_embeds(metadata, model, processor, list(set(ipa_target_phonemes)), device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/checkpoint_229_100000')
    parser.add_argument('--encoder_sampling_rate', type=int, default=50) #w2v2 encoder uses 49 Hz according to paper, but 50 Hz is better for longer files
    parser.add_argument('--metadata_file', type=str, default='metadata/news_books_phonemes_zs.tsv')
    parser.add_argument('--data_dir1', type=str, default='compressed_data/cgn_comp_k.zip')
    parser.add_argument('--data_dir2', type=str, default='compressed_data/cgn_comp_o.zip')
    parser.add_argument('--output_dir', type=str, default='phoneme_embeds_cgn_K+O')
    parser.add_argument('--n_phonemes', type=int, default=50)
    parser.add_argument('--n_speakers', type=int, default=10) #currently not using this argument
    parser.add_argument('--max_phonemes_total', type=int, default=100)
    args = parser.parse_args()
    main(args)

