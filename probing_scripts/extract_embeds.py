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


def apply_model(wav_file, model, processor):

    audio, sr = librosa.load(wav_file, sr=16000)
    input_values = processor(audio,
                             sampling_rate=sr,
                             return_tensors="pt",
                             padding="longest").input_values

    with torch.no_grad():
        output_dict = model(input_values, output_hidden_states=True)

    return output_dict.hidden_states


def get_phoneme_embeds(metadata, model, processor, target_phonemes, args):

    # Remove rows with parsing errors
    metadata = metadata[metadata['phoneme'].apply(lambda x: len(x) <= 4)]

    print('n speakers:', len(metadata['speaker_id'].unique()))

    all_embeddings = {}

    with zipfile.ZipFile(args.data_dir, 'r') as zipf:
        for s in metadata['speaker_id'].unique():

            if s.startswith('N'):

                # get df for current speaker
                speaker_df = metadata[metadata['speaker_id'] == s]

                # to track how many phoneme samples we have collected
                phoneme_counts = {p: 0 for p in target_phonemes}

                # initialize variables
                hidden_states = None
                prev_filename = None
                current_filename = speaker_df['audio_filename'].iloc[0]

                print(f'Extracting phoneme embeds for speaker {s}...')

                for index, row in speaker_df.iterrows():
                    if row['phoneme'] not in target_phonemes:
                        continue
                    if phoneme_counts[row['phoneme']] >= args.n_phonemes:
                        continue # continue to the next phoneme

                    current_filename = row['audio_filename']

                    if prev_filename != current_filename:

                        if f'split_files/{current_filename}' in zipf.namelist():
                            with zipf.open(f'split_files/{current_filename}') as file_data:
                                file_stream = io.BytesIO(file_data.read())
                                # print(f'Applying model to {current_filename}')
                                hidden_states = apply_model(file_stream, model, processor)
                                prev_filename = current_filename
                        else:
                            print(f'Error processing {current_filename}')
                            continue

                    # Get time indices
                    start_idx = math.floor(row['start_time'] * args.encoder_sampling_rate)
                    end_idx = math.ceil(row['end_time'] * args.encoder_sampling_rate)

                    for layer_idx, layer_states in enumerate(hidden_states):
                        phoneme_seq = layer_states[:, start_idx:end_idx, :]
                        phoneme_seq_np = phoneme_seq.squeeze(0).detach().numpy()  # remove batch dim

                        # Create a unique key
                        key = f'{s}/{row["phoneme"]}/layer{layer_idx:02d}/{phoneme_counts[row["phoneme"]]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{current_filename.strip(".wav")}'
                        all_embeddings[key] = phoneme_seq_np

                    phoneme_counts[row['phoneme']] += 1

                    if all([count >= args.n_phonemes for count in phoneme_counts.values()]):
                        print(f"Extracted {args.n_phonemes} examples for all target phonemes.")
                        continue  # Continue to the next speaker

    # Save all embeddings in one compressed file
    output_file = os.path.join(args.output_dir, 'phoneme_embeddings.npz')
    np.savez_compressed(output_file, **all_embeddings)
    print(f'All embeddings saved to: {output_file}')
 

def main(args):

    # load model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained(args.model_path)
    model.eval()

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load metadata with timestamps
    metadata = pd.read_csv(args.metadata_file, sep='\t', header=0)

    target_phonemes = ['p', 'b', 't', 'd', 'k', 'f', 'v', 's', 'z', 'G', 'x', 'm', 'n', 'N', 'l', 'r', 'j', 'w', 'h',
                       'I', 'E', 'A', 'O', 'Y', 'i', 'e', 'a', 'o', 'u', 'y', '@', 'E+', 'Y+', 'A+'] # 'S', 'Z', 'g', '2'
    
    print('Number of phoneme classes:', len(target_phonemes))

    # get phoneme embeddings
    get_phoneme_embeds(metadata, model, processor, list(set(target_phonemes)), args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/checkpoint_229_100000')
    parser.add_argument('--encoder_sampling_rate', default=49) #w2v2 encoder uses 49 Hz
    parser.add_argument('--metadata_file', default='metadata/news_phonemes_zs.tsv')
    parser.add_argument('--data_dir', default='split_files.zip')
    parser.add_argument('--output_dir', default='phoneme_embeds_cgn')
    parser.add_argument('--n_phonemes', default=50)
    parser.add_argument('--n_speakers', default=10) #currently not using this argument
    args = parser.parse_args()
    main(args)
