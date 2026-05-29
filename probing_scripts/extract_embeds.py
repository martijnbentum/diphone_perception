import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
import argparse
import torch
import pandas as pd
import numpy as np
import zipfile
import frame
from src.phoneme_mapper import Mapper
from sklearn.utils import shuffle
import logging
import tempfile
import shutil
import librosa
import h5py
import random


# ------------------------
# Seed utility
# ------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_error_log(rows, filename):
    if rows:
        pd.DataFrame(rows).to_csv(filename, index=False)


def extract_wav2vec2(audio, sr, model, processor, device):
    input_values = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding="longest"
    ).input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    return {
        "cnn": outputs.extract_features,        # (B, T, 512)
        "transformer": outputs.hidden_states   # tuple of 13 tensors
    }


def extract_mfcc(audio, sr, n_mfcc=39):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=320
    )
    return mfcc.T


def extract_subset(zip_path, target_dir, needed_files):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        all_files = zipf.namelist()

        for file in all_files:
            filename = os.path.basename(file)

            if filename in needed_files:
                zipf.extract(file, target_dir)


# ================================
# MAIN EXTRACTION FUNCTION
# ================================


def get_phoneme_embeds(metadata, processor, extractors, target_phonemes, device, args):

    # Remove rows with parsing errors & shuffle metadata
    metadata = metadata[metadata['phoneme'].apply(lambda x: len(x) <= 4)]
    metadata = shuffle(metadata, random_state=42)

    # # load duration threshold information (we use Q3 of each phoneme's duration distribution as the max. value)
    # duration_thresholds = pd.read_csv(args.phoneme_duration_threshold_file, sep=',', header=0)
    # duration_lookup = duration_thresholds.set_index('ipa_phoneme')['max_threshold'].to_dict() #TODO: change back to Q3 if necessary

    # Track how many phonemes we have collected globally (across all speakers)
    global_phoneme_counts = {p: 0 for p in target_phonemes}

    # Track various error cases (start frame bigger/equal/one smaller than end frame + too long duration)
    error_bigger, error_equal, error_one, error_duration, success, none = [], [], [], [], [], []

    # We will extract the CGN data to the TMP dir and clean up after we're done
    temp_data_dir = tempfile.mkdtemp()

    try:

        if args.use_subset:

            logging.info('Using subset mode...')

            subset = metadata.sample(n=100, random_state=42)  # pick 100 files
            needed_files = set(subset['audio_filename'])
            
            extract_subset(args.data_dir1, temp_data_dir, needed_files)
            extract_subset(args.data_dir2, temp_data_dir, needed_files)

            logging.info(f'Subset of files extracted to: {temp_data_dir}')
        
        else:

            logging.info('Using full dataset mode...')

            # unpack first zip file (comp k)
            with zipfile.ZipFile(args.data_dir1, 'r') as zipf1:
                zipf1.extractall(temp_data_dir)

            # unpack second zip file (comp o)
            with zipfile.ZipFile(args.data_dir2, 'r') as zipf2:
                zipf2.extractall(temp_data_dir)

            logging.info(f'All files extracted to: {temp_data_dir}')

        # Log number of files for component K and O
        for comp in ['k', 'o']:
            subset = metadata[metadata['comp'] == comp]
            num_files = subset['audio_filename'].nunique()
            logging.info(f'Number of unique audio files in subset metadata for component "{comp}": {num_files}')

        # Here we will store the phone embeddings in an HDF5 file with a hierarchical structure: 
        # model -> speaker -> phoneme -> layer [not included for MFCC] -> example_index
        output_file = os.path.join(
            f'{args.output_dir}',
            'phone_embeds.h5'
        )

        with h5py.File(output_file, "w") as h5f:

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

                    file_path = os.path.join(
                        temp_data_dir,
                        'data/cgn_sentences/split_files',
                        audio_filename
                    )

                    if not os.path.exists(file_path):
                        logging.warning(f"{file_path} not found")
                        continue

                    # ---- Load audio ONCE ----
                    audio, sr = librosa.load(file_path, sr=16000)

                    # ---- Extract features for all models ----
                    feature_outputs = {}

                    for name, extractor in extractors.items():

                        if extractor["type"] == "wav2vec2":
                            feature_outputs[name] = extract_wav2vec2(
                                audio, sr,
                                extractor["model"],
                                processor,
                                device
                            )
                            
                            # # Log number of frames per layer
                            # n_layers = len(feature_outputs[name])
                            # n_frames_per_layer = [layer.shape[1] for layer in feature_outputs[name]]  # shape: (1, frames, hidden)
                            # logging.info(f"{name}: {n_layers} layers, frames per layer = {n_frames_per_layer}")

                            # CNN number of frames
                            cnn_frames = feature_outputs[name]["cnn"].shape[1]
                            
                            # Transformer number of layers + number of frames
                            transformer_layers = feature_outputs[name]["transformer"]
                            n_layers = len(transformer_layers)
                            n_frames_per_layer = [layer.shape[1] for layer in transformer_layers]

                            logging.info(f"{name}: CNN frames = {cnn_frames}, Transformer layers = {n_layers}, frames per layer = {n_frames_per_layer}")

                        elif extractor["type"] == "mfcc":
                            feature_outputs[name] = extract_mfcc(audio, sr)
                            logging.info(f"{name}: frames = {feature_outputs[name].shape[0]}, features = {feature_outputs[name].shape[1]}")

                    # Process each row (phoneme) in the file
                    for _, row in file_df.iterrows():

                        phoneme = row['ipa_phoneme']

                        if global_phoneme_counts[phoneme] >= args.max_phonemes_total:
                            continue
                        if phoneme_counts[phoneme] >= args.n_phonemes:
                            continue

                        # # check if the phoneme duration is not too long
                        # q3_value = duration_lookup.get(phoneme)
                        # if row['duration'] > q3_value:
                        #     # log these cases
                        #     error_duration.append(row)
                        #     continue

                        model_results = {}   # store results temporarily
                        model_failed = False

                        for model_name, features in feature_outputs.items():

                            # --- Determine frame count per model ---
                            if model_name == "mfcc":
                                n_frames = features.shape[0] # shape: (frames, n_mfcc)
                            else:
                                # n_frames = features[0].shape[1] # layer 0, shape: (1, frames, hidden)

                                # CNN and transformer have same frame length
                                n_frames = features["cnn"].shape[1]

                            f = frame.Frames(n_frames=n_frames)

                            start = f.start_frame(start=row['start_time'], percentage_overlap=100)
                            end = f.end_frame(end=row['end_time'], percentage_overlap=100)

                            #logging.info(f"Processing {model_name} - phoneme '{phoneme}' from {row['start_time']}s to {row['end_time']}s: start frame = {start.index if start else None}, end frame = {end.index if end else None}")

                            if start is None or end is None:
                                none.append(row)
                                model_failed = True
                                #logging.info(f"Model failed because start or end frame is None for phoneme '{phoneme}' in file '{audio_filename}'")
                                break

                            if start.index > end.index:
                                error_bigger.append(row)
                                model_failed = True
                                #logging.info(f"Model failed because start frame is greater than end frame for phoneme '{phoneme}' in file '{audio_filename}'")
                                break

                            if start.index == end.index:
                                error_equal.append(row)
                                model_failed = True
                                #logging.info(f"Model failed because start frame equals end frame for phoneme '{phoneme}' in file '{audio_filename}'")
                                break
                            
                            ### UPDATE: single frame is okay
                            # if end.index - start.index == 1:
                            #     error_one.append(row)
                            #     model_failed = True
                            #     #logging.info(f"Model failed because only one frame is available for phoneme '{phoneme}' in file '{audio_filename}'")
                            #     break

                            # --- Extract phoneme slice ---
                            if model_name == "mfcc":
                                phoneme_seq = features[start.index:end.index, :]
                                model_results[model_name] = phoneme_seq

                            else:
                                # layer_slices = []
                                # for layer_tensor in features:
                                #     layer_np = layer_tensor.squeeze(0).detach().cpu().numpy()
                                #     phoneme_seq = layer_np[start.index:end.index, :]
                                #     layer_slices.append(phoneme_seq)

                                # model_results[model_name] = layer_slices
                                
                                model_results[model_name] = features
 
                        # If any model failed frame alignment → skip this phoneme
                        if model_failed:
                            continue

                        # If we reach here → SUCCESS
                        success.append(row)

                        # Save results to HDF5 with hierarchical keys
                        for model_name, result in model_results.items():
                            
                            if model_name == "mfcc":
                                
                                # model, speaker, phoneme, example_index + metadata for traceability
                                key = f'{model_name}/{s}/{phoneme}/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                                h5f.create_dataset(key, data=result, compression="gzip")

                            else:

                                # # we only save layer 9 
                                # layer_idx = 9
                                # phoneme_seq = result[layer_idx]
                                # # model, speaker, phoneme, layer, example_index + metadata for traceability
                                # key = f'{model_name}/{s}/{phoneme}/layer{layer_idx:02d}/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                                # h5f.create_dataset(key, data=phoneme_seq, compression="gzip")

                                for layer_spec in args.layers:

                                    if layer_spec == "cnn":
                                        cnn_tensor = result["cnn"]
                                        cnn_np = cnn_tensor.squeeze(0).detach().cpu().numpy()
                                        phoneme_seq = cnn_np[start.index:end.index, :]

                                        key = f'{model_name}/{s}/{phoneme}/cnn/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                                        h5f.create_dataset(key, data=phoneme_seq, compression="gzip")

                                    else:
                                        layer_idx = int(layer_spec)

                                        transformer_layers = result["transformer"]
                                        layer_tensor = transformer_layers[layer_idx]

                                        layer_np = layer_tensor.squeeze(0).detach().cpu().numpy()
                                        phoneme_seq = layer_np[start.index:end.index, :]

                                        key = f'{model_name}/{s}/{phoneme}/layer{layer_idx:02d}/{phoneme_counts[phoneme]}_{row["previous_phoneme"]}_{row["next_phoneme"]}_{audio_filename.strip(".wav")}'
                                        h5f.create_dataset(key, data=phoneme_seq, compression="gzip")

                        # Count how many examples we have for this phoneme
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
        
        logging.info(f'All embeddings saved to: {output_file}')

        # save csv logs with error & success cases
        os.makedirs(f'csv_logs', exist_ok=True)
        save_error_log(error_bigger, f'csv_logs/start_bigger_than_end_FEB2026_v4.csv')
        save_error_log(error_equal, f'csv_logs/start_equal_to_end_FEB2026_v4.csv')
        save_error_log(error_one, f'csv_logs/start_one_smaller_than_end_FEB2026_v4.csv')
        save_error_log(success, f'csv_logs/success_FEB2026_v4.csv')
        save_error_log(error_duration, f'csv_logs/too_long_FEB2026_v4.csv')
        save_error_log(none, f'csv_logs/none_error_FEB2026_v4.csv')

    finally:
        # clean up extracted files
        shutil.rmtree(temp_data_dir)


def main(args):

    logging.basicConfig(
        filename=f'extract_embeds_APRIL2026_v2.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Build extractors --------
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    extractors = {}

    # -------------------------
    # Optionally use PRETRAINED model
    # -------------------------
    if args.use_pretrained:
        pretrained_model = Wav2Vec2Model.from_pretrained(args.model_path_pretrained)
        pretrained_model.eval().to(device)
        logging.info(f'Using Pretrained model, moved to {device}')
        extractors["pretrained"] = {"type": "wav2vec2", "model": pretrained_model}

    # -------------------------
    # Optionally use CHECKPOINTS
    # -------------------------
    if args.use_checkpoints:
        for checkpoint in args.checkpoint_paths:

            # Load full Hugging Face model directly
            checkpoint_model = Wav2Vec2Model.from_pretrained(checkpoint)

            # Eval
            checkpoint_model.eval().to(device)

            logging.info(f"Loaded checkpoint {checkpoint} to {device}")
            
            if args.use_pretrained:
                # check if pretrained & checkpoint weights differ
                logging.info(f"Comparing pretrained model with checkpoint {checkpoint}...")

                weights_differ = any(
                    not torch.allclose(p1, p2)
                    for p1, p2 in zip(pretrained_model.parameters(), checkpoint_model.parameters())
                )
                logging.info(f"Weights differ: {weights_differ}")

                # check output difference
                input_values = torch.randn(1, 16000).to(device)

                with torch.no_grad():
                    out1 = pretrained_model(input_values).last_hidden_state
                    out2 = checkpoint_model(input_values).last_hidden_state

                diff = torch.mean(torch.abs(out1 - out2))
                logging.info(f"Output difference: {diff.item()}")

            name = os.path.basename(checkpoint)

            extractors[name] = {
                "type": "wav2vec2",
                "model": checkpoint_model
            }

            logging.info(
                f"Loaded checkpoint {name}, moved to {device}"
            )

    # -------------------------
    # Optionally use (multiple) UNTRAINED models
    # -------------------------
    if args.use_untrained:

        for i in range(args.num_untrained_models):

            seed = args.base_seed + i
            set_seed(seed)

            config = Wav2Vec2Config()
            untrained_model = Wav2Vec2Model(config)
            untrained_model.eval().to(device)

            name = f"untrained_{i}"

            extractors[name] = {
                "type": "wav2vec2",
                "model": untrained_model,
                "seed": seed
            }

            logging.info(
                f'Initialized {name} with seed {seed}, moved to {device}'
            )

    # -------------------------
    # Optionally use MFCC
    # -------------------------
    if args.use_mfcc:
        extractors["mfcc"] = {"type": "mfcc"}
        logging.info('Using MFCC')

    # -------------------------
    # Prepare output directory and filter metadata for target phonemes
    # -------------------------
    os.makedirs(f'{args.output_dir}', exist_ok=True)

    metadata = pd.read_csv(args.metadata_file, sep='\t', header=0)

    target_phonemes = [
        'p','b','t','d','k','f','v','s','z','G','x',
        'm','n','N','h','l','r','j','w',
        'I','E','A','O','i','e','a','o','u','@','E+','Y',
    ]

    mapper = Mapper(language='dutch')
    ipa_target_phonemes = [
        mapper.cgn_to_ipa[p]
        for p in target_phonemes
        if p in mapper.cgn_to_ipa
    ]

    missing = [
        p for p in target_phonemes
        if p not in mapper.cgn_to_ipa
    ]

    if missing:
        logging.warning(f"Missing phonemes in CGN->IPA mapping: {missing}")

    metadata['ipa_phoneme'] = metadata['phoneme'].map(mapper.cgn_to_ipa)
    metadata = metadata[metadata['ipa_phoneme'].isin(ipa_target_phonemes)]

    logging.info(str(ipa_target_phonemes))
    logging.info(str(metadata['ipa_phoneme'].dropna().unique()))

    # -------------------------
    # Embedding extraction
    # -------------------------
    get_phoneme_embeds(
        metadata,
        processor,
        extractors,
        list(set(ipa_target_phonemes)),
        device,
        args
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_subset', action='store_true', help='Whether to extract a small subset of the data for testing purposes')

    parser.add_argument('--model_path_pretrained', type=str, default='models/checkpoint_229_100000')
    parser.add_argument('--metadata_file', type=str, default='metadata/timestamps/news_books_phonemes_zs.tsv')
    parser.add_argument('--phoneme_duration_threshold_file', type=str, default='metadata/cgn_phoneme_duration_stats/phoneme_duration_thresholds.csv')
   
    parser.add_argument('--data_dir1', type=str, default='compressed_data/cgn_comp_k.zip')
    parser.add_argument('--data_dir2', type=str, default='compressed_data/cgn_comp_o.zip')
    parser.add_argument('--output_dir', type=str, default='phoneme_embeds_cgn_K+O')
    
    parser.add_argument('--n_phonemes', type=int, default=10) # per speaker
    parser.add_argument('--max_phonemes_total', type=int, default=1000)

    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--use_untrained', action='store_true')
    parser.add_argument('--use_mfcc', action='store_true')

    parser.add_argument(
        '--use_checkpoints',
        action='store_true',
        help='Whether to use additional wav2vec2-NL checkpoints (in addition to the pretrained model)'
    )

    parser.add_argument('--num_untrained_models', type=int, default=1)
    parser.add_argument('--base_seed', type=int, default=1234)

    parser.add_argument(
        '--checkpoint_paths',
        nargs='+',
        default=[],
        help='Paths to wav2vec2-NL model checkpoints'
    )

    parser.add_argument(
        '--layers',
        nargs='+',
        default=['9'],
        help="Layers to extract: 'cnn' or transformer layer indices (0–12)"
    )

    args = parser.parse_args()

    main(args)
