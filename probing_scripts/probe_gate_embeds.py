import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import math
import argparse
import torch
import pandas as pd
import torchaudio
import numpy as np
import json
import joblib
import random
import frame
from collections import defaultdict


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


def get_stimulus_embeds(audio_filename, start_timestamp, model, processor, num_layers, data_dir, device):
    '''
    Get target embedding of a stimulus audio file
    (first fully overlapping frame of the phoneme we are interested in - either phoneme 1 or phoneme 2)
    '''
    stimulus_embeds_per_layer = {
        layer_idx: None
        for layer_idx in range(num_layers)
    }

    hidden_states = apply_model(f'{data_dir}/{audio_filename}', model, processor, device)

    # time to frame indices
    f = frame.Frames(n_frames=hidden_states[0].shape[1])
    start = f.start_frame(start=start_timestamp, percentage_overlap=100)

    if start.index != None:
        print(f'start frame for {audio_filename} time {start_timestamp}: {start.index}')

        for i in range(num_layers):
            frame_seq = hidden_states[i].squeeze(0)
            # get target embedding (first fully overlapping frame of the target phoneme)
            target_embed = frame_seq[start.index]
            # save it in the dict
            stimulus_embeds_per_layer[i] = target_embed.detach().cpu().numpy()
    
    else:
        print(f'Error: start frame for {audio_filename} time {start_timestamp}: {start.index}')

    return stimulus_embeds_per_layer


def main(args):

    # load processor
    if args.model_type == 'finetuned':
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    else:
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    
    # load wav2vec2 model
    model = Wav2Vec2Model.from_pretrained(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f'Moved model to {device}')

    # load probing models
    layer_probes = {
        layer_idx: joblib.load(os.path.join(f'phoneme_probes/{args.results_dir}', f"probe_{args.probe_id}_layer_{layer_idx}.joblib"))
        for layer_idx in range(args.num_layers)
    }

    target_phonemes = layer_probes[0].classes_

    # load metadata
    with open(args.metadata_file) as f:
        d = json.load(f)

    # shuffle items
    items = list(d.items())
    random.seed(42)
    random.shuffle(items)
    shuffled_d = dict(items)

    # store model responses
    model_responses_dict = {}

    for i, (label, info) in enumerate(shuffled_d.items()):

        if info[args.diphone_target] in target_phonemes:
            audio_file_path = info[f'gate_{args.gate}_audio_filename'] if args.gate < 6 else info["original_audio_filename"]
            start_timestamp = info[f'{args.diphone_target}_start_time']

            # get embeddings for the stimulus
            embedding_dict = get_stimulus_embeds(audio_file_path, 
                                                start_timestamp,
                                                model, processor, args.num_layers, 
                                                args.data_dir, device)
            
            for layer_idx, embed in embedding_dict.items():
                if layer_idx == args.best_layer:
                    embed = embed.reshape(1, -1)
                    pred = layer_probes[layer_idx].predict(embed)

                    # create copy of metadata info
                    if label not in model_responses_dict:
                        model_responses_dict[label] = info.copy()
                        model_responses_dict[label]["responses"] = []

                    # add model response
                    model_responses_dict[label]["responses"].append(
                        {
                        "gt_phoneme1": info["phoneme_1"],
                        "gt_phoneme2": info["phoneme_2"],
                        f"response_{args.diphone_target}": pred.item(),
                        "participant": args.model_type,
                        "gate": args.gate
                        }
                    )


    output_path = f'results/{args.results_dir}/model_responses_gate_{args.gate}_{args.diphone_target}_layer_{args.best_layer}_{args.model_type}.json'

    with open(output_path, 'w') as f:
        json.dump(model_responses_dict, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='gate_1+4_K+O_pretrained')
    parser.add_argument('--probe_id', type=str, default='A')
    parser.add_argument('--diphone_target', type=str, default='phoneme_1')
    parser.add_argument('--model_path', type=str, default='models/checkpoint_229_100000')
    parser.add_argument('--num_layers', type=int, default=13)
    parser.add_argument('--metadata_file', type=str, default='metadata/info.json')
    parser.add_argument('--data_dir', type=str, default='stimuli')
    parser.add_argument('--best_layer', type=int, default=9)
    parser.add_argument('--model_type', type=str, default='pretrained')
    args = parser.parse_args()
    main(args)
