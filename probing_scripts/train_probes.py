import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
import argparse
import os
import random
import matplotlib.pyplot as plt
import joblib
import csv
import re


def load_embeds(npz_file_path, num_layers, args):
    '''
    Return dictionary with layers as keys, and dictionaries with phoneme: [embed_list] as values.
    '''
    layer_embeddings = {i: defaultdict(list) for i in range(num_layers)}

    # Load all embeddings
    data = np.load(npz_file_path)

    for key in data.files:
        # Example key format: 's01/p/layer00/0_t_k_news_00123'
        parts = key.split('/')
        _, phoneme, layer_str, _ = parts
        layer_idx = int(re.search(r'\d+', layer_str).group())  # e.g., 'layer00' -> 0

        # Stop saving embeddings if we've reached a specified amount
        if len(layer_embeddings[layer_idx][phoneme]) >= int(args.max_n_embeds):
            continue

        frame_embed = data[key]
        if frame_embed.ndim == 3:
            frame_embed = frame_embed[0]  # remove batch dimension

        num_time_steps = frame_embed.shape[0]

        # Apply gating
        if args.gate == 1 or args.gate == 4:
            layer_embeddings[layer_idx][phoneme].append(frame_embed[1])  # only save second frame (strict "first" frame of the phoneme)
        elif args.gate == 3:
            layer_embeddings[layer_idx][phoneme].append(frame_embed[-2])  # only save second-to-last frame (strict "last" frame of the phoneme)
        else:
            for t in range(num_time_steps):
                layer_embeddings[layer_idx][phoneme].append(frame_embed[t]) # save all frames

    return layer_embeddings



def downsample_train_embeddings(train_embeddings, n_embeds, seed=42):
    '''
    Limit training set to n_embeds per class
    '''
    random.seed(seed)
    sampled_train = {layer: defaultdict(list) for layer in train_embeddings}
    for layer in train_embeddings:
        for phoneme, embeds in train_embeddings[layer].items():
            if len(embeds) > n_embeds:
                sampled = random.sample(embeds, n_embeds)
            else:
                sampled = embeds  # Use all available if not enough
            sampled_train[layer][phoneme] = sampled
    return sampled_train


def train_test_split_embeddings(layer_embeddings, test_ratio=0.2, seed=42):
    '''
    Given a dict of {layer: {phoneme: [embeds]}}, return train/test split
    '''
    random.seed(seed)
    train_embeddings = {layer: defaultdict(list) for layer in layer_embeddings}
    test_embeddings = {layer: defaultdict(list) for layer in layer_embeddings}

    # Use first layer to get consistent indices per phoneme
    for phoneme, embeddings in layer_embeddings[0].items():
        n = len(embeddings)
        indices = list(range(n))
        random.shuffle(indices)
        split = int(n * (1 - test_ratio))
        train_idx = indices[:split]
        test_idx = indices[split:]

        # Apply split to all layers
        for layer in layer_embeddings:
            all_embeds = layer_embeddings[layer][phoneme]
            train_embeddings[layer][phoneme] = [all_embeds[i] for i in train_idx]
            test_embeddings[layer][phoneme] = [all_embeds[i] for i in test_idx]

    return train_embeddings, test_embeddings


def prepare_data_for_layer(embeddings_dict, layer_idx):
    X = []
    y = []
    for phoneme, embeds in embeddings_dict[layer_idx].items():
        X.extend(embeds)
        y.extend([phoneme] * len(embeds))
    return np.array(X), np.array(y)


def main(args):

    # Load embeddings
    embeds = load_embeds(args.embeds_dir, num_layers=args.num_layers, args=args)

    for phoneme in embeds[0].keys():
        print(phoneme, len(embeds[0][phoneme]))

    # Train test split
    train_full, test_embeds = train_test_split_embeddings(embeds, test_ratio=0.2)

    # Filter out unwanted classes
    unwanted_phonemes = {'S', 'Z', 'g', '2'}
    for layer in train_full:
        for phoneme in unwanted_phonemes:
            train_full[layer].pop(phoneme, None)
            test_embeds[layer].pop(phoneme, None)
    
    # Downsample train embeds
    train_embeds = downsample_train_embeddings(train_full, int(args.n_embeds))

    # Define layer-wise probes
    layer_probes = {
        layer_idx: LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000)
        for layer_idx in range(args.num_layers)
    }

    # Save layer-wise accuracies for each phoneme
    accs = []
    preds = []

    # Create a directory to save the probes
    os.makedirs(args.probe_save_dir, exist_ok=True)

    # Train and test an individual probe for each layer
    for layer_idx in range(args.num_layers):

        train_X, train_y = prepare_data_for_layer(train_embeds, layer_idx)

        print(Counter(train_y))

        layer_probes[layer_idx].fit(train_X, train_y)
        test_X, test_y = prepare_data_for_layer(test_embeds, layer_idx)
        test_pred = layer_probes[layer_idx].predict(test_X)

        print(Counter(test_y))

        # Evaluate
        test_acc = accuracy_score(test_y, test_pred)
        print(f'Accuracy for layer {layer_idx}:', test_acc)
        accs.append(test_acc)
        preds.append(Counter(test_pred))

    # Save probes
    if args.save_probes:
        for layer_idx, probe in layer_probes.items():
            filename = os.path.join(args.probe_save_dir, f"probe_{args.probe_id}_layer_{layer_idx}.joblib")
            joblib.dump(probe, filename)

    # Save results (specifying the number of embeddings that were used for training)
    with open(args.results_file, 'a') as outfile:
        for layer_idx, acc in enumerate(accs):
            outfile.write(str(args.n_embeds) + '\t' + str(layer_idx) + '\t' + str(acc) + '\n')
    
    # Save the prediction distribution (how often was each class predicted?)
    file_exists = os.path.exists(args.class_distribution_file)
    with open(args.class_distribution_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)

        # Write header only if file did not exist
        if not file_exists:
            all_classes = sorted(preds[0].keys())
            header = ['n_embeds', 'layer_idx'] + all_classes
            writer.writerow(header)

        # Write rows
        for layer_idx, counter in enumerate(preds):
            all_classes = sorted(counter.keys())
            row = [str(args.n_embeds)] + [str(layer_idx)] + [counter.get(cls, 0) for cls in all_classes]
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', default=1)
    parser.add_argument('--probe_id', default='A') # probe A: gate 1 phoneme 1
    parser.add_argument('--embeds_dir', default='phoneme_embeds_cgn/phoneme_embeddings.npz')
    parser.add_argument('--probe_save_dir', default='phoneme_probes')
    parser.add_argument('--save_probes', default=False)
    parser.add_argument('--results_file', default='results/probing_results_varying_n_embeds_2.csv')
    parser.add_argument('--num_layers', default=13)
    parser.add_argument('--n_embeds', default=100) # number of embeddings to use for each phoneme class
    parser.add_argument('--max_n_embeds', default=100)
    parser.add_argument('--class_distribution_file', default='results/probing_class_distribution_2.csv')
    args = parser.parse_args()
    main(args)
