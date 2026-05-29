import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, Counter
import argparse
import os
import re
import joblib
import logging
import h5py
from tqdm import tqdm
import random


def load_embeddings(h5_file_path, model_types, args):
    """
    Load phoneme embeddings with deterministic shuffling.

    Works with:
      - MFCC + transformers
      - transformers only

    Assumes HDF5 keys are already aligned across models.

    Returns:
        {model_type: (X, y)}
    """
    logging.info("Starting memory-safe aligned loading...")

    target_layer_str = args.target_layer
    target_phoneme = args.target_phoneme
    n_embeds = args.n_embeds

    has_mfcc = "mfcc" in model_types
    transformer_models = [m for m in model_types if m != "mfcc"]

    X_dict = {m: [] for m in model_types}
    y = []

    with h5py.File(h5_file_path, "r") as h5f:

        # -----------------------------
        # Choose reference model
        # -----------------------------
        if has_mfcc:
            reference_model = "mfcc"
        else:
            reference_model = model_types[0]

        logging.info(f"Using '{reference_model}' as reference model for key alignment")

        speakers = list(h5f[reference_model].keys())
        logging.info(f"Found {len(speakers)} speakers in {reference_model}")

        # -----------------------------
        # Collect phoneme set
        # -----------------------------
        phoneme_set = set()
        for speaker in speakers:
            phoneme_set.update(h5f[reference_model][speaker].keys())
        phoneme_list = sorted(phoneme_set)
        logging.info(f"Phonemes found: {phoneme_list}")

        if target_phoneme not in phoneme_list:
            raise ValueError(f"Target phoneme {target_phoneme} not found!")

        other_phonemes = [p for p in phoneme_list if p != target_phoneme]
        n_per_other = n_embeds // len(other_phonemes)
        remainder = n_embeds % len(other_phonemes)

        if remainder != 0:
            print(
                f"Using {n_per_other} embeddings per other phoneme. "
                f"Warning: {remainder} embeddings will be unused due to floor division."
            )

        total_needed = {target_phoneme: n_embeds}
        for p in other_phonemes:
            total_needed[p] = n_per_other

        logging.info(f"Target phoneme '{target_phoneme}': {n_embeds}, each other: {n_per_other}")

        # -----------------------------
        # Helper: get shared keys across MFCC and all transformer layers
        # -----------------------------
        def get_shared_keys(speaker, phoneme):
            """
            Returns the set of keys available in MFCC (if present) AND
            in every transformer model at target_layer_str.
            Returns an empty set if any model/layer is missing entirely.
            """
            shared = None

            if has_mfcc:
                if phoneme not in h5f["mfcc"][speaker]:
                    return set()
                shared = set(h5f["mfcc"][speaker][phoneme].keys())

            for model in transformer_models:
                if phoneme not in h5f[model][speaker]:
                    return set()
                phoneme_grp = h5f[model][speaker][phoneme]
                if target_layer_str not in phoneme_grp:
                    return set()
                model_keys = set(phoneme_grp[target_layer_str].keys())
                shared = model_keys if shared is None else shared & model_keys

            return shared if shared is not None else set()

        # -----------------------------
        # Collect all (speaker, phoneme, key) triples
        # -----------------------------
        logging.info("Collecting all available keys across speakers and phonemes...")
        all_keys = []

        if args.use_subset:
            max_per_phoneme = 5

            for speaker in speakers[:5]:  # first 5 speakers
                for phoneme in total_needed.keys():
                    shared_keys = get_shared_keys(speaker, phoneme)
                    if not shared_keys:
                        continue
                    sampled_keys = random.sample(sorted(shared_keys), min(len(shared_keys), max_per_phoneme))
                    all_keys.extend((speaker, phoneme, k) for k in sampled_keys)

            logging.info(f"Debug subset: {len(all_keys)} keys collected")

        else:
            for speaker in speakers:
                for phoneme in total_needed.keys():
                    shared_keys = get_shared_keys(speaker, phoneme)
                    if not shared_keys:
                        continue
                    for key in shared_keys:
                        all_keys.append((speaker, phoneme, key))

            logging.info(f"Total available keys across all speakers/phonemes: {len(all_keys)}")

        # Deterministic shuffle
        random.Random(42).shuffle(all_keys)

        counts = defaultdict(int)
        processed = 0
        X_dict = {m: [] for m in model_types}
        y = []

        # -----------------------------
        # Sample until quotas are filled
        # -----------------------------
        for speaker, phoneme, key in all_keys:
            if counts[phoneme] >= total_needed[phoneme]:
                continue

            # Load MFCC if present
            if has_mfcc:
                data_mfcc = h5f["mfcc"][speaker][phoneme][key][()]

            # Load transformer layers
            layer_data = {}
            for model in transformer_models:
                layer_data[model] = h5f[model][speaker][phoneme][target_layer_str][key][()]

            if processed == 0:
                if has_mfcc:
                    logging.info("mfcc shape: " + str(data_mfcc.shape))
                for model in layer_data:
                    logging.info(f"{model} shape: " + str(layer_data[model].shape))

            # Decide mid index from an available tensor
            if has_mfcc:
                base = data_mfcc
            else:
                base = layer_data[transformer_models[0]]

            mid_idx = base.shape[0] // 2

            # Append middle frame for each model
            for model in model_types:
                if model == "mfcc":
                    data = data_mfcc
                else:
                    data = layer_data[model]

                frame = data[mid_idx]
                X_dict[model].append(frame)

            y.append(phoneme)
            counts[phoneme] += 1
            processed += 1

            if processed % 500 == 0:
                logging.info(f"Collected {processed} samples so far...")

            if all(counts[p] >= total_needed[p] for p in total_needed):
                logging.info("All required samples collected. Stopping early.")
                break

    logging.info(f"Final counts per phoneme: {counts}")
    logging.info(f"Total samples collected: {processed}")

    return {m: (np.array(X_dict[m]), np.array(y)) for m in model_types}


# ============================================================
# Main
# ============================================================

def main(args):

    logging.basicConfig(
        filename=args.logging_filename,
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

    target_layer_str = args.target_layer
    print("Target layer:", target_layer_str)

    embeds_path = f'{args.embeds_dir}/phone_embeds.h5'

    # Load model types
    with h5py.File(embeds_path, "r") as h5f:
        model_types = list(h5f.keys())

    print(f"Models found in HDF5: {model_types}")

    # ---------------------------
    # Load aligned embeddings
    # ---------------------------
    data = load_embeddings(embeds_path, model_types, args)

    for model in model_types:

        X, y = data[model]
        print(f"{model} dataset size:", len(X))
        print(f"{model} distribution:", Counter(y))

        # ---------------------------
        # Stratified 5-Fold CV
        # ---------------------------
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test_full = y[train_idx], y[test_idx]

            y_train_binary = np.where(y_train_full == args.target_phoneme, "target", "other")
            y_test_binary = np.where(y_test_full == args.target_phoneme, "target", "other")

            probe = LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000)
            probe.fit(X_train, y_train_binary)

            y_pred = probe.predict(X_test)
            acc = accuracy_score(y_test_binary, y_pred)
            accuracies.append(acc)

            print(f"{model} Fold {fold_idx+1} Accuracy: {acc:.4f}")

            # Save probe
            if args.save_probes:
                save_dir = f"{args.probe_save_dir}/{model}/{args.target_phoneme}"
                os.makedirs(save_dir, exist_ok=True)
                if model == 'mfcc':
                    probe_filename = f"{save_dir}/probe_fold_{fold_idx+1:02d}.joblib"
                else:
                    probe_filename = f"{save_dir}/probe_{target_layer_str}_fold_{fold_idx+1:02d}.joblib"
                joblib.dump(probe, probe_filename)

            # Save predictions
            results_dir = f"{args.results_dir}/{model}/{args.target_phoneme}"
            os.makedirs(results_dir, exist_ok=True)
            pred_file = os.path.join(results_dir, f'{model}_{args.target_phoneme}_fold_{fold_idx+1}_predictions.txt')
            with open(pred_file, 'w') as f:
                f.write("true_phoneme\tbinary_true\tbinary_pred\tcorrect\n")
                for full_label, true_bin, pred_bin in zip(y_test_full, y_test_binary, y_pred):
                    correct = 1 if true_bin == pred_bin else 0
                    f.write(f"{full_label}\t{true_bin}\t{pred_bin}\t{correct}\n")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{model} 5-Fold Accuracy: Mean={mean_acc:.4f}, Std={std_acc:.4f}")
        logging.info(f"{model} Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_subset', action='store_true', help="Use a small subset of data for quick debugging")
    parser.add_argument('--target_phoneme', type=str, required=True)
    parser.add_argument('--embeds_dir', type=str, default='embeds')
    parser.add_argument('--probe_save_dir', type=str, default='phoneme_probes')
    parser.add_argument('--save_probes', action='store_true')
    parser.add_argument('--results_dir', type=str, default='results/probe_performance')
    parser.add_argument('--num_layers', type=int, default=13)
    parser.add_argument('--n_embeds', type=int, default=1000)
    parser.add_argument('--target_layer', type=str, default='layer09')
    parser.add_argument('--logging_filename', type=str, default='train_probes_binary.log')
    args = parser.parse_args()
    main(args)
