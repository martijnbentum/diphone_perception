import random
from collections import Counter, defaultdict
from pathlib import Path

import echoframe
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from probing.extract_embeddings import default_model_name, default_store_root
from probing.metadata import _data_dir

default_probe_save_dir = _data_dir / 'phone_probes'
default_results_dir = _data_dir / 'probe_results'


def _select_phones(phones, target_phoneme, n_embeds=None, seed=42):
    '''Deterministically sample up to n_embeds target-phoneme phones, plus an
    even split of n_embeds across every other phoneme class - mirrors the
    balanced-sampling scheme from the old train_probes_binary.py.

    n_embeds=None (the default) uses every available target-phoneme phone,
    rather than an arbitrary cap - pass a smaller number for a quick trial.

    Returns a list of (phone, phraser_phone, binary_label) tuples, where
    binary_label is 'target' or 'other'.
    '''
    by_label = defaultdict(list)
    for phone, phraser_phone in zip(phones.phones, phones.phraser_phones):
        by_label[phone.phoneme_ipa].append((phone, phraser_phone))

    if target_phoneme not in by_label:
        raise ValueError(
            f'target_phoneme {target_phoneme!r} not found among phones')

    if n_embeds is None:
        n_embeds = len(by_label[target_phoneme])

    other_labels = [label for label in by_label if label != target_phoneme]
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, remainder = divmod(n_embeds, len(other_labels))
    if n_per_other == 0:
        raise ValueError(
            f'n_embeds={n_embeds} is too small to split across '
            f'{len(other_labels)} other phoneme classes')

    rng = random.Random(seed)

    def _take(label, quota):
        pool = list(by_label[label])
        if len(pool) < quota:
            raise ValueError(
                f'phoneme {label!r} has only {len(pool)} phones available, '
                f'need {quota}')
        rng.shuffle(pool)
        return pool[:quota]

    selected = [(p, pp, 'target') for p, pp in _take(target_phoneme, n_embeds)]
    for label in other_labels:
        selected += [(p, pp, 'other') for p, pp in _take(label, n_per_other)]

    rng.shuffle(selected)
    return selected


def _load_middle_frame_vectors(store, selected, model_name, layer, collar):
    '''Batch-load stored embeddings for `selected` and reduce each to its
    middle frame. Phones missing from the store are dropped, not backfilled
    - their count is returned so callers can tell if a class came up short.
    '''
    phraser_keys = [phraser_phone.key for _, phraser_phone, _ in selected]
    embeddings = store.phraser_keys_to_embeddings(
        phraser_keys, model_name, layer, collar=collar)
    by_key = {e.phraser_key: e for e in embeddings.embeddings}

    X, y, true_labels, missing = [], [], [], []
    for phone, phraser_phone, binary_label in selected:
        embedding = by_key.get(phraser_phone.key)
        if embedding is None:
            missing.append(phone)
            continue
        X.append(embedding.middle_frame_segment(phraser_phone))
        y.append(binary_label)
        true_labels.append(phone.phoneme_ipa)
    return np.array(X), np.array(y), np.array(true_labels), missing


def train_binary_probe(
    phones,
    target_phoneme,
    store=None,
    store_root=default_store_root,
    model_name=default_model_name,
    layer=9,
    collar=500,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    save_probes=False,
    probe_save_dir=default_probe_save_dir,
    save_predictions=False,
    results_dir=default_results_dir,
    verbose=True,
):
    '''Train/test a binary (target-phoneme-vs-other) logistic regression
    probe on middle-frame wav2vec2 embeddings stored in echoframe, with
    5-fold StratifiedKFold(shuffle=True, random_state=random_state).

    phones:            probing.metadata.Phones (phraser_phones must be
                       complete)
    target_phoneme:    IPA label to classify against all other phones
    store:             existing echoframe.Store to read from; opened at
                       store_root if None
    model_name, layer, collar:  identify which stored embeddings to read -
                       must match what extract_phone_embeddings wrote
    n_embeds:          total target-class samples; each other phoneme class
                       gets n_embeds // (number of other phoneme classes).
                       None (default) uses every available target-phoneme
                       phone rather than an arbitrary cap.
    save_probes:       dump each fold's fitted probe under probe_save_dir
    save_predictions:  dump each fold's per-example predictions under
                       results_dir

    Returns a dict with per-fold accuracies, their mean/std, the fitted
    probes, and how many sampled phones had no stored embedding.
    '''
    if store is None:
        store = echoframe.Store(str(store_root))

    selected = _select_phones(
        phones, target_phoneme, n_embeds, seed=random_state)
    X, y, true_labels, missing = _load_middle_frame_vectors(
        store, selected, model_name, layer, collar)

    if verbose:
        print(f'{len(X)} embeddings loaded ({len(missing)} missing)')
        print(Counter(y))

    kf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies, probes, fold_predictions = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        probe = LogisticRegression(solver='liblinear', max_iter=1000)
        probe.fit(X[train_idx], y[train_idx])
        y_pred = probe.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)

        accuracies.append(acc)
        probes.append(probe)
        fold_predictions.append(
            list(zip(true_labels[test_idx], y[test_idx], y_pred)))

        if verbose:
            print(f'fold {fold_idx + 1}: accuracy={acc:.4f}')
        if save_probes:
            _save_probe(probe, probe_save_dir, model_name, target_phoneme,
                layer, fold_idx)
        if save_predictions:
            _save_predictions(fold_predictions[-1], results_dir, model_name,
                target_phoneme, layer, fold_idx)

    mean_acc, std_acc = float(np.mean(accuracies)), float(np.std(accuracies))
    if verbose:
        print(f'{target_phoneme} layer {layer}: '
            f'mean={mean_acc:.4f} std={std_acc:.4f}')

    return {
        'target_phoneme': target_phoneme,
        'layer': layer,
        'accuracies': accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'probes': probes,
        'n_samples': len(X),
        'n_missing': len(missing),
    }


def _save_probe(probe, probe_save_dir, model_name, target_phoneme, layer,
    fold_idx):
    probe_dir = Path(probe_save_dir) / model_name / target_phoneme
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / f'layer{layer:02d}_fold{fold_idx + 1:02d}.joblib'
    joblib.dump(probe, path)


def _save_predictions(predictions, results_dir, model_name, target_phoneme,
    layer, fold_idx):
    pred_dir = Path(results_dir) / model_name / target_phoneme
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f'layer{layer:02d}_fold{fold_idx + 1:02d}_predictions.txt'
    with open(path, 'w') as f:
        f.write('true_phoneme\tbinary_true\tbinary_pred\tcorrect\n')
        for true_full, true_bin, pred_bin in predictions:
            correct = int(true_bin == pred_bin)
            f.write(f'{true_full}\t{true_bin}\t{pred_bin}\t{correct}\n')
