'''Map phones and Echoframe embeddings to plain numpy arrays.

Loading every phone's embedding once produces arrays that are cheap to
pickle and share, instead of each per-phoneme probe re-loading its own
selected subset directly from the store.
'''

import random
from collections import Counter, namedtuple

import numpy as np

from probing import probe_utils

_random_state = 42

ProbeMatrix = namedtuple('ProbeMatrix',
    ['X', 'phone_labels', 'phraser_keys', 'missing'])


def build_probe_matrix(phones, store, model_name, layer, collar=2000,
    expected_target_count=13500):
    '''Load one middle-frame model feature per phone into aligned arrays.

    phones:                phone inventory paired with Phraser phones
    store:                 open Echoframe checkpoint store
    model_name:            embedding model identifier
    layer:                 hidden-state layer index or 'cnn'
    collar:                embedding context in milliseconds
    expected_target_count:  every phone label must have exactly this many
                            loaded tokens; raises otherwise

    Returns a ProbeMatrix whose X and phone_labels are aligned row for
    row. Phones with no stored feature are omitted from both and their
    Phraser keys are returned separately as missing.
    '''
    probe_utils.validate_probe_layer(layer)
    probe_utils.validate_unique_phraser_keys(phones)
    phraser_keys = [
        phraser_phone.key for phraser_phone in phones.phraser_phones]
    features = _load_model_features(store, phraser_keys, model_name, layer,
        collar)
    by_key = {}
    for feature in features:
        by_key[feature.phraser_key] = feature

    X, phone_labels, kept_keys, missing = [], [], [], []
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for phone, phraser_phone in pairs:
        feature = by_key.get(phraser_phone.key)
        if feature is None:
            missing.append(phraser_phone.key)
            continue
        X.append(feature.middle_frame_segment(phraser_phone))
        phone_labels.append(phone.phoneme_ipa)
        kept_keys.append(phraser_phone.key)

    matrix = ProbeMatrix(np.array(X), np.array(phone_labels), kept_keys,
        missing)
    _check_expected_target_count(phones, matrix.phone_labels,
        expected_target_count)
    return matrix


def _load_model_features(store, phraser_keys, model_name, layer, collar):
    if layer == 'cnn':
        features = store.phraser_keys_to_cnn_features(phraser_keys,
            model_name, collar=collar)
        return features.cnn_features
    embeddings = store.phraser_keys_to_embeddings(phraser_keys, model_name,
        layer, collar=collar)
    return embeddings.embeddings


def build_mfcc_probe_matrix(phones, store, frame='center',
    expected_target_count=13500):
    '''Load one MFCC frame reduction per phone into aligned arrays.

    phones:                phone inventory paired with Phraser phones
    store:                 open Echoframe MFCC store
    frame:                 MFCC frame reduction ('center', 'mean', 'first',
                           or 'last')
    expected_target_count:  every phone label must have exactly this many
                            loaded tokens; raises otherwise

    Returns a ProbeMatrix whose X and phone_labels are aligned row for
    row. Phones with no stored MFCC matrix are omitted from both and their
    Phraser keys are returned separately as missing.
    '''
    probe_utils.validate_unique_phraser_keys(phones)
    keys = [
        store.make_echoframe_key('acoustic_feature', feature_name='mfcc',
            phraser_key=phraser_phone.key)
        for phraser_phone in phones.phraser_phones]
    vectors = store.load_many_frames(keys, frame=frame, keep_missing=True)

    X, phone_labels, kept_keys, missing = [], [], [], []
    pairs = zip(phones.phones, phones.phraser_phones, vectors, strict=True)
    for phone, phraser_phone, vector in pairs:
        if vector is None:
            missing.append(phraser_phone.key)
            continue
        X.append(np.asarray(vector))
        phone_labels.append(phone.phoneme_ipa)
        kept_keys.append(phraser_phone.key)

    matrix = ProbeMatrix(np.array(X), np.array(phone_labels), kept_keys,
        missing)
    _check_expected_target_count(phones, matrix.phone_labels,
        expected_target_count)
    return matrix


def _check_expected_target_count(phones, phone_labels, expected_target_count):
    all_labels = {phone.phoneme_ipa for phone in phones.phones}
    counts = token_counts(phone_labels)
    mismatched = {label: counts.get(label, 0) for label in all_labels
        if counts.get(label, 0) != expected_target_count}
    if mismatched:
        message = f'expected {expected_target_count} tokens per label; '
        message += f'mismatched counts: {mismatched}'
        raise ValueError(message)


def token_counts(phone_labels):
    '''Count loaded tokens per phone label.

    phone_labels:  labels aligned with ProbeMatrix.X, e.g. from
                   build_probe_matrix

    Reflects every phone currently loaded, not the balanced
    target-vs-other subset a training run would actually select;
    see probe_utils.select_phones.
    '''
    return Counter(phone_labels)


def describe_probe_run(phone_labels, target_phoneme, representation,
    expected_token_count):
    '''Build a lightweight descriptive summary of one target phoneme.

    phone_labels:          labels aligned with ProbeMatrix.X, e.g. from
                           build_probe_matrix
    target_phoneme:        phoneme label treated as the positive class
    representation:        caller-supplied representation name, e.g.
                           'mfcc', 'embedding', or 'cnn'
    expected_token_count:  n_target must equal this; raises otherwise

    n_target and n_other count loaded tokens, not a balanced sample;
    see token_counts.
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    counts = token_counts(phone_labels)
    n_target = counts[target_phoneme]
    n_other = sum(counts.values()) - n_target
    if n_target != expected_token_count:
        message = f'expected {expected_token_count} tokens for '
        message += f'target_phoneme {target_phoneme!r}, found {n_target}'
        raise ValueError(message)
    return {'target_phoneme': target_phoneme, 'n_target': n_target,
        'n_other': n_other, 'representation': representation}


def select_balanced_vectors(matrix, target_phoneme):
    '''Build a balanced target-vs-other training set from a preloaded
    ProbeMatrix, with no store or phone-inventory access.

    Every row whose label matches target_phoneme is used as the positive
    class; every other label present in the matrix is randomly
    downsampled to an even split of that same total count.

    matrix:          probe_data.ProbeMatrix loaded via build_probe_matrix
    target_phoneme:  phoneme label used as the positive class

    Returns (X, y, true_labels, missing) matching probe_run.run()'s
    load_vectors contract. missing is always empty: selection only ever
    draws from rows already present in the matrix, so there's nothing
    left to be missing by the time this runs.
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    phone_labels = matrix.phone_labels
    target_mask = phone_labels == target_phoneme
    n_target = int(target_mask.sum())
    if n_target == 0:
        message = f'target_phoneme {target_phoneme!r} not found in matrix'
        raise ValueError(message)

    other_labels = sorted(set(phone_labels[~target_mask].tolist()))
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, _ = divmod(n_target, len(other_labels))
    if n_per_other == 0:
        message = f'n_target={n_target} is too small to split across '
        message += f'{len(other_labels)} other phoneme classes'
        raise ValueError(message)

    rng = random.Random(_random_state)
    selected = [(index, 'target')
        for index in np.flatnonzero(target_mask).tolist()]
    for label in other_labels:
        pool = np.flatnonzero(phone_labels == label).tolist()
        if len(pool) < n_per_other:
            message = f'phoneme {label!r} has only {len(pool)} phones '
            message += f'available, need {n_per_other}'
            raise ValueError(message)
        rng.shuffle(pool)
        for index in pool[:n_per_other]:
            selected.append((index, 'other'))

    rng.shuffle(selected)
    indices = [index for index, _ in selected]
    y = np.array([binary_label for _, binary_label in selected])
    return matrix.X[indices], y, phone_labels[indices], []
