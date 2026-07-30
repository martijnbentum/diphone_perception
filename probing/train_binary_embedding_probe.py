from pathlib import Path

import echoframe
import numpy as np

from probing import probe_utils
from probing.extract_embeddings import default_model_name, default_store_root

default_probe_save_dir = probe_utils.default_probe_save_dir
default_results_dir = probe_utils.default_results_dir


def _embedding_echoframe_keys(
    store, selected, model_name, layer, collar,
):
    return [
        store.make_echoframe_key(
            'hidden_state', model_name=model_name,
            phraser_key=phraser_phone.key, layer=layer, collar=collar)
        for _, phraser_phone, _ in selected
    ]


def _load_middle_frame_vectors(store, selected, model_name, layer, collar):
    '''Load embeddings and reduce each stored phone to its middle frame.'''
    phraser_keys = [
        phraser_phone.key for _, phraser_phone, _ in selected]
    embeddings = store.phraser_keys_to_embeddings(
        phraser_keys, model_name, layer, collar=collar)
    by_key = {
        embedding.phraser_key: embedding
        for embedding in embeddings.embeddings
    }

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


def _run_directory(
    root, model_name, target_phoneme, layer, collar, run_id,
):
    return (
        Path(root) / model_name / target_phoneme / f'layer{layer:02d}'
        / f'collar{collar}ms' / run_id
    )


def train_binary_embedding_probe(
    phones,
    target_phoneme,
    store=None,
    store_root=default_store_root,
    model_name=default_model_name,
    layer=9,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    verbose=True,
):
    '''Train a binary target-vs-other probe on middle-frame embeddings.

    `standardize=False` preserves the legacy raw-feature probe. When True,
    StandardScaler is fitted independently inside every cross-validation
    training fold through an sklearn Pipeline.
    '''
    probe_utils.validate_target_phoneme(target_phoneme)
    probe_utils.validate_probe_arguments(n_splits, standardize)
    probe_utils.validate_unique_phraser_keys(phones)
    selected = probe_utils.select_phones(
        phones, target_phoneme, n_embeds, seed=random_state)

    if store is None:
        store = echoframe.Store(str(store_root))
    echoframe_keys = _embedding_echoframe_keys(
        store, selected, model_name, layer, collar)
    feature_parameters = {
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
        'frame': 'middle',
    }
    manifest = probe_utils.build_probe_run_manifest(
        store, selected, echoframe_keys, 'embedding', feature_parameters,
        target_phoneme, n_embeds, n_splits, random_state, standardize)
    run_id = probe_utils.hash_run_manifest(manifest)
    probe_run_directory = _run_directory(
        probe_save_dir, model_name, target_phoneme, layer, collar, run_id)
    predictions_run_directory = _run_directory(
        results_dir, model_name, target_phoneme, layer, collar, run_id)

    def load_vectors():
        return _load_middle_frame_vectors(
            store, selected, model_name, layer, collar)

    result_fields = {
        'representation': 'embedding',
        'target_phoneme': target_phoneme,
        'model_name': model_name,
        'layer': layer,
        'collar': collar,
    }
    return probe_utils.run_binary_probe(
        load_vectors=load_vectors,
        manifest=manifest,
        probe_run_directory=probe_run_directory,
        predictions_run_directory=predictions_run_directory,
        result_fields=result_fields,
        display_name=f'{target_phoneme} layer {layer}',
        n_splits=n_splits,
        random_state=random_state,
        standardize=standardize,
        save_probes=save_probes,
        save_predictions=save_predictions,
        overwrite=overwrite,
        verbose=verbose,
    )


def train_binary_embedding_probes(
    phones,
    target_phonemes=None,
    store=None,
    store_root=default_store_root,
    model_name=default_model_name,
    layer=9,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    verbose=True,
):
    '''Train one binary embedding probe run for each target phoneme.

    When target_phonemes is None, all labels in
    phones.label_to_phraser_phone are used. The Phraser label inventory
    must contain exactly the same number of items for every label.
    '''
    probe_utils.validate_probe_arguments(n_splits, standardize)
    targets = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes, n_samples=n_embeds)
    probe_utils.validate_unique_phraser_keys(phones)

    owns_store = store is None
    if store is None:
        store = echoframe.Store(str(store_root))

    def train_one(target_phoneme):
        return train_binary_embedding_probe(
            phones,
            target_phoneme,
            store=store,
            model_name=model_name,
            layer=layer,
            collar=collar,
            n_embeds=n_embeds,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            save_probes=save_probes,
            probe_save_dir=probe_save_dir,
            save_predictions=save_predictions,
            results_dir=results_dir,
            overwrite=overwrite,
            verbose=verbose,
        )

    try:
        return probe_utils.run_probe_sweep(
            targets, train_one, 'embedding', verbose=verbose)
    finally:
        if owns_store:
            store.close()
