import random
import time
from collections import defaultdict

import echoframe
import numpy as np

import locations
from probing.extract_embeddings import default_model_name


def run_probe_sweep(target_phonemes, train_one, representation, verbose=True):
    '''Train one binary probe per target and report elapsed time and ETA.

    target_phonemes:  ordered labels to probe
    train_one:        callback that trains one target label
    representation:  name included in progress messages
    verbose:         whether to print progress
    '''
    targets = list(target_phonemes)
    started_at = time.monotonic()
    results = {}
    total = len(targets)
    for index, target in enumerate(targets, start=1):
        if verbose:
            print(f'[{representation} probes] {index}/{total} starting '
                f'{target!r}', flush=True)
        results[target] = train_one(target)
        elapsed = time.monotonic() - started_at
        eta = elapsed / index * (total - index)
        if verbose:
            elapsed_text = _format_duration(elapsed)
            eta_text = _format_duration(eta)
            print(f'[{representation} probes] {index}/{total} completed '
                f'{target!r}; elapsed {elapsed_text}; ETA {eta_text}',
                flush=True)
    return results


def select_phones(phones, target_phoneme, n_samples=None, seed=42):
    '''Select a balanced target-vs-other sample deterministically.

    phones:           phone inventory paired with Phraser phones
    target_phoneme:   label used as the positive class
    n_samples:        number of positive examples to select
    seed:             random seed used before sampling
    '''
    validate_target_phoneme(target_phoneme)
    by_label = defaultdict(list)
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for phone, phraser_phone in pairs:
        by_label[phone.phoneme_ipa].append((phone, phraser_phone))

    if target_phoneme not in by_label:
        message = f'target_phoneme {target_phoneme!r} not found among phones'
        raise ValueError(message)

    if n_samples is None:
        n_samples = len(by_label[target_phoneme])
    elif isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError('n_samples must be a positive integer or None')
    elif n_samples <= 0:
        raise ValueError('n_samples must be a positive integer or None')

    other_labels = [label for label in by_label if label != target_phoneme]
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, _ = divmod(n_samples, len(other_labels))
    if n_per_other == 0:
        other_count = len(other_labels)
        message = f'n_samples={n_samples} is too small to split across '
        message += f'{other_count} other phoneme classes'
        raise ValueError(message)

    rng = random.Random(seed)

    def _take(label, quota):
        pool = list(by_label[label])
        pool_count = len(pool)
        if pool_count < quota:
            message = f'phoneme {label!r} has only {pool_count} phones '
            message += f'available, need {quota}'
            raise ValueError(message)
        rng.shuffle(pool)
        return pool[:quota]

    selected = []
    target_phones = _take(target_phoneme, n_samples)
    for phone, phraser_phone in target_phones:
        selected.append((phone, phraser_phone, 'target'))
    for label in other_labels:
        other_phones = _take(label, n_per_other)
        for phone, phraser_phone in other_phones:
            selected.append((phone, phraser_phone, 'other'))

    rng.shuffle(selected)
    return selected


def prepare_balanced_probe_targets(phones, target_phonemes=None,
    n_samples=None):
    '''Validate the balanced label inventory and return requested targets.

    phones:            phone inventory paired with Phraser phones
    target_phonemes:   optional ordered subset of labels
    n_samples:         positive examples requested for each target
    '''
    grouped = phones.label_to_phraser_phone
    if not grouped: raise ValueError('phones.label_to_phraser_phone is empty')

    label_counts = {}
    for label, items in grouped.items():
        label_counts[label] = len(items)

    invalid_labels = []
    for label in label_counts:
        if not isinstance(label, str) or not label: invalid_labels.append(label)
    if invalid_labels:
        message = 'label_to_phraser_phone keys must be non-empty strings; '
        message += f'invalid labels: {invalid_labels}'
        raise TypeError(message)

    unique_counts = set(label_counts.values())
    if len(unique_counts) != 1:
        label_items = label_counts.items()
        sorted_items = sorted(label_items)
        count_descriptions = []
        for label, count in sorted_items:
            count_descriptions.append(f'{label!r}: {count}')
        counts = ', '.join(count_descriptions)
        message = 'label_to_phraser_phone is not balanced; every label must '
        message += f'have the same number of items. Counts: {counts}'
        raise ValueError(message)
    count_iterator = iter(unique_counts)
    items_per_label = next(count_iterator)
    if items_per_label == 0:
        raise ValueError('label_to_phraser_phone contains no items per label')

    if target_phonemes is None:
        targets = sorted(label_counts)
    else:
        if isinstance(target_phonemes, str):
            message = 'target_phonemes must be an iterable of phoneme strings'
            raise TypeError(message)
        targets = list(target_phonemes)
        for target in targets:
            validate_target_phoneme(target)
        unique_targets = set(targets)
        if len(unique_targets) != len(targets):
            raise ValueError('target_phonemes contains duplicate labels')
        known_labels = set(label_counts)
        missing = sorted(unique_targets - known_labels)
        if missing:
            message = 'target phonemes not found in label_to_phraser_phone: '
            message += f'{missing}'
            raise ValueError(message)
        if not targets: raise ValueError('target_phonemes must not be empty')

    requested = items_per_label if n_samples is None else n_samples
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError('n_samples must be a positive integer or None')
    if requested <= 0:
        raise ValueError('n_samples must be a positive integer or None')
    if requested > items_per_label:
        message = f'n_samples={requested} exceeds the balanced inventory of '
        message += f'{items_per_label} items per label'
        raise ValueError(message)
    other_label_count = len(label_counts) - 1
    if other_label_count == 0:
        raise ValueError('at least two phoneme labels are required')
    if requested // other_label_count == 0:
        message = f'n_samples={requested} is too small to split across '
        message += f'{other_label_count} other phoneme labels'
        raise ValueError(message)
    return targets


def inspect_feature_scale(phones, embedding_store=None, mfcc_store=None,
    embedding_store_root=locations.echoframe_store,
    mfcc_store_root=locations.echoframe_mfcc_store,
    model_name=default_model_name, layer=9, collar=2000, sample_size=1000,
    random_state=42, std_ratio_threshold=10.0, verbose=True):
    '''Inspect per-dimension scale on paired embedding and MFCC center frames.

    The same phone indices are sampled for both representations and only
    complete pairs are retained. The recommendation is heuristic.

    phones:               phone inventory paired with Phraser phones
    embedding_store:      optional open embedding store
    mfcc_store:           optional open MFCC store
    sample_size:          maximum number of paired phones to inspect
    std_ratio_threshold:  spread that triggers the scaling recommendation
    '''
    _validate_scale_arguments(sample_size, std_ratio_threshold)
    embedding_store_owned = embedding_store is None
    mfcc_store_owned = mfcc_store is None
    if embedding_store is None:
        embedding_root = str(embedding_store_root)
        embedding_store = echoframe.Store(embedding_root)
    if mfcc_store is None:
        mfcc_root = str(mfcc_store_root)
        mfcc_store = echoframe.Store(mfcc_root)

    try:
        segments = phones.phraser_phones
        if not segments: raise ValueError('phones.phraser_phones is empty')
        segment_count = len(segments)
        selected_count = min(sample_size, segment_count)
        rng = random.Random(random_state)
        population = range(segment_count)
        selected_indices = rng.sample(population, selected_count)
        selected = [segments[index] for index in selected_indices]

        embedding_keys = []
        mfcc_keys = []
        for segment in selected:
            embedding_key = embedding_store.make_echoframe_key(
                'hidden_state', model_name=model_name,
                phraser_key=segment.key, layer=layer, collar=collar)
            embedding_keys.append(embedding_key)
            mfcc_key = mfcc_store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=segment.key)
            mfcc_keys.append(mfcc_key)
        embedding_vectors = embedding_store.load_many_frames(embedding_keys,
            frame='center', keep_missing=True)
        mfcc_vectors = mfcc_store.load_many_frames(mfcc_keys, frame='center',
            keep_missing=True)

        paired_embeddings, paired_mfccs = [], []
        vector_pairs = zip(embedding_vectors, mfcc_vectors, strict=True)
        for embedding, mfcc in vector_pairs:
            if embedding is None or mfcc is None: continue
            embedding_array = np.asarray(embedding)
            mfcc_array = np.asarray(mfcc)
            paired_embeddings.append(embedding_array)
            paired_mfccs.append(mfcc_array)
        if len(paired_embeddings) < 2:
            message = 'fewer than two sampled phones have both embedding and '
            message += 'MFCC features; increase sample_size or complete '
            message += 'extraction'
            raise ValueError(message)

        embedding_matrix = np.stack(paired_embeddings)
        mfcc_matrix = np.stack(paired_mfccs)
        embedding_summary = _feature_scale_summary(embedding_matrix,
            std_ratio_threshold)
        mfcc_summary = _feature_scale_summary(mfcc_matrix,
            std_ratio_threshold)
        missing_embeddings = 0
        for vector in embedding_vectors:
            missing_embeddings += vector is None
        missing_mfccs = 0
        for vector in mfcc_vectors:
            missing_mfccs += vector is None
        report = {'n_requested': selected_count,
            'n_paired': len(paired_embeddings),
            'n_missing_embedding': missing_embeddings,
            'n_missing_mfcc': missing_mfccs, 'model_name': model_name,
            'layer': layer, 'collar': collar, 'frame': 'center',
            'std_ratio_threshold': std_ratio_threshold,
            'embedding': embedding_summary, 'mfcc': mfcc_summary}
        if verbose:
            paired_count = report['n_paired']
            requested_count = report['n_requested']
            print(f'paired feature-scale sample: {paired_count}/'
                f'{requested_count} requested phones')
            _print_feature_scale('embedding', embedding_summary,
                std_ratio_threshold)
            _print_feature_scale('mfcc', mfcc_summary, std_ratio_threshold)
        return report
    finally:
        if embedding_store_owned: embedding_store.close()
        if mfcc_store_owned: mfcc_store.close()


def validate_target_phoneme(target_phoneme):
    '''Validate a non-empty target phoneme label.'''
    message = 'target_phoneme must be a non-empty string'
    invalid = not isinstance(target_phoneme, str) or not target_phoneme
    if invalid: raise TypeError(message)


def validate_unique_phraser_keys(phones, example_count=5):
    '''Raise when two metadata phones refer to the same Phraser key.

    phones:         phone inventory paired with Phraser phones
    example_count:  maximum duplicate examples included in the error
    '''
    first_indices = {}
    duplicate_count = 0
    examples = []
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for index, (phone, phraser_phone) in enumerate(pairs):
        key = _hashable_key(phraser_phone.key)
        first_index = first_indices.get(key)
        if first_index is None:
            first_indices[key] = index
            continue
        duplicate_count += 1
        if len(examples) < example_count:
            stable_key = repr(key)
            phoneme = str(phone.phoneme_ipa)
            example = {'first_index': first_index, 'duplicate_index': index,
                'key': stable_key, 'phoneme': phoneme}
            examples.append(example)
    if duplicate_count:
        message = f'{duplicate_count} duplicate Phraser key occurrences '
        message += 'found; probe training requires unique keys. '
        message += f'Examples: {examples}'
        raise ValueError(message)


def _hashable_key(value):
    if isinstance(value, (bytearray, memoryview)): return bytes(value)
    return value


def _format_duration(seconds):
    rounded_seconds = round(seconds)
    seconds = max(0, rounded_seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _validate_scale_arguments(sample_size, std_ratio_threshold):
    message = 'sample_size must be a positive integer'
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError(message)
    if sample_size <= 0: raise ValueError(message)
    if std_ratio_threshold <= 1:
        raise ValueError('std_ratio_threshold must be greater than 1')


def _feature_scale_summary(matrix, std_ratio_threshold):
    matrix = np.asarray(matrix)
    message = 'feature-scale input must be a 2D matrix'
    if matrix.ndim != 2: raise ValueError(message)
    finite_values = np.isfinite(matrix)
    if not finite_values.all():
        raise ValueError('feature-scale input contains non-finite values')
    dimension_means = matrix.mean(axis=0)
    dimension_stds = matrix.std(axis=0, ddof=1)
    nonzero_stds = dimension_stds[dimension_stds > 0]
    if len(nonzero_stds) >= 2:
        maximum_std = nonzero_stds.max()
        minimum_std = nonzero_stds.min()
        std_ratio = float(maximum_std / minimum_std)
    else:
        std_ratio = 1.0

    quantiles = np.quantile(dimension_stds, [0, .25, .5, .75, 1])
    q_min = float(quantiles[0])
    q25 = float(quantiles[1])
    median = float(quantiles[2])
    q75 = float(quantiles[3])
    q_max = float(quantiles[4])
    std_quantiles = dict(min=q_min, q25=q25, median=median, q75=q75,
        max=q_max)
    zero_count = np.count_nonzero(dimension_stds == 0)
    zero_variance_dimensions = int(zero_count)
    absolute_means = np.abs(dimension_means)
    maximum_absolute_mean = np.max(absolute_means)
    max_abs_mean = float(maximum_absolute_mean)
    recommend_standardize = (
        len(nonzero_stds) >= 2 and std_ratio >= std_ratio_threshold)
    return {'n_items': matrix.shape[0],
        'n_dimensions': matrix.shape[1],
        'dimension_means': dimension_means, 'dimension_stds': dimension_stds,
        'std_quantiles': std_quantiles, 'std_ratio': std_ratio,
        'zero_variance_dimensions': zero_variance_dimensions,
        'max_abs_mean': max_abs_mean,
        'recommend_standardize': recommend_standardize}


def _print_feature_scale(name, summary, std_ratio_threshold):
    quantiles = summary['std_quantiles']
    if summary['recommend_standardize']:
        recommendation = 'consider standardizing'
    else:
        recommendation = 'scale spread is below the heuristic threshold'
    n_items = summary['n_items']
    n_dimensions = summary['n_dimensions']
    minimum = quantiles['min']
    median = quantiles['median']
    maximum = quantiles['max']
    std_ratio = summary['std_ratio']
    zero_variance = summary['zero_variance_dimensions']
    print(f'{name}: {n_items} items × {n_dimensions} dimensions')
    print(f'  dimension std min/median/max: {minimum:.6g} / '
        f'{median:.6g} / {maximum:.6g}')
    print(f'  nonzero std ratio: {std_ratio:.3g}; threshold: '
        f'{std_ratio_threshold:g} — {recommendation}')
    print(f'  zero-variance dimensions: {zero_variance}')
