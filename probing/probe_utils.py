import random
import time
from collections import defaultdict

_random_state = 42
_cnn_layer = 'cnn'


def validate_probe_layer(layer):
    '''Validate one hidden-state layer index or the CNN layer marker.'''
    if layer == _cnn_layer: return
    message = "layer must be a non-negative integer or 'cnn'"
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError(message)
    if layer < 0: raise ValueError(message)


def representation_for_layer(layer):
    '''Return the probe representation selected by one layer value.'''
    validate_probe_layer(layer)
    if layer == _cnn_layer: return _cnn_layer
    return 'embedding'


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


def select_phones(phones, target_phoneme):
    '''Select a balanced target-vs-other sample deterministically.

    Every available target-phoneme phone is used as the positive class;
    the "other" classes are sampled down to match that count.

    phones:          phone inventory paired with Phraser phones
    target_phoneme:  label used as the positive class
    '''
    validate_target_phoneme(target_phoneme)
    by_label = defaultdict(list)
    pairs = zip(phones.phones, phones.phraser_phones, strict=True)
    for phone, phraser_phone in pairs:
        by_label[phone.phoneme_ipa].append((phone, phraser_phone))

    if target_phoneme not in by_label:
        message = f'target_phoneme {target_phoneme!r} not found among phones'
        raise ValueError(message)

    n_samples = len(by_label[target_phoneme])

    other_labels = [label for label in by_label if label != target_phoneme]
    if not other_labels:
        raise ValueError('no other phoneme classes to sample as "other"')

    n_per_other, _ = divmod(n_samples, len(other_labels))
    if n_per_other == 0:
        other_count = len(other_labels)
        message = f'n_samples={n_samples} is too small to split across '
        message += f'{other_count} other phoneme classes'
        raise ValueError(message)

    rng = random.Random(_random_state)

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


def prepare_balanced_probe_targets(phones, target_phonemes=None):
    '''Validate the balanced label inventory and return requested targets.

    phones:            phone inventory paired with Phraser phones
    target_phonemes:   optional ordered subset of labels
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

    other_label_count = len(label_counts) - 1
    if other_label_count == 0:
        raise ValueError('at least two phoneme labels are required')
    if items_per_label // other_label_count == 0:
        message = f'n_samples={items_per_label} is too small to split across '
        message += f'{other_label_count} other phoneme labels'
        raise ValueError(message)
    return targets


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
