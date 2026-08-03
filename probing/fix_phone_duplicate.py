import json
import random
from collections import Counter
from pathlib import Path

from probing.metadata import load_phraser_keys


_data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
duplicate_phone_counts_file = _data_dir / 'duplicate_phone_counts.json'
duplicate_replacement_phraser_phone_key_file = (
    _data_dir / 'duplicate_replacement_phraser_phone_keys.bin'
)

_phraser_key_len = 22
_phones_per_label = 13_500
_total_phone_count = 418_500
_components = {'comp-k', 'comp-o'}

# Inclusive bounds in milliseconds, copied from
# notes/phone_duration_stats.md. Only labels that need replacements are
# included; an unknown label in the counts file is rejected.
_duration_bounds = {
    'aː': (46, 675),
    'eː': (47, 445),
    'f': (46, 439),
    'iː': (50, 965),
    'j': (46, 563),
    'l': (46, 358),
    'm': (46, 592),
    'n': (46, 296),
    'oː': (50, 530),
    'r': (46, 644),
    's': (50, 699),
    'v': (46, 253),
    'x': (46, 373),
    'ɣ': (50, 344),
    'ɪ': (46, 466),
    'ʋ': (46, 274),
}


def _load_replacement_counts(path):
    with open(path) as f:
        counts = json.load(f)
    if not isinstance(counts, dict) or not counts:
        raise ValueError('replacement counts must be a non-empty object')

    for label, count in counts.items():
        if not isinstance(label, str) or not label:
            raise ValueError('replacement count labels must be non-empty strings')
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f'replacement count for {label!r} must be a positive integer'
            )
        if count % 2:
            raise ValueError(
                f'replacement count for {label!r} must be even, got {count}'
            )
        if label not in _duration_bounds:
            raise ValueError(f'no duration bounds configured for {label!r}')
    return counts


def _is_component_audio(audio):
    return bool(_components.intersection(Path(audio.filename).parts))


def filter_component_audios(audios):
    '''Return audios stored below the exact comp-k or comp-o component.'''
    return [audio for audio in audios if _is_component_audio(audio)]


def _valid_key(key, description):
    if not isinstance(key, bytes) or len(key) != _phraser_key_len:
        raise ValueError(
            f'{description} must be a {_phraser_key_len}-byte value'
        )


def _load_current_keys(phones):
    keys = load_phraser_keys(phones.phraser_key_path)
    for key in keys:
        _valid_key(key, 'current Phraser phone key')
    return keys


def _duplicate_occurrences(keys, metadata_phones):
    if len(keys) != len(metadata_phones):
        raise ValueError(
            f'key/metadata length mismatch: {len(keys)} keys and '
            f'{len(metadata_phones)} metadata phones'
        )

    seen = set()
    duplicates = []
    for index, (key, phone) in enumerate(zip(keys, metadata_phones)):
        if key in seen:
            duplicates.append((index, phone.phoneme_ipa))
        else:
            seen.add(key)
    return duplicates


def _check_duplicate_counts(duplicates, counts):
    expected = {label: count // 2 for label, count in counts.items()}
    found = Counter(label for _, label in duplicates)
    if found != Counter(expected):
        raise ValueError(
            f'duplicate occurrence counts do not match requested counts: '
            f'found {dict(found)}, expected {expected}'
        )
    return expected


def _phone_is_in_duration_range(phone):
    minimum, maximum = _duration_bounds[phone.label]
    duration = phone.end - phone.start
    return minimum <= duration <= maximum


def _collect_candidate_phones(phones, labels, current_keys):
    candidates = {label: [] for label in labels}
    candidate_keys = set()

    for audio in filter_component_audios(phones.store.audios):
        for phone in audio.phones:
            if phone.label not in labels:
                continue
            if not _phone_is_in_duration_range(phone):
                continue
            key = phone.key
            _valid_key(key, 'candidate Phraser phone key')
            if key in current_keys or key in candidate_keys:
                continue
            candidate_keys.add(key)
            candidates[phone.label].append(phone)
    return candidates


def _sample_candidate_keys(candidates, required_counts, seed):
    random.seed(seed)
    selected = {}
    for label, count in required_counts.items():
        available = candidates[label]
        if len(available) < count:
            raise ValueError(
                f'not enough replacement candidates for {label!r}: '
                f'need {count}, found {len(available)}'
            )
        selected[label] = [
            phone.key for phone in random.sample(available, count)
        ]
    return selected


def _arrange_replacement_keys(duplicates, selected):
    by_label = {label: iter(keys) for label, keys in selected.items()}
    return [next(by_label[label]) for _, label in duplicates]


def _validate_final_keys(current_keys, metadata_phones, duplicates,
    replacement_keys):
    if len(replacement_keys) != len(duplicates):
        raise ValueError(
            f'replacement count mismatch: {len(replacement_keys)} keys for '
            f'{len(duplicates)} duplicate occurrences'
        )

    final_keys = list(current_keys)
    for (index, _), key in zip(duplicates, replacement_keys):
        _valid_key(key, 'replacement Phraser phone key')
        final_keys[index] = key

    if len(final_keys) != _total_phone_count:
        raise ValueError(
            f'final inventory has {len(final_keys)} keys, expected '
            f'{_total_phone_count}'
        )
    if len(set(final_keys)) != len(final_keys):
        raise ValueError('final Phraser phone keys are not unique')

    label_counts = Counter(phone.phoneme_ipa for phone in metadata_phones)
    invalid = {
        label: count for label, count in label_counts.items()
        if count != _phones_per_label
    }
    if invalid or sum(label_counts.values()) != _total_phone_count:
        raise ValueError(
            f'metadata labels must each have {_phones_per_label} phones; '
            f'found {dict(label_counts)}'
        )


def save_duplicate_replacement_phraser_keys(
    phones,
    path=duplicate_replacement_phraser_phone_key_file,
    counts_path=duplicate_phone_counts_file,
    seed=42,
    overwrite=False,
):
    '''Select unused Phraser phones and save keys that replace duplicates.

    The output contains one fixed-width key for each repeated occurrence in
    the current key file. Keys are ordered to match those occurrences in
    metadata order, so they can be substituted without changing alignment.
    '''
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'{path} already exists; pass overwrite=True to replace it'
        )

    counts = _load_replacement_counts(counts_path)
    current_keys = _load_current_keys(phones)
    metadata_phones = phones.phones
    duplicates = _duplicate_occurrences(current_keys, metadata_phones)
    required_counts = _check_duplicate_counts(duplicates, counts)

    candidates = _collect_candidate_phones(
        phones, set(required_counts), set(current_keys)
    )
    selected = _sample_candidate_keys(candidates, required_counts, seed)
    replacement_keys = _arrange_replacement_keys(duplicates, selected)
    _validate_final_keys(
        current_keys, metadata_phones, duplicates, replacement_keys
    )

    path.write_bytes(b''.join(replacement_keys))
