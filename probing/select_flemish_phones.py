import random
from pathlib import Path

from progressbar import progressbar


_data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
flemish_phraser_phone_key_file = (
    _data_dir / 'flemish_phraser_phone_keys.bin'
)

_phraser_key_len = 22
_components = {'comp-k', 'comp-o'}
_language_component = 'vl'

# This order is the binary layout contract: each label occupies one block of
# flemish_phones_per_label consecutive keys.
flemish_phone_labels = (
    'd', 'f', 'ə', 'z', 'p', 'l', 'ɛ', 'eː', 't', 'r', 'ʉ', 'ŋ', 'aː', 'v',
    'ɔ', 'm', 'ɪ', 'ɣ', 'x', 'oː', 'ɑ', 'n', 'h', 'k', 'iː', 'j', 'b', 's',
    'ɛi', 'ʋ', 'uː',
)
flemish_phones_per_label = 5_000
flemish_phone_count = len(flemish_phone_labels) * flemish_phones_per_label

# Inclusive bounds in milliseconds, copied from
# notes/phone_duration_stats.md, regenerated from data/metadata.csv.
_duration_bounds = {
    'aː': (46, 675),
    'b': (46, 304),
    'd': (46, 273),
    'eː': (47, 445),
    'f': (46, 439),
    'h': (46, 414),
    'iː': (50, 965),
    'j': (46, 563),
    'k': (46, 457),
    'l': (46, 358),
    'm': (46, 592),
    'n': (46, 296),
    'oː': (50, 530),
    'p': (46, 324),
    'r': (46, 644),
    's': (50, 699),
    't': (46, 366),
    'uː': (50, 351),
    'v': (46, 253),
    'x': (46, 373),
    'z': (46, 354),
    'ɑ': (47, 355),
    'ɔ': (46, 283),
    'ə': (46, 1077),
    'ɛ': (46, 273),
    'ɛi': (50, 429),
    'ɪ': (46, 466),
    'ɣ': (50, 344),
    'ŋ': (50, 654),
    'ʉ': (50, 304),
    'ʋ': (46, 274),
}


def _show_progress(items, prefix, enabled):
    if not enabled:
        return items
    return progressbar(items, prefix=prefix)


def _is_flemish_component_audio(audio):
    parts = set(Path(audio.filename).parts)
    has_component = bool(_components.intersection(parts))
    return has_component and _language_component in parts


def filter_flemish_audios(audios, show_progress=True):
    '''Return Flemish Dutch audios from exact comp-k/comp-o components.'''
    audios = _show_progress(
        audios, prefix='Filtering Flemish component audios: ',
        enabled=show_progress,
    )
    return [audio for audio in audios if _is_flemish_component_audio(audio)]


def _valid_key(key, description):
    if not isinstance(key, bytes) or len(key) != _phraser_key_len:
        raise ValueError(
            f'{description} must be a {_phraser_key_len}-byte value'
        )


def _phone_is_in_duration_range(phone):
    minimum, maximum = _duration_bounds[phone.label]
    duration = phone.end - phone.start
    return minimum <= duration <= maximum


def _collect_candidate_keys(store, show_progress=True):
    candidates = {label: [] for label in flemish_phone_labels}
    candidate_keys = set()
    labels = set(flemish_phone_labels)

    audios = filter_flemish_audios(
        store.audios, show_progress=show_progress
    )
    audios = _show_progress(
        audios, prefix='Scanning Flemish phone candidates: ',
        enabled=show_progress,
    )
    for audio in audios:
        for phone in audio.phones:
            if phone.label not in labels:
                continue
            if not _phone_is_in_duration_range(phone):
                continue

            key = phone.key
            _valid_key(key, 'candidate Phraser phone key')
            if key in candidate_keys:
                continue
            candidate_keys.add(key)
            candidates[phone.label].append(key)
    return candidates


def _available_counts(candidates):
    counts = {
        label: len(candidates[label]) for label in flemish_phone_labels}
    print('Available unique Flemish phone tokens per label:')
    for label, count in counts.items():
        print(f'  {label}: {count}')
    return counts


def _sample_candidate_keys(candidates, seed, show_progress=True):
    random.seed(seed)
    selected = {}
    labels = _show_progress(
        flemish_phone_labels, prefix='Sampling Flemish phone labels: ',
        enabled=show_progress,
    )
    for label in labels:
        selected[label] = random.sample(
            candidates[label], flemish_phones_per_label
        )
    return selected


def _selection_result(path, available_counts, selected_counts, written):
    return {
        'available_counts': available_counts,
        'selected_counts': selected_counts,
        'path': path,
        'written': written,
    }


def _validate_selected_keys(selected):
    selected_keys = []
    for label in flemish_phone_labels:
        keys = selected[label]
        if len(keys) != flemish_phones_per_label:
            raise ValueError(
                f'selected {len(keys)} keys for {label!r}, expected '
                f'{flemish_phones_per_label}'
            )
        for key in keys:
            _valid_key(key, 'selected Phraser phone key')
        selected_keys.extend(keys)

    if len(selected_keys) != flemish_phone_count:
        raise ValueError(
            f'selected {len(selected_keys)} keys, expected '
            f'{flemish_phone_count}'
        )
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError('selected Phraser phone keys are not globally unique')
    return selected_keys


def save_flemish_phraser_phone_keys(
    store,
    path=flemish_phraser_phone_key_file,
    seed=42,
    overwrite=False,
    show_progress=True,
):
    '''Select a balanced Flemish phone inventory and save its Phraser keys.

    The output is label-major in ``flemish_phone_labels`` order, with 5,000
    consecutive 22-byte keys per label. If any label has too few eligible
    unique tokens, counts are reported and the output is untouched. Set
    show_progress=False to suppress progress bars.
    '''
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'{path} already exists; pass overwrite=True to replace it'
        )

    candidates = _collect_candidate_keys(
        store, show_progress=show_progress
    )
    available_counts = _available_counts(candidates)
    selected_counts = {label: 0 for label in flemish_phone_labels}
    if any(
        count < flemish_phones_per_label
        for count in available_counts.values()
    ):
        return _selection_result(
            path, available_counts, selected_counts, written=False
        )

    selected = _sample_candidate_keys(
        candidates, seed, show_progress=show_progress
    )
    selected_keys = _validate_selected_keys(selected)
    path.write_bytes(b''.join(selected_keys))

    selected_counts = {
        label: len(selected[label]) for label in flemish_phone_labels
    }
    return _selection_result(
        path, available_counts, selected_counts, written=True
    )
