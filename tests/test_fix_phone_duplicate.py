import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from phraser import SEGMENT_KEY_LENGTH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import fix_phone_duplicate


def make_key(index):
    return f'key-{index:018d}'.encode()


def make_phraser_phone(label, key, start=0, end=10):
    return SimpleNamespace(label=label, key=key, start=start, end=end)


def make_audio(filename, phones=()):
    return SimpleNamespace(filename=filename, phones=list(phones))


def make_phones(tmp_path, current_keys, labels, audios=()):
    key_path = tmp_path / 'phraser_phone_keys.bin'
    key_path.write_bytes(b''.join(current_keys))
    return SimpleNamespace(
        phraser_key_path=key_path,
        phones=[SimpleNamespace(phoneme_ipa=label) for label in labels],
        store=SimpleNamespace(audios=list(audios)),
    )


def write_counts(tmp_path, counts):
    path = tmp_path / 'duplicate_phone_counts.json'
    path.write_text(json.dumps(counts))
    return path


def read_keys(path):
    size = SEGMENT_KEY_LENGTH
    data = path.read_bytes()
    assert len(data) % size == 0
    return [data[index:index + size] for index in range(0, len(data), size)]


def configure_small_inventory(monkeypatch, total, per_label, bounds):
    monkeypatch.setattr(fix_phone_duplicate, '_total_phone_count', total)
    monkeypatch.setattr(fix_phone_duplicate, '_phones_per_label', per_label)
    monkeypatch.setattr(fix_phone_duplicate, '_duration_bounds', bounds)


def test_filter_component_audios_requires_component_and_nl_path():
    comp_k = make_audio('/corpus/comp-k/nl/speaker/one.wav')
    comp_o = make_audio('corpus/comp-o/nl/two.wav')
    nested = make_audio('/comp-k/comp-o/nl/three.wav')
    similar = make_audio('/corpus/comp-k-extra/nl/four.wav')
    suffix = make_audio('/corpus/my-comp-o/nl/five.wav')
    wrong_case = make_audio('/corpus/comp-K/nl/six.wav')
    flemish = make_audio('/corpus/comp-k/vl/seven.wav')
    language_variant = make_audio('/corpus/comp-o/nl-BE/eight.wav')
    language_similar = make_audio('/corpus/comp-k/my-nl/nine.wav')

    result = fix_phone_duplicate.filter_component_audios(
        [
            similar, comp_k, suffix, comp_o, wrong_case, nested, flemish,
            language_variant, language_similar,
        ],
        show_progress=False,
    )

    assert result == [comp_k, comp_o, nested]


def test_candidate_duration_bounds_are_inclusive(monkeypatch):
    configure_small_inventory(monkeypatch, 0, 0, {'a': (10, 20)})
    phones = SimpleNamespace(store=SimpleNamespace(audios=[make_audio(
        '/corpus/comp-k/nl/audio.wav',
        [
            make_phraser_phone('a', make_key(1), end=9),
            make_phraser_phone('a', make_key(2), end=10),
            make_phraser_phone('a', make_key(3), start=5, end=25),
            make_phraser_phone('a', make_key(4), end=21),
        ],
    )]))

    candidates = fix_phone_duplicate._collect_candidate_phones(
        phones, {'a'}, set()
    )

    assert [phone.key for phone in candidates['a']] == [
        make_key(2),
        make_key(3),
    ]


def test_candidates_exclude_current_and_repeated_keys(monkeypatch):
    configure_small_inventory(monkeypatch, 0, 0, {'a': (1, 20)})
    current_key = make_key(1)
    repeated_key = make_key(2)
    unique_key = make_key(3)
    phones = SimpleNamespace(store=SimpleNamespace(audios=[
        make_audio('/corpus/comp-k/nl/one.wav', [
            make_phraser_phone('a', current_key),
            make_phraser_phone('a', repeated_key),
        ]),
        make_audio('/corpus/comp-o/nl/two.wav', [
            make_phraser_phone('a', repeated_key),
            make_phraser_phone('a', unique_key),
        ]),
    ]))

    candidates = fix_phone_duplicate._collect_candidate_phones(
        phones, {'a'}, {current_key}
    )

    assert [phone.key for phone in candidates['a']] == [
        repeated_key,
        unique_key,
    ]


def test_sampling_uses_module_random_seed_and_sample(monkeypatch):
    calls = []
    candidates = {
        'a': [
            make_phraser_phone('a', make_key(1)),
            make_phraser_phone('a', make_key(2)),
        ],
        'b': [make_phraser_phone('b', make_key(3))],
    }

    monkeypatch.setattr(
        fix_phone_duplicate.random,
        'seed',
        lambda seed: calls.append(('seed', seed)),
    )

    def sample(population, count):
        calls.append(('sample', [phone.key for phone in population], count))
        return list(reversed(population))[:count]

    monkeypatch.setattr(fix_phone_duplicate.random, 'sample', sample)

    selected = fix_phone_duplicate._sample_candidate_keys(
        candidates, {'a': 2, 'b': 1}, seed=42, show_progress=False,
    )

    assert calls == [
        ('seed', 42),
        ('sample', [make_key(1), make_key(2)], 2),
        ('sample', [make_key(3)], 1),
    ]
    assert selected == {
        'a': [make_key(2), make_key(1)],
        'b': [make_key(3)],
    }


def test_collection_and_sampling_report_progress(monkeypatch):
    prefixes = []
    monkeypatch.setattr(
        fix_phone_duplicate, '_duration_bounds', {'a': (1, 20)})

    def fake_progressbar(items, prefix):
        prefixes.append(prefix)
        return items

    monkeypatch.setattr(fix_phone_duplicate, 'progressbar', fake_progressbar)
    monkeypatch.setattr(
        fix_phone_duplicate.random,
        'sample',
        lambda population, count: list(population)[:count],
    )
    phone = make_phraser_phone('a', make_key(1))
    phones = SimpleNamespace(store=SimpleNamespace(audios=[
        make_audio('/corpus/comp-k/nl/audio.wav', [phone]),
    ]))

    candidates = fix_phone_duplicate._collect_candidate_phones(
        phones, {'a'}, set())
    selected = fix_phone_duplicate._sample_candidate_keys(
        candidates, {'a': 1}, seed=42)

    assert selected == {'a': [phone.key]}
    assert prefixes == [
        'Filtering component audios: ',
        'Scanning candidate phones in audios: ',
        'Sampling replacement labels: ',
    ]


@pytest.mark.parametrize(
    'counts',
    [
        [],
        {},
        {'n': 3},
        {'n': 0},
        {'n': -2},
        {'n': True},
        {'n': 2.0},
        {'': 2},
        {'unknown': 2},
    ],
    ids=[
        'not-an-object',
        'empty',
        'odd',
        'zero',
        'negative',
        'boolean',
        'not-an-integer',
        'empty-label',
        'unknown-label',
    ],
)
def test_save_rejects_invalid_replacement_counts(
    tmp_path, monkeypatch, counts,
):
    counts_path = write_counts(tmp_path, counts)
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file',
        tmp_path / 'replacements.bin')
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)

    with pytest.raises(ValueError):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(
            object(),
        )


def test_save_rejects_counts_that_do_not_match_duplicate_occurrences(
    tmp_path, monkeypatch,
):
    configure_small_inventory(monkeypatch, 2, 2, {'a': (1, 20)})
    key = make_key(1)
    phones = make_phones(tmp_path, [key, key], ['a', 'a'])
    counts_path = write_counts(tmp_path, {'a': 4})
    output_path = tmp_path / 'replacements.bin'
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file', output_path)
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)

    with pytest.raises(ValueError, match='do not match'):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(phones)

    assert not output_path.exists()


def test_save_rejects_key_metadata_length_mismatch(tmp_path, monkeypatch):
    configure_small_inventory(monkeypatch, 2, 2, {'a': (1, 20)})
    phones = make_phones(tmp_path, [make_key(1)], ['a', 'a'])
    counts_path = write_counts(tmp_path, {'a': 2})
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file',
        tmp_path / 'replacements.bin')
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)

    with pytest.raises(ValueError, match='key/metadata length mismatch'):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(phones)


def test_save_rejects_insufficient_candidate_pool(tmp_path, monkeypatch):
    configure_small_inventory(monkeypatch, 2, 2, {'a': (1, 20)})
    key = make_key(1)
    phones = make_phones(tmp_path, [key, key], ['a', 'a'])
    counts_path = write_counts(tmp_path, {'a': 2})
    output_path = tmp_path / 'replacements.bin'
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file', output_path)
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)

    with pytest.raises(ValueError, match="not enough.*'a'.*need 1, found 0"):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(phones)

    assert not output_path.exists()


def test_save_requires_explicit_permission_to_overwrite(
    tmp_path, monkeypatch,
):
    output_path = tmp_path / 'replacements.bin'
    output_path.write_bytes(b'original contents')
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file', output_path)
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file',
        tmp_path / 'does-not-need-to-exist.json')

    with pytest.raises(FileExistsError, match='overwrite=True'):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(object())

    assert output_path.read_bytes() == b'original contents'


def test_save_writes_records_in_duplicate_metadata_order(
    tmp_path, monkeypatch,
):
    configure_small_inventory(
        monkeypatch, total=8, per_label=4,
        bounds={'a': (1, 20), 'b': (1, 20)},
    )
    current_keys = [
        make_key(1),
        make_key(2),
        make_key(1),  # duplicate a
        make_key(3),
        make_key(2),  # duplicate b
        make_key(4),
        make_key(5),
        make_key(4),  # duplicate a
    ]
    labels = ['a', 'b', 'a', 'b', 'b', 'a', 'b', 'a']
    replacement_a1 = make_key(10)
    replacement_a2 = make_key(11)
    replacement_b = make_key(12)
    phones = make_phones(tmp_path, current_keys, labels, [
        make_audio('/corpus/comp-k/nl/a.wav', [
            make_phraser_phone('a', replacement_a1),
            make_phraser_phone('b', replacement_b),
        ]),
        make_audio('/corpus/comp-o/nl/b.wav', [
            make_phraser_phone('a', replacement_a2),
        ]),
    ])
    counts_path = write_counts(tmp_path, {'b': 2, 'a': 4})
    output_path = tmp_path / 'replacements.bin'
    output_path.write_bytes(b'old')
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file', output_path)
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)
    seeds = []
    monkeypatch.setattr(
        fix_phone_duplicate.random, 'seed', lambda seed: seeds.append(seed)
    )
    monkeypatch.setattr(
        fix_phone_duplicate.random,
        'sample',
        lambda population, count: list(population)[:count],
    )

    fix_phone_duplicate.save_duplicate_replacement_phraser_keys(
        phones,
        overwrite=True,
    )

    assert seeds == [42]
    assert read_keys(output_path) == [
        replacement_a1,
        replacement_b,
        replacement_a2,
    ]


def test_save_validates_final_label_balance_before_writing(
    tmp_path, monkeypatch,
):
    configure_small_inventory(monkeypatch, 4, 2, {'a': (1, 20)})
    current_key = make_key(1)
    phones = make_phones(
        tmp_path,
        [current_key, current_key, make_key(2), make_key(3)],
        ['a', 'a', 'a', 'a'],
        [make_audio('/corpus/comp-k/nl/a.wav', [
            make_phraser_phone('a', make_key(10)),
        ])],
    )
    counts_path = write_counts(tmp_path, {'a': 2})
    output_path = tmp_path / 'replacements.bin'
    monkeypatch.setattr(
        locations, 'duplicate_replacement_phraser_key_file', output_path)
    monkeypatch.setattr(
        locations, 'duplicate_phone_counts_file', counts_path)

    with pytest.raises(ValueError, match='metadata labels must each have 2'):
        fix_phone_duplicate.save_duplicate_replacement_phraser_keys(phones)

    assert not output_path.exists()


def test_final_validation_rejects_duplicate_replacement_keys(monkeypatch):
    configure_small_inventory(monkeypatch, 4, 4, {'a': (1, 20)})
    original = make_key(1)
    current_keys = [original, original, original, make_key(2)]
    metadata_phones = [
        SimpleNamespace(phoneme_ipa='a') for _ in current_keys
    ]
    duplicates = [(1, 'a'), (2, 'a')]
    replacement = make_key(10)

    with pytest.raises(ValueError, match='not unique'):
        fix_phone_duplicate._validate_final_keys(
            current_keys,
            metadata_phones,
            duplicates,
            [replacement, replacement],
        )


def test_final_validation_rejects_wrong_replacement_record_count(monkeypatch):
    configure_small_inventory(monkeypatch, 2, 2, {'a': (1, 20)})
    original = make_key(1)

    with pytest.raises(ValueError, match='replacement count mismatch'):
        fix_phone_duplicate._validate_final_keys(
            [original, original],
            [
                SimpleNamespace(phoneme_ipa='a'),
                SimpleNamespace(phoneme_ipa='a'),
            ],
            [(1, 'a')],
            [],
        )
