from pathlib import Path
from types import SimpleNamespace

import pytest

from probing import select_flemish_phones


def make_key(index):
    return f'flemish-{index:014d}'.encode()


def make_phone(label, key, start=0, end=10):
    return SimpleNamespace(label=label, key=key, start=start, end=end)


def make_audio(filename, phones=()):
    return SimpleNamespace(filename=filename, phones=list(phones))


def make_store(audios):
    return SimpleNamespace(audios=list(audios))


def configure_small_inventory(
    monkeypatch, labels, per_label, bounds,
):
    monkeypatch.setattr(
        select_flemish_phones, 'flemish_phone_labels', tuple(labels))
    monkeypatch.setattr(
        select_flemish_phones, 'flemish_phones_per_label', per_label)
    monkeypatch.setattr(
        select_flemish_phones, 'flemish_phone_count',
        len(labels) * per_label,
    )
    monkeypatch.setattr(select_flemish_phones, '_duration_bounds', bounds)


def read_keys(path):
    data = path.read_bytes()
    size = select_flemish_phones._phraser_key_len
    assert len(data) % size == 0
    return [
        data[index:index + size]
        for index in range(0, len(data), size)
    ]


def test_default_path_labels_and_duration_bounds_are_complete():
    expected_labels = (
        'd', 'f', 'ə', 'z', 'p', 'l', 'ɛ', 'eː', 't', 'r', 'ʉ', 'ŋ', 'aː',
        'v', 'ɔ', 'm', 'ɪ', 'ɣ', 'x', 'oː', 'ɑ', 'n', 'h', 'k', 'iː', 'j',
        'b', 's', 'ɛi', 'ʋ', 'uː',
    )

    assert select_flemish_phones.flemish_phone_labels == expected_labels
    assert set(select_flemish_phones._duration_bounds) == set(expected_labels)
    assert select_flemish_phones._duration_bounds['ə'] == (46, 1077)
    assert select_flemish_phones.flemish_phones_per_label == 5_000
    assert select_flemish_phones.flemish_phone_count == 155_000
    assert (
        select_flemish_phones.flemish_phraser_phone_key_file.name
        == 'flemish_phraser_phone_keys.bin'
    )


def test_filter_flemish_audios_requires_component_and_vl_path():
    comp_k = make_audio('/corpus/comp-k/vl/speaker/one.wav')
    comp_o = make_audio('corpus/comp-o/vl/two.wav')
    wrong_language = make_audio('/corpus/comp-k/nl/three.wav')
    language_variant = make_audio('/corpus/comp-o/vl-BE/four.wav')
    language_similar = make_audio('/corpus/comp-k/my-vl/five.wav')
    component_variant = make_audio('/corpus/comp-k-extra/vl/six.wav')
    wrong_case = make_audio('/corpus/comp-K/vl/seven.wav')

    result = select_flemish_phones.filter_flemish_audios(
        [
            wrong_language, comp_k, language_variant, comp_o,
            language_similar, component_variant, wrong_case,
        ],
        show_progress=False,
    )

    assert result == [comp_k, comp_o]


def test_collect_candidates_uses_inclusive_bounds_and_global_keys(monkeypatch):
    configure_small_inventory(
        monkeypatch, labels=('a', 'b'), per_label=1,
        bounds={'a': (10, 20), 'b': (10, 20)},
    )
    shared_key = make_key(2)
    store = make_store([make_audio('/corpus/comp-k/vl/audio.wav', [
        make_phone('a', make_key(1), end=9),
        make_phone('a', make_key(1), end=10),
        make_phone('a', shared_key, start=5, end=25),
        make_phone('a', make_key(3), end=21),
        make_phone('b', shared_key, end=10),
        make_phone('b', make_key(4), end=20),
        make_phone('not-requested', b'bad'),
    ])])

    candidates = select_flemish_phones._collect_candidate_keys(
        store, show_progress=False)

    assert candidates == {
        'a': [make_key(1), shared_key],
        'b': [make_key(4)],
    }


def test_collect_candidates_rejects_malformed_key(monkeypatch):
    configure_small_inventory(
        monkeypatch, labels=('a',), per_label=1, bounds={'a': (1, 20)})
    store = make_store([make_audio('/corpus/comp-o/vl/audio.wav', [
        make_phone('a', b'too-short'),
    ])])

    with pytest.raises(ValueError, match='22-byte'):
        select_flemish_phones._collect_candidate_keys(
            store, show_progress=False)


def test_collection_and_sampling_report_progress(monkeypatch):
    configure_small_inventory(
        monkeypatch, labels=('a',), per_label=1, bounds={'a': (1, 20)})
    prefixes = []

    def fake_progressbar(items, prefix):
        prefixes.append(prefix)
        return items

    monkeypatch.setattr(
        select_flemish_phones, 'progressbar', fake_progressbar)
    monkeypatch.setattr(
        select_flemish_phones.random,
        'sample',
        lambda population, count: list(population)[:count],
    )
    store = make_store([make_audio('/corpus/comp-k/vl/audio.wav', [
        make_phone('a', make_key(1)),
    ])])

    candidates = select_flemish_phones._collect_candidate_keys(store)
    selected = select_flemish_phones._sample_candidate_keys(
        candidates, seed=42)

    assert selected == {'a': [make_key(1)]}
    assert prefixes == [
        'Filtering Flemish component audios: ',
        'Scanning Flemish phone candidates: ',
        'Sampling Flemish phone labels: ',
    ]


def test_insufficient_inventory_reports_all_counts_without_writing(
    tmp_path, monkeypatch, capsys,
):
    configure_small_inventory(
        monkeypatch, labels=('a', 'b'), per_label=2,
        bounds={'a': (1, 20), 'b': (1, 20)},
    )
    store = make_store([make_audio('/corpus/comp-k/vl/audio.wav', [
        make_phone('a', make_key(1)),
        make_phone('a', make_key(2)),
        make_phone('b', make_key(3)),
    ])])
    path = tmp_path / 'flemish.bin'
    path.write_bytes(b'original')

    result = select_flemish_phones.save_flemish_phraser_phone_keys(
        store, path=path, overwrite=True, show_progress=False)

    assert result == {
        'available_counts': {'a': 2, 'b': 1},
        'selected_counts': {'a': 0, 'b': 0},
        'path': path,
        'written': False,
    }
    assert path.read_bytes() == b'original'
    report = capsys.readouterr().out
    assert 'a: 2' in report
    assert 'b: 1' in report


def test_insufficient_inventory_does_not_create_output(
    tmp_path, monkeypatch,
):
    configure_small_inventory(
        monkeypatch, labels=('a',), per_label=2, bounds={'a': (1, 20)})
    store = make_store([make_audio('/corpus/comp-k/vl/audio.wav', [
        make_phone('a', make_key(1)),
    ])])
    path = tmp_path / 'flemish.bin'

    result = select_flemish_phones.save_flemish_phraser_phone_keys(
        store, path=path, show_progress=False)

    assert result['written'] is False
    assert not path.exists()


def test_existing_output_requires_overwrite(tmp_path):
    path = tmp_path / 'flemish.bin'
    path.write_bytes(b'original')

    with pytest.raises(FileExistsError, match='overwrite=True'):
        select_flemish_phones.save_flemish_phraser_phone_keys(
            object(), path=path, show_progress=False)

    assert path.read_bytes() == b'original'


def test_final_validation_rejects_keys_reused_across_labels(
    monkeypatch,
):
    configure_small_inventory(
        monkeypatch, labels=('a', 'b'), per_label=1,
        bounds={'a': (1, 20), 'b': (1, 20)},
    )
    shared_key = make_key(1)

    with pytest.raises(ValueError, match='not globally unique'):
        select_flemish_phones._validate_selected_keys({
            'a': [shared_key],
            'b': [shared_key],
        })


def test_success_uses_seed_and_writes_label_major_keys(
    tmp_path, monkeypatch,
):
    configure_small_inventory(
        monkeypatch, labels=('a', 'b'), per_label=2,
        bounds={'a': (1, 20), 'b': (1, 20)},
    )
    store = make_store([make_audio('/corpus/comp-o/vl/audio.wav', [
        make_phone('a', make_key(1)),
        make_phone('a', make_key(2)),
        make_phone('a', make_key(3)),
        make_phone('b', make_key(4)),
        make_phone('b', make_key(5)),
        make_phone('b', make_key(6)),
    ])])
    calls = []
    monkeypatch.setattr(
        select_flemish_phones.random,
        'seed',
        lambda seed: calls.append(('seed', seed)),
    )

    def sample(population, count):
        calls.append(('sample', list(population), count))
        return list(reversed(population))[:count]

    monkeypatch.setattr(select_flemish_phones.random, 'sample', sample)
    path = tmp_path / 'flemish.bin'

    result = select_flemish_phones.save_flemish_phraser_phone_keys(
        store, path=path, seed=42, show_progress=False)

    assert calls == [
        ('seed', 42),
        ('sample', [make_key(1), make_key(2), make_key(3)], 2),
        ('sample', [make_key(4), make_key(5), make_key(6)], 2),
    ]
    assert read_keys(path) == [
        make_key(3), make_key(2), make_key(6), make_key(5)]
    assert result == {
        'available_counts': {'a': 3, 'b': 3},
        'selected_counts': {'a': 2, 'b': 2},
        'path': path,
        'written': True,
    }
