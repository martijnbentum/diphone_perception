import sys
from pathlib import Path

import numpy as np
import pytest
from echoframe import EchoframeMetadata, Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import move_embeddings_based_on_phraser_key as mover


MODEL_NAME = 'wav2vec2_nl1_checkpoint-1000'


def make_key(value):
    return value.to_bytes(22, byteorder='big')


def save_output(
    store, phraser_key, data, *, output_type='hidden_state', layer=9,
):
    echoframe_key = store.make_echoframe_key(
        output_type,
        model_name=MODEL_NAME,
        phraser_key=phraser_key,
        layer=layer,
        collar=2000,
    )
    metadata = EchoframeMetadata(
        echoframe_key,
        store=store,
        tags=['original'],
        model_name=MODEL_NAME,
        phraser_source_id='cgn-awd',
    )
    return store.save(echoframe_key, metadata, np.asarray(data))


def make_source_store(path, records):
    store = Store(str(path), max_shard_size_bytes=100_000_000)
    store.register_model(
        MODEL_NAME,
        huggingface_id='example/wav2vec2',
        language='nl',
        size='base',
        architecture='wav2vec2',
    )
    store.register_phraser_store('cgn-awd', path.parent / 'phraser')
    saved = [save_output(store, *record) for record in records]
    store.close()
    return saved


def test_move_copies_verifies_deletes_and_compacts_mixed_shard(tmp_path):
    flemish_key = make_key(1)
    dutch_key = make_key(2)
    source_path = tmp_path / 'source'
    destination_path = tmp_path / 'destination'
    source_records = make_source_store(source_path, [
        (flemish_key, [[1.0, np.nan], [2.0, 3.0]]),
        (dutch_key, [[4.0, 5.0]]),
    ])
    source = Store(str(source_path))
    attention = save_output(
        source, flemish_key, [[8.0]], output_type='attention')
    source.close()

    report = mover.move_embeddings_based_on_phraser_keys(
        [flemish_key], source_path, destination_path,
        batch_size=1, verbose=False)

    source = Store(str(source_path))
    destination = Store(str(destination_path))
    try:
        destination_key = destination.make_echoframe_key(
            'hidden_state', model_name=MODEL_NAME,
            phraser_key=flemish_key, layer=9, collar=2000)
        dutch_source_key = source.make_echoframe_key(
            'hidden_state', model_name=MODEL_NAME,
            phraser_key=dutch_key, layer=9, collar=2000)
        copied = destination.load(destination_key)

        assert np.array_equal(
            copied,
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            equal_nan=True,
        )
        assert source.load(source_records[0].echoframe_key) is None
        assert np.array_equal(source.load(dutch_source_key), [[4.0, 5.0]])
        assert np.array_equal(source.load(attention.echoframe_key), [[8.0]])
        assert len(destination.metadatas) == 1
        assert destination.phraser_registry.load_path('cgn-awd') == str(
            (tmp_path / 'phraser').resolve())
        assert source.verify_integrity()['unreferenced_shard_files'] == []
        assert destination.verify_integrity()['ok'] is True
    finally:
        destination.close()
        source.close()

    assert report['status'] == 'moved'
    assert report['selected_embedding_count'] == 1
    assert report['copied_count'] == 1
    assert report['verified_count'] == 1
    assert report['deleted_count'] == 1
    assert report['mixed_shard_count'] == 1
    assert report['pure_flemish_shard_count'] == 0
    assert report['compacted_shard_count'] == 1


def test_move_removes_a_pure_flemish_shard(tmp_path):
    flemish_key = make_key(1)
    source_path = tmp_path / 'source'
    destination_path = tmp_path / 'destination'
    saved = make_source_store(
        source_path, [(flemish_key, [[1.0, 2.0]])])
    source_shard = source_path / 'shards' / f'{saved[0].shard_id}.h5'

    report = mover.move_embeddings_based_on_phraser_keys(
        [flemish_key], source_path, destination_path, verbose=False)

    assert report['pure_flemish_shard_count'] == 1
    assert report['mixed_shard_count'] == 0
    assert report['compacted_shard_count'] == 1
    assert not source_shard.exists()


def test_move_leaves_source_untouched_when_payload_verification_fails(
    tmp_path, monkeypatch,
):
    flemish_key = make_key(1)
    source_path = tmp_path / 'source'
    destination_path = tmp_path / 'destination'
    saved = make_source_store(
        source_path, [(flemish_key, [[1.0, 2.0]])])
    monkeypatch.setattr(mover, '_payloads_match', lambda first, second: False)

    with pytest.raises(RuntimeError, match='does not exactly match'):
        mover.move_embeddings_based_on_phraser_keys(
            [flemish_key], source_path, destination_path, verbose=False)

    source = Store(str(source_path))
    try:
        assert source.load(saved[0].echoframe_key) is not None
    finally:
        source.close()
    assert destination_path.is_dir()


def test_move_requires_a_new_destination_before_opening_source(
    tmp_path, monkeypatch,
):
    source_path = tmp_path / 'source'
    source_path.mkdir()
    destination_path = tmp_path / 'destination'
    destination_path.mkdir()
    monkeypatch.setattr(
        mover,
        'Store',
        lambda path, **kwargs: pytest.fail('Store should not be opened'),
    )

    with pytest.raises(FileExistsError, match='must be new'):
        mover.move_embeddings_based_on_phraser_keys(
            [make_key(1)], source_path, destination_path, verbose=False)


@pytest.mark.parametrize(
    ('keys', 'message'),
    [
        ([make_key(1), make_key(1)], 'globally unique'),
        ([b'\x00' * 22], 'non-placeholder'),
    ],
)
def test_move_rejects_invalid_key_lists_before_opening_a_store(
    tmp_path, monkeypatch, keys, message,
):
    monkeypatch.setattr(
        mover,
        'Store',
        lambda path, **kwargs: pytest.fail('Store should not be opened'),
    )

    with pytest.raises(ValueError, match=message):
        mover.move_embeddings_based_on_phraser_keys(
            keys, tmp_path / 'source', tmp_path / 'destination',
            verbose=False,
        )


def test_no_matching_embeddings_does_not_create_destination(tmp_path):
    source_path = tmp_path / 'source'
    destination_path = tmp_path / 'destination'
    make_source_store(source_path, [(make_key(2), [[1.0, 2.0]])])

    report = mover.move_embeddings_based_on_phraser_keys(
        [make_key(1)], source_path, destination_path, verbose=False)

    assert report['status'] == 'no_matches'
    assert report['destination_created'] is False
    assert not destination_path.exists()


def test_load_flemish_keys_uses_the_binary_loader_and_validator(monkeypatch):
    keys = [make_key(1), make_key(2)]
    calls = []
    monkeypatch.setattr(
        mover.metadata, 'load_phraser_keys',
        lambda path: calls.append(('load', path)) or keys,
    )
    monkeypatch.setattr(
        mover.metadata, '_validate_flemish_phraser_keys',
        lambda value: calls.append(('validate', value)),
    )

    loaded = mover.load_flemish_phraser_keys('flemish.bin')

    assert loaded is keys
    assert calls == [
        ('load', 'flemish.bin'),
        ('validate', keys),
    ]


def test_default_roots_are_the_dutch_and_flemish_model_store_roots():
    assert mover.default_netherlandic_stores_root.name == (
        'echoframe_model_stores')
    assert mover.default_flemish_stores_root.name == (
        'echoframe_model_flemish_stores')


def test_source_discovery_and_destination_mapping_are_model_specific(
    tmp_path,
):
    source_root = tmp_path / 'dutch'
    source_root.mkdir()
    second = source_root / 'model-b'
    first = source_root / 'model-a'
    second.mkdir()
    first.mkdir()
    (source_root / 'not-a-store').write_text('file')
    flemish_root = tmp_path / 'flemish'

    paths = mover.netherlandic_source_paths(source_root)

    assert paths == [first.resolve(), second.resolve()]
    assert mover.flemish_destination_path(first, flemish_root) == (
        flemish_root.resolve() / 'model-a')


def test_move_flemish_data_skips_existing_destinations(tmp_path, monkeypatch):
    source_root = tmp_path / 'dutch'
    flemish_root = tmp_path / 'flemish'
    source_root.mkdir()
    flemish_root.mkdir()
    for model_name in ('model-a', 'model-b'):
        (source_root / model_name).mkdir()
    (flemish_root / 'model-b').mkdir()
    monkeypatch.setattr(
        mover, 'load_flemish_phraser_keys', lambda path: [make_key(1)])
    calls = []

    def move(keys, source_path, destination_path, **kwargs):
        calls.append((source_path, destination_path))
        return _synthetic_store_report(source_path, destination_path, 3)

    monkeypatch.setattr(
        mover, 'move_embeddings_based_on_phraser_keys', move)

    report = mover.move_flemish_data(
        phraser_key_path=tmp_path / 'keys.bin',
        netherlandic_root=source_root,
        flemish_root=flemish_root,
        verbose=False,
    )

    assert calls == [(
        source_root / 'model-a', flemish_root.resolve() / 'model-a')]
    assert [item['status'] for item in report['stores']] == [
        'moved', 'skipped_existing']
    assert report['stores'][1]['reason'] == (
        'destination store already exists')
    assert report['summary']['moved_stores'] == 1
    assert report['summary']['skipped_existing_stores'] == 1
    assert report['status'] == 'complete'


def _synthetic_store_report(source_path, destination_path, count):
    return {
        'status': 'moved',
        'source_path': str(source_path),
        'destination_path': str(destination_path),
        'selected_embedding_count': count,
        'copied_count': count,
        'verified_count': count,
        'deleted_count': count,
        'affected_shard_count': 2,
        'pure_flemish_shard_count': 1,
        'mixed_shard_count': 1,
        'compacted_shard_count': 2,
    }


def test_move_flemish_data_reports_progress_and_aggregate_counts(
    tmp_path, monkeypatch, capsys,
):
    source_root = tmp_path / 'dutch'
    flemish_root = tmp_path / 'flemish'
    source_root.mkdir()
    for model_name in ('model-a', 'model-b'):
        (source_root / model_name).mkdir()
    keys = [make_key(1)]
    calls = []
    monkeypatch.setattr(
        mover, 'load_flemish_phraser_keys', lambda path: keys)

    def move(phraser_keys, source_path, destination_path, **kwargs):
        calls.append((phraser_keys, source_path, destination_path, kwargs))
        return _synthetic_store_report(source_path, destination_path, 3)

    monkeypatch.setattr(
        mover, 'move_embeddings_based_on_phraser_keys', move)

    report = mover.move_flemish_data(
        phraser_key_path=tmp_path / 'keys.bin',
        netherlandic_root=source_root,
        flemish_root=flemish_root,
        batch_size=17,
        verbose=True,
    )

    assert len(calls) == 2
    assert calls[0][0] is keys
    assert calls[0][3] == {'batch_size': 17, 'verbose': True}
    assert report['status'] == 'complete'
    assert report['summary'] == {
        'n_stores': 2,
        'moved_stores': 2,
        'skipped_existing_stores': 0,
        'no_match_stores': 0,
        'failed_stores': 0,
        'selected_embedding_count': 6,
        'copied_count': 6,
        'verified_count': 6,
        'deleted_count': 6,
        'affected_shard_count': 4,
        'pure_flemish_shard_count': 2,
        'mixed_shard_count': 2,
        'compacted_shard_count': 4,
    }
    output = capsys.readouterr().out
    assert '[1/2] model-a' in output
    assert '[2/2] model-b' in output
    assert 'Flemish embedding move report' in output
    assert '6 copied and verified' in output
