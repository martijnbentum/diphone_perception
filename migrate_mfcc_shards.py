'''Rebuild the random-frame MFCC store with smaller HDF5 shards.'''

import argparse
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from time import monotonic

import echoframe
import h5py
import numpy as np
from echoframe.metadata import EchoframeMetadata

import locations


MAX_SHARD_ITEMS = 20_000
DEFAULT_BATCH_SIZE = 5_000
READ_CHUNK_SIZE = 100


def migrate_mfcc_shards(root, batch_size=DEFAULT_BATCH_SIZE, resume=False):
    '''Rebuild an MFCC store and retain the original as a backup.

    root:        path to the existing Echoframe MFCC store
    batch_size:  number of MFCCs copied per write batch
    resume:      continue a partial replacement after an interrupted read

    Stop all writers before running this one-time migration. The replacement
    and backup are siblings of root, on the same filesystem. A failed build
    leaves the original store in place and keeps the partial replacement.
    '''
    root = Path(root).resolve()
    if batch_size < 1: raise ValueError('batch_size must be positive')
    if not root.is_dir(): raise ValueError(f'MFCC store does not exist: {root}')
    rebuilt = root.with_name(f'{root.name}.resharded')
    backup = root.with_name(f'{root.name}.before_reshard')
    if backup.exists(): raise FileExistsError(backup)
    if rebuilt.exists() and not resume: raise FileExistsError(rebuilt)
    if resume and not rebuilt.is_dir():
        raise FileNotFoundError(f'partial replacement not found: {rebuilt}')

    source = echoframe.Store(str(root))
    destination = None
    try:
        keys = source.index.all_echoframe_keys
        if not keys: raise ValueError('MFCC store is empty')
        source_txnid = source.index.last_txnid()
        if not resume and source.config_path.exists():
            rebuilt.mkdir()
            shutil.copy2(source.config_path, rebuilt / 'config.json')
        destination = echoframe.Store(str(rebuilt),
            max_shard_items=MAX_SHARD_ITEMS)
        expected, samples, initial_saved = _resume_state(source, destination,
            keys, batch_size) if resume else (defaultdict(set), {}, 0)
        start = monotonic()
        print(f'Migrating {len(keys)} MFCCs from item {initial_saved}; '
            f'write batch {batch_size}, '
            f'read chunk {READ_CHUNK_SIZE}', flush=True)
        for offset in range(initial_saved, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            metadata = source.index.load_many(batch_keys, store=source)
            _check_metadata(metadata, batch_keys)
            payloads = []
            for read_offset in range(0, len(metadata), READ_CHUNK_SIZE):
                chunk = metadata[read_offset:read_offset + READ_CHUNK_SIZE]
                payloads.extend(source.storage.load_many(chunk))
                read_count = offset + len(payloads)
                _report_progress(read_count, offset, len(keys), start,
                    initial_saved)
            items = []
            for position, (key, record, payload) in enumerate(zip(
                    batch_keys, metadata, payloads, strict=True)):
                if np.shape(payload) != (1, 39):
                    raise ValueError(f'invalid MFCC shape for {key.hex()}')
                copied = EchoframeMetadata.from_dict(record.to_dict(), key,
                    store=destination)
                items.append({'echoframe_key': key, 'metadata': copied,
                    'data': payload})
                if (offset + position) % 1000 == 0:
                    samples[key] = np.asarray(payload).copy()
            destination.save_many(items)
            copied = destination.load_many_metadata(batch_keys)
            for record, new_record in zip(metadata, copied, strict=True):
                _check_copy(record, new_record)
                expected[new_record.shard_id].add(
                    new_record.echoframe_key.hex())
            saved_now = offset + len(batch_keys)
            _report_progress(saved_now, saved_now, len(keys), start,
                initial_saved)

        if source.index.last_txnid() != source_txnid:
            raise RuntimeError('source store changed during migration')
        _verify_destination(destination, keys, expected, samples)
    finally:
        if destination is not None: destination.close()
        source.close()

    root.rename(backup)
    try:
        rebuilt.rename(root)
        switched = echoframe.Store(str(root))
        try:
            _check_samples(switched, samples)
        finally:
            switched.close()
    except Exception:
        if root.exists(): root.rename(rebuilt)
        backup.rename(root)
        raise
    print(f'Migrated MFCC store: {root}', flush=True)
    print(f'Original store retained at: {backup}', flush=True)
    return backup


def _report_progress(read, saved, total, start, initial_saved=0):
    '''Print read/write counts and a rough time-to-completion estimate.'''
    elapsed = monotonic() - start
    completed = (saved - initial_saved) or (read - initial_saved)
    remaining = total - (saved if saved > initial_saved else read)
    eta = elapsed * remaining / completed
    elapsed_text = str(timedelta(seconds=round(elapsed)))
    eta_text = str(timedelta(seconds=round(eta)))
    print(f'Read {read}/{total}; saved {saved}/{total}; '
        f'elapsed {elapsed_text}; rough ETA {eta_text}', flush=True)


def _resume_state(source, destination, keys, batch_size):
    '''Check completed batches in a partial replacement before resuming.'''
    saved_keys = destination.index.all_echoframe_keys
    count = len(saved_keys)
    if saved_keys != keys[:count]:
        raise ValueError('partial replacement keys are not a source prefix')
    if count % batch_size and count != len(keys):
        raise ValueError('partial replacement ends inside a write batch')
    print(f'Checking {count} saved MFCCs before resuming', flush=True)
    expected = defaultdict(set)
    samples = {}
    for shard_id in destination.index.list_shards():
        records = destination.index.find_by_shard(shard_id,
            store=destination)
        for record in records:
            expected[shard_id].add(record.echoframe_key.hex())
    for offset in range(0, count, batch_size):
        batch_keys = saved_keys[offset:offset + batch_size]
        source_records = source.index.load_many(batch_keys, store=source)
        copied_records = destination.index.load_many(batch_keys,
            store=destination)
        _check_metadata(source_records, batch_keys)
        _check_metadata(copied_records, batch_keys)
        for original, copied in zip(source_records, copied_records,
                strict=True):
            _check_copy(original, copied)
        checked = min(offset + batch_size, count)
        print(f'Checked metadata {checked}/{count}', flush=True)
    sample_keys = saved_keys[::1000]
    for offset in range(0, len(sample_keys), READ_CHUNK_SIZE):
        batch_keys = sample_keys[offset:offset + READ_CHUNK_SIZE]
        records = source.index.load_many(batch_keys, store=source)
        payloads = source.storage.load_many(records)
        for key, payload in zip(batch_keys, payloads, strict=True):
            samples[key] = np.asarray(payload).copy()
        checked = min(offset + READ_CHUNK_SIZE, len(sample_keys))
        print(f'Checked payload samples {checked}/{len(sample_keys)}',
            flush=True)
    _verify_destination(destination, saved_keys, expected, samples)
    print(f'Resuming after {count} verified MFCCs', flush=True)
    return expected, samples, count


def _check_metadata(metadata, keys):
    '''Require every indexed item to be a marker MFCC.'''
    if len(metadata) != len(keys):
        raise ValueError('source index is missing requested MFCC metadata')
    for key, record in zip(keys, metadata, strict=True):
        if (record.echoframe_key != key
                or record.output_type != 'acoustic_feature'
                or record.feature_name != 'mfcc'):
            raise ValueError(f'non-MFCC record in source: {key.hex()}')


def _check_copy(source, destination):
    '''Check marker link, shape, tags, and creation time after copying.'''
    fields = ('echoframe_key', 'phraser_source_id', 'shape', 'tags',
        'created_at')
    if any(getattr(source, field) != getattr(destination, field)
            for field in fields):
        raise ValueError(f'MFCC metadata changed: {source.echoframe_key.hex()}')


def _verify_destination(store, keys, expected, samples):
    '''Verify indexed keys, HDF5 dataset names, and sampled payloads.'''
    if set(store.index.all_echoframe_keys) != set(keys):
        raise ValueError('replacement index keys differ from source')
    shard_files = {path.stem for path in store.storage.root.glob('*.h5')}
    if shard_files != set(expected):
        raise ValueError('replacement shard files differ from index')
    for shard_id, names in expected.items():
        indexed = store.index.shard_entry_count(shard_id)
        if indexed != len(names):
            raise ValueError(f'index count differs for {shard_id}')
        filename = store.storage.root / f'{shard_id}.h5'
        with h5py.File(filename, 'r') as handle:
            actual = set(handle['/items'].keys())
        if actual != names:
            raise ValueError(f'HDF5 dataset names differ for {shard_id}')
        print(f'Verified {shard_id}: {len(names)} MFCCs', flush=True)
    _check_samples(store, samples)


def _check_samples(store, samples):
    '''Compare sampled MFCC payloads exactly with the source.'''
    keys = list(samples)
    payloads = store.load_many(keys, keep_missing=True)
    for key, payload in zip(keys, payloads, strict=True):
        expected = samples[key]
        if (payload is None or payload.dtype != expected.dtype
                or not np.array_equal(payload, expected)):
            raise ValueError(f'MFCC payload differs for {key.hex()}')


def main():
    '''Parse migration options and rebuild the configured MFCC store.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=locations.decomposition_random_frames_echoframe_mfcc_store)
    parser.add_argument('--batch-size', type=int,
        default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--resume', action='store_true',
        help='verify and continue an interrupted partial replacement')
    args = parser.parse_args()
    migrate_mfcc_shards(args.root, batch_size=args.batch_size,
        resume=args.resume)


if __name__ == '__main__':
    main()
