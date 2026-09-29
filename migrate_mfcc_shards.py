'''Rebuild the random-frame MFCC store with smaller HDF5 shards.'''

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from time import monotonic

import echoframe
import h5py
import numpy as np
from echoframe.metadata import EchoframeMetadata

import locations


MAX_SHARD_ITEMS = 20_000
DEFAULT_BATCH_SIZE = 5_000


def migrate_mfcc_shards(root, batch_size=DEFAULT_BATCH_SIZE):
    '''Rebuild an MFCC store and retain the original as a backup.

    root:        path to the existing Echoframe MFCC store
    batch_size:  number of MFCCs copied per write batch

    Stop all writers before running this one-time migration. The replacement
    and backup are siblings of root, on the same filesystem. A failed build
    leaves the original store in place and keeps the partial replacement.
    '''
    root = Path(root).resolve()
    if batch_size < 1: raise ValueError('batch_size must be positive')
    if not root.is_dir(): raise ValueError(f'MFCC store does not exist: {root}')
    rebuilt = root.with_name(f'{root.name}.resharded')
    backup = root.with_name(f'{root.name}.before_reshard')
    if rebuilt.exists(): raise FileExistsError(rebuilt)
    if backup.exists(): raise FileExistsError(backup)

    source = echoframe.Store(str(root))
    destination = None
    try:
        keys = source.index.all_echoframe_keys
        if not keys: raise ValueError('MFCC store is empty')
        source_txnid = source.index.last_txnid()
        if source.config_path.exists():
            rebuilt.mkdir()
            shutil.copy2(source.config_path, rebuilt / 'config.json')
        destination = echoframe.Store(str(rebuilt),
            max_shard_items=MAX_SHARD_ITEMS)
        expected = defaultdict(set)
        samples = {}
        start = monotonic()
        for offset in range(0, len(keys), batch_size):
            batch_keys = keys[offset:offset + batch_size]
            metadata = source.index.load_many(batch_keys, store=source)
            _check_metadata(metadata, batch_keys)
            payloads = source.storage.load_many(metadata)
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
            done = min(offset + batch_size, len(keys))
            elapsed = monotonic() - start
            eta = elapsed * (len(keys) - done) / done
            print(f'Copied {done}/{len(keys)} MFCCs; ETA {eta / 60:.1f} min',
                flush=True)

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
    args = parser.parse_args()
    migrate_mfcc_shards(args.root, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
