'''Move hidden-state embeddings selected by Phraser key between stores.'''

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from echoframe import EchoframeMetadata, Store
from progressbar import (
    Bar, ETA, Percentage, ProgressBar, SimpleProgress, Variable,
)

from probing import metadata
from probing.extract_embeddings import (
    default_flemish_model_stores_root,
    default_model_stores_root,
)


default_netherlandic_stores_root = default_model_stores_root
default_flemish_stores_root = default_flemish_model_stores_root
default_flemish_phraser_key_path = (
    metadata.flemish_phraser_phone_key_file)
default_batch_size = 100
_source_verification_batch_size = 10_000


def _resolved_path(path):
    return Path(path).expanduser().resolve()


def _move_progress_bar(max_value, label, verbose):
    if not verbose:
        return None
    return ProgressBar(
        max_value=max(max_value, 1),
        widgets=[
            Variable('label', format='{formatted_value}', width=62),
            ' ', Bar(), ' ', SimpleProgress(), ' ', Percentage(), ' ', ETA(),
        ],
        variables={'label': label},
    ).start()


def _update_move_progress(bar, value, label):
    if bar is not None:
        bar.update(value, label=label)


def _finish_move_progress(bar, label):
    if bar is not None:
        bar.variables['label'] = label
        bar.finish()


def _validate_batch_size(batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('batch_size must be a positive integer')
    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')


def _validate_phraser_keys(phraser_keys):
    if not isinstance(phraser_keys, list):
        raise TypeError('phraser_keys must be a list')
    if not phraser_keys:
        raise ValueError('phraser_keys must not be empty')
    invalid = [
        index for index, key in enumerate(phraser_keys)
        if not isinstance(key, bytes)
        or len(key) != metadata._phraser_key_len
        or key == metadata._phraser_key_placeholder
    ]
    if invalid:
        examples = ', '.join(map(str, invalid[:5]))
        raise ValueError(
            'every Phraser key must be a non-placeholder '
            f'{metadata._phraser_key_len}-byte value; invalid indices: '
            f'{examples}')
    if len(set(phraser_keys)) != len(phraser_keys):
        raise ValueError('phraser_keys must be globally unique')
    return set(phraser_keys)


def load_flemish_phraser_keys(
    path=default_flemish_phraser_key_path,
):
    '''Load and validate the selected Flemish Phraser-key inventory.'''
    keys = metadata.load_phraser_keys(path)
    metadata._validate_flemish_phraser_keys(keys)
    return keys


def netherlandic_source_paths(
    root=default_netherlandic_stores_root,
):
    '''Return every model-specific store directory below the Dutch root.'''
    root = _resolved_path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f'Netherlandic Echoframe store root does not exist: {root}')
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


def flemish_destination_path(
    netherlandic_source_path,
    root=default_flemish_stores_root,
):
    '''Map one Dutch model-store path to its Flemish destination path.'''
    source_path = Path(netherlandic_source_path).expanduser()
    if not source_path.name:
        raise ValueError('netherlandic_source_path must name a model store')
    return _resolved_path(root) / source_path.name


def _validate_store_paths(source_path, destination_path):
    source_path = _resolved_path(source_path)
    destination_path = _resolved_path(destination_path)
    if not source_path.is_dir():
        raise FileNotFoundError(
            f'source Echoframe store does not exist: {source_path}')
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f'destination Echoframe store must be new: {destination_path}')
    if (
        source_path == destination_path
        or source_path in destination_path.parents
    ):
        raise ValueError('source and destination stores must be independent')
    return source_path, destination_path


def _selected_hidden_states(source, phraser_key_set):
    all_metadatas = source.metadatas
    selected = [
        item for item in all_metadatas
        if item.output_type == 'hidden_state'
        and item.phraser_key in phraser_key_set
    ]
    selected.sort(key=lambda item: (
        str(item.shard_id), item.model_name, item.layer,
        item.collar, item.echoframe_key,
    ))
    return all_metadatas, selected


def _register_models(source, destination, selected):
    model_names = sorted({item.model_name for item in selected})
    for model_name in model_names:
        source_model = source.load_model_metadata(model_name)
        if source_model is None:
            raise ValueError(
                f'source model is not registered: {model_name!r}')
        destination.register_model(
            model_name,
            local_path=source_model.local_path,
            huggingface_id=source_model.huggingface_id,
            language=source_model.language,
            size=source_model.size,
            architecture=getattr(source_model, 'architecture', None),
        )
    return model_names


def _register_phraser_sources(source, destination, selected):
    source_ids = sorted({
        item.phraser_source_id for item in selected
        if item.phraser_source_id is not None
    })
    for source_id in source_ids:
        path = source.phraser_registry.load_path(source_id)
        if path is None:
            raise ValueError(
                'selected embedding refers to an unregistered Phraser '
                f'source: {source_id!r}')
        destination.register_phraser_store(source_id, path)
    return source_ids


def _destination_item(destination, source_metadata, payload):
    destination_key = destination.make_echoframe_key(
        'hidden_state',
        model_name=source_metadata.model_name,
        phraser_key=source_metadata.phraser_key,
        layer=source_metadata.layer,
        collar=source_metadata.collar,
    )
    destination_metadata = EchoframeMetadata(
        destination_key,
        store=destination,
        tags=source_metadata.tags,
        model_name=source_metadata.model_name,
        phraser_source_id=source_metadata.phraser_source_id,
    )
    destination_metadata.created_at = source_metadata.created_at
    return {
        'echoframe_key': destination_key,
        'metadata': destination_metadata,
        'data': payload,
    }


def _metadata_signature(item):
    return {
        'model_name': item.model_name,
        'output_type': item.output_type,
        'phraser_key': item.phraser_key,
        'phraser_source_id': item.phraser_source_id,
        'layer': item.layer,
        'collar': item.collar,
        'tags': item.tags,
        'created_at': item.created_at,
        'shape': item.shape,
    }


def _payloads_match(source_payload, destination_payload):
    source_array = np.asarray(source_payload)
    destination_array = np.asarray(destination_payload)
    if source_array.dtype != destination_array.dtype:
        return False
    if source_array.shape != destination_array.shape:
        return False
    return np.array_equal(
        source_array, destination_array, equal_nan=True)


def _copy_and_verify_batches(
    source, destination, selected, batch_size, label, verbose,
):
    destination_keys = []
    n_total = len(selected)
    progress_label = f'[{label}] copying and verifying embeddings'
    progress = _move_progress_bar(n_total, progress_label, verbose)
    completed = False
    try:
        for start in range(0, n_total, batch_size):
            batch = selected[start:start + batch_size]
            source_payloads = source.metadatas_to_payloads(batch)
            if len(source_payloads) != len(batch):
                raise RuntimeError(
                    'source returned an unexpected number of payloads')
            items = [
                _destination_item(destination, source_metadata, payload)
                for source_metadata, payload in zip(
                    batch, source_payloads, strict=True)
            ]
            destination.save_many(items)
            batch_keys = [item['echoframe_key'] for item in items]
            copied_metadatas = destination.load_many_metadata(
                batch_keys, keep_missing=True)
            copied_payloads = destination.metadatas_to_payloads(
                copied_metadatas)
            if len(copied_metadatas) != len(batch):
                raise RuntimeError(
                    'destination returned an unexpected number of metadata '
                    'records')
            if len(copied_payloads) != len(batch):
                raise RuntimeError(
                    'destination returned an unexpected number of payloads')

            records = zip(
                batch,
                source_payloads,
                copied_metadatas,
                copied_payloads,
                strict=True,
            )
            for (
                source_metadata,
                source_payload,
                copied_metadata,
                copied_payload,
            ) in records:
                if copied_metadata is None:
                    raise RuntimeError(
                        'destination metadata verification found a missing '
                        'record')
                if _metadata_signature(
                    source_metadata,
                ) != _metadata_signature(copied_metadata):
                    raise RuntimeError(
                        'destination metadata does not match its source '
                        f'record: {copied_metadata.echoframe_key.hex()}')
                if not _payloads_match(source_payload, copied_payload):
                    key = copied_metadata.echoframe_key.hex()
                    raise RuntimeError(
                        'destination payload does not exactly match its '
                        f'source record: {key}')

            destination_keys.extend(batch_keys)
            finished = min(start + len(batch), n_total)
            _update_move_progress(progress, finished, progress_label)
        completed = True
    finally:
        final_label = (
            f'[{label}] embeddings copied and verified'
            if completed else f'[{label}] embedding copy stopped'
        )
        _finish_move_progress(progress, final_label)
    return destination_keys


def _verify_destination(destination, selected, destination_keys):
    expected_keys = set(destination_keys)
    actual_keys = {item.echoframe_key for item in destination.metadatas}
    if actual_keys != expected_keys:
        raise RuntimeError(
            'destination key set does not match the selected embeddings')
    if len(expected_keys) != len(selected):
        raise RuntimeError('destination contains duplicate embedding keys')
    integrity = destination.verify_integrity()
    if not integrity.get('ok', False):
        raise RuntimeError(
            'destination Echoframe integrity verification failed: '
            f'{integrity.get("broken_metadata_references", [])!r}')
    if integrity.get('unreferenced_shard_files'):
        raise RuntimeError(
            'destination contains unreferenced shard files after copying')
    return integrity


def _selected_by_shard(selected):
    output = defaultdict(list)
    for item in selected:
        if item.shard_id is None:
            raise ValueError(
                'selected embedding metadata has no source shard identifier')
        output[item.shard_id].append(item)
    return dict(output)


def _count_remaining_selected(
    source, selected, label='source', verbose=False,
):
    remaining_count = 0
    n_total = len(selected)
    progress_label = f'[{label}] verifying source deletions'
    progress = _move_progress_bar(n_total, progress_label, verbose)
    completed = False
    try:
        for start in range(
            0, n_total, _source_verification_batch_size,
        ):
            batch = selected[start:start + _source_verification_batch_size]
            echoframe_keys = [item.echoframe_key for item in batch]
            metadatas = source.load_many_metadata(
                echoframe_keys, keep_missing=True)
            if len(metadatas) != len(echoframe_keys):
                raise RuntimeError(
                    'source returned an unexpected number of metadata '
                    'records during deletion verification')
            remaining_count += sum(item is not None for item in metadatas)
            finished = min(start + len(batch), n_total)
            _update_move_progress(progress, finished, progress_label)
        completed = True
    finally:
        final_label = (
            f'[{label}] source deletions checked'
            if completed else f'[{label}] source-deletion check stopped'
        )
        _finish_move_progress(progress, final_label)
    return remaining_count


def _delete_and_compact(
    source, all_metadatas, selected, label, verbose,
):
    by_shard = _selected_by_shard(selected)
    all_counts = Counter(
        item.shard_id for item in all_metadatas if item.shard_id is not None)
    pure_shards = {
        shard_id for shard_id, items in by_shard.items()
        if len(items) == all_counts[shard_id]
    }
    affected_shards = sorted(by_shard)
    compacted_shards = []
    deleted_count = 0
    progress_label = f'[{label}] compacting affected source shards'
    progress = _move_progress_bar(
        len(affected_shards), progress_label, verbose)
    completed = False
    try:
        for index, shard_id in enumerate(affected_shards, start=1):
            items = by_shard[shard_id]
            source.index.delete_many(items)
            deleted_count += len(items)
            compacted = source.compact_shards(
                shard_ids=[shard_id],
                min_garbage_bytes=0,
                min_garbage_ratio=0,
            )
            compacted_shards.extend(compacted)
            current_label = (
                f'[{label}] compacting; '
                f'{deleted_count:,}/{len(selected):,} deleted'
            )
            _update_move_progress(progress, index, current_label)
        completed = True
    finally:
        final_label = (
            f'[{label}] affected source shards compacted'
            if completed else f'[{label}] source compaction stopped'
        )
        _finish_move_progress(progress, final_label)

    remaining_count = _count_remaining_selected(
        source, selected, label=label, verbose=verbose)
    if remaining_count:
        raise RuntimeError(
            f'{remaining_count:,} selected embeddings remain in the source')
    if verbose:
        print(
            f'[{label}] selected source entries deleted; '
            'verifying source-store integrity',
            flush=True,
        )
    integrity = source.verify_integrity()
    if not integrity.get('ok', False):
        raise RuntimeError(
            'source Echoframe integrity verification failed after deletion: '
            f'{integrity.get("broken_metadata_references", [])!r}')
    if integrity.get('unreferenced_shard_files'):
        raise RuntimeError(
            'source contains unreferenced shard files after compaction')
    if verbose:
        print(f'[{label}] source-store integrity verified', flush=True)
    return {
        'deleted_count': deleted_count,
        'affected_shard_count': len(affected_shards),
        'pure_flemish_shard_count': len(pure_shards),
        'mixed_shard_count': len(affected_shards) - len(pure_shards),
        'compacted_shard_count': len(compacted_shards),
        'source_integrity': integrity,
    }


def _empty_move_report(source_path, destination_path, n_keys, elapsed):
    return {
        'status': 'no_matches',
        'source_path': str(source_path),
        'destination_path': str(destination_path),
        'requested_phraser_key_count': n_keys,
        'selected_embedding_count': 0,
        'copied_count': 0,
        'verified_count': 0,
        'deleted_count': 0,
        'model_names': [],
        'phraser_source_ids': [],
        'affected_shard_count': 0,
        'pure_flemish_shard_count': 0,
        'mixed_shard_count': 0,
        'compacted_shard_count': 0,
        'destination_created': False,
        'elapsed_seconds': round(elapsed, 6),
    }


def move_embeddings_based_on_phraser_keys(
    phraser_keys,
    source_path,
    destination_path,
    batch_size=default_batch_size,
    verbose=True,
):
    '''Move source hidden states whose Phraser key occurs in ``phraser_keys``.

    The destination must not exist. Every selected payload and metadata record
    is copied and checked exactly before any source index entry is removed.
    Only affected source shards are compacted: pure selected shards disappear,
    while mixed shards are rewritten with their non-selected records.
    '''
    _validate_batch_size(batch_size)
    phraser_key_set = _validate_phraser_keys(phraser_keys)
    source_path, destination_path = _validate_store_paths(
        source_path, destination_path)
    label = source_path.name
    started = time.perf_counter()
    source = None
    destination = None
    try:
        source = Store(str(source_path))
        all_metadatas, selected = _selected_hidden_states(
            source, phraser_key_set)
        if not selected:
            report = _empty_move_report(
                source_path,
                destination_path,
                len(phraser_key_set),
                time.perf_counter() - started,
            )
            if verbose:
                print(
                    f'[{label}] no matching hidden states; skipped',
                    flush=True,
                )
            return report

        if verbose:
            print(
                f'[{label}] selected {len(selected):,} hidden states in '
                f'{len(_selected_by_shard(selected)):,} source shards',
                flush=True,
            )
        destination = Store(
            str(destination_path), max_shard_size_bytes=100_000_000)
        model_names = _register_models(source, destination, selected)
        source_ids = _register_phraser_sources(
            source, destination, selected)
        destination_keys = _copy_and_verify_batches(
            source,
            destination,
            selected,
            batch_size,
            label,
            verbose,
        )
        destination_integrity = _verify_destination(
            destination, selected, destination_keys)
        deletion = _delete_and_compact(
            source, all_metadatas, selected, label, verbose)
        elapsed = time.perf_counter() - started
        report = {
            'status': 'moved',
            'source_path': str(source_path),
            'destination_path': str(destination_path),
            'requested_phraser_key_count': len(phraser_key_set),
            'selected_embedding_count': len(selected),
            'copied_count': len(destination_keys),
            'verified_count': len(destination_keys),
            'model_names': model_names,
            'phraser_source_ids': source_ids,
            'destination_created': True,
            'destination_integrity': destination_integrity,
            'elapsed_seconds': round(elapsed, 6),
            **deletion,
        }
        if verbose:
            print(
                f'[{label}] moved {len(selected):,} embeddings in '
                f'{elapsed:.1f}s',
                flush=True,
            )
        return report
    finally:
        try:
            if destination is not None:
                destination.close()
        finally:
            if source is not None:
                source.close()


def _preflight_destinations(source_paths, flemish_root):
    pairs = [
        (source_path, flemish_destination_path(source_path, flemish_root))
        for source_path in source_paths
    ]
    destination_paths = [destination for _, destination in pairs]
    if len(set(destination_paths)) != len(destination_paths):
        raise ValueError('multiple source stores map to one destination path')
    return pairs


def _skipped_existing_report(source_path, destination_path):
    return {
        'status': 'skipped_existing',
        'source_path': str(_resolved_path(source_path)),
        'destination_path': str(_resolved_path(destination_path)),
        'selected_embedding_count': 0,
        'copied_count': 0,
        'verified_count': 0,
        'deleted_count': 0,
        'affected_shard_count': 0,
        'pure_flemish_shard_count': 0,
        'mixed_shard_count': 0,
        'compacted_shard_count': 0,
        'reason': 'destination store already exists',
    }


def _print_flemish_report(report):
    summary = report['summary']
    print('Flemish embedding move report', flush=True)
    print(
        f'  Stores: {summary["moved_stores"]:,} moved, '
        f'{summary["skipped_existing_stores"]:,} already existing, '
        f'{summary["no_match_stores"]:,} without matches, '
        f'{summary["failed_stores"]:,} failed',
        flush=True,
    )
    print(
        f'  Embeddings: {summary["copied_count"]:,} copied and verified, '
        f'{summary["deleted_count"]:,} deleted from Dutch stores',
        flush=True,
    )
    print(
        f'  Source shards: {summary["pure_flemish_shard_count"]:,} pure '
        f'Flemish, {summary["mixed_shard_count"]:,} mixed, '
        f'{summary["compacted_shard_count"]:,} compacted',
        flush=True,
    )
    print(
        f'  Elapsed: {report["elapsed_seconds"]:.1f}s', flush=True)
    for item in report['stores']:
        detail = item.get('error', '')
        if item['status'] == 'moved':
            detail = f'{item["selected_embedding_count"]:,} embeddings'
        elif item['status'] == 'skipped_existing':
            detail = item['reason']
        print(
            f'  {Path(item["source_path"]).name}: '
            f'{item["status"]} {detail}',
            flush=True,
        )


def move_flemish_data(
    phraser_key_path=default_flemish_phraser_key_path,
    netherlandic_root=default_netherlandic_stores_root,
    flemish_root=default_flemish_stores_root,
    batch_size=default_batch_size,
    verbose=True,
):
    '''Move Flemish-key hidden states from every Dutch model-specific store.

    Existing destination paths are skipped and reported. Individual store
    failures are recorded and later stores continue, so the returned report
    describes the entire run.
    '''
    _validate_batch_size(batch_size)
    started = time.perf_counter()
    phraser_keys = load_flemish_phraser_keys(phraser_key_path)
    source_paths = netherlandic_source_paths(netherlandic_root)
    if not source_paths:
        raise ValueError(
            'no model-specific Netherlandic Echoframe stores were found')
    pairs = _preflight_destinations(source_paths, flemish_root)
    if verbose:
        print(
            f'Moving embeddings for {len(phraser_keys):,} Flemish Phraser '
            f'keys across {len(pairs):,} model stores',
            flush=True,
        )

    store_reports = []
    for index, (source_path, destination_path) in enumerate(pairs, start=1):
        if verbose:
            print(
                f'[{index:,}/{len(pairs):,}] {source_path.name}', flush=True)
        if destination_path.exists() or destination_path.is_symlink():
            item = _skipped_existing_report(
                source_path, destination_path)
            store_reports.append(item)
            if verbose:
                print(
                    f'[{source_path.name}] skipped: {item["reason"]}',
                    flush=True,
                )
            continue
        try:
            item = move_embeddings_based_on_phraser_keys(
                phraser_keys,
                source_path,
                destination_path,
                batch_size=batch_size,
                verbose=verbose,
            )
        except Exception as error:
            item = {
                'status': 'failed',
                'source_path': str(_resolved_path(source_path)),
                'destination_path': str(_resolved_path(destination_path)),
                'selected_embedding_count': 0,
                'copied_count': 0,
                'verified_count': 0,
                'deleted_count': 0,
                'affected_shard_count': 0,
                'pure_flemish_shard_count': 0,
                'mixed_shard_count': 0,
                'compacted_shard_count': 0,
                'error': f'{type(error).__name__}: {error}',
            }
            if verbose:
                print(
                    f'[{source_path.name}] failed: {item["error"]}',
                    flush=True,
                )
        store_reports.append(item)

    summary = {
        'n_stores': len(store_reports),
        'moved_stores': sum(
            item['status'] == 'moved' for item in store_reports),
        'skipped_existing_stores': sum(
            item['status'] == 'skipped_existing'
            for item in store_reports),
        'no_match_stores': sum(
            item['status'] == 'no_matches' for item in store_reports),
        'failed_stores': sum(
            item['status'] == 'failed' for item in store_reports),
        'selected_embedding_count': sum(
            item['selected_embedding_count'] for item in store_reports),
        'copied_count': sum(item['copied_count'] for item in store_reports),
        'verified_count': sum(
            item['verified_count'] for item in store_reports),
        'deleted_count': sum(item['deleted_count'] for item in store_reports),
        'affected_shard_count': sum(
            item['affected_shard_count'] for item in store_reports),
        'pure_flemish_shard_count': sum(
            item['pure_flemish_shard_count'] for item in store_reports),
        'mixed_shard_count': sum(
            item['mixed_shard_count'] for item in store_reports),
        'compacted_shard_count': sum(
            item['compacted_shard_count'] for item in store_reports),
    }
    report = {
        'status': (
            'completed_with_errors' if summary['failed_stores']
            else 'complete'
        ),
        'phraser_key_path': str(_resolved_path(phraser_key_path)),
        'netherlandic_root': str(_resolved_path(netherlandic_root)),
        'flemish_root': str(_resolved_path(flemish_root)),
        'phraser_key_count': len(phraser_keys),
        'stores': store_reports,
        'summary': summary,
        'elapsed_seconds': round(time.perf_counter() - started, 6),
    }
    if verbose:
        _print_flemish_report(report)
    return report
