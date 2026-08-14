import echoframe
from echoframe.acoustic_features import store_mfcc_batch
from phraser.audio.batch import mfcc_batch

import locations

default_phraser_source_id = 'cgn-awd'


def _notify(verbose, message):
    if verbose:
        print(f'[mfcc] {message}', flush=True)


def _counted(count, noun):
    suffix = '' if count == 1 else 's'
    return f'{count} {noun}{suffix}'


def _recording_batches(segments, recordings_per_batch):
    groups = {}
    for segment in segments:
        audio = segment.audio
        key = str(audio.filename), audio.sample_rate, audio.duration
        groups.setdefault(key, []).append(segment)
    groups = list(groups.values())
    total_recordings = len(groups)
    total_batches = (
        total_recordings + recordings_per_batch - 1
    ) // recordings_per_batch
    for start in range(0, len(groups), recordings_per_batch):
        selected = groups[start:start + recordings_per_batch]
        batch = [segment for group in selected for segment in group]
        batch_number = start // recordings_per_batch + 1
        yield (
            batch_number, total_batches, len(selected), total_recordings,
            batch,
        )


def _missing_mfcc_segments(segments, store):
    keys = [
        store.make_echoframe_key(
            'acoustic_feature', feature_name='mfcc',
            phraser_key=segment.key)
        for segment in segments
    ]
    metadatas = store.load_many_metadata(keys, keep_missing=True)
    return [
        segment for segment, metadata in zip(segments, metadatas, strict=True)
        if metadata is None
    ]


def extract_phone_mfcc(
    phones,
    store=None,
    phraser_source_id=default_phraser_source_id,
    workers=None,
    recordings_per_batch=30,
    keep_segment_cache=False,
    tags=None,
    verbose=True,
):
    '''Compute and store MFCCs for every phone in `phones` into an echoframe
    Store, via phraser's mfcc_batch (parallel, grouped per recording) and
    echoframe's store_mfcc_batch.

    phones:             probing.metadata.Phones - phones.phraser_phones must
                        be complete (raises otherwise)
    store:              existing echoframe.Store to write into; if None, one
                        is opened at locations.echoframe_mfcc_store
    phraser_source_id:  label to register phones.store under in this store
    workers:            max worker processes for mfcc_batch; defaults to
                        available CPUs
    recordings_per_batch:  number of audio recordings computed/stored at
                        once (default 30)
    keep_segment_cache: retain computed matrices as segment._mfcc after they
                        have been stored (default False)
    tags:               optional tags stored on new metadata
    verbose:            print batch progress
    '''
    if (
        isinstance(recordings_per_batch, bool)
        or not isinstance(recordings_per_batch, int)
    ):
        raise TypeError('recordings_per_batch must be a positive integer')
    if recordings_per_batch <= 0:
        raise ValueError('recordings_per_batch must be a positive integer')
    if not isinstance(keep_segment_cache, bool):
        raise TypeError('keep_segment_cache must be a boolean')
    if store is None:
        store_root = locations.echoframe_mfcc_store
        _notify(verbose, f'opening Echoframe store at {store_root}')
        store = echoframe.Store(str(store_root))
    _notify(
        verbose, f'attaching Phraser source {phraser_source_id!r}')
    store.attach_phraser_store(phraser_source_id, phones.store)

    _notify(verbose, 'loading matched Phraser phones')
    segments = phones.phraser_phones
    _notify(verbose, f'loaded {_counted(len(segments), "matched phone")}')
    if not segments:
        _notify(verbose, 'no phones to process')
        return store

    _notify(
        verbose,
        'grouping phones into batches of '
        f'{_counted(recordings_per_batch, "recording")}',
    )
    already_stored = 0
    stored = 0
    groups_announced = False
    batches = _recording_batches(segments, recordings_per_batch)
    for (
        batch_number, total_batches, recording_count, total_recordings, batch,
    ) in batches:
        if not groups_announced:
            _notify(
                verbose,
                f'grouped phones from '
                f'{_counted(total_recordings, "recording")} into '
                f'{_counted(total_batches, "batch")}',
            )
            groups_announced = True
        stage = f'batch {batch_number}/{total_batches}'
        _notify(
            verbose,
            f'{stage}: checking {_counted(len(batch), "phone")} from '
            f'{_counted(recording_count, "recording")} against the '
            'Echoframe store',
        )
        missing = _missing_mfcc_segments(batch, store)
        present_count = len(batch) - len(missing)
        already_stored += present_count
        if not missing:
            _notify(
                verbose,
                f'{stage}: all {_counted(len(batch), "MFCC")} already stored',
            )
            continue
        worker_description = (
            'available CPU cores' if workers is None
            else f'up to {workers} workers'
        )
        _notify(
            verbose,
            f'{stage}: computing {_counted(len(missing), "missing MFCC")} '
            'using '
            f'{worker_description}',
        )
        try:
            mfcc_batch(missing, workers=workers, cache_on_segment=True)
            _notify(
                verbose,
                f'{stage}: computation finished; preparing and writing '
                f'{_counted(len(missing), "MFCC")}',
            )
            store_mfcc_batch(
                missing, store, tags=tags, verbose=verbose)
            stored += len(missing)
        finally:
            if not keep_segment_cache:
                for segment in missing:
                    segment.__dict__.pop('_mfcc', None)
        cache_status = (
            'retained segment cache' if keep_segment_cache
            else 'cleared temporary segment cache'
        )
        _notify(
            verbose,
            f'{stage}: complete; stored {_counted(len(missing), "MFCC")} and '
            f'{cache_status}',
        )
    _notify(
        verbose,
        f'complete: stored {_counted(stored, "MFCC")}; skipped '
        f'{_counted(already_stored, "MFCC")} already present',
    )
    return store
