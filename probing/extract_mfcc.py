import echoframe
from echoframe.acoustic_features import store_mfcc_batch
from phraser.audio.batch import mfcc_batch

from probing.metadata import _data_dir

default_store_root = _data_dir / 'echoframe_mfcc_store'
default_phraser_source_id = 'cgn-awd'


def extract_phone_mfcc(
    phones,
    store=None,
    store_root=default_store_root,
    phraser_source_id=default_phraser_source_id,
    workers=None,
    cache_on_segment=True,
    tags=None,
    verbose=True,
):
    '''Compute and store MFCCs for every phone in `phones` into an echoframe
    Store, via phraser's mfcc_batch (parallel, grouped per recording) and
    echoframe's store_mfcc_batch.

    phones:             probing.metadata.Phones - phones.phraser_phones must
                        be complete (raises otherwise)
    store:              existing echoframe.Store to write into; if None, one
                        is opened at store_root
    store_root:         path for a new echoframe.Store, used only when store
                        is None
    phraser_source_id:  label to register phones.store under in this store
    workers:            max worker processes for mfcc_batch; defaults to
                        available CPUs
    cache_on_segment:   passed through to mfcc_batch - read and populate
                        segment._mfcc for the wav2vec2-aligned 20 ms grid
    tags:               optional tags stored on new metadata
    verbose:            print batch progress
    '''
    if store is None:
        store = echoframe.Store(str(store_root))
    store.attach_phraser_store(phraser_source_id, phones.store)

    segments = phones.phraser_phones
    mfcc_batch(segments, workers=workers, cache_on_segment=cache_on_segment)
    store_mfcc_batch(segments, store, tags=tags, verbose=verbose)
    return store
