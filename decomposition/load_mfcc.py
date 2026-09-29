'''Load stored random-frame marker MFCCs from Echoframe.'''

import echoframe
import numpy as np
from phraser.audio.mfcc import FEATURE_DIM

import locations


def load_mfcc(markers, store=None):
    '''Load stored MFCCs as a (markers, 39) matrix in marker order.

    markers:  iterable of saved markers whose MFCCs were extracted
    store:    open Echoframe store; None opens the decomposition MFCC store

    Raise ValueError for empty input or missing or malformed stored rows.
    A store opened here is closed before returning; a supplied store stays
    open. The markers' Phraser store is not opened or closed.
    '''
    markers = list(markers)
    if not markers: raise ValueError('markers must not be empty')
    owns_store = store is None
    if owns_store:
        root = locations.decomposition_random_frames_echoframe_mfcc_store
        store = echoframe.Store(str(root))
    try:
        keys = [store.make_echoframe_key('acoustic_feature',
            feature_name='mfcc', phraser_key=marker.key)
            for marker in markers]
        vectors = store.load_many_frames(keys, frame='center',
            keep_missing=True)
        for index, vector in enumerate(vectors):
            if vector is None:
                raise ValueError(f'missing MFCC for marker {index}')
            if np.shape(vector) != (FEATURE_DIM,):
                raise ValueError(f'invalid MFCC shape for marker {index}')
        return np.stack(vectors)
    finally:
        if owns_store: store.close()
