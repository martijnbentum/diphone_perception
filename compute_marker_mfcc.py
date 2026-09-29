'''Compute random-frame marker MFCCs in the configured Echoframe store.

Move the existing MFCC store aside before the first run. Re-running this
script then skips markers whose MFCCs are already stored.
'''

import echoframe

import locations
from decomposition.extract_mfcc import extract_marker_mfcc
from decomposition.load_embeddings import load_cgn, load_markers


def main():
    '''Extract MFCCs for all saved random-frame markers.'''
    cgn = load_cgn()
    try:
        markers = load_markers(cgn)
        root = locations.decomposition_random_frames_echoframe_mfcc_store
        store = echoframe.Store(str(root), max_shard_items=20_000)
        try:
            print(f'Extracting MFCCs for {len(markers)} markers', flush=True)
            extract_marker_mfcc(markers, store=store, workers=16)
            print('MFCC extraction complete', flush=True)
        finally:
            store.close()
    finally:
        cgn.close()


if __name__ == '__main__':
    main()
