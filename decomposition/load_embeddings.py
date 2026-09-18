'''Load saved random-frame markers and their Echoframe features.'''

import numpy as np

import locations
import model_store
from decomposition.sampling import load_cgn_store, load_manifest

default_model_name = 'wav2vec2_nl1_checkpoint-200000'
default_phraser_source_id = 'cgn-awd'


def load_embeddings(markers, store, model_name=default_model_name, layer=9,
    collar=2000, to_matrix=False):
    '''Bulk-load an Echoframe Embeddings or CNNFeatures collection.

    Returns CNNFeatures when layer is 'cnn', otherwise Embeddings. Individual
    objects are available through .cnn_features or .embeddings, respectively.
    Uses batched payload reads and preserves the order of retained markers.
    Echoframe warns and skips missing or invalid features; the collection can
    therefore contain fewer items than markers. Empty input or no valid
    features raises ValueError, as do duplicate keys among loaded features.
    Supply unique marker keys. collar is in milliseconds and must match
    extraction. With to_matrix=True, return a NumPy matrix with one row per
    retained marker, averaging its frames through embeddings_to_matrix.
    '''
    keys = [marker.key for marker in markers]
    if layer == 'cnn':
        o = store.phraser_keys_to_cnn_features(keys, model_name, collar=collar)
    else:
        o = store.phraser_keys_to_embeddings(keys, model_name, layer, collar=collar)
    if to_matrix: return embeddings_to_matrix(o)
    return o

def load_embedding(marker, store, model_name=default_model_name, layer=9,
    collar=2000):
    '''Load one stored Embedding, or CNNFeature when layer is 'cnn'.

    collar is in milliseconds and must match extraction. Raises ValueError
    for missing metadata or payload. Returns the full stored feature object
    without pooling or selecting frames; its array is available as .data.
    '''
    if layer == 'cnn':
        o = store.phraser_key_to_cnn_feature(marker.key, model_name, collar=collar)
    else:
        o = store.phraser_key_to_embedding(marker.key, model_name, layer, collar=collar)
    return o

def check_marker_alignment(markers, manifest=None, region='nl',
    components=('comp-k', 'comp-o'), label='decomp_random_frames'):
    '''Check marker count and order against the manifest; return True on success.

    markers:     iterable of markers in the order to check
    manifest:    manifest dictionary; None loads it through sampling
    region:      region passed to load_manifest when manifest is omitted
    components:  components passed to load_manifest when manifest is omitted
    label:       marker label prefix, followed by _{frame_index}

    Compare marker.label and marker.audio.filename against every selected
    frame in audio_infos order. Filenames are compared as strings. Raise
    ValueError on a count mismatch or the first mismatching marker. Does not
    reorder markers. A supplied iterator is consumed by this check.
    '''
    if manifest is None:
        manifest = load_manifest(region=region, components=components)
    markers = list(markers)
    n_expected = sum(len(info['frame_indices'])
        for info in manifest['audio_infos'])
    if len(markers) != n_expected:
        message = f'expected {n_expected} markers, got {len(markers)}'
        raise ValueError(message)
    index = 0
    for info in manifest['audio_infos']:
        for frame_index in info['frame_indices']:
            marker = markers[index]
            expected = (str(info['filename']), f'{label}_{frame_index}')
            actual = (str(marker.audio.filename), marker.label)
            if actual != expected:
                message = f'marker {index}: expected {expected!r}, '
                raise ValueError(message + f'got {actual!r}')
            index += 1
    return True


def embeddings_to_matrix(embeddings):
    '''Convert Embeddings or CNNFeatures to a (markers, dimensions) matrix.

    Average each marker's stored frames, preserving collection order.
    '''
    if hasattr(embeddings, 'embeddings'):
        embeddings = embeddings.embeddings
    else:
        embeddings = embeddings.cnn_features
    return np.stack([embedding.data.mean(axis=0) for embedding in embeddings])


def load_store(model_name=default_model_name, phraser_store=None,
    phraser_source_id=default_phraser_source_id):
    '''Open a model's decomposition store and attach a live Phraser store.

    Opens default CGN when phraser_store is omitted. The caller closes both
    stores: store.close() closes Echoframe only. Retrieve attached Phraser
    with store.load_phraser_store(phraser_source_id) to use or close it.
    '''
    if phraser_store is None: phraser_store = load_cgn()
    root = locations.decomposition_random_frames_echoframe_model_stores
    store = model_store.open_model_store(model_name, stores_root=root)
    store.attach_phraser_store(phraser_source_id, phraser_store)
    return store

def load_markers(phraser_store, label='decomp_random_frames'):
    '''Return saved markers matching the label prefix as a reusable list.'''
    markers = phraser_store.markers.filter(label__startswith=label)
    return list(markers)

def load_cgn():
    '''Open the default CGN Phraser store; the caller closes it.'''
    return load_cgn_store()
