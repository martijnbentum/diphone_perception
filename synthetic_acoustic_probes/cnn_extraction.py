'''Sequential CNN extraction for synthetic-stimulus experiments.'''

import echoframe.segment_features as segment_features
from progressbar import progressbar


def extract_cnn_checkpoint(phrases, model_name, store, *, collar=0,
    gpu=False, overwrite=False):
    '''Extract and store CNN features for one registered checkpoint.

    phrases:     Iterable of native Phraser Phrase objects.
    model_name:  Registered Echoframe model name.
    store:       Echoframe Store receiving the CNN features.
    collar:      Context in milliseconds around each Phrase.
    gpu:         Whether Echoframe should run the model on a GPU.
    overwrite:   Whether Echoframe should replace stored CNN features.

    Extraction is sequential and releases the cached model before returning.
    Returns None.
    '''
    phrases = tuple(phrases)
    if not phrases: raise ValueError('phrases must not be empty')
    try:
        for phrase in progressbar(phrases):
            segment_features.compute_cnn(phrase, model_name, store,
                collar=collar, gpu=gpu, overwrite=overwrite)
    finally:
        store.remove_cached_model()


def extract_cnn_checkpoints(phrases, model_names, store, *, collar=0,
    gpu=False, overwrite=False):
    '''Extract and store CNN features for registered checkpoints in order.

    phrases:     Iterable of native Phraser Phrase objects.
    model_names:  Iterable of registered Echoframe model names.
    store:       Echoframe Store receiving the CNN features.
    collar:      Context in milliseconds around each Phrase.
    gpu:         Whether Echoframe should run models on a GPU.
    overwrite:   Whether Echoframe should replace stored CNN features.

    Existing features are skipped by Echoframe when overwrite is false, so a
    failed run can be resumed by calling this function again. Returns None.
    '''
    phrases = tuple(phrases)
    if not phrases: raise ValueError('phrases must not be empty')
    if isinstance(model_names, str):
        raise TypeError('model_names must be an iterable of strings')
    try: model_names = tuple(model_names)
    except TypeError as error:
        message = 'model_names must be an iterable of strings'
        raise TypeError(message) from error
    if not model_names: raise ValueError('model_names must not be empty')

    for model_name in progressbar(model_names):
        extract_cnn_checkpoint(phrases, model_name, store, collar=collar,
            gpu=gpu, overwrite=overwrite)
