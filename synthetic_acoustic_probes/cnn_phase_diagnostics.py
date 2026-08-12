'''Temporal cancellation diagnostics for stored CNN features.'''

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CNNPhaseDiagnostics:
    '''Frame-aggregation measurements for one stored CNN representation.'''

    stimulus_id: str
    mean_vector: np.ndarray
    middle_vector: np.ndarray
    even_mean_vector: np.ndarray
    odd_mean_vector: np.ndarray
    mean_norm: float
    middle_norm: float
    mean_frame_norm: float
    cancellation_ratio: float | None
    even_mean_norm: float
    odd_mean_norm: float
    even_odd_cosine_distance: float | None


def diagnose_cnn_phase(phrases, model_name, store, *, collar=0):
    '''Measure frame cancellation in stored CNN features for each Phrase.

    phrases:     Ordered iterable of native Phraser Phrase objects.
    model_name:  Registered Echoframe model name.
    store:       Echoframe Store containing the CNN features.
    collar:      Context in milliseconds used during extraction.

    Returns one ``CNNPhaseDiagnostics`` record per Phrase, in input order.
    '''

    phrases = tuple(phrases)
    if not phrases: raise ValueError('phrases must not be empty')

    diagnostics = []
    for phrase in phrases:
        feature = store.phraser_key_to_cnn_feature(
            phrase.key,
            model_name,
            collar=collar,
        )
        diagnostics.append(_diagnose_feature(feature, phrase))
    return tuple(diagnostics)


def _diagnose_feature(feature, phrase):
    frames = feature.data
    mean_vector = np.asarray(
        feature.aggregate_segment(phrase, method='mean')
    )
    middle_vector = np.asarray(
        feature.aggregate_segment(phrase, method='middle')
    )
    even_mean_vector = frames[::2].mean(axis=0)
    odd_mean_vector = frames[1::2].mean(axis=0)

    mean_norm = _norm(mean_vector)
    middle_norm = _norm(middle_vector)
    frame_norms = np.linalg.norm(frames, axis=1)
    mean_frame_norm = float(frame_norms.mean())
    even_mean_norm = _norm(even_mean_vector)
    odd_mean_norm = _norm(odd_mean_vector)

    cancellation_ratio = None
    if mean_frame_norm:
        cancellation_ratio = mean_norm / mean_frame_norm

    return CNNPhaseDiagnostics(
        stimulus_id=phrase.label,
        mean_vector=mean_vector,
        middle_vector=middle_vector,
        even_mean_vector=even_mean_vector,
        odd_mean_vector=odd_mean_vector,
        mean_norm=mean_norm,
        middle_norm=middle_norm,
        mean_frame_norm=mean_frame_norm,
        cancellation_ratio=cancellation_ratio,
        even_mean_norm=even_mean_norm,
        odd_mean_norm=odd_mean_norm,
        even_odd_cosine_distance=_cosine_distance(
            even_mean_vector,
            odd_mean_vector,
        ),
    )


def _norm(vector):
    return float(np.linalg.norm(vector))


def _cosine_distance(left, right):
    denominator = _norm(left) * _norm(right)
    if not denominator: return None
    similarity = float(np.dot(left, right) / denominator)
    return 1 - similarity
