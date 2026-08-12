from types import SimpleNamespace

import numpy as np
import pytest

from synthetic_acoustic_probes import cnn_phase_diagnostics


class _CNNFeature:
    def __init__(self, data, mean, middle):
        self.data = np.asarray(data, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.middle = np.asarray(middle, dtype=float)
        self.calls = []

    def aggregate_segment(self, phrase, method='mean'):
        self.calls.append((phrase, method))
        if method == 'middle': return self.middle
        return self.mean


class _Store:
    def __init__(self, features):
        self.features = features
        self.calls = []

    def phraser_key_to_cnn_feature(self, key, model_name, collar=0):
        self.calls.append((key, model_name, collar))
        return self.features[key]


def test_diagnose_cnn_phase_measures_frame_cancellation():
    '''Mean, middle, and alternating-frame measurements are returned.'''

    phrase = SimpleNamespace(key=b'a', label='pure-tone_f-3200')
    feature = _CNNFeature(
        data=[[2, 0], [-2, 0], [2, 0], [-2, 0]],
        mean=[0, 0],
        middle=[7, 0],
    )
    store = _Store({phrase.key: feature})

    result = cnn_phase_diagnostics.diagnose_cnn_phase(
        [phrase],
        'checkpoint',
        store,
        collar=25,
    )[0]

    assert store.calls == [(b'a', 'checkpoint', 25)]
    assert feature.calls == [(phrase, 'mean'), (phrase, 'middle')]
    assert result.stimulus_id == 'pure-tone_f-3200'
    np.testing.assert_array_equal(result.mean_vector, [0, 0])
    np.testing.assert_array_equal(result.middle_vector, [7, 0])
    np.testing.assert_array_equal(result.even_mean_vector, [2, 0])
    np.testing.assert_array_equal(result.odd_mean_vector, [-2, 0])
    assert result.mean_norm == 0
    assert result.middle_norm == 7
    assert result.mean_frame_norm == 2
    assert result.cancellation_ratio == 0
    assert result.even_mean_norm == 2
    assert result.odd_mean_norm == 2
    assert result.even_odd_cosine_distance == pytest.approx(2)


def test_diagnose_cnn_phase_preserves_phrase_order():
    '''Each Phrase loads its own payload and retains caller order.'''

    first = SimpleNamespace(key=b'b', label='stimulus-b')
    second = SimpleNamespace(key=b'a', label='stimulus-a')
    features = {
        b'a': _CNNFeature([[1], [1]], [1], [1]),
        b'b': _CNNFeature([[2], [2]], [2], [2]),
    }
    store = _Store(features)

    results = cnn_phase_diagnostics.diagnose_cnn_phase(
        (first, second),
        'checkpoint',
        store,
    )

    assert [result.stimulus_id for result in results] == [
        'stimulus-b', 'stimulus-a'
    ]
    assert store.calls == [
        (b'b', 'checkpoint', 0),
        (b'a', 'checkpoint', 0),
    ]


def test_diagnose_cnn_phase_reports_undefined_zero_norm_metrics():
    '''Zero frame norms make ratios and cosine distances undefined.'''

    phrase = SimpleNamespace(key=b'a', label='silence')
    feature = _CNNFeature(
        data=np.zeros((4, 2)),
        mean=np.zeros(2),
        middle=np.zeros(2),
    )
    store = _Store({phrase.key: feature})

    result = cnn_phase_diagnostics.diagnose_cnn_phase(
        [phrase], 'checkpoint', store
    )[0]

    assert result.cancellation_ratio is None
    assert result.even_odd_cosine_distance is None


def test_diagnose_cnn_phase_rejects_empty_phrases():
    '''At least one Phrase is required.'''

    with pytest.raises(ValueError, match='phrases must not be empty'):
        cnn_phase_diagnostics.diagnose_cnn_phase((), 'checkpoint', object())
