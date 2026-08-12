from types import SimpleNamespace

import numpy as np
import pytest

import locations
from synthetic_acoustic_probes import cnn_phase_diagnostics
from synthetic_acoustic_probes.storage import write_stimuli


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


class _ClosableStore:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_phase_diagnostic_stimuli_cover_five_sample_alignments():
    '''The default panel contains 34 frequencies by five input offsets.'''

    stimuli = cnn_phase_diagnostics.phase_diagnostic_stimuli()

    assert len(stimuli) == 170
    assert len({stimulus.stimulus_id for stimulus in stimuli}) == 170
    frequencies = tuple(dict.fromkeys(
        stimulus.parameters['frequencies_hz'][0] for stimulus in stimuli
    ))
    assert frequencies == (
        3190, 3200, 3210,
        3490, 3500, 3510,
        3590, 3600, 3610,
        *range(3950, 4060, 10),
        *range(4750, 4860, 10),
        6390, 6400, 6410,
    )
    first_five = stimuli[:5]
    assert [
        stimulus.parameters['sample_offset'] for stimulus in first_five
    ] == list(range(5))
    assert all(
        stimulus.parameters['frequencies_hz'] == [3190]
        for stimulus in first_five
    )


def test_phase_diagnostic_sample_offset_advances_waveform():
    '''Offset one is the original sinusoid advanced by one input sample.'''

    original, advanced = cnn_phase_diagnostics.phase_diagnostic_stimuli(
        frequencies=(3200,),
        sample_offsets=(0, 1),
    )

    expected_phase = 2 * np.pi * 3200 / 16_000
    assert advanced.parameters['phases_radians'] == [expected_phase]
    np.testing.assert_allclose(
        advanced.waveform[:-1],
        original.waveform[1:],
        atol=1e-6,
    )


def test_create_phase_diagnostic_stimuli_writes_default_panel(monkeypatch):
    '''Stimulus creation writes the generated panel to the requested path.'''

    expected = (object(),)
    calls = []
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'phase_diagnostic_stimuli',
        lambda: expected,
    )
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'write_stimuli',
        lambda stimuli, path, overwrite: calls.append(
            (stimuli, path, overwrite)
        ),
    )

    result = cnn_phase_diagnostics.create_phase_diagnostic_stimuli()

    assert result is expected
    assert calls == [(
        expected,
        locations.f0_phase_diagnostic_stimuli,
        False,
    )]


def test_create_phase_diagnostic_stores(monkeypatch, tmp_path):
    '''Phraser and one-model Echoframe stores are created and attached.'''

    phraser_store = _ClosableStore()
    phraser_store.path = tmp_path / 'phraser'
    add_calls = []
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'Store',
        lambda path: phraser_store,
    )
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'add_stimuli',
        lambda package, store: add_calls.append((package, store)),
    )
    package = tmp_path / 'stimuli'

    created_phraser = (
        cnn_phase_diagnostics.create_phase_diagnostic_phraser_store(
            stimulus_package=package,
            store_path=phraser_store.path,
        )
    )

    assert created_phraser is phraser_store
    assert add_calls == [(package, phraser_store)]

    echoframe_store = _ClosableStore()
    attach_calls = []
    echoframe_store.attach_phraser_store = (
        lambda source_id, store: attach_calls.append((source_id, store))
    )
    create_calls = []

    def fake_create_store(path, model_names, model_paths_file):
        create_calls.append((path, model_names, model_paths_file))
        return echoframe_store

    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'create_store',
        fake_create_store,
    )
    model_paths = tmp_path / 'models.json'
    store_path = tmp_path / 'echoframe'

    created_echoframe = (
        cnn_phase_diagnostics.create_phase_diagnostic_echoframe_store(
            phraser_store,
            store_path=store_path,
            model_name='checkpoint',
            model_paths_file=model_paths,
        )
    )

    assert created_echoframe is echoframe_store
    assert create_calls == [(store_path, ('checkpoint',), model_paths)]
    assert attach_calls == [(
        cnn_phase_diagnostics.PHASE_DIAGNOSTIC_PHRASER_SOURCE_ID,
        phraser_store,
    )]


def test_run_phase_diagnostics_saves_offset_zero_comparisons(
    tmp_path,
    monkeypatch,
):
    '''Saved results follow manifest order and compare only within frequency.'''

    package = tmp_path / 'stimuli'
    stimuli = cnn_phase_diagnostics.phase_diagnostic_stimuli(
        frequencies=(3200,),
        sample_offsets=(0, 1),
    )
    write_stimuli(stimuli, package)
    zero = SimpleNamespace(key=b'zero', label=stimuli[0].stimulus_id)
    one = SimpleNamespace(key=b'one', label=stimuli[1].stimulus_id)
    features = {
        zero.key: _CNNFeature([[1, 0], [1, 0]], [1, 0], [1, 0]),
        one.key: _CNNFeature([[0, 1], [0, 1]], [0, 1], [0, 1]),
    }
    store = _Store(features)
    store.load_phraser_store = lambda source_id: object()
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'load_stimuli',
        lambda store: (one, zero),
    )
    output_path = tmp_path / 'diagnostics.npz'

    diagnostics = cnn_phase_diagnostics.run_phase_diagnostics(
        store,
        model_name='checkpoint',
        stimulus_package=package,
        output_path=output_path,
    )

    assert [item.stimulus_id for item in diagnostics] == [
        zero.label,
        one.label,
    ]
    with np.load(output_path, allow_pickle=False) as saved:
        assert saved['stimulus_ids'].tolist() == [zero.label, one.label]
        assert saved['sample_offsets'].tolist() == [0, 1]
        np.testing.assert_allclose(
            saved['mean_offset_zero_cosine_distances'],
            [0, 1],
        )
        np.testing.assert_allclose(
            saved['mean_offset_zero_euclidean_distances'],
            [0, np.sqrt(2)],
        )
        assert saved['model_name'] == 'checkpoint'
        assert saved['collar_ms'] == 0


def test_run_phase_diagnostic_experiment_orchestrates_and_closes(
    tmp_path,
    monkeypatch,
):
    '''The one-call runner creates, extracts, diagnoses, and closes stores.'''

    phraser_store = _ClosableStore()
    echoframe_store = _ClosableStore()
    expected = (object(),)
    calls = []
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'create_phase_diagnostic_stimuli',
        lambda **kwargs: calls.append(('stimuli', kwargs)),
    )

    def fake_create_phraser(**kwargs):
        calls.append(('phraser', kwargs))
        return phraser_store

    def fake_create_echoframe(store, **kwargs):
        calls.append(('echoframe', store, kwargs))
        return echoframe_store

    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'create_phase_diagnostic_phraser_store',
        fake_create_phraser,
    )
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'create_phase_diagnostic_echoframe_store',
        fake_create_echoframe,
    )
    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'extract_phase_diagnostic_cnn_features',
        lambda store, **kwargs: calls.append(('extract', store, kwargs)),
    )

    def fake_run(store, **kwargs):
        calls.append(('diagnose', store, kwargs))
        return expected

    monkeypatch.setattr(
        cnn_phase_diagnostics,
        'run_phase_diagnostics',
        fake_run,
    )

    result = cnn_phase_diagnostics.run_phase_diagnostic_experiment(
        output_root=tmp_path / 'phase_diagnostics',
        model_name='checkpoint',
        model_paths_file=tmp_path / 'models.json',
        gpu=True,
    )

    assert result is expected
    assert [call[0] for call in calls] == [
        'stimuli', 'phraser', 'echoframe', 'extract', 'diagnose'
    ]
    assert phraser_store.closed is True
    assert echoframe_store.closed is True


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
