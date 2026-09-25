from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from phraser.audio.mfcc import _mfcc_matrix, recording_mfcc

import locations
from decomposition import extract_mfcc


def make_marker(key, start=20, sample_rate=16_000,
    duration=1_000, filename='audio.wav', store=None):
    '''Create a minimal marker with recording metadata.'''
    audio = SimpleNamespace(filename=filename, sample_rate=sample_rate,
        duration=duration)
    return SimpleNamespace(key=key, start=start, end=start + 20,
        audio=audio, store=store)


def make_store(existing=()):
    '''Create a store mock that reports the requested cached features.'''
    store = Mock()
    def make_key(kind, feature_name, phraser_key):
        return kind, feature_name, phraser_key
    def load_metadata(keys, keep_missing):
        result = []
        for key in keys:
            metadata = None
            if key[2] in existing:
                metadata = SimpleNamespace(shape=(1, 39))
            result.append(metadata)
        return result
    store.make_echoframe_key.side_effect = make_key
    store.load_many_metadata.side_effect = load_metadata
    return store


def test_find_unaligned_markers_reports_and_preserves_order(capsys):
    '''Check sample-grid alignment without requiring a complete MFCC window.'''
    aligned = make_marker('aligned', start=20)
    shifted_16k = make_marker('shifted-16k', start=21)
    shifted_22k = make_marker('shifted-22k', start=20,
        sample_rate=22_050)
    late_aligned = make_marker('late-aligned', start=980)
    markers = [aligned, shifted_16k, shifted_22k, late_aligned]

    result = extract_mfcc.find_unaligned_markers(markers)

    assert result == [shifted_16k, shifted_22k]
    assert capsys.readouterr().out == 'Unaligned marker starts: 2 / 4\n'


def test_find_unaligned_markers_accepts_empty_list(capsys):
    '''Report zero without opening audio or a store.'''
    assert extract_mfcc.find_unaligned_markers([]) == []
    assert capsys.readouterr().out == 'Unaligned marker starts: 0 / 0\n'


def test_default_store_root_and_lifetime(monkeypatch):
    '''Use the dedicated location and leave a returned store open.'''
    phraser_store = object()
    marker = make_marker('saved', store=phraser_store)
    store = make_store(existing={'saved'})
    constructor = Mock(return_value=store)
    monkeypatch.setattr(extract_mfcc.echoframe, 'Store', constructor)

    result = extract_mfcc.extract_marker_mfcc([marker], verbose=False)

    assert result is store
    root = locations.decomposition_random_frames_echoframe_mfcc_store
    root_string = str(root)
    constructor.assert_called_once_with(root_string)
    store.attach_phraser_store.assert_called_once_with('cgn-awd', phraser_store)
    store.save_many.assert_not_called()
    store.close.assert_not_called()


def test_aligned_and_shifted_markers_use_their_respective_paths(monkeypatch):
    '''Route each marker to the correct extractor and save its selected row.'''
    phraser_store = object()
    aligned = make_marker('aligned', store=phraser_store)
    shifted = make_marker('shifted', sample_rate=22_050,
        filename='shifted.wav', store=phraser_store)
    store = make_store()
    batch_calls = []
    shifted_calls = []

    def fake_batch(markers, workers, cache_on_segment):
        batch_calls.append((markers, workers, cache_on_segment))
        matrices = []
        for marker in markers:
            _, n_rows = extract_mfcc._recording_grid_row(marker)
            rows = np.arange(n_rows)[:, None]
            matrix = np.repeat(rows, 39, axis=1)
            matrices.append(matrix)
        return matrices

    def fake_shifted(marker):
        shifted_calls.append(marker)
        return np.full((1, 39), 3.0)

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_batch)
    monkeypatch.setattr(extract_mfcc, '_marker_start_mfcc', fake_shifted)
    monkeypatch.setattr(extract_mfcc, 'make_acoustic_feature_item',
        lambda key, name, data, store, **kwargs: (key, data))

    extract_mfcc.extract_marker_mfcc([aligned, shifted], store=store,
        workers=3, verbose=False)

    assert batch_calls == [([aligned], 3, False)]
    assert shifted_calls == [shifted]
    items = store.save_many.call_args.args[0]
    assert [item[0][2] for item in items] == ['aligned', 'shifted']
    expected_aligned = np.full((1, 39), 1.0)
    expected_shifted = np.full((1, 39), 3.0)
    np.testing.assert_array_equal(items[0][1], expected_aligned)
    np.testing.assert_array_equal(items[1][1], expected_shifted)


def test_aligned_marker_selects_recording_frame(tmp_path, monkeypatch):
    '''Match the marker row to the full recording MFCC grid.'''
    sample_rate = 16_000
    seconds = np.arange(sample_rate) / sample_rate
    signal = np.sin(2 * np.pi * (200 + 300 * seconds) * seconds)
    filename = tmp_path / 'aligned.wav'
    sf.write(filename, signal, sample_rate)
    marker = make_marker('aligned', start=200, filename=filename)
    store = make_store()
    monkeypatch.setattr(extract_mfcc, 'make_acoustic_feature_item',
        lambda key, name, data, store, **kwargs: (key, data))

    extract_mfcc.extract_marker_mfcc([marker], store=store,
        workers=1, verbose=False)

    actual = store.save_many.call_args.args[0][0][1]
    expected = recording_mfcc(marker.audio)[20:21]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)


def test_shifted_marker_uses_audio_on_both_sides_for_deltas(tmp_path):
    '''Anchor an off-grid window and include neighboring audio for deltas.'''
    sample_rate = 22_050
    seconds = np.arange(sample_rate) / sample_rate
    signal = (0.4 * np.sin(2 * np.pi * 280 * seconds)
        + 0.2 * np.sin(2 * np.pi * 1_200 * seconds))
    filename = tmp_path / 'changing.wav'
    sf.write(filename, signal, sample_rate)
    recorded_signal, _ = sf.read(filename)
    marker = make_marker('shifted', start=200, sample_rate=sample_rate,
        filename=filename)

    result = extract_mfcc._marker_start_mfcc(marker)

    start = round(marker.start / 1000 * sample_rate)
    hop = round(0.01 * sample_rate)
    window = round(0.025 * sample_rate)
    expected = _mfcc_matrix(recorded_signal[
        start - 4 * hop:start + window + 4 * hop], sample_rate)[:, 4]
    assert result.shape == (1, 39)
    np.testing.assert_allclose(result[0], expected, rtol=1e-4, atol=1e-3)


def test_rejects_marker_without_complete_window():
    '''Reject a start too close to the end of the recording.'''
    marker = make_marker('too-close', start=980)
    store = make_store()

    with pytest.raises(ValueError, match='complete MFCC window'):
        extract_mfcc.extract_marker_mfcc([marker], store=store, verbose=False)


def test_owned_store_closes_after_extraction_error(monkeypatch):
    '''Close an internally opened store if extraction fails.'''
    phraser_store = object()
    marker = make_marker('missing', store=phraser_store)
    store = make_store()
    constructor = Mock(return_value=store)
    failure = RuntimeError('compute failed')
    compute = Mock(side_effect=failure)
    monkeypatch.setattr(extract_mfcc.echoframe, 'Store', constructor)
    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', compute)

    with pytest.raises(RuntimeError, match='compute failed'):
        extract_mfcc.extract_marker_mfcc([marker], verbose=False)

    store.close.assert_called_once_with()
