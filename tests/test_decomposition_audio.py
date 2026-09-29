'''Check marker acoustic measurements and their SQLite alignment.'''

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from decomposition.audio import database, extract, frequency_band, intensity


def make_marker(key, filename, start=100, sample_rate=16_000,
    duration=500):
    '''Make a marker with recording-relative times in milliseconds.

    key:          marker's Phraser-style byte key
    filename:     audio file path
    start:        frame start in milliseconds
    sample_rate:  audio samples per second
    duration:     recording duration in milliseconds
    '''
    audio = SimpleNamespace(filename=filename, sample_rate=sample_rate,
        duration=duration)
    return SimpleNamespace(key=key, start=start, end=start + 25, audio=audio)


def write_tone(filename, sample_rate=16_000, frequency=200):
    '''Write half a second of a full-scale-compatible sine wave.

    filename:     output WAV path
    sample_rate:  audio samples per second
    frequency:    sine frequency in Hz
    '''
    times = np.arange(sample_rate // 2) / sample_rate
    signal = 0.5 * np.sin(2 * np.pi * frequency * times)
    soundfile.write(filename, signal, sample_rate, subtype='FLOAT')


def test_intensity_uses_local_mean_and_reports_unavailable_context():
    '''Constant pressure survives only when local mean removal is disabled.'''
    signal = np.full(2_000, 0.5, dtype=np.float32)
    measured = intensity.praat_intensity(signal, 16_000, 1_000)
    raw = intensity.praat_intensity(signal, 16_000, 1_000,
        subtract_mean=False)

    assert measured == -300.0
    assert raw == pytest.approx(10 * np.log10(0.25 / (2e-5) ** 2))
    boundary_value = intensity.praat_intensity(signal, 16_000, 0)
    assert np.isnan(boundary_value)


def test_frequency_bands_keep_the_tone_in_its_band():
    '''A 200 Hz tone puts most 25 ms frame power below 500 Hz.'''
    sample_rate = 16_000
    times = np.arange(400) / sample_rate
    signal = np.sin(2 * np.pi * 200 * times)

    powers = frequency_band.frequency_band_power(signal, sample_rate)

    assert powers.shape == (4,)
    assert powers[0] == pytest.approx(0.5, rel=0.1)
    assert powers[0] > 100 * powers[1:].sum()
    silence = np.zeros(400)
    silent_powers = frequency_band.frequency_band_power(silence, sample_rate)
    expected_silence = np.zeros(4)
    np.testing.assert_array_equal(silent_powers, expected_silence)


def test_default_worker_count_is_sixteen(tmp_path, monkeypatch):
    '''Use 16 worker processes unless the caller specifies otherwise.

    tmp_path:     pytest temporary directory
    monkeypatch:  pytest patch fixture
    '''
    calls = []
    monkeypatch.setattr(extract, '_process_tasks',
        lambda tasks, path, workers, verbose: calls.append(workers))
    marker = make_marker(b'a' * 22, tmp_path / 'unused.wav')

    extract.extract_marker_acoustics([marker],
        database=tmp_path / 'acoustics.sqlite', verbose=False)

    assert calls == [16]


def test_loader_preserves_marker_order_and_missing_intensity(tmp_path):
    '''Match rows by key, independently of their SQLite insertion order.'''
    path = tmp_path / 'acoustics.sqlite'
    with sqlite3.connect(path) as connection:
        database._prepare_database(connection, {}, overwrite=False)
        connection.executemany('''INSERT INTO marker_acoustics
            VALUES (?, ?, ?, ?, ?, ?)''', [
            (b'a' * 22, None, 1.0, 2.0, 3.0, 4.0),
            (b'b' * 22, 65.0, 5.0, 6.0, 7.0, 8.0),
        ])
    marker_a = make_marker(b'a' * 22, 'unused.wav')
    marker_b = make_marker(b'b' * 22, 'unused.wav')

    values = database.load_marker_acoustics([marker_b, marker_a], path)

    assert values['intensity_db'][0] == 65.0
    assert np.isnan(values['intensity_db'][1])
    np.testing.assert_array_equal(values['power_0_500'], [5.0, 1.0])
    with pytest.raises(ValueError, match='missing acoustics'):
        database.load_marker_acoustics([
            make_marker(b'c' * 22, 'unused.wav')], path)


@pytest.mark.multicore
def test_extraction_resumes_and_overwrites_in_the_same_file(tmp_path):
    '''Read two markers from one recording and replace rows when requested.'''
    filename = tmp_path / 'tone.wav'
    write_tone(filename)
    edge = make_marker(b'a' * 22, filename, start=0)
    middle = make_marker(b'b' * 22, filename, start=101)
    markers = [edge, middle]
    path = tmp_path / 'acoustics.sqlite'

    extract.extract_marker_acoustics(markers, path, workers=1, verbose=False)
    original = database.load_marker_acoustics(markers, path)
    assert np.isnan(original['intensity_db'][0])
    assert np.isfinite(original['intensity_db'][1])
    assert original['power_0_500'][1] > 0
    expected_intensity = intensity.marker_to_intensity(middle)
    expected_bands = frequency_band.marker_to_frequency_bands(middle)
    assert original['intensity_db'][1] == pytest.approx(
        expected_intensity, rel=1e-5)
    assert 10 * np.log10(original['power_0_500'][1]) == pytest.approx(
        expected_bands[0], rel=1e-5)

    silence = np.zeros(8_000)
    soundfile.write(filename, silence, 16_000, subtype='FLOAT')
    extract.extract_marker_acoustics(markers, path, workers=1, verbose=False)
    retained = database.load_marker_acoustics(markers, path)
    np.testing.assert_array_equal(retained['power_0_500'],
        original['power_0_500'])

    extract.extract_marker_acoustics(markers, path, workers=1,
        overwrite=True, verbose=False)
    replaced = database.load_marker_acoustics(markers, path)
    assert replaced['intensity_db'][1] == -300.0
    np.testing.assert_array_equal(replaced['power_0_500'], [0.0, 0.0])
    with sqlite3.connect(path) as connection:
        count = connection.execute(
            'SELECT COUNT(*) FROM marker_acoustics').fetchone()[0]
        settings_text = connection.execute(
            "SELECT value FROM metadata WHERE key = 'settings'").fetchone()[0]
    assert count == 2
    assert json.loads(settings_text)['pitch_floor'] == 100.0


@pytest.mark.multicore
def test_changed_settings_recompute_across_recordings(tmp_path):
    '''Reset the same database when pitch settings change with two workers.'''
    first_file = tmp_path / 'first.wav'
    second_file = tmp_path / 'second.wav'
    write_tone(first_file)
    write_tone(second_file, frequency=800)
    markers = [make_marker(b'a' * 22, first_file),
        make_marker(b'b' * 22, second_file)]
    path = tmp_path / 'acoustics.sqlite'

    extract.extract_marker_acoustics(markers, path, workers=2, verbose=False)
    original = database.load_marker_acoustics(markers, path)
    silence = np.zeros(8_000)
    soundfile.write(first_file, silence, 16_000, subtype='FLOAT')
    extract.extract_marker_acoustics(markers, path, workers=2,
        pitch_floor=200, verbose=False)
    updated = database.load_marker_acoustics(markers, path)

    assert original['power_0_500'][0] > 0
    assert updated['power_0_500'][0] == 0
    assert updated['power_500_1000'][1] > 0
    with sqlite3.connect(path) as connection:
        settings_text = connection.execute(
            "SELECT value FROM metadata WHERE key = 'settings'").fetchone()[0]
    assert json.loads(settings_text)['pitch_floor'] == 200.0
