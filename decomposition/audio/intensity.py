'''Measure marker-aligned intensity with Praat's Kaiser-20 window.'''

from functools import lru_cache

import numpy as np
import soundfile


def marker_to_intensity(marker, pitch_floor=100, subtract_mean=True):
    '''Return intensity at the center of a marker's 25 ms MFCC window.

    marker:         marker with recording-relative times in milliseconds
    pitch_floor:    controls the 6.4 / pitch_floor second analysis window
    subtract_mean:  remove the local unweighted mean before windowing

    Return NaN when the full analysis window crosses a recording boundary.
    The result is dB relative to 2e-5 of the waveform's amplitude units;
    digital audio requires calibration before interpreting it as dB SPL.
    '''
    audio = marker.audio
    sample_rate = audio.sample_rate
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('audio must have a positive sample rate')
    if (not np.isfinite(marker.start) or not np.isfinite(marker.end)
            or marker.start < 0 or marker.end <= marker.start
            or marker.end > audio.duration):
        raise ValueError('marker must have valid recording-relative timing')

    start_sample = round(marker.start / 1000 * sample_rate)
    frame_samples = round(0.025 * sample_rate)
    if frame_samples < 1:
        raise ValueError('sample rate is too low for a 25 ms frame')
    center_sample = start_sample + frame_samples // 2
    weights = _window(sample_rate, pitch_floor)
    half_samples = len(weights) // 2
    first = center_sample - half_samples
    if first < 0: return np.nan

    signal, loaded_rate = soundfile.read(audio.filename, start=first,
        stop=center_sample + half_samples + 1, dtype='float64',
        always_2d=True)
    if loaded_rate != sample_rate:
        raise ValueError('audio samples do not match marker sample rate')
    if len(signal) != len(weights): return np.nan
    return praat_intensity(signal, sample_rate, half_samples,
        pitch_floor=pitch_floor, subtract_mean=subtract_mean)


def praat_intensity(signal, sample_rate, center_sample, pitch_floor=100,
    subtract_mean=True):
    '''Return Praat-style intensity at a sample index in a waveform.

    signal:         mono samples or (samples, channels) array
    sample_rate:    waveform sample rate in Hz
    center_sample:  zero-based sample index of the analysis center
    pitch_floor:    controls the 6.4 / pitch_floor second window
    subtract_mean:  remove each channel's local unweighted mean

    Return NaN if the full window is unavailable. Praat's contour frame
    placement is not reproduced; this measures at the requested sample.
    '''
    weights = _window(sample_rate, pitch_floor)
    if (isinstance(center_sample, bool)
            or not isinstance(center_sample, (int, np.integer))):
        raise ValueError('center_sample must be an integer')
    signal = np.asarray(signal)
    if signal.ndim == 1: signal = signal[:, None]
    if signal.ndim != 2 or signal.shape[1] < 1:
        raise ValueError('signal must have samples and at least one channel')
    half_samples = len(weights) // 2
    first = center_sample - half_samples
    last = center_sample + half_samples + 1
    if first < 0 or last > len(signal): return np.nan

    samples = np.asarray(signal[first:last], dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError('analysis samples must be finite')
    if subtract_mean: samples = samples - samples.mean(axis=0)
    mean_square = np.sum(samples**2 * weights[:, None])
    mean_square /= weights.sum() * samples.shape[1]
    ratio = mean_square / (2e-5)**2
    if ratio < 1e-30: return -300.0
    return float(10 * np.log10(ratio))


@lru_cache(maxsize=64)
def _window(sample_rate, pitch_floor):
    '''Return normalized Kaiser-20 weights for Praat's physical window.'''
    if (not np.isfinite(sample_rate) or sample_rate <= 0
            or not np.isfinite(pitch_floor) or pitch_floor <= 0):
        raise ValueError('sample_rate and pitch_floor must be positive')
    window_samples = 3.2 * sample_rate / pitch_floor
    half_samples = int(np.floor(window_samples))
    if half_samples < 1:
        raise ValueError('sample rate is too low for this pitch floor')
    half_duration = 3.2 / pitch_floor
    positions = np.arange(-half_samples, half_samples + 1)
    x = positions / (sample_rate * half_duration)
    root = np.sqrt(np.maximum(0, 1 - x**2))
    weights = np.i0((2 * np.pi**2 + 0.5) * root)
    weights /= weights.max()
    return weights
