'''Measure energy in four frequency bands of a marker's 25 ms frame.'''

import numpy as np
from phraser.audio.audio import load_audio_samples
from phraser.audio.mfcc import WINDOW_SECONDS


BANDS_HZ = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000))


def frequency_band_power(signal, sample_rate):
    '''Return mean-square power in the four bands as a NumPy array.

    signal:       one-dimensional audio samples from a single frame
    sample_rate:  samples per second; Nyquist must be at least 4000 Hz

    Bands are 0–500, 500–1000, 1000–2000, and 2000–4000 Hz. The final
    band includes 4000 Hz; each other band's upper edge is exclusive.
    A Hann window limits leakage. Parseval normalization by the window's
    squared sum makes power across all FFT bins equal the frame's
    window-weighted mean-square power after removal of its DC offset.
    The four bands omit power above 4000 Hz.
    '''
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 3:
        raise ValueError('signal must be a one-dimensional frame')
    if not np.all(np.isfinite(signal)):
        raise ValueError('signal must contain finite samples')
    if not np.isfinite(sample_rate) or sample_rate < 8000:
        raise ValueError('sample_rate must be at least 8000 Hz')

    signal = signal - signal.mean()
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window)
    sample_spacing = 1 / sample_rate
    n_samples = len(signal)
    frequencies = np.fft.rfftfreq(n_samples, sample_spacing)
    power = np.abs(spectrum) ** 2
    if len(signal) % 2:
        power[1:] *= 2
    else:
        power[1:-1] *= 2
    power /= len(signal) * np.sum(window ** 2)

    n_bands = len(BANDS_HZ)
    bands = np.empty(n_bands, dtype=np.float64)
    for index, (lower, upper) in enumerate(BANDS_HZ):
        within = (frequencies >= lower) & (frequencies < upper)
        if index == len(BANDS_HZ) - 1:
            within |= frequencies == upper
        bands[index] = power[within].sum()
    return bands


def frequency_band_db(signal, sample_rate):
    '''Return four band levels in dBFS; silent bands are negative infinity.

    signal:       one-dimensional audio samples from a single frame
    sample_rate:  samples per second; Nyquist must be at least 4000 Hz

    The reference is mean-square amplitude 1.0 for normalized PCM audio.
    These values are not Praat dB SPL, which uses a physical sound-pressure
    reference. Subtracting two band levels gives a spectral-balance measure.
    '''
    power = frequency_band_power(signal, sample_rate)
    with np.errstate(divide='ignore'):
        return 10 * np.log10(power)


def marker_to_frequency_bands(marker):
    '''Return four dBFS band levels for a frame starting at marker.start.

    marker:  marker with recording-relative start time in milliseconds

    Uses the same 25 ms start alignment and sample rounding as marker_to_mfcc.
    A complete analysis window must fit within the recording.
    '''
    audio = marker.audio
    sample_rate = audio.sample_rate
    if sample_rate < 8000 or marker.start < 0:
        raise ValueError('marker must have valid audio and start time')
    start_sample = round(marker.start / 1000 * sample_rate)
    window_samples = round(WINDOW_SECONDS * sample_rate)
    audio_samples = round(audio.duration / 1000 * sample_rate)
    stop_sample = start_sample + window_samples
    if stop_sample > audio_samples:
        raise ValueError('no complete analysis window starts at marker.start')
    signal, loaded_rate = load_audio_samples(audio.filename,
        start_sample=start_sample, stop_sample=stop_sample)
    if loaded_rate != sample_rate or len(signal) != window_samples:
        raise ValueError('audio samples do not match marker metadata')
    return frequency_band_db(signal, sample_rate)
