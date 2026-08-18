'''Acoustic verification helpers for generated waveforms.'''

from dataclasses import dataclass

import numpy as np


def measure_waveform(waveform, sample_rate):
    '''Measure basic level and duration properties of a waveform.
    waveform:     One-dimensional array of audio samples.
    sample_rate:  Number of waveform samples per second.
    '''
    samples = _validated_waveform(waveform, sample_rate)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    energy = float(np.sum(np.square(samples)))
    dbfs = float(20 * np.log10(rms)) if rms > 0 else -np.inf
    crest_factor = float(peak / rms) if rms > 0 else np.inf
    measurements = WaveformMeasurements(peak=peak, rms=rms, energy=energy,
        dbfs=dbfs, crest_factor=crest_factor,
        dc_mean=float(np.mean(samples)),
        duration_seconds=samples.size / sample_rate)
    return measurements


def dominant_fft_peaks(waveform, sample_rate, count=5, minimum_frequency=0.0):
    '''Return the strongest non-repeated real-FFT frequency bins.
    waveform:           One-dimensional array of audio samples.
    sample_rate:        Number of waveform samples per second.
    count:              Maximum number of peaks to return.
    minimum_frequency:  Frequencies below this Hz value are excluded.
    '''
    samples = _validated_waveform(waveform, sample_rate)
    if count < 1:
        raise ValueError('count must be at least 1')
    spectrum = np.abs(np.fft.rfft(samples))
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    spectrum[frequencies < minimum_frequency] = 0
    indices = np.argsort(spectrum)[::-1][:count]
    frequencies = frequencies.tolist()
    spectrum = spectrum.tolist()
    peaks = [(frequencies[index], spectrum[index]) for index in indices]
    return peaks


def designed_formant_response(frequencies_hz, formants_hz, bandwidths_hz):
    '''Approximate cascaded resonator response for design verification.
    frequencies_hz:  Frequencies in Hz at which to evaluate the response.
    formants_hz:     Center frequency in Hz for each resonator.
    bandwidths_hz:   Bandwidth in Hz for each resonator.
    '''
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError('frequencies_hz must be a finite one-dimensional array')
    if len(formants_hz) != len(bandwidths_hz) or not formants_hz:
        raise ValueError('formants and bandwidths must have matching lengths')
    response = np.ones_like(frequencies)
    for formant, bandwidth in zip(formants_hz, bandwidths_hz):
        if formant <= 0 or bandwidth <= 0:
            raise ValueError('formants and bandwidths must be positive')
        ratio = 2 * (frequencies - formant) / bandwidth
        response *= 1 / np.sqrt(1 + np.square(ratio))
    maximum = response.max(initial=0)
    normalized = response / maximum if maximum else response
    return normalized


@dataclass(frozen=True)
class WaveformMeasurements:
    peak: float
    rms: float
    energy: float
    dbfs: float
    crest_factor: float
    dc_mean: float
    duration_seconds: float


def _validated_waveform(waveform, sample_rate):
    '''Validate waveform as a finite, non-empty one-dimensional array.'''
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1 or not samples.size:
        raise ValueError('waveform must be a non-empty one-dimensional array')
    if not np.all(np.isfinite(samples)):
        raise ValueError('waveform contains non-finite values')
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    return samples
