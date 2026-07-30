'''Acoustic verification helpers for generated waveforms.'''

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveformMeasurements:
    peak: float
    rms: float
    energy: float
    dbfs: float
    crest_factor: float
    dc_mean: float
    duration_seconds: float


def measure_waveform(waveform, sample_rate):
    '''Measure basic level and duration properties of a waveform.'''

    samples = _validated_waveform(waveform, sample_rate)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    energy = float(np.sum(np.square(samples)))
    dbfs = float(20 * np.log10(rms)) if rms > 0 else -np.inf
    crest_factor = float(peak / rms) if rms > 0 else np.inf
    return WaveformMeasurements(
        peak=peak,
        rms=rms,
        energy=energy,
        dbfs=dbfs,
        crest_factor=crest_factor,
        dc_mean=float(np.mean(samples)),
        duration_seconds=samples.size / sample_rate,
    )


def dominant_fft_peaks(
    waveform,
    sample_rate,
    count=5,
    minimum_frequency=0.0,
):
    '''Return the strongest non-repeated real-FFT frequency bins.'''

    samples = _validated_waveform(waveform, sample_rate)
    if count < 1:
        raise ValueError('count must be at least 1')
    spectrum = np.abs(np.fft.rfft(samples))
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    spectrum[frequencies < minimum_frequency] = 0
    indices = np.argsort(spectrum)[::-1][:count]
    return [
        (float(frequencies[index]), float(spectrum[index]))
        for index in indices
    ]


def designed_formant_response(
    frequencies_hz,
    formants_hz,
    bandwidths_hz,
):
    '''Approximate cascaded resonator response for design verification.'''

    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError('frequencies_hz must be a finite one-dimensional array')
    if len(formants_hz) != len(bandwidths_hz) or not formants_hz:
        raise ValueError('formants and bandwidths must have matching lengths')
    response = np.ones_like(frequencies)
    for formant, bandwidth in zip(formants_hz, bandwidths_hz):
        if formant <= 0 or bandwidth <= 0:
            raise ValueError('formants and bandwidths must be positive')
        response *= 1 / np.sqrt(
            1 + np.square(2 * (frequencies - formant) / bandwidth)
        )
    maximum = response.max(initial=0)
    return response / maximum if maximum else response


def _validated_waveform(waveform, sample_rate):
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1 or not samples.size:
        raise ValueError('waveform must be a non-empty one-dimensional array')
    if not np.all(np.isfinite(samples)):
        raise ValueError('waveform contains non-finite values')
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    return samples
