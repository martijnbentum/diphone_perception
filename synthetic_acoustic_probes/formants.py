'''Praat source-filter monophthong synthesis.'''

from itertools import product

import numpy as np
import parselmouth
from parselmouth.praat import call

from .stimuli import DURATION, SAMPLE_RATE, Stimulus


def praat_vowel_stimulus(f0_hz, f1_hz, f2_hz, bandwidths_hz=(80.0, 100.0),
    duration=DURATION, sample_rate=SAMPLE_RATE, target_rms=0.1,
    fade_duration=0.01, minimum_formant_separation=100.0, stimulus_id=None):
    '''Synthesize a voiced source filtered by a two-formant Praat grid.
    f0_hz:                       Voice fundamental frequency in Hz.
    f1_hz:                       First formant frequency in Hz.
    f2_hz:                       Second formant frequency in Hz.
    bandwidths_hz:               Bandwidths in Hz for F1 and F2.
    duration:                    Signal duration in seconds.
    sample_rate:                 Number of waveform samples per second.
    target_rms:                  RMS target, or None to skip normalization.
    fade_duration:               Onset/offset fade duration in seconds.
    minimum_formant_separation:  Minimum required F2 minus F1 in Hz.
    stimulus_id:                 Explicit ID, or a derived one when omitted.
    '''
    _validate_formants(f0_hz, f1_hz, f2_hz, bandwidths_hz, duration,
        sample_rate, target_rms, fade_duration, minimum_formant_separation)
    # Lay down a flat F0 trajectory and convert it to glottal pulse instants.
    pitch_tier = call('Create PitchTier', 'f0', 0, duration)
    call(pitch_tier, 'Add point', 0, f0_hz)
    call(pitch_tier, 'Add point', duration, f0_hz)
    point_process = call(pitch_tier, 'To PointProcess')
    # Synthesize the glottal source (voicing) waveform at those pulses.
    source = call(point_process, 'To Sound (phonation)', sample_rate, 1.0,
        0.01, 0.7, 0.01, 3.0, 4.0)
    bandwidth_1, bandwidth_2 = bandwidths_hz
    # Build the two-formant vocal-tract filter.
    formant_grid = call('Create FormantGrid', 'filter', 0, duration, 2, f1_hz,
        f2_hz - f1_hz, bandwidth_1, bandwidth_2 - bandwidth_1)
    # Shape the source through the filter to produce the vowel.
    filtered = call([source, formant_grid], 'Filter (no scale)')
    n_samples = round(duration * sample_rate)
    waveform = np.asarray(filtered.values[0, :n_samples], dtype=np.float64)
    waveform = _apply_fade(waveform, sample_rate, fade_duration)
    waveform = _apply_target_rms(waveform, target_rms)
    _reject_clipping(waveform)
    f0_hz, f1_hz, f2_hz = float(f0_hz), float(f1_hz), float(f2_hz)
    parameters = _praat_vowel_parameters(f0_hz, f1_hz, f2_hz, bandwidth_1,
        bandwidth_2, duration, target_rms, fade_duration)
    if stimulus_id is None:
        stimulus_id = f'praat-vowel_f0-{f0_hz:g}_f1-{f1_hz:g}_f2-{f2_hz:g}'
    waveform = waveform.astype(np.float32)
    sample_rate = int(sample_rate)
    return Stimulus(waveform, sample_rate, parameters, stimulus_id)


def praat_formant_stimuli(f1_values, f2_values, f0_values=(120,),
    bandwidths_hz=(80.0, 100.0), minimum_formant_separation=100.0, **kwargs):
    '''Generate a speech-plausible F0/F1/F2 grid with crossed pairs omitted.
    f1_values:                   F1 values in Hz.
    f2_values:                   F2 values in Hz.
    f0_values:                   F0 values in Hz.
    bandwidths_hz:               Bandwidths in Hz for F1 and F2.
    minimum_formant_separation:  Minimum required F2 minus F1 in Hz; pairs
                                  below it are skipped.
    '''
    output = []
    for f0_hz, f1_hz, f2_hz in product(f0_values, f1_values, f2_values):
        if f2_hz - f1_hz < minimum_formant_separation:
            continue
        stimulus = praat_vowel_stimulus(f0_hz=f0_hz, f1_hz=f1_hz,
            f2_hz=f2_hz, bandwidths_hz=bandwidths_hz,
            minimum_formant_separation=minimum_formant_separation, **kwargs)
        output.append(stimulus)
    return output


def _praat_vowel_parameters(f0_hz, f1_hz, f2_hz, bandwidth_1, bandwidth_2,
    duration, target_rms, fade_duration):
    '''Return the provenance parameters for a Praat vowel stimulus.'''
    return {
        'generator': 'praat_source_filter',
        'family': 'praat_formants',
        'f0_hz': f0_hz,
        'f1_hz': f1_hz,
        'f2_hz': f2_hz,
        'bandwidth_1_hz': float(bandwidth_1),
        'bandwidth_2_hz': float(bandwidth_2),
        'duration_seconds': float(duration),
        'target_rms': None if target_rms is None else float(target_rms),
        'fade_duration_seconds': float(fade_duration),
        'parselmouth_version': parselmouth.__version__,
        'praat_version': parselmouth.PRAAT_VERSION,
    }


def _validate_formants(f0_hz, f1_hz, f2_hz, bandwidths_hz, duration,
    sample_rate, target_rms, fade_duration, minimum_formant_separation):
    '''Raise if any Praat synthesis parameter is invalid.'''
    numeric = [
        f0_hz, f1_hz, f2_hz, duration, sample_rate,
        fade_duration, minimum_formant_separation,
        *bandwidths_hz,
    ]
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError('all synthesis parameters must be finite')
    if f0_hz <= 0:
        raise ValueError('f0_hz must be positive')
    if f1_hz <= 0 or f2_hz - f1_hz < minimum_formant_separation:
        raise ValueError('formants must satisfy separated F1 < F2')
    if f2_hz >= sample_rate / 2:
        raise ValueError('F2 must be below Nyquist')
    if len(bandwidths_hz) != 2 or min(bandwidths_hz) <= 0:
        raise ValueError('two positive formant bandwidths are required')
    if duration <= 0:
        raise ValueError('duration must be positive')
    if fade_duration < 0 or 2 * fade_duration > duration:
        raise ValueError('fade_duration does not fit within duration')
    if target_rms is not None and not 0 < target_rms < 1:
        raise ValueError('target_rms must be in (0, 1) or None')


def _apply_fade(waveform, sample_rate, fade_duration):
    '''Apply a raised-cosine onset/offset fade in place.'''
    fade_samples = min(round(fade_duration * sample_rate), waveform.size // 2)
    if fade_samples <= 0:
        return waveform
    fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
    waveform[:fade_samples] *= fade
    waveform[-fade_samples:] *= fade[::-1]
    return waveform


def _apply_target_rms(waveform, target_rms):
    '''Scale the waveform to target_rms, raising on a silent input.'''
    if target_rms is None:
        return waveform
    current_rms = np.sqrt(np.mean(np.square(waveform)))
    if current_rms == 0:
        raise ValueError('Praat produced a silent waveform')
    return waveform * (target_rms / current_rms)


def _reject_clipping(waveform):
    '''Raise if any sample exceeds the +/-1 amplitude range.'''
    if np.max(np.abs(waveform)) > 1:
        m = 'synthesized waveform clips; lower target_rms or change formants'
        raise ValueError(m)
