'''Deterministic stimuli from the wav2vec feature-encoder paper.

Experiment-specific generators: ``pure_tone_stimuli``, ``bias_stimuli``,
``sinusoidal_component_formant_stimuli``, ``amplitude_stimuli``, and
``temporal_burst_stimuli``. All are built on the shared ``sum_of_sinusoids``
primitive and the ``Stimulus`` container.
'''

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

import locations


SAMPLE_RATE = 16_000
DURATION = 1.0


def pure_tone_stimuli(frequencies=None, duration=DURATION,
    sample_rate=SAMPLE_RATE, amplitude=1.0, save=False,
    output_root=None, overwrite=False):
    '''Generate the paper's pure-tone grid, corrected to stop below Nyquist.
    frequencies:  Frequencies in Hz, or the paper grid when omitted.
    duration:  Signal duration in seconds.
    sample_rate:  Number of waveform samples per second.
    amplitude:  Amplitude shared by every pure tone.
    save:  Persist the generated stimuli when true.
    output_root:  Optional package directory used when saving.
    overwrite:  Replace an existing package only when saving.
    '''
    _validate_save_options(save, output_root, overwrite)
    if frequencies is None:
        frequencies = np.arange(10, sample_rate / 2, 10)
    extra_parameters = {'family': 'pure_tone'}
    stimuli = []
    for frequency in frequencies:
        stimulus_id = f'pure-tone_f-{float(frequency):g}'
        stimulus = sum_of_sinusoids(frequency, amplitudes=amplitude,
            duration=duration, sample_rate=sample_rate,
            stimulus_id=stimulus_id, extra_parameters=extra_parameters)
        stimuli.append(stimulus)
    if save:
        from .storage import write_stimuli

        if output_root is None: destination = locations.f0_pure_tone_stimuli
        else: destination = output_root
        write_stimuli(stimuli, destination, overwrite=overwrite)
    return stimuli


def bias_stimuli(frequencies=(100, 200, 300, 400, 500),
    biases=(-0.5, -0.25, 0.0, 0.25, 0.5), amplitude=0.5,
    duration=DURATION, sample_rate=SAMPLE_RATE):
    '''Generate the paper's F0-by-DC-bias grid.
    frequencies:  Base F0 frequencies in Hz.
    biases:       DC offsets applied to each frequency.
    amplitude:    Amplitude shared by every tone.
    duration:     Signal duration in seconds.
    sample_rate:  Number of waveform samples per second.
    '''
    extra_parameters = {'family': 'bias'}
    stimuli = []
    for frequency, bias in product(frequencies, biases):
        stimulus_id = f'bias_f-{float(frequency):g}_b-{float(bias):g}'
        stimulus = sum_of_sinusoids(frequency, amplitudes=amplitude,
            duration=duration, sample_rate=sample_rate, dc_bias=bias,
            stimulus_id=stimulus_id, extra_parameters=extra_parameters)
        stimuli.append(stimulus)
    return stimuli


def sinusoidal_component_formant_stimuli(f0_values=(120,), f1_values=None,
    f2_values=None, amplitudes=(0.5, 0.35, 0.15),
    duration=DURATION, sample_rate=SAMPLE_RATE):
    '''Reproduce the paper's three-sinusoid “formant” experiment.
    f0_values:    F0 values in Hz for the fixed source component.
    f1_values:    F1 values in Hz, or the paper grid when omitted.
    f2_values:    F2 values in Hz, or the paper grid when omitted.
    amplitudes:   Amplitude for the F0, F1, and F2 components.
    duration:     Signal duration in seconds.
    sample_rate:  Number of waveform samples per second.
    '''
    f1_values = np.linspace(235, 850, 30) if f1_values is None else f1_values
    f2_values = np.linspace(595, 2400, 30) if f2_values is None else f2_values
    output = []
    for f0, f1, f2 in product(f0_values, f1_values, f2_values):
        f0, f1, f2 = float(f0), float(f1), float(f2)
        extra_parameters = {'family': 'sinusoidal_component_formants',
            'f0_hz': f0, 'f1_hz': f1, 'f2_hz': f2,
            'acoustically_valid': f1 < f2}
        stimulus_id = f'sinusoidal-formants_f0-{f0:g}_f1-{f1:g}_f2-{f2:g}'
        stimulus = sum_of_sinusoids((f0, f1, f2), amplitudes=amplitudes,
            duration=duration, sample_rate=sample_rate,
            stimulus_id=stimulus_id, extra_parameters=extra_parameters)
        output.append(stimulus)
    return output


def amplitude_stimuli(frequencies=(100, 700), amplitudes=None,
    duration=DURATION, sample_rate=SAMPLE_RATE, exact_paper_code=False):
    '''Generate the two-component amplitude grid.
    frequencies:       The two component frequencies in Hz.
    amplitudes:        Explicit amplitude grid, or the default grid when
                       omitted.
    duration:          Signal duration in seconds.
    sample_rate:       Number of waveform samples per second.
    exact_paper_code:  Use the paper's amplitude transformation instead of even
                       energy spacing.
    Correct energy spacing is primary. Set `exact_paper_code=True` to reproduce
    the transformation in the public notebook.
    '''
    if len(frequencies) != 2:
        raise ValueError('amplitude probe requires exactly two frequencies')
    if amplitudes is not None:
        amplitude_values = _as_1d_array(amplitudes, 'amplitudes')
    elif exact_paper_code: amplitude_values = paper_code_amplitudes()
    else: amplitude_values = energy_spaced_amplitudes()
    exact_paper_code = bool(exact_paper_code)
    output = []
    amplitude_pairs = product(amplitude_values, amplitude_values)
    for amplitude_0, amplitude_1 in amplitude_pairs:
        amplitude_0, amplitude_1 = float(amplitude_0), float(amplitude_1)
        energy_0 = amplitude_0 ** 2 / 2
        energy_1 = amplitude_1 ** 2 / 2
        extra_parameters = {'family': 'amplitude',
            'amplitude_0': amplitude_0, 'amplitude_1': amplitude_1,
            'energy_0': energy_0, 'energy_1': energy_1,
            'exact_paper_code': exact_paper_code}
        stimulus_id = (f'amplitude_a0-{amplitude_0:.8g}'
            f'_a1-{amplitude_1:.8g}')
        stimulus = sum_of_sinusoids(frequencies,
            amplitudes=(amplitude_0, amplitude_1), duration=duration,
            sample_rate=sample_rate, stimulus_id=stimulus_id,
            extra_parameters=extra_parameters)
        output.append(stimulus)
    return output


def energy_spaced_amplitudes(start=0.1, stop=0.5, count=20):
    '''Amplitudes whose squared values are evenly spaced.
    start:  Lower amplitude bound.
    stop:   Upper amplitude bound.
    count:  Number of amplitude values to generate.
    '''
    if start < 0 or stop < start or count < 1:
        raise ValueError('invalid amplitude grid bounds or count')
    return np.sqrt(np.linspace(start ** 2, stop ** 2, count))


def paper_code_amplitudes(start=0.1, stop=0.5, count=20):
    '''Exact amplitude transformation used by the paper's public code.
    start:  Lower amplitude bound.
    stop:   Upper amplitude bound.
    count:  Number of amplitude values to generate.
    '''
    if start < 0 or stop < start or count < 1:
        raise ValueError('invalid amplitude grid bounds or count')
    return np.square(np.linspace(np.sqrt(start), np.sqrt(stop), count))


def temporal_burst_stimuli(
    burst_durations=(0.32, 0.16, 0.08, 0.04, 0.02, 0.01),
    base_frequency=200, burst_frequency=800, duration=DURATION,
    sample_rate=SAMPLE_RATE, amplitude=1.0):
    '''Replace the centered portion of a pure tone with another frequency.
    burst_durations:  Durations in seconds of the replaced center segment.
    base_frequency:   Frequency in Hz of the surrounding tone.
    burst_frequency:  Frequency in Hz of the replaced center segment.
    duration:         Signal duration in seconds.
    sample_rate:      Number of waveform samples per second.
    amplitude:        Amplitude shared by both frequencies.
    '''
    base_frequency = float(base_frequency)
    burst_frequency = float(burst_frequency)
    duration = float(duration)
    amplitude = float(amplitude)
    output = []
    for burst_duration in burst_durations:
        burst_duration = float(burst_duration)
        waveform, start, stop = _centered_frequency_replacement(
            base_frequency=base_frequency,
            replacement_frequency=burst_frequency,
            replacement_duration=burst_duration, duration=duration,
            sample_rate=sample_rate, amplitude=amplitude)
        parameters = {'generator': 'centered_frequency_replacement',
            'family': 'temporal_burst',
            'base_frequency_hz': base_frequency,
            'burst_frequency_hz': burst_frequency,
            'burst_duration_seconds': burst_duration,
            'burst_start_sample': start, 'burst_stop_sample': stop,
            'duration_seconds': duration, 'amplitude': amplitude}
        stimulus_id = f'temporal-burst_d-{burst_duration:g}'
        stimulus = Stimulus(waveform, sample_rate, parameters, stimulus_id)
        output.append(stimulus)
    return output


def sum_of_sinusoids(frequencies, amplitudes=1.0, phases=0.0,
    duration=DURATION, sample_rate=SAMPLE_RATE, dc_bias=0.0,
    allow_clipping=False, stimulus_id=None, extra_parameters=None):
    '''Create an exact-length sum of sinusoids.
    frequencies:       Frequencies in Hz for each sinusoid component.
    amplitudes:        Amplitude per component, or one shared amplitude.
    phases:            Phase in radians per component, or one shared phase.
    duration:          Signal duration in seconds.
    sample_rate:       Number of waveform samples per second.
    dc_bias:           Constant offset added to the summed waveform.
    allow_clipping:    Whether the waveform may exceed +/-1 before scaling.
    stimulus_id:       Explicit ID, or a derived one when omitted.
    extra_parameters:  Extra manifest fields merged into the parameters.
    '''
    frequencies = _as_1d_array(frequencies, 'frequencies')
    amplitudes = _repeat_or_match_length(amplitudes, frequencies.size,
        'amplitudes')
    phases = _repeat_or_match_length(phases, frequencies.size, 'phases')
    _validate_signal_parameters(frequencies, amplitudes, phases, duration,
        sample_rate, dc_bias)
    n_samples = round(duration * sample_rate)
    if n_samples <= 0:
        raise ValueError('duration and sample_rate produce no samples')
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    waveform = _sinusoid_waveform(frequencies, amplitudes, phases, time,
        dc_bias)
    _reject_clipping(waveform, allow_clipping)
    parameters = _sinusoid_parameters(frequencies, amplitudes, phases,
        duration, dc_bias)
    if extra_parameters:
        parameters.update(extra_parameters)
    stimulus_id = stimulus_id or _stimulus_id(parameters)
    return Stimulus(waveform, int(sample_rate), parameters, stimulus_id)


@dataclass(frozen=True)
class Stimulus:
    '''Immutable waveform plus controlled parameters and stable identifier.
    waveform:     Float32 samples, made read-only after construction.
    sample_rate:  Number of waveform samples per second.
    parameters:   Provenance dict, copied into an immutable mapping.
    stimulus_id:  Stable identifier used as the manifest and filename key.
    '''
    waveform: np.ndarray
    sample_rate: int
    parameters: dict
    stimulus_id: str

    def __post_init__(self):
        waveform = np.asarray(self.waveform, dtype=np.float32).copy()
        if waveform.ndim != 1:
            raise ValueError('Stimulus.waveform must be one-dimensional')
        if not waveform.size:
            raise ValueError('Stimulus.waveform must not be empty')
        if not np.all(np.isfinite(waveform)):
            raise ValueError('Stimulus.waveform contains non-finite samples')
        if not isinstance(self.sample_rate, (int, np.integer)):
            raise ValueError('Stimulus.sample_rate must be an integer')
        if self.sample_rate <= 0:
            raise ValueError('Stimulus.sample_rate must be positive')
        waveform.setflags(write=False)
        object.__setattr__(self, 'waveform', waveform)
        object.__setattr__(self, 'parameters',
            MappingProxyType(dict(self.parameters)))


def _centered_frequency_replacement(base_frequency, replacement_frequency,
    replacement_duration, duration, sample_rate, amplitude):
    '''Replace the centered segment of a base tone with another frequency.'''
    values_by_name = {'base_frequency': base_frequency,
        'replacement_frequency': replacement_frequency,
        'replacement_duration': replacement_duration, 'duration': duration,
        'amplitude': amplitude}
    for name, value in values_by_name.items():
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if not 0 <= replacement_duration <= duration:
        raise ValueError('replacement_duration must be between 0 and duration')
    if amplitude < 0:
        raise ValueError('amplitude must be non-negative')
    _validate_below_nyquist(base_frequency, sample_rate, 'base_frequency')
    _validate_below_nyquist(replacement_frequency, sample_rate,
        'replacement_frequency')
    n_samples = round(duration * sample_rate)
    burst_samples = round(replacement_duration * sample_rate)
    start = (n_samples - burst_samples) // 2
    stop = start + burst_samples
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    base = amplitude * np.sin(2 * np.pi * base_frequency * time)
    replacement = amplitude * np.sin(2 * np.pi * replacement_frequency * time)
    base[start:stop] = replacement[start:stop]
    _reject_clipping(base, allow_clipping=False)
    return base.astype(np.float32), start, stop


def _as_1d_array(values, name):
    '''Validate values as a finite, non-empty one-dimensional array.'''
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or not values.size:
        raise ValueError(f'{name} must be a non-empty one-dimensional array')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} contains non-finite values')
    return values


def _validate_save_options(save, output_root, overwrite):
    '''Validate the save/output_root/overwrite argument combination.'''
    if not isinstance(save, (bool, np.bool_)):
        raise TypeError('save must be a boolean')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError('overwrite must be a boolean')
    if output_root is not None and not save:
        raise ValueError('output_root requires save=True')
    if overwrite and not save:
        raise ValueError('overwrite=True requires save=True')


def _repeat_or_match_length(values, size, name):
    '''Repeat a length-one array or require an exact length match.'''
    values = _as_1d_array(values, name)
    if values.size == 1:
        return np.repeat(values, size)
    if values.size != size:
        raise ValueError(f'{name} must have length 1 or {size}')
    return values


def _validate_signal_parameters(frequencies, amplitudes, phases, duration,
    sample_rate, dc_bias):
    '''Validate the shared numeric constraints for a sinusoid signal.'''
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive')
    if np.any(frequencies < 0) or np.any(frequencies >= sample_rate / 2):
        raise ValueError('frequencies must lie in [0, Nyquist)')
    if np.any(amplitudes < 0):
        raise ValueError('amplitudes must be non-negative')
    if not np.all(np.isfinite(phases)):
        raise ValueError('phases contains non-finite values')
    if not np.isfinite(dc_bias):
        raise ValueError('dc_bias must be finite')


def _sinusoid_waveform(frequencies, amplitudes, phases, time, dc_bias):
    '''Sum the sinusoid components and add the DC bias.'''
    sinusoids = np.sin(
        2 * np.pi * frequencies[:, None] * time + phases[:, None])
    waveform = np.sum(amplitudes[:, None] * sinusoids, axis=0)
    return waveform + dc_bias


def _sinusoid_parameters(frequencies, amplitudes, phases, duration,
    dc_bias):
    '''Return the provenance parameters for a sum-of-sinusoids stimulus.'''
    return {'generator': 'sum_of_sinusoids',
        'frequencies_hz': frequencies.tolist(),
        'amplitudes': amplitudes.tolist(), 'phases_radians': phases.tolist(),
        'duration_seconds': float(duration), 'dc_bias': float(dc_bias)}


def _reject_clipping(waveform, allow_clipping):
    '''Raise if the waveform exceeds the clipping threshold.'''
    if not allow_clipping and np.max(np.abs(waveform)) > 1 + 1e-7:
        raise ValueError('waveform would clip; lower amplitudes or allow clipping')


def _validate_below_nyquist(frequency, sample_rate, name):
    '''Raise if frequency is not strictly below the Nyquist frequency.'''
    if not 0 <= frequency < sample_rate / 2:
        raise ValueError(f'{name} must be below Nyquist')


def _stimulus_id(parameters):
    '''Derive a stable stimulus ID from a hash of the parameters.'''
    serial = json.dumps(parameters, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(serial.encode('utf-8')).hexdigest()[:16]
    return f'stimulus-{digest}'
