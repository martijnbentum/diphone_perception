'''Deterministic stimuli from the wav2vec feature-encoder paper.'''

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

import locations


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION = 1.0
_DEFAULT_PURE_TONE_OUTPUT_ROOT = locations.f0_pure_tone_stimuli


@dataclass(frozen=True)
class Stimulus:
    '''Immutable waveform plus controlled parameters and stable identifier.'''

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
        object.__setattr__(
            self, 'parameters', MappingProxyType(dict(self.parameters))
        )


def sum_of_sinusoids(
    frequencies,
    amplitudes=1.0,
    phases=0.0,
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
    dc_bias=0.0,
    allow_clipping=False,
    stimulus_id=None,
    extra_parameters=None,
):
    '''Create an exact-length sum of sinusoids.'''

    frequencies = _one_dimensional_values(frequencies, 'frequencies')
    amplitudes = _broadcast_values(amplitudes, frequencies.size, 'amplitudes')
    phases = _broadcast_values(phases, frequencies.size, 'phases')
    _validate_signal_parameters(
        frequencies, amplitudes, phases, duration, sample_rate, dc_bias
    )
    n_samples = round(duration * sample_rate)
    if n_samples <= 0:
        raise ValueError('duration and sample_rate produce no samples')
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    waveform = np.sum(
        amplitudes[:, None] * np.sin(
            2 * np.pi * frequencies[:, None] * time + phases[:, None]
        ),
        axis=0,
    )
    waveform += dc_bias
    _reject_clipping(waveform, allow_clipping)
    parameters = {
        'generator': 'sum_of_sinusoids',
        'frequencies_hz': frequencies.tolist(),
        'amplitudes': amplitudes.tolist(),
        'phases_radians': phases.tolist(),
        'duration_seconds': float(duration),
        'dc_bias': float(dc_bias),
    }
    if extra_parameters:
        parameters.update(extra_parameters)
    stimulus_id = stimulus_id or _stimulus_id(parameters)
    return Stimulus(waveform, int(sample_rate), parameters, stimulus_id)


def pure_tone_stimuli(
    frequencies=None,
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
    amplitude=1.0,
    save=False,
    output_root=None,
    overwrite=False,
):
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
    stimuli = [
        sum_of_sinusoids(
            frequency,
            amplitudes=amplitude,
            duration=duration,
            sample_rate=sample_rate,
            stimulus_id=f'pure-tone_f-{float(frequency):g}',
            extra_parameters={'family': 'pure_tone'},
        )
        for frequency in frequencies
    ]
    if save:
        from .storage import write_stimuli

        destination = (
            _DEFAULT_PURE_TONE_OUTPUT_ROOT
            if output_root is None else output_root
        )
        write_stimuli(stimuli, destination, overwrite=overwrite)
    return stimuli


def bias_stimuli(
    frequencies=(100, 200, 300, 400, 500),
    biases=(-0.5, -0.25, 0.0, 0.25, 0.5),
    amplitude=0.5,
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
):
    '''Generate the paper's F0-by-DC-bias grid.'''

    return [
        sum_of_sinusoids(
            frequency,
            amplitudes=amplitude,
            duration=duration,
            sample_rate=sample_rate,
            dc_bias=bias,
            stimulus_id=f'bias_f-{float(frequency):g}_b-{float(bias):g}',
            extra_parameters={'family': 'bias'},
        )
        for frequency, bias in product(frequencies, biases)
    ]


def sinusoidal_component_formant_stimuli(
    f0_values=(120,),
    f1_values=None,
    f2_values=None,
    amplitudes=(0.5, 0.35, 0.15),
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
):
    '''Reproduce the paper's three-sinusoid “formant” experiment.'''

    f1_values = (
        np.linspace(235, 850, 30)
        if f1_values is None else f1_values
    )
    f2_values = (
        np.linspace(595, 2400, 30)
        if f2_values is None else f2_values
    )
    output = []
    for f0, f1, f2 in product(f0_values, f1_values, f2_values):
        acoustically_valid = bool(f1 < f2)
        output.append(sum_of_sinusoids(
            (f0, f1, f2),
            amplitudes=amplitudes,
            duration=duration,
            sample_rate=sample_rate,
            stimulus_id=(
                f'sinusoidal-formants_f0-{float(f0):g}'
                f'_f1-{float(f1):g}_f2-{float(f2):g}'
            ),
            extra_parameters={
                'family': 'sinusoidal_component_formants',
                'f0_hz': float(f0),
                'f1_hz': float(f1),
                'f2_hz': float(f2),
                'acoustically_valid': acoustically_valid,
            },
        ))
    return output


def energy_spaced_amplitudes(start=0.1, stop=0.5, count=20):
    '''Amplitudes whose squared values are evenly spaced.'''

    if start < 0 or stop < start or count < 1:
        raise ValueError('invalid amplitude grid bounds or count')
    return np.sqrt(np.linspace(start ** 2, stop ** 2, count))


def paper_code_amplitudes(start=0.1, stop=0.5, count=20):
    '''Exact amplitude transformation used by the paper's public code.'''

    if start < 0 or stop < start or count < 1:
        raise ValueError('invalid amplitude grid bounds or count')
    return np.square(np.linspace(np.sqrt(start), np.sqrt(stop), count))


def amplitude_stimuli(
    frequencies=(100, 700),
    amplitudes=None,
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
    exact_paper_code=False,
):
    '''Generate the two-component amplitude grid.

    Correct energy spacing is primary. Set `exact_paper_code=True` to reproduce
    the transformation in the public notebook.
    '''

    if len(frequencies) != 2:
        raise ValueError('amplitude probe requires exactly two frequencies')
    if amplitudes is None:
        amplitude_values = (
            paper_code_amplitudes()
            if exact_paper_code else energy_spaced_amplitudes()
        )
    else:
        amplitude_values = _one_dimensional_values(amplitudes, 'amplitudes')
    output = []
    for amplitude_0, amplitude_1 in product(
        amplitude_values, amplitude_values
    ):
        output.append(sum_of_sinusoids(
            frequencies,
            amplitudes=(amplitude_0, amplitude_1),
            duration=duration,
            sample_rate=sample_rate,
            stimulus_id=(
                f'amplitude_a0-{float(amplitude_0):.8g}'
                f'_a1-{float(amplitude_1):.8g}'
            ),
            extra_parameters={
                'family': 'amplitude',
                'amplitude_0': float(amplitude_0),
                'amplitude_1': float(amplitude_1),
                'energy_0': float(amplitude_0 ** 2 / 2),
                'energy_1': float(amplitude_1 ** 2 / 2),
                'exact_paper_code': bool(exact_paper_code),
            },
        ))
    return output


def temporal_burst_stimuli(
    burst_durations=(0.32, 0.16, 0.08, 0.04, 0.02, 0.01),
    base_frequency=200,
    burst_frequency=800,
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
    amplitude=1.0,
):
    '''Replace the centered portion of a pure tone with another frequency.'''

    output = []
    for burst_duration in burst_durations:
        waveform, start, stop = _centered_frequency_replacement(
            base_frequency=base_frequency,
            replacement_frequency=burst_frequency,
            replacement_duration=burst_duration,
            duration=duration,
            sample_rate=sample_rate,
            amplitude=amplitude,
        )
        parameters = {
            'generator': 'centered_frequency_replacement',
            'family': 'temporal_burst',
            'base_frequency_hz': float(base_frequency),
            'burst_frequency_hz': float(burst_frequency),
            'burst_duration_seconds': float(burst_duration),
            'burst_start_sample': start,
            'burst_stop_sample': stop,
            'duration_seconds': float(duration),
            'amplitude': float(amplitude),
        }
        output.append(Stimulus(
            waveform,
            sample_rate,
            parameters,
            f'temporal-burst_d-{float(burst_duration):g}',
        ))
    return output


def _centered_frequency_replacement(
    base_frequency,
    replacement_frequency,
    replacement_duration,
    duration,
    sample_rate,
    amplitude,
):
    for name, value in (
        ('base_frequency', base_frequency),
        ('replacement_frequency', replacement_frequency),
        ('replacement_duration', replacement_duration),
        ('duration', duration),
        ('amplitude', amplitude),
    ):
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if not 0 <= replacement_duration <= duration:
        raise ValueError('replacement_duration must be between 0 and duration')
    if amplitude < 0:
        raise ValueError('amplitude must be non-negative')
    nyquist = sample_rate / 2
    if not 0 <= base_frequency < nyquist:
        raise ValueError('base_frequency must be below Nyquist')
    if not 0 <= replacement_frequency < nyquist:
        raise ValueError('replacement_frequency must be below Nyquist')
    n_samples = round(duration * sample_rate)
    burst_samples = round(replacement_duration * sample_rate)
    start = (n_samples - burst_samples) // 2
    stop = start + burst_samples
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    base = amplitude * np.sin(2 * np.pi * base_frequency * time)
    replacement = amplitude * np.sin(
        2 * np.pi * replacement_frequency * time
    )
    base[start:stop] = replacement[start:stop]
    _reject_clipping(base, allow_clipping=False)
    return base.astype(np.float32), start, stop


def _one_dimensional_values(values, name):
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or not values.size:
        raise ValueError(f'{name} must be a non-empty one-dimensional array')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} contains non-finite values')
    return values


def _validate_save_options(save, output_root, overwrite):
    if not isinstance(save, (bool, np.bool_)):
        raise TypeError('save must be a boolean')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError('overwrite must be a boolean')
    if output_root is not None and not save:
        raise ValueError('output_root requires save=True')
    if overwrite and not save:
        raise ValueError('overwrite=True requires save=True')


def _broadcast_values(values, size, name):
    values = _one_dimensional_values(values, name)
    if values.size == 1:
        return np.repeat(values, size)
    if values.size != size:
        raise ValueError(f'{name} must have length 1 or {size}')
    return values


def _validate_signal_parameters(
    frequencies,
    amplitudes,
    phases,
    duration,
    sample_rate,
    dc_bias,
):
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


def _reject_clipping(waveform, allow_clipping):
    if not allow_clipping and np.max(np.abs(waveform)) > 1 + 1e-7:
        raise ValueError('waveform would clip; lower amplitudes or allow clipping')


def _stimulus_id(parameters):
    serial = json.dumps(parameters, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(serial.encode('utf-8')).hexdigest()[:16]
    return f'stimulus-{digest}'
