import numpy as np
import pytest

from synthetic_acoustic_probes.acoustics import dominant_fft_peaks
from synthetic_acoustic_probes.stimuli import (
    amplitude_stimuli,
    bias_stimuli,
    energy_spaced_amplitudes,
    pure_tone_stimuli,
    sinusoidal_component_formant_stimuli,
    sum_of_sinusoids,
    temporal_burst_stimuli,
)


def test_sum_of_sinusoids_is_exact_and_repeatable():
    left = sum_of_sinusoids(
        [100, 300],
        amplitudes=[0.2, 0.3],
        duration=0.25,
        sample_rate=16_000,
    )
    right = sum_of_sinusoids(
        [100, 300],
        amplitudes=[0.2, 0.3],
        duration=0.25,
        sample_rate=16_000,
    )
    assert left.waveform.shape == (4000,)
    assert left.waveform.dtype == np.float32
    assert np.array_equal(left.waveform, right.waveform)
    assert left.stimulus_id == right.stimulus_id
    assert not left.waveform.flags.writeable


def test_default_stimulus_identifier_is_a_stable_sha256_prefix():
    stimulus = sum_of_sinusoids(100, amplitudes=0.2)
    assert stimulus.stimulus_id == 'stimulus-37887db06cc7bd7f'


def test_integer_frequency_peaks_are_recovered():
    stimulus = sum_of_sinusoids(
        [100, 700],
        amplitudes=[0.5, 0.25],
    )
    peaks = dominant_fft_peaks(
        stimulus.waveform,
        stimulus.sample_rate,
        count=2,
        minimum_frequency=1,
    )
    assert {frequency for frequency, _ in peaks} == {100, 700}


def test_temporal_replacement_has_exact_centered_sample_span():
    stimulus = temporal_burst_stimuli([0.01])[0]
    assert stimulus.parameters['burst_stop_sample'] - (
        stimulus.parameters['burst_start_sample']
    ) == 160
    assert stimulus.parameters['burst_start_sample'] == (16_000 - 160) // 2


def test_default_paper_grid_sizes_and_endpoints():
    tones = pure_tone_stimuli()
    assert len(tones) == 799
    assert tones[0].parameters['frequencies_hz'] == [10]
    assert tones[-1].parameters['frequencies_hz'] == [7990]

    assert len(bias_stimuli()) == 25
    assert len(sinusoidal_component_formant_stimuli()) == 900
    assert len(amplitude_stimuli()) == 400
    assert len(temporal_burst_stimuli()) == 6


def test_corrected_amplitudes_are_evenly_spaced_in_energy():
    amplitudes = energy_spaced_amplitudes(count=20)
    energy_steps = np.diff(amplitudes ** 2)
    assert np.allclose(energy_steps, energy_steps[0])


def test_invalid_signals_fail_clearly():
    with pytest.raises(ValueError, match='Nyquist'):
        sum_of_sinusoids(8000, sample_rate=16_000)
    with pytest.raises(ValueError, match='clip'):
        sum_of_sinusoids([100, 200], amplitudes=[1, 1])
    with pytest.raises(ValueError, match='non-negative'):
        sum_of_sinusoids(100, amplitudes=-1)
