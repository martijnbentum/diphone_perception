import numpy as np
import pytest

from synthetic_acoustic_probes.acoustics import (
    designed_formant_response,
    measure_waveform,
)
from synthetic_acoustic_probes.formants import (
    praat_formant_stimuli,
    praat_vowel_stimulus,
)


def test_praat_vowel_is_exact_length_deterministic_and_equal_rms():
    left = praat_vowel_stimulus(
        120, 500, 1500, duration=0.25, target_rms=0.1
    )
    right = praat_vowel_stimulus(
        120, 500, 1500, duration=0.25, target_rms=0.1
    )
    assert left.waveform.shape == (4000,)
    assert np.array_equal(left.waveform, right.waveform)
    assert measure_waveform(left.waveform, left.sample_rate).rms == pytest.approx(
        0.1, abs=1e-7
    )
    assert left.parameters['f1_hz'] == 500
    assert left.parameters['f2_hz'] == 1500
    assert left.parameters['praat_version']
    assert left.waveform[0] == pytest.approx(0)


def test_praat_grid_rejects_crossed_or_too_close_formants():
    stimuli = praat_formant_stimuli(
        f0_values=[120],
        f1_values=[500, 1000],
        f2_values=[550, 1500],
        duration=0.1,
        minimum_formant_separation=100,
    )
    pairs = {
        (item.parameters['f1_hz'], item.parameters['f2_hz'])
        for item in stimuli
    }
    assert pairs == {(500, 1500), (1000, 1500)}

    with pytest.raises(ValueError, match='separated'):
        praat_vowel_stimulus(120, 1000, 1050)


def test_designed_filter_response_has_peaks_near_formants():
    frequencies = np.arange(100, 3001)
    response = designed_formant_response(
        frequencies,
        formants_hz=[500, 1500],
        bandwidths_hz=[80, 100],
    )
    for formant in (500, 1500):
        local = np.abs(frequencies - formant) <= 100
        peak = frequencies[local][np.argmax(response[local])]
        assert abs(peak - formant) <= 5
