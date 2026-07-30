import numpy as np
import pytest

from synthetic_acoustic_probes.acoustics import measure_waveform
from synthetic_acoustic_probes.stimuli import sum_of_sinusoids


def test_waveform_measurements_match_sine_analytics():
    stimulus = sum_of_sinusoids(100, amplitudes=0.5)
    measurements = measure_waveform(
        stimulus.waveform, stimulus.sample_rate
    )
    assert measurements.peak == pytest.approx(0.5, abs=1e-6)
    assert measurements.rms == pytest.approx(0.5 / np.sqrt(2), abs=1e-6)
    assert measurements.energy == pytest.approx(
        16_000 * 0.5 ** 2 / 2, abs=1e-3
    )
    assert measurements.dc_mean == pytest.approx(0, abs=1e-7)
    assert measurements.crest_factor == pytest.approx(np.sqrt(2), abs=1e-6)


def test_silence_level_is_explicit():
    measurements = measure_waveform(np.zeros(100), 100)
    assert measurements.rms == 0
    assert measurements.dbfs == -np.inf
    assert measurements.crest_factor == np.inf
