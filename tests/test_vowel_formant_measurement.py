import numpy as np

from synthetic_acoustic_probes.formants import praat_vowel_stimulus
from vowel_formant_reference.measurement import (
    MeasurementSettings,
    measure_formants,
)


def test_measure_formants_rejects_silence_and_short_audio():
    silent = measure_formants(np.zeros(1600), 16_000)
    assert not silent.success
    assert silent.rejection_reason == 'waveform is silent'

    short = measure_formants(np.ones(100), 16_000)
    assert not short.success
    assert 'shorter than' in short.rejection_reason


def test_measure_formants_on_controlled_praat_vowel_is_plausible():
    stimulus = praat_vowel_stimulus(
        f0_hz=120,
        f1_hz=500,
        f2_hz=1500,
        duration=0.5,
    )
    result = measure_formants(
        stimulus.waveform,
        stimulus.sample_rate,
        gender='male',
    )
    assert result.success
    assert abs(result.f0_hz - 120) < 3
    assert 100 < result.f1_hz < result.f2_hz < result.f3_hz < 8000
    assert result.central_start_seconds == 0.15
    assert result.central_end_seconds == 0.35


def test_measurement_settings_use_gender_specific_ranges():
    settings = MeasurementSettings()
    assert settings.formant_ceiling('male') == 5000
    assert settings.formant_ceiling('female') == 5500
    assert settings.pitch_range('male') == (60, 300)
    assert settings.pitch_range('female') == (100, 500)
