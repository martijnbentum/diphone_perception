'''Praat source-filter monophthong synthesis.'''

from itertools import product

import numpy as np

from .stimuli import DEFAULT_DURATION, DEFAULT_SAMPLE_RATE, Stimulus


def praat_vowel_stimulus(
    f0_hz,
    f1_hz,
    f2_hz,
    bandwidths_hz=(80.0, 100.0),
    duration=DEFAULT_DURATION,
    sample_rate=DEFAULT_SAMPLE_RATE,
    target_rms=0.1,
    fade_duration=0.01,
    minimum_formant_separation=100.0,
    stimulus_id=None,
):
    '''Synthesize a voiced source filtered by a two-formant Praat grid.'''

    _validate_formants(
        f0_hz,
        f1_hz,
        f2_hz,
        bandwidths_hz,
        duration,
        sample_rate,
        target_rms,
        fade_duration,
        minimum_formant_separation,
    )
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError as error:
        raise ImportError(
            'Praat vowel synthesis requires praat-parselmouth'
        ) from error

    pitch_tier = call('Create PitchTier', 'f0', 0, duration)
    call(pitch_tier, 'Add point', 0, f0_hz)
    call(pitch_tier, 'Add point', duration, f0_hz)
    point_process = call(pitch_tier, 'To PointProcess')
    source = call(
        point_process,
        'To Sound (phonation)',
        sample_rate,
        1.0,
        0.01,
        0.7,
        0.01,
        3.0,
        4.0,
    )
    bandwidth_1, bandwidth_2 = bandwidths_hz
    formant_grid = call(
        'Create FormantGrid',
        'filter',
        0,
        duration,
        2,
        f1_hz,
        f2_hz - f1_hz,
        bandwidth_1,
        bandwidth_2 - bandwidth_1,
    )
    filtered = call([source, formant_grid], 'Filter (no scale)')
    n_samples = round(duration * sample_rate)
    waveform = np.asarray(filtered.values[0, :n_samples], dtype=np.float64)
    waveform = _apply_fade(waveform, sample_rate, fade_duration)
    if target_rms is not None:
        current_rms = np.sqrt(np.mean(np.square(waveform)))
        if current_rms == 0:
            raise ValueError('Praat produced a silent waveform')
        waveform *= target_rms / current_rms
    if np.max(np.abs(waveform)) > 1:
        raise ValueError(
            'synthesized waveform clips; lower target_rms or change formants'
        )
    parameters = {
        'generator': 'praat_source_filter',
        'family': 'praat_formants',
        'f0_hz': float(f0_hz),
        'f1_hz': float(f1_hz),
        'f2_hz': float(f2_hz),
        'bandwidth_1_hz': float(bandwidth_1),
        'bandwidth_2_hz': float(bandwidth_2),
        'duration_seconds': float(duration),
        'target_rms': None if target_rms is None else float(target_rms),
        'fade_duration_seconds': float(fade_duration),
        'parselmouth_version': parselmouth.__version__,
        'praat_version': parselmouth.PRAAT_VERSION,
    }
    stimulus_id = stimulus_id or (
        f'praat-vowel_f0-{float(f0_hz):g}'
        f'_f1-{float(f1_hz):g}_f2-{float(f2_hz):g}'
    )
    return Stimulus(
        waveform.astype(np.float32),
        int(sample_rate),
        parameters,
        stimulus_id,
    )


def praat_formant_stimuli(
    f1_values,
    f2_values,
    f0_values=(120,),
    bandwidths_hz=(80.0, 100.0),
    minimum_formant_separation=100.0,
    **kwargs,
):
    '''Generate a speech-plausible F0/F1/F2 grid with crossed pairs omitted.'''

    output = []
    for f0_hz, f1_hz, f2_hz in product(
        f0_values, f1_values, f2_values
    ):
        if f2_hz - f1_hz < minimum_formant_separation:
            continue
        output.append(praat_vowel_stimulus(
            f0_hz=f0_hz,
            f1_hz=f1_hz,
            f2_hz=f2_hz,
            bandwidths_hz=bandwidths_hz,
            minimum_formant_separation=minimum_formant_separation,
            **kwargs,
        ))
    return output


def _validate_formants(
    f0_hz,
    f1_hz,
    f2_hz,
    bandwidths_hz,
    duration,
    sample_rate,
    target_rms,
    fade_duration,
    minimum_formant_separation,
):
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
    fade_samples = min(round(fade_duration * sample_rate), waveform.size // 2)
    if fade_samples <= 0:
        return waveform
    fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
    waveform[:fade_samples] *= fade
    waveform[-fade_samples:] *= fade[::-1]
    return waveform
