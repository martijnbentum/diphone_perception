'''Praat-based F0 and formant measurements for monophthong waveforms.'''

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MeasurementSettings:
    '''Settings for robust measurements over a stable central window.'''

    time_step: float = 0.005
    central_fraction: float = 0.4
    minimum_duration: float = 0.05
    max_number_of_formants: float = 5.0
    male_max_formant: float = 5000.0
    female_max_formant: float = 5500.0
    other_max_formant: float = 5500.0
    window_length: float = 0.025
    pre_emphasis_from: float = 50.0
    male_pitch_floor: float = 60.0
    male_pitch_ceiling: float = 300.0
    female_pitch_floor: float = 100.0
    female_pitch_ceiling: float = 500.0
    other_pitch_floor: float = 60.0
    other_pitch_ceiling: float = 500.0

    def __post_init__(self):
        if not 0 < self.central_fraction <= 1:
            raise ValueError('central_fraction must be in (0, 1]')
        if self.minimum_duration <= 0:
            raise ValueError('minimum_duration must be positive')
        if self.time_step <= 0:
            raise ValueError('time_step must be positive')

    def formant_ceiling(self, gender):
        if gender == 'male':
            return self.male_max_formant
        if gender == 'female':
            return self.female_max_formant
        return self.other_max_formant

    def pitch_range(self, gender):
        if gender == 'male':
            return self.male_pitch_floor, self.male_pitch_ceiling
        if gender == 'female':
            return self.female_pitch_floor, self.female_pitch_ceiling
        return self.other_pitch_floor, self.other_pitch_ceiling


@dataclass(frozen=True)
class FormantMeasurement:
    '''A successful or rejected waveform measurement.'''

    success: bool
    f0_hz: float = np.nan
    f1_hz: float = np.nan
    f2_hz: float = np.nan
    f3_hz: float = np.nan
    b1_hz: float = np.nan
    b2_hz: float = np.nan
    b3_hz: float = np.nan
    n_samples: int = 0
    sample_rate: int = 0
    central_start_seconds: float = np.nan
    central_end_seconds: float = np.nan
    rejection_reason: str | None = None

    def to_dict(self):
        return asdict(self)


def measure_formants(
    waveform,
    sample_rate,
    gender=None,
    settings=None,
):
    '''Measure median F0 and F1-F3 in a waveform's stable central portion.

    Failures are returned as structured rejected measurements so a corpus run
    can retain and audit every token.
    '''

    settings = settings or MeasurementSettings()
    try:
        samples = _validate_waveform(waveform, sample_rate)
    except ValueError as error:
        return FormantMeasurement(
            success=False,
            n_samples=np.asarray(waveform).size,
            sample_rate=int(sample_rate) if sample_rate else 0,
            rejection_reason=str(error),
        )

    duration = samples.size / sample_rate
    if duration < settings.minimum_duration:
        return FormantMeasurement(
            success=False,
            n_samples=samples.size,
            sample_rate=sample_rate,
            rejection_reason=(
                f'duration {duration:.6f}s is shorter than '
                f'{settings.minimum_duration:.6f}s'
            ),
        )

    try:
        import parselmouth
    except ImportError as error:
        raise ImportError(
            'formant measurement requires praat-parselmouth'
        ) from error

    central_start, central_end = _central_window(
        duration, settings.central_fraction
    )
    times = np.arange(
        central_start,
        central_end + settings.time_step / 2,
        settings.time_step,
    )
    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(
            time_step=settings.time_step,
            max_number_of_formants=settings.max_number_of_formants,
            maximum_formant=settings.formant_ceiling(gender),
            window_length=settings.window_length,
            pre_emphasis_from=settings.pre_emphasis_from,
        )
        pitch_floor, pitch_ceiling = settings.pitch_range(gender)
        pitch = sound.to_pitch(
            time_step=settings.time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        values = {}
        for number in (1, 2, 3):
            values[f'f{number}_hz'] = _finite_median([
                formant.get_value_at_time(number, time)
                for time in times
            ])
            values[f'b{number}_hz'] = _finite_median([
                formant.get_bandwidth_at_time(number, time)
                for time in times
            ])
        values['f0_hz'] = _finite_median([
            pitch.get_value_at_time(time)
            for time in times
        ])
    except Exception as error:
        return FormantMeasurement(
            success=False,
            n_samples=samples.size,
            sample_rate=sample_rate,
            central_start_seconds=central_start,
            central_end_seconds=central_end,
            rejection_reason=f'Praat measurement failed: {error}',
        )

    reason = _plausibility_rejection(values, sample_rate)
    return FormantMeasurement(
        success=reason is None,
        n_samples=samples.size,
        sample_rate=sample_rate,
        central_start_seconds=central_start,
        central_end_seconds=central_end,
        rejection_reason=reason,
        **values,
    )


def _validate_waveform(waveform, sample_rate):
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError('waveform must be one-dimensional')
    if samples.size == 0:
        raise ValueError('waveform must not be empty')
    if not np.all(np.isfinite(samples)):
        raise ValueError('waveform contains non-finite samples')
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    if np.max(np.abs(samples)) == 0:
        raise ValueError('waveform is silent')
    return samples


def _central_window(duration, fraction):
    margin = duration * (1 - fraction) / 2
    return margin, duration - margin


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    return float(np.median(values))


def _plausibility_rejection(values, sample_rate):
    formants = [values[f'f{number}_hz'] for number in (1, 2, 3)]
    if not all(np.isfinite(formants)):
        return 'one or more formants could not be measured'
    if not formants[0] < formants[1] < formants[2]:
        return 'formants are not strictly ordered F1 < F2 < F3'
    if formants[0] < 100:
        return 'F1 is below 100 Hz'
    if formants[2] >= sample_rate / 2:
        return 'F3 is at or above Nyquist'
    if not np.isfinite(values['f0_hz']):
        return 'F0 could not be measured'
    return None
