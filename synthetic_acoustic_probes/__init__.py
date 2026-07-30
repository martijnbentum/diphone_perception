'''Model-independent synthetic acoustic probes.'''

from .acoustics import (
    WaveformMeasurements,
    designed_formant_response,
    dominant_fft_peaks,
    measure_waveform,
)
from .formants import (
    praat_formant_stimuli,
    praat_vowel_stimulus,
)
from .metrics import (
    accumulated_adjacent_cosine_scale,
    compare_frequency_scales,
    conditional_axis_monotonicity,
    cosine_distance_matrix,
    cross_validated_ridge_scores,
    local_neighbor_preservation,
    pairwise_geometry_spearman,
    structure_report,
)
from .stimuli import (
    Stimulus,
    amplitude_stimuli,
    bias_stimuli,
    energy_spaced_amplitudes,
    paper_code_amplitudes,
    pure_tone_stimuli,
    sinusoidal_component_formant_stimuli,
    sum_of_sinusoids,
    temporal_burst_stimuli,
)

__all__ = [
    'Stimulus',
    'WaveformMeasurements',
    'accumulated_adjacent_cosine_scale',
    'amplitude_stimuli',
    'bias_stimuli',
    'compare_frequency_scales',
    'conditional_axis_monotonicity',
    'cosine_distance_matrix',
    'cross_validated_ridge_scores',
    'designed_formant_response',
    'dominant_fft_peaks',
    'energy_spaced_amplitudes',
    'local_neighbor_preservation',
    'measure_waveform',
    'pairwise_geometry_spearman',
    'paper_code_amplitudes',
    'praat_formant_stimuli',
    'praat_vowel_stimulus',
    'pure_tone_stimuli',
    'sinusoidal_component_formant_stimuli',
    'sum_of_sinusoids',
    'structure_report',
    'temporal_burst_stimuli',
]
