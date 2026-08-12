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
from .f0_metrics import (
    f0_checkpoint_metrics,
    f0_checkpoint_step,
    f0_smoothness_metrics,
    load_f0_checkpoint_metrics,
)
from .f0_plot import (
    plot_f0_checkpoint_distance_heatmap,
    plot_f0_checkpoint_result,
    plot_f0_checkpoint_smoothness,
    plot_f0_umap,
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
from .storage import write_stimuli
from .umap_projection import project_umap
from .vowel_materials import (
    DEFAULT_SOURCE_IDS,
    vowel_anchor_stimuli,
    write_vowel_anchor_materials,
)
from .vowel_plots import (
    DEFAULT_VOWEL_PLOT_GENDERS,
    DEFAULT_VOWEL_PLOT_SOURCE_IDS,
    plot_vowel_formant_space,
)

__all__ = [
    'DEFAULT_SOURCE_IDS',
    'DEFAULT_VOWEL_PLOT_GENDERS',
    'DEFAULT_VOWEL_PLOT_SOURCE_IDS',
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
    'f0_checkpoint_metrics',
    'f0_checkpoint_step',
    'f0_smoothness_metrics',
    'load_f0_checkpoint_metrics',
    'local_neighbor_preservation',
    'measure_waveform',
    'pairwise_geometry_spearman',
    'paper_code_amplitudes',
    'plot_f0_checkpoint_distance_heatmap',
    'plot_f0_checkpoint_result',
    'plot_f0_checkpoint_smoothness',
    'plot_f0_umap',
    'plot_vowel_formant_space',
    'praat_formant_stimuli',
    'praat_vowel_stimulus',
    'pure_tone_stimuli',
    'project_umap',
    'sinusoidal_component_formant_stimuli',
    'sum_of_sinusoids',
    'structure_report',
    'temporal_burst_stimuli',
    'vowel_anchor_stimuli',
    'write_stimuli',
    'write_vowel_anchor_materials',
]
