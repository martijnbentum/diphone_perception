'''Sample-offset experiment and temporal diagnostics for CNN features.'''

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from phraser import Store

import locations

from .cnn_extraction import extract_cnn_checkpoint
from .echoframe_store import create_store
from .phraser_store import add_stimuli, load_stimuli
from .stimuli import DURATION, SAMPLE_RATE
from .stimuli import sum_of_sinusoids
from .storage import write_stimuli


PHASE_DIAGNOSTIC_MODEL_NAME = 'wav2vec2_nl1_checkpoint-200000'
PHASE_DIAGNOSTIC_PHRASER_SOURCE_ID = 'f0-phase-diagnostics'
PHASE_DIAGNOSTIC_FREQUENCIES_HZ = (
    3190, 3200, 3210,
    3490, 3500, 3510,
    3590, 3600, 3610,
    3950, 3960, 3970, 3980, 3990, 4000,
    4010, 4020, 4030, 4040, 4050,
    4750, 4760, 4770, 4780, 4790, 4800,
    4810, 4820, 4830, 4840, 4850,
    6390, 6400, 6410,
)
PHASE_DIAGNOSTIC_SAMPLE_OFFSETS = tuple(range(5))


@dataclass(frozen=True)
class CNNPhaseDiagnostics:
    '''Frame-aggregation measurements for one stored CNN representation.'''
    stimulus_id: str
    mean_vector: np.ndarray
    middle_vector: np.ndarray
    even_mean_vector: np.ndarray
    odd_mean_vector: np.ndarray
    mean_norm: float
    middle_norm: float
    mean_frame_norm: float
    cancellation_ratio: float | None
    even_mean_norm: float
    odd_mean_norm: float
    even_odd_cosine_distance: float | None


def phase_diagnostic_stimuli(
    frequencies=PHASE_DIAGNOSTIC_FREQUENCIES_HZ,
    sample_offsets=PHASE_DIAGNOSTIC_SAMPLE_OFFSETS,
    *,
    duration=DURATION,
    sample_rate=SAMPLE_RATE,
    amplitude=1.0,
):
    '''Generate pure tones at five input-sample alignments.
    A sample offset advances the original sinusoid by that many samples:
    ``x_offset[n] = x_original[n + sample_offset]``. For the default
    one-second, integer-frequency tones, this changes phase while retaining
    frequency, amplitude, duration, and FFT magnitude.
    '''
    frequencies = tuple(frequencies)
    sample_offsets = tuple(sample_offsets)
    if not frequencies: raise ValueError('frequencies must not be empty')
    if 0 not in sample_offsets:
        raise ValueError('sample_offsets must include zero')
    stimuli = []
    for frequency in frequencies:
        for sample_offset in sample_offsets:
            if (
                isinstance(sample_offset, bool)
                or not isinstance(sample_offset, (int, np.integer))
                or sample_offset < 0
            ):
                message = 'sample offsets must be non-negative integers'
                raise ValueError(message)
            phase = 2 * np.pi * frequency * sample_offset / sample_rate
            stimulus_id = (
                f'pure-tone_f-{float(frequency):g}'
                f'_sample-offset-{int(sample_offset)}'
            )
            stimulus = sum_of_sinusoids(
                frequency,
                amplitudes=amplitude,
                phases=phase,
                duration=duration,
                sample_rate=sample_rate,
                stimulus_id=stimulus_id,
                extra_parameters={
                    'family': 'f0_phase_diagnostics',
                    'sample_offset': int(sample_offset),
                },
            )
            stimuli.append(stimulus)
    return tuple(stimuli)


def create_phase_diagnostic_stimuli(*, output_root=None, overwrite=False):
    '''Generate and persist the 170-stimulus phase-diagnostic panel.'''
    if output_root is None:
        output_root = locations.f0_phase_diagnostic_stimuli
    stimuli = phase_diagnostic_stimuli()
    write_stimuli(stimuli, output_root, overwrite=overwrite)
    return stimuli


def create_phase_diagnostic_phraser_store(
    *,
    stimulus_package=None,
    store_path=None,
):
    '''Create a Phraser store containing the phase-diagnostic stimuli.'''
    if stimulus_package is None:
        stimulus_package = locations.f0_phase_diagnostic_stimuli
    if store_path is None:
        store_path = locations.f0_phase_diagnostic_phraser_store
    store = Store(store_path)
    try: add_stimuli(stimulus_package, store)
    except Exception:
        store.close()
        raise
    return store


def create_phase_diagnostic_echoframe_store(
    phraser_store,
    *,
    store_path=None,
    model_name=PHASE_DIAGNOSTIC_MODEL_NAME,
):
    '''Create a one-model Echoframe store and attach the Phraser store.'''
    if store_path is None:
        store_path = locations.f0_phase_diagnostic_echoframe_store
    store = create_store(store_path, (model_name,))
    try:
        store.attach_phraser_store(
            PHASE_DIAGNOSTIC_PHRASER_SOURCE_ID,
            phraser_store,
        )
    except Exception:
        store.close()
        raise
    return store


def extract_phase_diagnostic_cnn_features(
    store,
    *,
    model_name=PHASE_DIAGNOSTIC_MODEL_NAME,
    stimulus_package=None,
    gpu=False,
    overwrite=False,
):
    '''Extract final CNN features for the phase-diagnostic stimuli.'''
    _, phrases = _phase_rows_and_phrases(store, stimulus_package)
    extract_cnn_checkpoint(
        phrases,
        model_name,
        store,
        collar=0,
        gpu=gpu,
        overwrite=overwrite,
    )


def run_phase_diagnostics(
    store,
    *,
    model_name=PHASE_DIAGNOSTIC_MODEL_NAME,
    stimulus_package=None,
    output_path=None,
    overwrite=False,
):
    '''Run diagnostics and persist aligned vectors and measurements.
    In addition to the per-stimulus frame measurements, the NPZ stores cosine
    and Euclidean distances from each mean vector to offset zero at the same
    frequency. Those within-frequency comparisons are the direct test of
    sample-alignment sensitivity; no cross-frequency all-pairs matrix is made.
    '''
    if output_path is None:
        output_path = locations.f0_phase_diagnostic_results
    rows, phrases = _phase_rows_and_phrases(store, stimulus_package)
    diagnostics = diagnose_cnn_phase(phrases, model_name, store, collar=0)
    _write_phase_diagnostics(
        diagnostics,
        rows,
        model_name,
        output_path,
        overwrite=overwrite,
    )
    return diagnostics


def run_phase_diagnostic_experiment(
    *,
    output_root=None,
    model_name=PHASE_DIAGNOSTIC_MODEL_NAME,
    gpu=False,
):
    '''Create every artifact, extract features, and save diagnostics.
    Creation is intentionally one-shot: existing stimulus or store paths are
    rejected. Interrupted CNN extraction can be resumed with
    ``extract_phase_diagnostic_cnn_features`` and diagnostics can then be run
    separately.
    '''
    if output_root is None: output_root = locations.f0_phase_diagnostics
    output_root = Path(output_root)
    stimulus_package = output_root / 'stimuli'
    phraser_store_path = output_root / 'phraser_store'
    echoframe_store_path = output_root / 'echoframe_store'
    result_path = output_root / 'cnn_phase_diagnostics.npz'
    _reject_existing_experiment_paths((
        stimulus_package,
        phraser_store_path,
        echoframe_store_path,
        result_path,
    ))
    create_phase_diagnostic_stimuli(output_root=stimulus_package)
    phraser_store = create_phase_diagnostic_phraser_store(
        stimulus_package=stimulus_package,
        store_path=phraser_store_path,
    )
    store = None
    try:
        store = create_phase_diagnostic_echoframe_store(
            phraser_store,
            store_path=echoframe_store_path,
            model_name=model_name,
        )
        extract_phase_diagnostic_cnn_features(
            store,
            model_name=model_name,
            stimulus_package=stimulus_package,
            gpu=gpu,
        )
        return run_phase_diagnostics(
            store,
            model_name=model_name,
            stimulus_package=stimulus_package,
            output_path=result_path,
        )
    finally:
        if store is not None: store.close()
        phraser_store.close()


def diagnose_cnn_phase(phrases, model_name, store, *, collar=0):
    '''Measure frame cancellation in stored CNN features for each Phrase.
    phrases:     Ordered iterable of native Phraser Phrase objects.
    model_name:  Registered Echoframe model name.
    store:       Echoframe Store containing the CNN features.
    collar:      Context in milliseconds used during extraction.
    Returns one ``CNNPhaseDiagnostics`` record per Phrase, in input order.
    '''
    phrases = tuple(phrases)
    if not phrases: raise ValueError('phrases must not be empty')
    diagnostics = []
    for phrase in phrases:
        feature = store.phraser_key_to_cnn_feature(
            phrase.key,
            model_name,
            collar=collar,
        )
        diagnostics.append(_diagnose_feature(feature, phrase))
    return tuple(diagnostics)


def _diagnose_feature(feature, phrase):
    frames = feature.data
    mean_vector = np.asarray(
        feature.aggregate_segment(phrase, method='mean')
    )
    middle_vector = np.asarray(
        feature.aggregate_segment(phrase, method='middle')
    )
    even_mean_vector = frames[::2].mean(axis=0)
    odd_mean_vector = frames[1::2].mean(axis=0)
    mean_norm = _norm(mean_vector)
    middle_norm = _norm(middle_vector)
    frame_norms = np.linalg.norm(frames, axis=1)
    mean_frame_norm = float(frame_norms.mean())
    even_mean_norm = _norm(even_mean_vector)
    odd_mean_norm = _norm(odd_mean_vector)
    cancellation_ratio = None
    if mean_frame_norm:
        cancellation_ratio = mean_norm / mean_frame_norm
    return CNNPhaseDiagnostics(
        stimulus_id=phrase.label,
        mean_vector=mean_vector,
        middle_vector=middle_vector,
        even_mean_vector=even_mean_vector,
        odd_mean_vector=odd_mean_vector,
        mean_norm=mean_norm,
        middle_norm=middle_norm,
        mean_frame_norm=mean_frame_norm,
        cancellation_ratio=cancellation_ratio,
        even_mean_norm=even_mean_norm,
        odd_mean_norm=odd_mean_norm,
        even_odd_cosine_distance=_cosine_distance(
            even_mean_vector,
            odd_mean_vector,
        ),
    )


def _norm(vector):
    return float(np.linalg.norm(vector))


def _cosine_distance(left, right):
    denominator = _norm(left) * _norm(right)
    if not denominator: return None
    similarity = float(np.dot(left, right) / denominator)
    return 1 - similarity


def _phase_rows_and_phrases(store, stimulus_package):
    if stimulus_package is None:
        stimulus_package = locations.f0_phase_diagnostic_stimuli
    rows = _phase_manifest_rows(stimulus_package)
    phraser_store = store.load_phraser_store(
        PHASE_DIAGNOSTIC_PHRASER_SOURCE_ID
    )
    phrases = load_stimuli(phraser_store)
    phrases_by_id = {phrase.label: phrase for phrase in phrases}
    try:
        ordered = tuple(phrases_by_id[row['stimulus_id']] for row in rows)
    except KeyError as error:
        message = f'phase-diagnostic Phrase not found: {error.args[0]!r}'
        raise ValueError(message) from error
    return rows, ordered


def _phase_manifest_rows(stimulus_package):
    manifest_path = Path(stimulus_package) / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'stimulus manifest not found: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid stimulus manifest: {error}') from error
    rows = manifest.get('stimuli')
    if manifest.get('schema_version') != 1 or not isinstance(rows, list):
        raise ValueError(f'invalid stimulus manifest: {manifest_path}')
    return tuple(rows)


def _write_phase_diagnostics(
    diagnostics,
    rows,
    model_name,
    output_path,
    *,
    overwrite,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'wb' if overwrite else 'xb'
    frequencies = np.asarray([
        row['parameters']['frequencies_hz'][0] for row in rows
    ], dtype=float)
    sample_offsets = np.asarray([
        row['parameters']['sample_offset'] for row in rows
    ], dtype=int)
    phases = np.asarray([
        row['parameters']['phases_radians'][0] for row in rows
    ], dtype=float)
    sample_rates = np.asarray([
        row['sample_rate_hz'] for row in rows
    ], dtype=int)
    mean_vectors = np.stack([
        item.mean_vector for item in diagnostics
    ])
    cosine, euclidean = _offset_zero_distances(
        mean_vectors,
        frequencies,
        sample_offsets,
    )
    arrays = {
        'stimulus_ids': np.asarray([
            item.stimulus_id for item in diagnostics
        ]),
        'frequencies_hz': frequencies,
        'sample_offsets': sample_offsets,
        'phases_radians': phases,
        'sample_rates_hz': sample_rates,
        'mean_vectors': mean_vectors,
        'middle_vectors': np.stack([
            item.middle_vector for item in diagnostics
        ]),
        'even_mean_vectors': np.stack([
            item.even_mean_vector for item in diagnostics
        ]),
        'odd_mean_vectors': np.stack([
            item.odd_mean_vector for item in diagnostics
        ]),
        'mean_norms': _diagnostic_values(diagnostics, 'mean_norm'),
        'middle_norms': _diagnostic_values(diagnostics, 'middle_norm'),
        'mean_frame_norms': _diagnostic_values(
            diagnostics,
            'mean_frame_norm',
        ),
        'cancellation_ratios': _diagnostic_values(
            diagnostics,
            'cancellation_ratio',
        ),
        'even_mean_norms': _diagnostic_values(
            diagnostics,
            'even_mean_norm',
        ),
        'odd_mean_norms': _diagnostic_values(
            diagnostics,
            'odd_mean_norm',
        ),
        'even_odd_cosine_distances': _diagnostic_values(
            diagnostics,
            'even_odd_cosine_distance',
        ),
        'mean_offset_zero_cosine_distances': cosine,
        'mean_offset_zero_euclidean_distances': euclidean,
        'model_name': np.asarray(model_name),
        'collar_ms': np.asarray(0),
    }
    with output_path.open(mode) as stream:
        np.savez_compressed(stream, **arrays)


def _offset_zero_distances(vectors, frequencies, sample_offsets):
    baseline_by_frequency = {}
    for index, (frequency, sample_offset) in enumerate(zip(
        frequencies,
        sample_offsets,
        strict=True,
    )):
        if sample_offset == 0: baseline_by_frequency[frequency] = vectors[index]
    cosine = []
    euclidean = []
    for vector, frequency in zip(vectors, frequencies, strict=True):
        try: baseline = baseline_by_frequency[frequency]
        except KeyError as error:
            message = f'offset-zero stimulus missing for {frequency:g} Hz'
            raise ValueError(message) from error
        distance = _cosine_distance(vector, baseline)
        cosine.append(np.nan if distance is None else distance)
        euclidean.append(_norm(vector - baseline))
    return np.asarray(cosine), np.asarray(euclidean)


def _diagnostic_values(diagnostics, attribute):
    values = []
    for item in diagnostics:
        value = getattr(item, attribute)
        values.append(np.nan if value is None else value)
    return np.asarray(values, dtype=float)


def _reject_existing_experiment_paths(paths):
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        formatted = ', '.join(str(path) for path in existing)
        raise FileExistsError(f'phase-diagnostic output exists: {formatted}')
