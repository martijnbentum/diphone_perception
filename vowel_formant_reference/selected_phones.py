'''Selection and measurement of monophthongs from the linked phone corpus.'''

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import (
    aggregate_gender_measurements,
    aggregate_speaker_measurements,
)
from .formant_tables import write_formant_table, write_manifest
from .measurement import MeasurementSettings, measure_formants


MONOPHTHONGS = frozenset({
    'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
    'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
    'ə',
})
FULL_MONOPHTHONGS = MONOPHTHONGS.difference({'ə'})


def is_monophthong(label):
    return label in MONOPHTHONGS


def select_monophthong_rows(
    data,
    label_column='ipa_phoneme',
    overlap_column='overlap',
):
    '''Select only non-overlapping monophthong rows from a DataFrame.'''

    frame = pd.DataFrame(data).copy()
    if label_column not in frame:
        raise ValueError(f'missing label column {label_column!r}')
    selected = frame[label_column].isin(MONOPHTHONGS)
    if overlap_column in frame:
        selected &= ~_boolean_series(frame[overlap_column])
    return frame.loc[selected].copy()


def measure_selected_phones(
    phones,
    phraser_phones=None,
    audio_loader=None,
    settings=None,
    limit=None,
):
    '''Measure linked selected-phone monophthongs.

    `phones` can be a `probing.metadata.Phones` instance or an iterable of
    metadata Phone objects. The returned table includes rejected rows.
    '''

    settings = settings or MeasurementSettings()
    metadata_phones, segments = _aligned_phones(phones, phraser_phones)
    audio_loader = audio_loader or _load_segment_audio
    rows = []
    measured = 0
    for index, (phone, segment) in enumerate(zip(metadata_phones, segments)):
        label = _phone_label(phone)
        if not is_monophthong(label):
            continue
        if getattr(phone, 'overlap', False):
            rows.append(_rejected_row(
                index, phone, segment, label, 'overlapping speech'
            ))
            continue
        stress = getattr(segment, 'stress', 'unknown')
        stress_rejection = _stress_rejection(label, stress)
        if stress_rejection:
            rows.append(_rejected_row(
                index, phone, segment, label, stress_rejection
            ))
            continue
        if limit is not None and measured >= limit:
            break
        measured += 1
        base = _base_row(index, phone, segment, label)
        try:
            waveform, sample_rate = audio_loader(segment)
            result = measure_formants(
                waveform,
                sample_rate,
                gender=base['gender'],
                settings=settings,
            )
            base.update(result.to_dict())
        except Exception as error:
            base.update({
                'success': False,
                'rejection_reason': f'audio loading failed: {error}',
            })
        base['measurement_settings'] = repr(settings)
        rows.append(base)
    return pd.DataFrame(rows)


def write_selected_phone_measurements(
    token_data,
    data_root=None,
    n_bootstrap=1000,
    confidence=0.95,
    seed=0,
):
    '''Write token, speaker, and gender tables as separate artifacts.'''

    tokens = pd.DataFrame(token_data)
    speakers = aggregate_speaker_measurements(tokens)
    genders = aggregate_gender_measurements(
        speakers,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    paths = {
        'tokens': write_formant_table(
            'selected_phone_tokens', tokens, data_root
        ),
        'speakers': write_formant_table(
            'selected_phone_speakers', speakers, data_root
        ),
        'genders': write_formant_table(
            'selected_phone_genders', genders, data_root
        ),
    }
    paths['manifest'] = write_manifest(data_root)
    return paths


def _aligned_phones(phones, phraser_phones):
    if hasattr(phones, 'phones'):
        metadata_phones = phones.phones
        segments = phones.phraser_phones if phraser_phones is None else phraser_phones
    else:
        metadata_phones = list(phones)
        if phraser_phones is None:
            segments = [
                phone.phraser_phone()
                for phone in metadata_phones
            ]
        else:
            segments = list(phraser_phones)
    if len(metadata_phones) != len(segments):
        raise ValueError('metadata and Phraser phone counts do not match')
    return metadata_phones, segments


def _phone_label(phone):
    return getattr(
        phone,
        'phoneme_ipa',
        getattr(phone, 'ipa_phoneme', None),
    )


def _stress_rejection(label, stress):
    if label == 'ə' and stress != 'unstressed':
        return f'schwa requires unstressed context, got {stress!r}'
    if label in FULL_MONOPHTHONGS and stress != 'primary':
        return f'full vowel requires primary stress, got {stress!r}'
    return None


def _base_row(index, phone, segment, label):
    speaker = getattr(phone, 'speaker', None)
    return {
        'source': 'selected_phone_tokens',
        'record_level': 'token',
        'token_index': index,
        'audio_filename': getattr(phone, 'audio_filename', None),
        'speaker_id': getattr(phone, 'speaker_id', None),
        'gender': getattr(speaker, 'gender', None),
        'age': getattr(speaker, 'age', None),
        'ipa': label,
        'stress': getattr(segment, 'stress', 'unknown'),
        'duration_seconds': (
            getattr(phone, 'duration_seconds', None)
            or getattr(segment, 'duration', np.nan) / 1000
        ),
        'start_seconds': getattr(phone, 'start_seconds', np.nan),
        'end_seconds': getattr(phone, 'end_seconds', np.nan),
    }


def _rejected_row(index, phone, segment, label, reason):
    row = _base_row(index, phone, segment, label)
    row.update({'success': False, 'rejection_reason': reason})
    return row


def _load_segment_audio(segment):
    try:
        from phraser.audio import load_audio_samples
    except ImportError as error:
        raise ImportError(
            'default selected-phone loading requires phraser'
        ) from error
    audio = segment.audio
    if audio is None:
        raise ValueError('Phraser phone has no linked audio')
    sample_rate = int(audio.sample_rate)
    start_sample = round(segment.start / 1000 * sample_rate)
    stop_sample = round(segment.end / 1000 * sample_rate)
    waveform, loaded_rate = load_audio_samples(
        Path(audio.filename),
        start_sample=start_sample,
        stop_sample=stop_sample,
    )
    if loaded_rate != sample_rate:
        raise ValueError(
            f'loaded sample rate {loaded_rate} != metadata {sample_rate}'
        )
    return waveform, sample_rate


def _boolean_series(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.fillna(False).map(
        lambda value: (
            value.strip().lower() in {'true', '1', 'yes'}
            if isinstance(value, str) else bool(value)
        )
    )
    return normalized.astype(bool)
