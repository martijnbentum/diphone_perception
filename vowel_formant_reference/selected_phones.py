'''Measure monophthongs directly from Phraser Phone objects.'''

from collections import Counter
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
from progressbar import progressbar

from .aggregation import MEASUREMENT_COLUMNS, aggregate_gender_measurements
from .formant_tables import DEFAULT_DATA_ROOT, write_manifest
from .measurement import MeasurementSettings, measure_formants


MONOPHTHONGS = frozenset({
    'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
    'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
    'ə',
})

_MONOPHTHONG_ORDER = (
    'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
    'iː', 'yː', 'eː', 'øː', 'aː', 'oː', 'uː',
    'ə',
)


@dataclass(frozen=True)
class PhoneFormantMeasurement:
    '''Formant results tied to a Phraser phone by its stable key.'''

    phone_key: bytes
    gender: str
    success: bool
    f0_hz: float | None = None
    f1_hz: float | None = None
    f2_hz: float | None = None
    f3_hz: float | None = None
    b1_hz: float | None = None
    b2_hz: float | None = None
    b3_hz: float | None = None
    rejection_reason: str | None = None

    def to_csv_record(self):
        record = asdict(self)
        record['phone_key'] = _phone_key_bytes(self.phone_key).hex()
        return record


def is_monophthong(label):
    return label in MONOPHTHONGS


def measure_and_write_phone_formants(
    phraser_phones,
    data_root=None,
    audio_loader=None,
    settings=None,
    limit=None,
    n_bootstrap=1000,
    confidence=0.95,
    seed=0,
):
    '''Measure Phraser phones, aggregate gender anchors, and write artifacts.

    Phraser is the source of truth for label, speaker, gender, audio, and
    timing. Stress is deliberately ignored. Overlapping speech is excluded.
    The function prints selected-vowel counts and every written path.
    '''

    settings = settings or MeasurementSettings()
    selected = _selected_phones(phraser_phones, limit=limit)
    counts = Counter(phone.label for phone in selected)
    ordered_counts = {
        label: counts[label]
        for label in _MONOPHTHONG_ORDER
        if counts[label]
    }
    print(f'Selected vowels: {ordered_counts}')

    audio_loader = audio_loader or _load_phone_audio
    measured = [
        (phone, _measure_phone(phone, audio_loader, settings))
        for phone in progressbar(selected)
    ]
    gender_formants = _aggregate_gender_formants(
        measured,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    paths = _write_measurement_artifacts(
        measured,
        gender_formants,
        ordered_counts,
        settings,
        data_root=data_root,
    )
    print('Written:')
    for name, path in paths.items():
        print(f'  {name}: {path.resolve()}')
    return paths


def _selected_phones(phraser_phones, limit=None):
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError('limit must be a non-negative integer or None')
    selected = []
    seen_keys = set()
    for phone in phraser_phones:
        if not is_monophthong(phone.label) or phone.overlap:
            continue
        key = _phone_key_bytes(phone.key)
        if key in seen_keys:
            raise ValueError(f'duplicate Phraser phone key: {key.hex()}')
        seen_keys.add(key)
        selected.append(phone)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _measure_phone(phone, audio_loader, settings):
    gender = phone.speaker.gender()
    try:
        waveform, sample_rate = audio_loader(phone)
        result = measure_formants(
            waveform,
            sample_rate,
            gender=gender,
            settings=settings,
        )
    except Exception as error:
        return PhoneFormantMeasurement(
            phone_key=_phone_key_bytes(phone.key),
            gender=gender,
            success=False,
            rejection_reason=f'audio loading failed: {error}',
        )
    values = {
        name: _finite_or_none(getattr(result, name))
        for name in MEASUREMENT_COLUMNS
    }
    return PhoneFormantMeasurement(
        phone_key=_phone_key_bytes(phone.key),
        gender=gender,
        success=result.success,
        rejection_reason=result.rejection_reason,
        **values,
    )


def _aggregate_gender_formants(
    measured,
    n_bootstrap,
    confidence,
    seed,
):
    groups = {}
    for phone, measurement in measured:
        if not measurement.success:
            continue
        key = (phone.speaker_id, measurement.gender, phone.label)
        groups.setdefault(key, []).append(measurement)

    speaker_rows = []
    for (speaker_id, gender, ipa), measurements in groups.items():
        row = {
            'speaker_id': _identifier(speaker_id),
            'gender': gender,
            'ipa': ipa,
            'n_tokens': len(measurements),
        }
        for name in MEASUREMENT_COLUMNS:
            values = [
                getattr(measurement, name)
                for measurement in measurements
                if getattr(measurement, name) is not None
            ]
            row[name] = float(np.median(values)) if values else None
        speaker_rows.append(row)

    if not speaker_rows:
        return []
    rows = aggregate_gender_measurements(
        speaker_rows,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    vowel_order = {
        label: index for index, label in enumerate(_MONOPHTHONG_ORDER)
    }
    rows.sort(key=lambda row: (
        row['gender'],
        vowel_order[row['ipa']],
    ))
    return rows


def _write_measurement_artifacts(
    measured,
    gender_formants,
    selected_counts,
    settings,
    data_root,
):
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    output_directory = root / 'selected_phones'
    output_directory.mkdir(parents=True, exist_ok=True)
    phone_path = output_directory / 'phone_formants.csv'
    metadata_path = output_directory / 'phone_formants_metadata.json'
    gender_path = output_directory / 'gender_formants.csv'

    phone_fields = [field.name for field in fields(PhoneFormantMeasurement)]
    phone_records = [
        measurement.to_csv_record()
        for _, measurement in measured
    ]
    _atomic_write_csv(phone_path, phone_fields, phone_records)
    gender_fields = _gender_formant_fields()
    _atomic_write_csv(gender_path, gender_fields, gender_formants)
    metadata = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'phone_key_encoding': 'hexadecimal Phraser LMDB key',
        'selection': {
            'monophthongs': list(_MONOPHTHONG_ORDER),
            'exclude_overlap': True,
            'check_stress': False,
            'selected_vowel_counts': selected_counts,
        },
        'measurements': {
            'n_selected': len(measured),
            'n_successful': sum(
                measurement.success for _, measurement in measured
            ),
            'n_rejected': sum(
                not measurement.success for _, measurement in measured
            ),
            'settings': asdict(settings),
        },
        'aggregation': {
            'method': 'median of per-speaker medians',
            'bootstrap_replicates': (
                gender_formants[0]['bootstrap_replicates']
                if gender_formants else None
            ),
            'confidence': (
                gender_formants[0]['confidence']
                if gender_formants else None
            ),
            'bootstrap_seed': (
                gender_formants[0]['bootstrap_seed']
                if gender_formants else None
            ),
        },
        'software_versions': _software_versions(),
    }
    _atomic_write_json(metadata_path, metadata)
    manifest_path = write_manifest(root)
    return {
        'phone_formants': phone_path,
        'phone_formants_metadata': metadata_path,
        'gender_formants': gender_path,
        'manifest': manifest_path,
    }


def _load_phone_audio(phone):
    try:
        from phraser.audio import load_audio_samples
    except ImportError as error:
        raise ImportError(
            'phone formant measurement requires phraser'
        ) from error
    audio = phone.audio
    if audio is None:
        raise ValueError('Phraser phone has no linked audio')
    sample_rate = int(audio.sample_rate)
    start_sample = round(phone.start_seconds * sample_rate)
    stop_sample = round(phone.end_seconds * sample_rate)
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


def _atomic_write_csv(path, fieldnames, records):
    temporary_path = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                lineterminator='\n',
            )
            writer.writeheader()
            writer.writerows([
                {
                    name: _csv_cell(record.get(name))
                    for name in fieldnames
                }
                for record in records
            ])
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path, value):
    temporary_path = path.with_name(f'.{path.name}.tmp')
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _gender_formant_fields():
    fields = [
        'source', 'record_level', 'gender', 'ipa',
        'n_speakers', 'n_tokens', 'aggregation',
        'bootstrap_seed', 'bootstrap_replicates', 'confidence',
    ]
    for name in MEASUREMENT_COLUMNS:
        fields.extend((name, f'{name}_ci_low', f'{name}_ci_high'))
    return fields


def _phone_key_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError('Phraser phone key must be bytes-like')


def _identifier(value):
    return value.hex() if isinstance(value, bytes) else str(value)


def _finite_or_none(value):
    number = float(value)
    return number if math.isfinite(number) else None


def _csv_cell(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, np.generic):
        return value.item()
    return value


def _software_versions():
    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    try:
        import parselmouth
    except ImportError:
        return versions
    versions.update({
        'parselmouth': parselmouth.__version__,
        'praat': parselmouth.PRAAT_VERSION,
    })
    return versions
