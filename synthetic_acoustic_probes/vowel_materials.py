'''Source-specific Dutch vowel-anchor materials.'''

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
from scipy.io import wavfile

from vowel_formant_reference.formant_tables import (
    LITERATURE_MONOPHTHONGS,
    adank_2004_formants,
    load_formant_table,
    weenink_1985_formants,
)

from .formants import praat_vowel_stimulus
from .stimuli import Stimulus


DEFAULT_SOURCE_IDS = (
    'cgn_selected_phones',
    'weenink_1985',
    'adank_2004_nsd',
    'adank_2004_ssd',
)

_DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / 'data'
    / 'synthetic_acoustic_probes'
    / 'vowel_formants'
)

_CGN_MONOPHTHONGS = (
    'ɪ', 'ɛ', 'ɑ', 'ɔ', 'ʉ',
    'iː', 'eː', 'aː', 'oː', 'uː', 'ə',
)


@dataclass(frozen=True)
class _SourceSpec:
    table_name: str
    population: str
    vowels: tuple[str, ...]


_SOURCE_SPECS = {
    'cgn_selected_phones': _SourceSpec(
        table_name='gender_formants',
        population='CGN selected phones',
        vowels=_CGN_MONOPHTHONGS,
    ),
    'weenink_1985': _SourceSpec(
        table_name='weenink_1985',
        population='Dutch adult',
        vowels=LITERATURE_MONOPHTHONGS,
    ),
    'adank_2004_nsd': _SourceSpec(
        table_name='adank_2004_table_1_monophthongs',
        population='Northern Standard Dutch',
        vowels=LITERATURE_MONOPHTHONGS,
    ),
    'adank_2004_ssd': _SourceSpec(
        table_name='adank_2004_table_1_monophthongs',
        population='Southern Standard Dutch',
        vowels=LITERATURE_MONOPHTHONGS,
    ),
}


def vowel_anchor_stimuli(
    source_id,
    *,
    data_root=None,
    duration=1.0,
    sample_rate=16_000,
    target_rms=0.1,
    bandwidths_hz=(80, 100),
):
    '''Synthesize every gender/vowel anchor for one supported source.'''

    spec = _source_spec(source_id)
    table, rows = _load_anchor_rows(source_id, spec, data_root)
    rows = _validated_anchor_rows(source_id, spec, rows)
    provenance = _anchor_provenance(table, data_root)

    stimuli = []
    for row in rows:
        stimulus_id = (
            f'vowel_{source_id}_{row["gender"]}_{row["ipa"]}'
        )
        synthesized = praat_vowel_stimulus(
            f0_hz=row['f0_hz'],
            f1_hz=row['f1_hz'],
            f2_hz=row['f2_hz'],
            bandwidths_hz=bandwidths_hz,
            duration=duration,
            sample_rate=sample_rate,
            target_rms=target_rms,
            stimulus_id=stimulus_id,
        )
        parameters = dict(synthesized.parameters)
        parameters.update({
            'source_id': source_id,
            'population': spec.population,
            'gender': row['gender'],
            'ipa': row['ipa'],
            'aggregation': row['aggregation'],
            'anchor_provenance': provenance.copy(),
        })
        stimuli.append(Stimulus(
            synthesized.waveform,
            synthesized.sample_rate,
            parameters,
            stimulus_id,
        ))
    return stimuli


def write_vowel_anchor_materials(
    source_ids=DEFAULT_SOURCE_IDS,
    *,
    output_root=None,
    overwrite=False,
):
    '''Write source-separated float32 WAV packages and JSON manifests.

    Existing source directories are left untouched unless ``overwrite`` is
    true. All requested packages are built in a staging directory before any
    destination is replaced.
    '''

    source_ids = _validated_source_ids(source_ids)
    output_root = (
        Path(output_root) if output_root is not None
        else _DEFAULT_OUTPUT_ROOT
    )
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        output_root / source_id
        for source_id in source_ids
        if (output_root / source_id).exists()
    ]
    if existing and not overwrite:
        paths = ', '.join(str(path) for path in existing)
        raise FileExistsError(
            f'refusing to replace existing source directories: {paths}; '
            'pass overwrite=True to replace them'
        )

    staging_root = Path(tempfile.mkdtemp(
        prefix='.vowel-formants-staging-',
        dir=output_root,
    ))
    try:
        for source_id in source_ids:
            _stage_source_package(staging_root, source_id)
        _commit_source_packages(
            staging_root,
            output_root,
            source_ids,
            overwrite,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {
        source_id: output_root / source_id
        for source_id in source_ids
    }


def _source_spec(source_id):
    try:
        return _SOURCE_SPECS[source_id]
    except (KeyError, TypeError) as error:
        normalized = str(source_id).lower()
        if 'pols' in normalized or 'nierop' in normalized:
            raise ValueError(
                f'F0 is unavailable for {source_id!r}; the Pols and Van '
                'Nierop sources are comparison-only'
            ) from error
        choices = ', '.join(DEFAULT_SOURCE_IDS)
        raise ValueError(
            f'unsupported source_id {source_id!r}; choose from {choices}'
        ) from error


def _load_anchor_rows(source_id, spec, data_root):
    if source_id == 'cgn_selected_phones':
        table = load_formant_table(
            spec.table_name,
            view='native',
            data_root=data_root,
        )
        rows = [
            {**row, 'population': spec.population}
            for row in table.data
        ]
    elif source_id == 'weenink_1985':
        table = weenink_1985_formants(data_root=data_root)
        rows = table.data
    else:
        table = adank_2004_formants(
            population=spec.population,
            data_root=data_root,
        )
        rows = table.data
    return table, rows


def _validated_anchor_rows(source_id, spec, rows):
    expected = {
        (gender, ipa)
        for gender in ('female', 'male')
        for ipa in spec.vowels
    }
    observed = []
    for row in rows:
        if row.get('population') != spec.population:
            continue
        key = (row.get('gender'), row.get('ipa'))
        if key not in expected:
            raise ValueError(
                f'{source_id!r} has unexpected anchor {key!r}'
            )
        observed.append(key)
        for name in ('f0_hz', 'f1_hz', 'f2_hz'):
            value = row.get(name)
            if not _finite_number(value):
                raise ValueError(
                    f'{source_id!r} anchor {key!r} has no finite {name}'
                )
        if not row.get('aggregation'):
            raise ValueError(
                f'{source_id!r} anchor {key!r} has no aggregation method'
            )

    duplicates = sorted({key for key in observed if observed.count(key) > 1})
    observed_set = set(observed)
    if duplicates or observed_set != expected:
        missing = sorted(expected - observed_set)
        raise ValueError(
            f'{source_id!r} does not contain exactly one anchor per '
            f'gender/vowel combination; missing={missing!r}, '
            f'duplicates={duplicates!r}'
        )
    order = {ipa: index for index, ipa in enumerate(spec.vowels)}
    return sorted(rows, key=lambda row: (
        ('female', 'male').index(row['gender']),
        order[row['ipa']],
    ))


def _anchor_provenance(table, data_root):
    source = table.source
    path = source.path(data_root)
    return {
        'table_name': source.name,
        'table_path': source.relative_path,
        'table_sha256': _file_sha256(path),
        'citation': source.reference,
        'url': source.url,
        'record_level': source.record_level,
    }


def _validated_source_ids(source_ids):
    if isinstance(source_ids, str):
        raise TypeError('source_ids must be an iterable of source IDs')
    source_ids = tuple(source_ids)
    for source_id in source_ids:
        _source_spec(source_id)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('source_ids must not contain duplicates')
    return source_ids


def _stage_source_package(staging_root, source_id):
    stimuli = vowel_anchor_stimuli(source_id)
    source_root = staging_root / source_id
    audio_root = source_root / 'audio'
    audio_root.mkdir(parents=True)
    rows = []
    for stimulus in stimuli:
        relative_path = Path('audio') / f'{stimulus.stimulus_id}.wav'
        path = source_root / relative_path
        wavfile.write(
            path,
            stimulus.sample_rate,
            stimulus.waveform.astype(np.float32, copy=False),
        )
        rows.append({
            'stimulus_id': stimulus.stimulus_id,
            'path': relative_path.as_posix(),
            'sha256': _file_sha256(path),
            'sample_rate_hz': stimulus.sample_rate,
            'dtype': str(stimulus.waveform.dtype),
            'n_samples': int(stimulus.waveform.size),
            'population': stimulus.parameters['population'],
            'gender': stimulus.parameters['gender'],
            'ipa': stimulus.parameters['ipa'],
            'f0_hz': stimulus.parameters['f0_hz'],
            'f1_hz': stimulus.parameters['f1_hz'],
            'f2_hz': stimulus.parameters['f2_hz'],
            'aggregation': stimulus.parameters['aggregation'],
        })

    first = stimuli[0]
    provenance = first.parameters['anchor_provenance']
    manifest = {
        'schema_version': 1,
        'source_id': source_id,
        'source_citation': provenance['citation'],
        'source_url': provenance['url'],
        'population': first.parameters['population'],
        'anchor_table': {
            'name': provenance['table_name'],
            'path': provenance['table_path'],
            'sha256': provenance['table_sha256'],
            'record_level': provenance['record_level'],
        },
        'synthesis_settings': {
            'generator': first.parameters['generator'],
            'duration_seconds': first.parameters['duration_seconds'],
            'sample_rate_hz': first.sample_rate,
            'target_rms': first.parameters['target_rms'],
            'fade_duration_seconds': (
                first.parameters['fade_duration_seconds']
            ),
            'bandwidths_hz': [
                first.parameters['bandwidth_1_hz'],
                first.parameters['bandwidth_2_hz'],
            ],
            'formants_hz': ['F1', 'F2'],
        },
        'software_versions': _software_versions(),
        'stimuli': rows,
    }
    _atomic_write_json(source_root / 'manifest.json', manifest)


def _commit_source_packages(
    staging_root,
    output_root,
    source_ids,
    overwrite,
):
    backup_root = Path(tempfile.mkdtemp(
        prefix='.vowel-formants-backup-',
        dir=output_root,
    ))
    committed = []
    try:
        for source_id in source_ids:
            staged = staging_root / source_id
            target = output_root / source_id
            backup = None
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f'refusing to replace existing source directory '
                        f'{target}; pass overwrite=True to replace it'
                    )
                backup = backup_root / source_id
                os.replace(target, backup)
            try:
                os.replace(staged, target)
            except Exception:
                if backup is not None:
                    os.replace(backup, target)
                raise
            committed.append((target, backup))
    except Exception:
        for target, backup in reversed(committed):
            shutil.rmtree(target)
            if backup is not None:
                os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


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


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _software_versions():
    import scipy

    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
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


def _finite_number(value):
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )
