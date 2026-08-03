'''Shared schemas and persistence helpers for binary phone probes.'''

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from probing import metadata, probe_utils


_hash_chunk_size = 1024 * 1024
_maximum_error_message_length = 2_000
_expected_phone_label_count = 31
_selected_run_pointer_schema_version = 1
_selected_run_pointer_prefix = 'selected_run_'


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(text)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolved_path(path):
    return str(Path(path).expanduser().resolve())


def _file_fingerprint(path):
    path = Path(path).expanduser().resolve()
    output = {'path': str(path), 'exists': path.is_file()}
    if not output['exists']:
        return output

    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(_hash_chunk_size):
            digest.update(chunk)
    output.update({
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    })
    return output


def _replacement_fingerprint(phraser_key_path, replacement_path):
    if replacement_path is None:
        return {'mode': 'disabled', 'configured_path': None}

    key_path = Path(phraser_key_path).expanduser().resolve()
    replacement_path = Path(replacement_path).expanduser().resolve()
    default_key_path = Path(metadata.phraser_key_file).resolve()
    default_replacement_path = Path(
        metadata.duplicate_replacement_phraser_key_file).resolve()
    if (
        key_path != default_key_path
        and replacement_path == default_replacement_path
    ):
        return {
            'mode': 'ignored_default_for_custom_keys',
            'configured_path': str(replacement_path),
        }
    if not replacement_path.is_file():
        return {
            'mode': 'unavailable',
            'configured_path': str(replacement_path),
        }
    return {
        'mode': 'applied',
        'configured_path': str(replacement_path),
        'file': _file_fingerprint(replacement_path),
    }


def _phone_inventory_fingerprint(
    phraser_key_path, duplicate_replacement_phraser_key_path,
):
    inputs = {
        'phraser_keys': _file_fingerprint(phraser_key_path),
        'duplicate_replacements': _replacement_fingerprint(
            phraser_key_path, duplicate_replacement_phraser_key_path),
    }
    serialized = json.dumps(
        inputs, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')
    return {
        'fingerprint': hashlib.sha256(serialized).hexdigest(),
        'inputs': inputs,
    }


def _task_identity(phone, model_name, layer):
    return {
        'phone': phone,
        'model_name': model_name,
        'layer': layer,
    }


def _selected_run_selector(
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
):
    return {
        'representation': 'embedding',
        'target_phoneme': phone,
        'feature_parameters': {
            'model_name': model_name,
            'layer': layer,
            'collar': collar,
            'frame': 'middle',
        },
        'n_samples': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'classifier': probe_utils.classifier_manifest(standardize),
    }


def _selected_run_selector_id(selector):
    serialized = json.dumps(
        selector,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()[:16]


def _selected_run_pointer_path(probe_save_dir, selector):
    features = selector['feature_parameters']
    directory = (
        Path(probe_save_dir) / features['model_name']
        / selector['target_phoneme'] / f'layer{features["layer"]:02d}'
        / f'collar{features["collar"]}ms'
    )
    selector_id = _selected_run_selector_id(selector)
    return directory / f'{_selected_run_pointer_prefix}{selector_id}.json'


def _write_selected_run_pointer(
    *,
    phone,
    model_name,
    layer,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    probe_save_dir,
    run_id,
    worker_status,
):
    selector = _selected_run_selector(
        phone,
        model_name,
        layer,
        collar,
        n_embeds,
        n_splits,
        random_state,
        standardize,
    )
    selector_id = _selected_run_selector_id(selector)
    path = _selected_run_pointer_path(probe_save_dir, selector)
    pointer = {
        'schema_version': _selected_run_pointer_schema_version,
        'kind': 'phone_binary_probe_selected_run',
        'selector_id': selector_id,
        'selector': selector,
        'run_id': run_id,
        'worker_status': worker_status,
        'updated_at': _utc_timestamp(),
    }
    _write_json_atomic(path, pointer)
    return path


def _compact_error(error):
    message = str(error)
    if len(message) > _maximum_error_message_length:
        message = message[:_maximum_error_message_length - 3] + '...'
    return {
        'type': type(error).__name__,
        'message': message,
    }


def _close_phones_store(phones):
    store = getattr(phones, '_store', None)
    if store is not None:
        store.close()


def _validated_phone_label_inventory(phones, n_embeds=None):
    labels = probe_utils.prepare_balanced_probe_targets(
        phones, target_phonemes=None, n_samples=n_embeds)
    if len(labels) != _expected_phone_label_count:
        raise ValueError(
            f'expected {_expected_phone_label_count} Netherlandic phone '
            f'labels, found {len(labels)}')
    counts = {
        len(items) for items in phones.label_to_phraser_phone.values()}
    if len(counts) != 1:
        raise ValueError('phone inventory is not balanced')
    return labels, next(iter(counts))


def _validate_cached_phone_labels(labels, phones_per_label, n_embeds):
    labels = _report_phone_labels(labels)
    if len(labels) != _expected_phone_label_count:
        raise ValueError(
            f'expected {_expected_phone_label_count} Netherlandic phone '
            f'labels, found {len(labels)}')
    if (
        isinstance(phones_per_label, bool)
        or not isinstance(phones_per_label, int)
        or phones_per_label <= 0
    ):
        raise ValueError('cached phones_per_label must be a positive integer')
    requested = phones_per_label if n_embeds is None else n_embeds
    if requested > phones_per_label:
        raise ValueError(
            f'n_embeds={requested} exceeds the balanced inventory of '
            f'{phones_per_label} items per label')
    if requested // (len(labels) - 1) == 0:
        raise ValueError(
            f'n_embeds={requested} is too small to split across '
            f'{len(labels) - 1} other phone labels')
    return labels


def _sweep_counts(outcomes):
    return {
        status: sum(outcome['status'] == status for outcome in outcomes)
        for status in (
            'completed', 'already_complete', 'failed', 'interrupted')
    }


def _report_phone_labels(phone_labels):
    labels = list(phone_labels)
    if not labels:
        raise ValueError('phone_labels must not be empty')
    for label in labels:
        probe_utils.validate_target_phoneme(label)
    if len(set(labels)) != len(labels):
        raise ValueError('phone_labels contains duplicate labels')
    return labels
