from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import result


def _manifest(**overrides):
    manifest = {
        'cache_schema_version': 2, 'trainer_version': 2,
        'representation': 'embedding',
        'feature_parameters': {'model_name': 'model-a', 'layer': 9},
        'target_phoneme': 'p', 'classifier': 'logreg',
        'selected_sample_count': 30, 'selected_samples_hash': 'abc',
        'feature_set_hash': 'def',
    }
    manifest.update(overrides)
    return manifest


def _phone_result(root):
    return result.PhoneResult.embedding('p', 'model-a', layer=9, collar=500,
        root=root)


# -- manifests_match ---------------------------------------------------

def test_manifests_match_ignores_selection_and_version_fields():
    a = _manifest(selected_sample_count=30, selected_samples_hash='abc')
    b = _manifest(selected_sample_count=29, selected_samples_hash='xyz',
        cache_schema_version=3, trainer_version=3)

    assert result.manifests_match(a, b) is True


def test_manifests_match_detects_identity_field_difference():
    a = _manifest(classifier='logreg')
    b = _manifest(classifier='other')

    assert result.manifests_match(a, b) is False


def test_manifests_match_detects_feature_set_hash_difference():
    a = _manifest(feature_set_hash='def')
    b = _manifest(feature_set_hash='uvw')

    assert result.manifests_match(a, b) is False


# -- PhoneResult identity ------------------------------------------------

def test_phone_result_embedding_path_includes_identity_components(tmp_path):
    phone_result = _phone_result(tmp_path)

    assert phone_result.path == (
        tmp_path / 'model-a' / 'p' / 'layer09' / 'collar500')


def test_phone_result_mfcc_path_includes_identity_components(tmp_path):
    phone_result = result.PhoneResult.mfcc('p', frame='center',
        root=tmp_path)

    assert phone_result.path == tmp_path / 'mfcc' / 'p' / 'frame-center'


def test_phone_result_equality_follows_identity(tmp_path):
    first = _phone_result(tmp_path)
    same = _phone_result(tmp_path)
    different = result.PhoneResult.embedding('t', 'model-a', layer=9,
        collar=500, root=tmp_path)

    assert first == same
    assert first != different


# -- check_manifest -------------------------------------------------------

def test_check_manifest_saves_when_none_stored(tmp_path):
    phone_result = _phone_result(tmp_path)
    manifest = _manifest()

    phone_result.check_manifest(manifest)

    assert phone_result.load_run() == manifest


def test_check_manifest_accepts_irrelevant_field_drift(tmp_path):
    phone_result = _phone_result(tmp_path)
    phone_result.check_manifest(_manifest(selected_samples_hash='abc'))

    fresh = _phone_result(tmp_path)
    fresh.check_manifest(_manifest(selected_samples_hash='xyz'))


def test_check_manifest_raises_on_identity_mismatch(tmp_path):
    phone_result = _phone_result(tmp_path)
    phone_result.check_manifest(_manifest(classifier='logreg'))

    fresh = _phone_result(tmp_path)
    with pytest.raises(ValueError, match='manifest does not match'):
        fresh.check_manifest(_manifest(classifier='other'))


# -- folds ------------------------------------------------------------

def test_fold_save_and_load_round_trip(tmp_path):
    phone_result = _phone_result(tmp_path)
    fold = result.Fold(phone_result, 1)
    fold.save_results([('p', 1, 1), ('t', 0, 1)])

    lines = fold.load_tsv()

    assert [line.correct for line in lines] == [True, False]
    assert fold.accuracy == pytest.approx(0.5)


def test_phone_result_tracks_completeness_and_accuracy(tmp_path):
    phone_result = _phone_result(tmp_path)
    result.Fold(phone_result, 1).save_results([('p', 1, 1), ('t', 0, 1)])

    assert phone_result.complete is False
    assert phone_result.missing_fold_numbers == [2, 3, 4, 5]

    fresh = _phone_result(tmp_path)
    for number in (2, 3, 4, 5):
        result.Fold(fresh, number).save_results([('p', 1, 1), ('t', 0, 0)])

    reloaded = _phone_result(tmp_path)
    assert reloaded.complete is True
    assert reloaded.accuracies == pytest.approx([0.5, 1.0, 1.0, 1.0, 1.0])
    assert reloaded.mean_accuracy == pytest.approx(0.9)
