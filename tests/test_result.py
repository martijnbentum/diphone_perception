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


def test_phone_result_cnn_path_includes_identity_components(tmp_path):
    phone_result = result.PhoneResult.model_feature(
        'p', 'model-a', layer='cnn', collar=500, root=tmp_path)

    assert phone_result.representation == 'cnn'
    assert phone_result.identity == (
        'p', 'cnn', 'model-a', 'cnn', 500, 'middle')
    assert phone_result.path == (
        tmp_path / 'model-a' / 'p' / 'layer-cnn' / 'collar500')
    assert result.PhoneResult.cnn(
        'p', 'model-a', collar=500, root=tmp_path) == phone_result
    assert result.cnn_result_path(
        'p', 'model-a', collar=500, root=tmp_path) == phone_result.path


def test_embedding_apis_reject_cnn_layer(tmp_path):
    with pytest.raises(ValueError, match='non-negative integer'):
        result.PhoneResult.embedding(
            'p', 'model-a', layer='cnn', collar=500, root=tmp_path)
    with pytest.raises(ValueError, match='non-negative integer'):
        result.embedding_result_path(
            'p', 'model-a', layer='cnn', collar=500, root=tmp_path)


@pytest.mark.parametrize(
    ('options', 'message'),
    [
        ({'representation': 'cnn', 'model_name': 'model-a', 'layer': 9,
            'collar': 500, 'frame': 'middle'}, "requires layer='cnn'"),
        ({'representation': 'embedding', 'model_name': 'model-a',
            'layer': 'cnn', 'collar': 500, 'frame': 'middle'},
            'non-negative integer'),
        ({'representation': 'mfcc', 'model_name': 'model-a', 'layer': None,
            'collar': None, 'frame': 'center'}, 'requires model_name'),
    ],
)
def test_phone_result_rejects_inconsistent_identity(
    tmp_path, options, message,
):
    with pytest.raises(ValueError, match=message):
        result.PhoneResult('p', root=tmp_path, **options)


def test_valid_representations_have_distinct_paths(tmp_path):
    results = [
        result.PhoneResult.embedding(
            'p', 'model-a', layer=9, collar=500, root=tmp_path),
        result.PhoneResult.cnn(
            'p', 'model-a', collar=500, root=tmp_path),
        result.PhoneResult.mfcc('p', frame='center', root=tmp_path),
    ]

    assert len({phone_result.identity for phone_result in results}) == 3
    assert len({phone_result.path for phone_result in results}) == 3


def test_layer_directory_name_preserves_numeric_convention():
    assert result.layer_directory_name(9) == 'layer09'
    assert result.layer_directory_name('cnn') == 'layer-cnn'


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


# -- checkpoint result inventory ----------------------------------------

def test_find_missing_checkpoint_layer_results_aggregates_phone_results(
    tmp_path,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    complete = result.PhoneResult.model_feature(
        'p', model_name, 9, 500, root=tmp_path)
    for fold_number in range(1, 6):
        result.Fold(complete, fold_number).save_results(
            [('p', 'target', 'target')])

    partial = result.PhoneResult.model_feature(
        'a', model_name, 9, 500, root=tmp_path)
    partial.save_run({'representation': 'embedding'})
    result.Fold(partial, 1).save_results([('a', 'target', 'target')])

    missing = result.find_missing_checkpoint_layer_results(
        ['p', 'a'], model_name, 9, collar=500, root=tmp_path)

    assert missing == [{
        'target_phoneme': 'a',
        'missing_fold_numbers': [2, 3, 4, 5],
        'run_manifest_missing': False,
    }]


def test_find_missing_checkpoint_results_groups_missing_results_by_layer(
    tmp_path,
):
    model_name = 'wav2vec2_nl1_checkpoint-1000'
    complete = result.PhoneResult.model_feature(
        'p', model_name, 9, 500, root=tmp_path)
    for fold_number in range(1, 6):
        result.Fold(complete, fold_number).save_results(
            [('p', 'target', 'target')])

    missing = result.find_missing_checkpoint_results(
        ['p'], model_name, (9, 'cnn'), collar=500, root=tmp_path)

    assert missing == {'cnn': [{
        'target_phoneme': 'p',
        'missing_fold_numbers': [1, 2, 3, 4, 5],
        'run_manifest_missing': True,
    }]}
