from pathlib import Path

import pytest

from synthetic_acoustic_probes._checkpoint_naming import checkpoint_step
from synthetic_acoustic_probes._checkpoint_naming import discover_checkpoints
from synthetic_acoustic_probes._checkpoint_naming import model_name_to_path
from synthetic_acoustic_probes._checkpoint_naming import path_to_model_name


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    (
        ('wav2vec2_checkpoint-0', 0),
        ('wav2vec2_nl1_checkpoint-1', 1),
        ('wav2vec2_nl1_checkpoint-200000', 200000),
    ),
)
def test_checkpoint_step_parses_supported_models(model_name, expected):
    '''Random and trained checkpoint names map to numeric steps.'''

    assert checkpoint_step(model_name) == expected


@pytest.mark.parametrize('model_name', ('', None, 'checkpoint-20'))
def test_checkpoint_step_rejects_unsupported_models(model_name):
    '''Unsupported checkpoint names are not silently assigned a step.'''

    with pytest.raises(ValueError):
        checkpoint_step(model_name)


def test_model_name_to_path_finds_existing_result(tmp_path):
    '''An existing npz result resolves to output_directory/model_name.npz.'''

    model_name = 'wav2vec2_nl1_checkpoint-200000'
    expected = tmp_path / f'{model_name}.npz'
    expected.touch()
    assert model_name_to_path(model_name, tmp_path) == expected


def test_model_name_to_path_rejects_missing_result(tmp_path):
    '''A model name with no npz result under output_directory fails.'''

    with pytest.raises(FileNotFoundError):
        model_name_to_path('wav2vec2_nl1_checkpoint-200000', tmp_path)


def test_path_to_model_name_returns_the_stem():
    '''The model name is the npz result path's filename without suffix.'''

    path = Path('/some/dir/wav2vec2_nl1_checkpoint-200000.npz')
    assert path_to_model_name(path) == 'wav2vec2_nl1_checkpoint-200000'


def test_discover_checkpoints_sorts_by_training_step(tmp_path):
    '''Discovered model names are ordered by step, not directory order.'''

    for model_name in ('wav2vec2_nl1_checkpoint-200000',
        'wav2vec2_checkpoint-0', 'wav2vec2_nl1_checkpoint-100000'):
        (tmp_path / f'{model_name}.npz').touch()
    assert discover_checkpoints(tmp_path) == (
        'wav2vec2_checkpoint-0', 'wav2vec2_nl1_checkpoint-100000',
        'wav2vec2_nl1_checkpoint-200000')


def test_discover_checkpoints_rejects_missing_directory(tmp_path):
    '''A non-existent output_directory fails clearly.'''

    with pytest.raises(FileNotFoundError, match='directory not found'):
        discover_checkpoints(tmp_path / 'missing')


def test_discover_checkpoints_rejects_empty_directory(tmp_path):
    '''An existing but empty output_directory fails clearly.'''

    with pytest.raises(FileNotFoundError, match='no checkpoint results'):
        discover_checkpoints(tmp_path)
