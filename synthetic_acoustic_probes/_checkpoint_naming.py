'''Wav2vec2 checkpoint name/step/path conversions shared across probes.'''

from pathlib import Path
import re

import locations


def checkpoint_step(model_name):
    '''Return the numeric training step encoded by a checkpoint model name.
    model_name:  Registered wav2vec2 NL1 checkpoint or random-init name.
    '''
    if not isinstance(model_name, str) or not model_name:
        raise ValueError('model_name must be a non-empty string')
    if model_name == locations.wav2vec2_random_checkpoint_name: return 0
    match = re.fullmatch(locations.wav2vec2_nl1_checkpoint_pattern,
        model_name)
    if match is None:
        raise ValueError(f'unsupported checkpoint model: {model_name!r}')
    return int(match.group(1))


def model_name_to_path(model_name, output_directory):
    '''Return the existing npz result path for one checkpoint model name.
    model_name:        Registered wav2vec2 checkpoint model name.
    output_directory:  Directory holding one npz result file per checkpoint.
    '''
    result_path = Path(output_directory) / f'{model_name}.npz'
    if not result_path.is_file():
        message = f'checkpoint result not found: {result_path}'
        raise FileNotFoundError(message)
    return result_path


def path_to_model_name(result_path):
    '''Return the checkpoint model name encoded by an npz result path.
    result_path:  Path to one checkpoint result .npz bundle.
    '''
    return Path(result_path).stem
