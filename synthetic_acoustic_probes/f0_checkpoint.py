'''Loaded, validated F0 checkpoint results read from result archives.'''

from pathlib import Path
import re

import numpy as np

import locations

from .f0_distances import (f0_adjacent_distances, f0_pairwise_distances,
    f0_pairwise_frequency_correlation)


class F0Checkpoint:
    '''One checkpoint's CNN features, their frequencies, and provenance.
    model_name:        F0 checkpoint model name; selects {model_name}.npz
                       under locations.f0_output_data.
    frequencies:       Per-sample frequencies in Hz, as loaded.
    cnn:               Samples by CNN features matrix.
    aggregation:       Frame-aggregation method used to build the features.
    umap_metric:       Distance metric used for the stored UMAP projection.
    umap_coordinates:  Samples by 2 UMAP coordinates, computed from cnn
                       with umap_metric.
    random_state:      Random seed recorded with the result.
    result_path:       Path the checkpoint was loaded from.
    checkpoint_step:   Numeric training step encoded by model_name.
    '''
    def __init__(self, model_name):
        self.model_name = model_name
        self.result_path = _model_name_to_path(model_name)
        with np.load(self.result_path, allow_pickle=False) as result:
            self._validate(result)
            self._set_info(result)
        self._adjacent_distances = None
        self._pairwise_distances = None
        self._pairwise_frequency_correlation = {}

    def __repr__(self):
        low, high = self.frequencies.min(), self.frequencies.max()
        return f'F0Checkpoint({self.model_name!r}, {low:g}-{high:g} Hz)'

    def _set_info(self, result):
        '''Extract fields from a field-checked npz result.'''
        self.frequencies = result['frequencies']
        self.cnn = result['mean_cnn_features']
        self.aggregation = result['aggregation'].item()
        self.umap_metric = result['metric'].item()
        self.umap_coordinates = result['coordinates']
        self.random_state = result['random_state'].item()
        self.checkpoint_step = _checkpoint_step(self.model_name)

    def adjacent_distances(self):
        '''Cached f0_adjacent_distances for this checkpoint.'''
        if self._adjacent_distances is not None: return self._adjacent_distances
        self._adjacent_distances = f0_adjacent_distances(self.cnn,
            self.frequencies)
        return self._adjacent_distances

    def pairwise_distances(self):
        '''Cached f0_pairwise_distances for this checkpoint.'''
        if self._pairwise_distances is not None: return self._pairwise_distances
        self._pairwise_distances = f0_pairwise_distances(self.cnn,
            self.frequencies)
        return self._pairwise_distances

    def pairwise_frequency_correlation(self, scale='hz'):
        '''Cached f0_pairwise_frequency_correlation, keyed by scale.'''
        cache = self._pairwise_frequency_correlation
        if scale in cache: return cache[scale]
        cache[scale] = f0_pairwise_frequency_correlation(self.cnn,
            self.frequencies, scale=scale)
        return cache[scale]

    def _validate(self, result):
        '''Check that a loaded npz result has the expected fields.'''
        required_fields = {'aggregation', 'coordinates', 'frequencies',
            'mean_cnn_features', 'metric', 'model_name', 'random_state'}
        missing = required_fields - set(result.files)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'F0 result is missing fields: {names}')
        stored_name = result['model_name'].item()
        if stored_name != self.model_name:
            m = f'F0 result model_name mismatch: {stored_name!r} '
            m += f'!= {self.model_name!r}'
            raise ValueError(m)


class F0Checkpoints:
    '''Every F0 checkpoint result under locations.f0_output_data.'''
    def __init__(self):
        self.output_directory = Path(locations.f0_output_data)
        self.result_paths = tuple(self.output_directory.glob('*.npz'))
        self._validate()
        model_names = [_path_to_model_name(p) for p in self.result_paths]
        checkpoints = [F0Checkpoint(name) for name in model_names]
        checkpoints.sort(key=lambda c: (c.checkpoint_step, c.model_name))
        self.checkpoints = tuple(checkpoints)
        checkpoint_numbers = [c.checkpoint_step for c in self.checkpoints]
        self.checkpoint_numbers = tuple(checkpoint_numbers)

    def __repr__(self):
        low, high = self.checkpoint_numbers[0], self.checkpoint_numbers[-1]
        n = len(self.checkpoints)
        return f'F0Checkpoints({n} checkpoints, {low}-{high})'

    def adjacent_distances(self):
        '''checkpoint_numbers paired with each adjacent_distances result.'''
        results = [c.adjacent_distances() for c in self.checkpoints]
        return self.checkpoint_numbers, results

    def pairwise_distances(self):
        '''checkpoint_numbers paired with each pairwise_distances result.'''
        results = [c.pairwise_distances() for c in self.checkpoints]
        return self.checkpoint_numbers, results

    def pairwise_frequency_correlation(self, scale='hz'):
        '''checkpoint_numbers paired with each pairwise correlation.'''
        results = []
        for checkpoint in self.checkpoints:
            correlation, _, _ = checkpoint.pairwise_frequency_correlation(scale)
            results.append(correlation)
        return self.checkpoint_numbers, results

    def _validate(self):
        '''Check that the output directory exists and has results.'''
        if not self.output_directory.is_dir():
            m = f'F0 output-data directory not found: {self.output_directory}'
            raise FileNotFoundError(m)
        if not self.result_paths:
            m = f'no F0 checkpoint results found in: {self.output_directory}'
            raise FileNotFoundError(m)


def _model_name_to_path(model_name):
    '''Return the existing npz path for one F0 checkpoint model name.'''
    result_path = Path(locations.f0_output_data) / f'{model_name}.npz'
    if not result_path.is_file():
        message = f'F0 checkpoint result not found: {result_path}'
        raise FileNotFoundError(message)
    return result_path


def _path_to_model_name(result_path):
    '''Return the F0 checkpoint model name encoded by an npz path.'''
    return result_path.stem


def _checkpoint_step(model_name):
    '''Return the numeric training step encoded by an F0 model name.'''
    if model_name == locations.wav2vec2_random_checkpoint_name:
        return 0
    match = re.fullmatch(locations.wav2vec2_nl1_checkpoint_pattern,
        model_name)
    if match is None:
        raise ValueError(f'unsupported F0 checkpoint model: {model_name!r}')
    return int(match.group(1))
