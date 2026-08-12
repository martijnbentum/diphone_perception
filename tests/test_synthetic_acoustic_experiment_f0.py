import numpy as np
import pytest

import locations
import synthetic_acoustic_probes.experiment_f0 as experiment_f0


class FakeStore:
    '''Small Store stand-in recording its path and close state.'''

    def __init__(self, path):
        '''Create an open stand-in at path.'''

        self.path = path
        self.closed = False

    def close(self):
        '''Record that the stand-in was closed.'''

        self.closed = True


def test_f0_locations_share_experiment_directory():
    '''F0 artifacts and stores live below the dedicated experiment root.'''

    root = locations.synthetic_acoustic_probes_data / 'experiment_f0'

    assert locations.f0_experiment == root
    assert locations.f0_output_data.parent == root
    assert locations.f0_plots.parent == root
    assert locations.f0_umap_plot.parent == root
    assert locations.f0_pure_tone_stimuli.parent == root
    assert locations.f0_pure_tone_phraser_store.parent == root
    assert locations.synthetic_acoustic_probes_echoframe_store.parent == root
    phase_root = root / 'phase_diagnostics'
    assert locations.f0_phase_diagnostics == phase_root
    assert locations.f0_phase_diagnostic_stimuli.parent == phase_root
    assert locations.f0_phase_diagnostic_phraser_store.parent == phase_root
    assert locations.f0_phase_diagnostic_echoframe_store.parent == phase_root
    assert locations.f0_phase_diagnostic_results.parent == phase_root


def test_create_auditory_stimuli_saves_to_default_location(monkeypatch):
    '''Stimulus creation delegates with saving and the shared default path.

    monkeypatch:  Pytest fixture used to replace pure-tone generation.
    '''

    calls = []
    expected = object()

    def fake_pure_tone_stimuli(save, output_root, overwrite):
        call = {}
        call['save'] = save
        call['output_root'] = output_root
        call['overwrite'] = overwrite
        calls.append(call)
        return expected

    monkeypatch.setattr(
        experiment_f0,
        'pure_tone_stimuli',
        fake_pure_tone_stimuli,
    )

    result = experiment_f0.create_auditory_stimuli()

    assert result is expected
    expected_call = {}
    expected_call['save'] = True
    expected_call['output_root'] = locations.f0_pure_tone_stimuli
    expected_call['overwrite'] = False
    assert calls == [expected_call]


def test_create_f0_pure_tone_phraser_store_populates_store(monkeypatch):
    '''Store creation fills and returns the open experiment store.

    monkeypatch:  Pytest fixture used to replace Phraser persistence.
    '''

    stores = []
    add_calls = []

    def fake_store(path):
        store = FakeStore(path)
        stores.append(store)
        return store

    monkeypatch.setattr(experiment_f0, 'Store', fake_store)
    monkeypatch.setattr(
        experiment_f0,
        'add_stimuli',
        lambda package, store: add_calls.append((package, store)),
    )

    result = experiment_f0.create_f0_pure_tone_phraser_store()

    assert result is stores[0]
    assert result.path == locations.f0_pure_tone_phraser_store
    assert result.closed is False
    assert add_calls == [(locations.f0_pure_tone_stimuli, result)]


def test_load_f0_pure_tone_phraser_store(tmp_path, monkeypatch):
    '''Loading requires an existing directory and returns its opened store.

    tmp_path:  Temporary directory supplied by pytest.
    monkeypatch:  Pytest fixture used to replace the Phraser Store.
    '''

    missing = tmp_path / 'missing'
    monkeypatch.setattr(locations, 'f0_pure_tone_phraser_store', missing)
    with pytest.raises(FileNotFoundError, match='F0 Phraser store not found'):
        experiment_f0.load_f0_pure_tone_phraser_store()

    store_path = tmp_path / 'phraser'
    store_path.mkdir()
    monkeypatch.setattr(locations, 'f0_pure_tone_phraser_store', store_path)
    stores = []

    def fake_store(path):
        store = FakeStore(path)
        stores.append(store)
        return store

    monkeypatch.setattr(experiment_f0, 'Store', fake_store)

    result = experiment_f0.load_f0_pure_tone_phraser_store()

    assert result is stores[0]
    assert result.path == store_path


def test_save_f0_checkpoint_result_writes_expected_bundle(
    tmp_path,
    monkeypatch,
):
    '''One checkpoint result contains the agreed arrays and metadata.'''
    model_name = 'wav2vec2_nl1_checkpoint-200000'
    store = object()
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.array([10, 20, 30, 40], dtype=float)
    coordinates = np.arange(8, dtype=float).reshape(4, 2)
    make_calls = []
    projection_calls = []

    def fake_make(model, received_store, *, aggregation):
        make_calls.append((model, received_store, aggregation))
        return X, y

    def fake_project(values, *, metric, random_state):
        projection_calls.append((values, metric, random_state))
        return coordinates

    monkeypatch.setattr(experiment_f0, 'make_f0_x_y', fake_make)
    monkeypatch.setattr(experiment_f0, 'project_umap', fake_project)
    output_directory = tmp_path / 'output_data'

    output_path = experiment_f0.save_f0_checkpoint_result(
        model_name,
        store,
        output_directory=output_directory,
    )

    assert output_path == output_directory / f'{model_name}.npz'
    assert make_calls == [(model_name, store, 'mean')]
    assert projection_calls == [(X, 'cosine', 42)]
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == {
            'mean_cnn_features',
            'coordinates',
            'frequencies',
            'random_state',
            'metric',
            'model_name',
            'aggregation',
        }
        np.testing.assert_array_equal(saved['mean_cnn_features'], X)
        np.testing.assert_array_equal(saved['coordinates'], coordinates)
        np.testing.assert_array_equal(saved['frequencies'], y)
        assert saved['random_state'].item() == 42
        assert saved['metric'].item() == 'cosine'
        assert saved['model_name'].item() == model_name
        assert saved['aggregation'].item() == 'mean'


def test_save_f0_checkpoint_result_skips_existing_file(tmp_path, monkeypatch):
    '''An existing checkpoint bundle is returned without recomputation.'''
    model_name = 'wav2vec2_checkpoint-0'
    output_directory = tmp_path / 'output_data'
    output_directory.mkdir()
    output_path = output_directory / f'{model_name}.npz'
    output_path.write_bytes(b'existing result')

    def fail(*args, **kwargs):
        pytest.fail('existing output should skip computation')

    monkeypatch.setattr(experiment_f0, 'make_f0_x_y', fail)
    monkeypatch.setattr(experiment_f0, 'project_umap', fail)

    result = experiment_f0.save_f0_checkpoint_result(
        model_name,
        object(),
        output_directory=output_directory,
    )

    assert result == output_path
    assert output_path.read_bytes() == b'existing result'


def test_save_f0_checkpoint_results_saves_catalog_and_reports_skips(
    tmp_path,
    monkeypatch,
):
    '''The checkpoint sweep preserves catalog order and skips prior output.'''
    model_names = (
        'wav2vec2_checkpoint-0',
        'wav2vec2_nl1_checkpoint-1',
        'wav2vec2_nl1_checkpoint-2',
    )
    output_directory = tmp_path / 'output_data'
    output_directory.mkdir()
    skipped_path = output_directory / f'{model_names[1]}.npz'
    skipped_path.write_bytes(b'existing result')
    calls = []

    def fake_save(model_name, store, *, output_directory):
        calls.append((model_name, store, output_directory))
        return output_directory / f'{model_name}.npz'

    monkeypatch.setattr(
        experiment_f0,
        'select_wav2vec2_nl1_checkpoints',
        lambda: model_names,
    )
    monkeypatch.setattr(
        experiment_f0,
        'save_f0_checkpoint_result',
        fake_save,
    )
    store = object()

    result = experiment_f0.save_f0_checkpoint_results(
        store,
        output_directory=output_directory,
    )

    saved_paths = tuple(
        output_directory / f'{model_name}.npz'
        for model_name in (model_names[0], model_names[2])
    )
    assert calls == [
        (model_names[0], store, output_directory),
        (model_names[2], store, output_directory),
    ]
    assert result == {
        'saved': saved_paths,
        'skipped': (skipped_path,),
    }
