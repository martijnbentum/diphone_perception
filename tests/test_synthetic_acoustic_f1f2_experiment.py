import numpy as np
import pytest

import locations
import synthetic_acoustic_probes.f1f2_experiment as f1f2_experiment


class FakeStore:
    '''Small Store stand-in recording its path and close state.'''

    def __init__(self, path):
        '''Create an open stand-in at path.'''

        self.path = path
        self.closed = False

    def close(self):
        '''Record that the stand-in was closed.'''

        self.closed = True


def test_f1f2_locations_share_experiment_directory():
    '''F1/F2 artifacts and stores live below the dedicated experiment root.'''

    root = locations.synthetic_acoustic_probes_data / 'experiment_f1f2'

    assert locations.f1f2_experiment == root
    assert locations.f1f2_output_data.parent == root
    assert locations.f1f2_plots.parent == root
    assert locations.f1f2_stimuli.parent == root
    assert locations.f1f2_phraser_store.parent == root
    assert locations.f1f2_echoframe_store.parent == root


def test_create_auditory_stimuli_saves_to_default_location(monkeypatch):
    '''Stimulus creation delegates with saving and the shared default path.

    monkeypatch:  Pytest fixture used to replace formant-grid generation.
    '''

    calls = []
    expected = object()

    def fake_formant_stimuli(save, output_root, overwrite):
        call = {}
        call['save'] = save
        call['output_root'] = output_root
        call['overwrite'] = overwrite
        calls.append(call)
        return expected

    monkeypatch.setattr(
        f1f2_experiment,
        'sinusoidal_component_formant_stimuli',
        fake_formant_stimuli,
    )

    result = f1f2_experiment.create_auditory_stimuli()

    assert result is expected
    expected_call = {}
    expected_call['save'] = True
    expected_call['output_root'] = locations.f1f2_stimuli
    expected_call['overwrite'] = False
    assert calls == [expected_call]


def test_create_f1f2_phraser_store_populates_store(monkeypatch):
    '''Store creation fills and returns the open experiment store.

    monkeypatch:  Pytest fixture used to replace Phraser persistence.
    '''

    stores = []
    add_calls = []

    def fake_store(path):
        store = FakeStore(path)
        stores.append(store)
        return store

    monkeypatch.setattr(f1f2_experiment, 'Store', fake_store)
    monkeypatch.setattr(
        f1f2_experiment,
        'add_stimuli',
        lambda package, store: add_calls.append((package, store)),
    )

    result = f1f2_experiment.create_f1f2_phraser_store()

    assert result is stores[0]
    assert result.path == locations.f1f2_phraser_store
    assert result.closed is False
    assert add_calls == [(locations.f1f2_stimuli, result)]


def test_load_f1f2_phraser_store(tmp_path, monkeypatch):
    '''Loading requires an existing directory and returns its opened store.

    tmp_path:  Temporary directory supplied by pytest.
    monkeypatch:  Pytest fixture used to replace the Phraser Store.
    '''

    missing = tmp_path / 'missing'
    monkeypatch.setattr(locations, 'f1f2_phraser_store', missing)
    with pytest.raises(FileNotFoundError, match='F1/F2 Phraser store'):
        f1f2_experiment.load_f1f2_phraser_store()

    store_path = tmp_path / 'phraser'
    store_path.mkdir()
    monkeypatch.setattr(locations, 'f1f2_phraser_store', store_path)
    stores = []

    def fake_store(path):
        store = FakeStore(path)
        stores.append(store)
        return store

    monkeypatch.setattr(f1f2_experiment, 'Store', fake_store)

    result = f1f2_experiment.load_f1f2_phraser_store()

    assert result is stores[0]
    assert result.path == store_path


def test_save_f1f2_checkpoint_result_writes_expected_bundle(
    tmp_path,
    monkeypatch,
):
    '''One checkpoint result contains the agreed arrays and metadata.'''
    model_name = 'wav2vec2_nl1_checkpoint-200000'
    store = object()
    X = np.arange(12, dtype=float).reshape(4, 3)
    f1_hz = np.array([300, 400, 500, 600], dtype=float)
    f2_hz = np.array([900, 1000, 1100, 1200], dtype=float)
    coordinates = np.arange(8, dtype=float).reshape(4, 2)
    make_calls = []
    projection_calls = []

    def fake_make(model, received_store):
        make_calls.append((model, received_store))
        return X, f1_hz, f2_hz

    def fake_project(values, *, metric, random_state):
        projection_calls.append((values, metric, random_state))
        return coordinates

    monkeypatch.setattr(f1f2_experiment, 'make_f1f2_x_y', fake_make)
    monkeypatch.setattr(f1f2_experiment, 'project_umap', fake_project)
    output_directory = tmp_path / 'output_data'
    monkeypatch.setattr(locations, 'f1f2_output_data', output_directory)

    output_path = f1f2_experiment.save_f1f2_checkpoint_result(
        model_name, store)

    assert output_path == output_directory / f'{model_name}.npz'
    assert make_calls == [(model_name, store)]
    assert projection_calls == [(X, 'cosine', 42)]
    with np.load(output_path, allow_pickle=False) as saved:
        assert set(saved.files) == {
            'cnn',
            'umap_coordinates',
            'umap_metric',
            'umap_random_state',
            'model_name',
            'f1_hz',
            'f2_hz',
        }
        np.testing.assert_array_equal(saved['cnn'], X)
        np.testing.assert_array_equal(saved['umap_coordinates'], coordinates)
        np.testing.assert_array_equal(saved['f1_hz'], f1_hz)
        np.testing.assert_array_equal(saved['f2_hz'], f2_hz)
        assert saved['umap_random_state'].item() == 42
        assert saved['umap_metric'].item() == 'cosine'
        assert saved['model_name'].item() == model_name


def test_save_f1f2_checkpoint_result_skips_existing_file(
    tmp_path,
    monkeypatch,
):
    '''An existing checkpoint bundle is returned without recomputation.'''
    model_name = 'wav2vec2_checkpoint-0'
    output_directory = tmp_path / 'output_data'
    output_directory.mkdir()
    output_path = output_directory / f'{model_name}.npz'
    output_path.write_bytes(b'existing result')
    monkeypatch.setattr(locations, 'f1f2_output_data', output_directory)

    def fail(*args, **kwargs):
        pytest.fail('existing output should skip computation')

    monkeypatch.setattr(f1f2_experiment, 'make_f1f2_x_y', fail)
    monkeypatch.setattr(f1f2_experiment, 'project_umap', fail)

    result = f1f2_experiment.save_f1f2_checkpoint_result(
        model_name, object())

    assert result == output_path
    assert output_path.read_bytes() == b'existing result'


def test_save_f1f2_checkpoint_results_saves_catalog_and_reports_skips(
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
    monkeypatch.setattr(locations, 'f1f2_output_data', output_directory)
    calls = []

    def fake_save(model_name, store):
        calls.append((model_name, store))
        return output_directory / f'{model_name}.npz'

    monkeypatch.setattr(
        f1f2_experiment,
        'select_wav2vec2_nl1_checkpoints',
        lambda: model_names,
    )
    monkeypatch.setattr(
        f1f2_experiment,
        'save_f1f2_checkpoint_result',
        fake_save,
    )
    store = object()

    result = f1f2_experiment.save_f1f2_checkpoint_results(store)

    saved_paths = tuple(
        output_directory / f'{model_name}.npz'
        for model_name in (model_names[0], model_names[2])
    )
    assert calls == [
        (model_names[0], store),
        (model_names[2], store),
    ]
    assert result == {
        'saved': saved_paths,
        'skipped': (skipped_path,),
    }
