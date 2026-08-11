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
