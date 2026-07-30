from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_utils


class FakePhraserPhone:
    def __init__(self, key):
        self.key = key


class FakePhones:
    def __init__(self, count):
        self.phraser_phones = [
            FakePhraserPhone(index) for index in range(count)
        ]


class FakeFrameStore:
    def __init__(self, vectors_by_key):
        self.vectors_by_key = vectors_by_key
        self.calls = []

    def make_echoframe_key(self, output_type, phraser_key, **parameters):
        return phraser_key

    def load_many_frames(self, keys, frame, keep_missing):
        self.calls.append({
            'keys': list(keys),
            'frame': frame,
            'keep_missing': keep_missing,
        })
        return [self.vectors_by_key.get(key) for key in keys]


class FakeBalancedPhones:
    def __init__(self, counts):
        self.label_to_phraser_phone = {
            label: [object()] * count for label, count in counts.items()
        }


def test_prepare_balanced_probe_targets_uses_phraser_label_inventory():
    phones = FakeBalancedPhones({'t': 10, 'a': 10, 'p': 10})

    targets = probe_utils.prepare_balanced_probe_targets(
        phones, n_samples=6)

    assert targets == ['a', 'p', 't']


def test_prepare_balanced_probe_targets_rejects_unequal_label_counts():
    phones = FakeBalancedPhones({'a': 10, 'p': 9, 't': 10})

    with pytest.raises(
        ValueError, match='not balanced.*every label.*same number',
    ) as error:
        probe_utils.prepare_balanced_probe_targets(phones)

    assert "'a': 10" in str(error.value)
    assert "'p': 9" in str(error.value)
    assert "'t': 10" in str(error.value)


def test_prepare_balanced_probe_targets_validates_requested_targets():
    phones = FakeBalancedPhones({'a': 10, 'p': 10, 't': 10})

    assert probe_utils.prepare_balanced_probe_targets(
        phones, ['t', 'p'], n_samples=6) == ['t', 'p']
    with pytest.raises(ValueError, match='not found'):
        probe_utils.prepare_balanced_probe_targets(
            phones, ['x'], n_samples=6)


def test_run_probe_sweep_reports_elapsed_time_and_eta(monkeypatch, capsys):
    times = iter([0, 10, 30])
    monkeypatch.setattr(
        probe_utils.time, 'monotonic', lambda: next(times))

    results = probe_utils.run_probe_sweep(
        ['a', 'p'], lambda target: f'result-{target}', 'embedding')

    assert results == {'a': 'result-a', 'p': 'result-p'}
    output = capsys.readouterr().out
    assert "[embedding probes] 1/2 starting 'a'" in output
    assert (
        "[embedding probes] 1/2 completed 'a'; elapsed 00:00:10; "
        "ETA 00:00:10"
    ) in output
    assert (
        "[embedding probes] 2/2 completed 'p'; elapsed 00:00:30; "
        "ETA 00:00:00"
    ) in output


def test_inspect_feature_scale_compares_paired_center_frames(capsys):
    phones = FakePhones(6)
    embedding_store = FakeFrameStore({
        0: np.array([0., 0.]),
        1: np.array([1., 2.]),
        2: np.array([2., 1.]),
        3: np.array([3., 3.]),
        4: np.array([4., 4.]),
    })
    mfcc_store = FakeFrameStore({
        0: np.array([0., 0.]),
        1: np.array([1., 100.]),
        2: np.array([2., 200.]),
        3: np.array([3., 300.]),
        5: np.array([5., 500.]),
    })

    report = probe_utils.inspect_feature_scale(
        phones,
        embedding_store=embedding_store,
        mfcc_store=mfcc_store,
        sample_size=6,
        std_ratio_threshold=10,
    )

    assert report['n_requested'] == 6
    assert report['n_paired'] == 4
    assert report['n_missing_embedding'] == 1
    assert report['n_missing_mfcc'] == 1
    assert report['embedding']['n_dimensions'] == 2
    assert report['embedding']['std_ratio'] == pytest.approx(1)
    assert report['embedding']['recommend_standardize'] is False
    assert report['mfcc']['std_ratio'] == pytest.approx(100)
    assert report['mfcc']['recommend_standardize'] is True
    assert all(call['frame'] == 'center' for call in (
        embedding_store.calls[0], mfcc_store.calls[0]))
    assert all(call['keep_missing'] for call in (
        embedding_store.calls[0], mfcc_store.calls[0]))
    output = capsys.readouterr().out
    assert 'embedding:' in output
    assert 'mfcc:' in output
    assert 'consider standardizing' in output


def test_feature_scale_does_not_recommend_scaling_constant_features():
    summary = probe_utils._feature_scale_summary(
        np.ones((4, 3)), std_ratio_threshold=10)

    assert summary['zero_variance_dimensions'] == 3
    assert summary['recommend_standardize'] is False


@pytest.mark.parametrize('sample_size', [0, -1])
def test_inspect_feature_scale_rejects_nonpositive_sample_size(sample_size):
    with pytest.raises(ValueError, match='positive integer'):
        probe_utils.inspect_feature_scale(
            FakePhones(2),
            embedding_store=FakeFrameStore({}),
            mfcc_store=FakeFrameStore({}),
            sample_size=sample_size,
        )
