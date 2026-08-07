from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import probe_utils


class FakeBalancedPhones:
    def __init__(self, counts):
        self.label_to_phraser_phone = {
            label: [object()] * count for label, count in counts.items()
        }


def test_prepare_balanced_probe_targets_uses_phraser_label_inventory():
    phones = FakeBalancedPhones({'t': 10, 'a': 10, 'p': 10})

    targets = probe_utils.prepare_balanced_probe_targets(phones)

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
        phones, ['t', 'p']) == ['t', 'p']
    with pytest.raises(ValueError, match='not found'):
        probe_utils.prepare_balanced_probe_targets(phones, ['x'])


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
