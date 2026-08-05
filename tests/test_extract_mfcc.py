import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locations
from probing import extract_mfcc


class FakeStore:
    def __init__(self, existing_keys=None):
        self.attach_phraser_store_calls = []
        self.existing_keys = set(existing_keys or [])

    def attach_phraser_store(self, source_id, phraser_store):
        self.attach_phraser_store_calls.append((source_id, phraser_store))

    def make_echoframe_key(
        self, output_type, feature_name=None, phraser_key=None,
    ):
        return output_type, feature_name, phraser_key

    def load_many_metadata(self, keys, keep_missing=False):
        metadatas = []
        for key in keys:
            metadata = object() if key in self.existing_keys else None
            if metadata is not None or keep_missing:
                metadatas.append(metadata)
        return metadatas


def test_default_store_root_uses_locations():
    store_root = inspect.signature(
        extract_mfcc.extract_phone_mfcc,
    ).parameters['store_root'].default

    assert store_root == locations.echoframe_mfcc_store


class FakeAudio:
    def __init__(self, filename):
        self.filename = filename
        self.sample_rate = 16_000
        self.duration = 10_000


class FakeSegment:
    def __init__(self, key, filename='audio.wav'):
        self.key = key
        self.audio = FakeAudio(filename)


class FakePhones:
    def __init__(self, phraser_phones, store):
        self.phraser_phones = phraser_phones
        self.store = store


# -- extract_phone_mfcc ------------------------------------------------------

def test_extract_phone_mfcc_attaches_store_and_computes(monkeypatch):
    store = FakeStore()
    phraser_phones = [FakeSegment('phone-a'), FakeSegment('phone-b')]
    phones = FakePhones(phraser_phones, store='cgn-store')

    mfcc_batch_calls = []
    store_mfcc_batch_calls = []

    def fake_mfcc_batch(segments, workers=None, cache_on_segment=True):
        mfcc_batch_calls.append(dict(
            segments=segments, workers=workers,
            cache_on_segment=cache_on_segment))
        for segment in segments:
            segment._mfcc = f'mfcc-{segment.key}'

    def fake_store_mfcc_batch(segments, store_arg, tags=None, verbose=True):
        assert all(hasattr(segment, '_mfcc') for segment in segments)
        store_mfcc_batch_calls.append(dict(
            segments=segments, store=store_arg, tags=tags, verbose=verbose))

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', fake_store_mfcc_batch)

    result = extract_mfcc.extract_phone_mfcc(
        phones, store=store, workers=4, recordings_per_batch=30,
        tags=['exp-a'], verbose=False)

    assert result is store
    assert store.attach_phraser_store_calls == [
        (extract_mfcc.default_phraser_source_id, 'cgn-store')]
    assert mfcc_batch_calls == [dict(
        segments=phraser_phones, workers=4, cache_on_segment=True)]
    assert store_mfcc_batch_calls == [dict(
        segments=phraser_phones, store=store, tags=['exp-a'], verbose=False)]
    assert all(not hasattr(segment, '_mfcc') for segment in phraser_phones)


def test_extract_phone_mfcc_uses_custom_phraser_source_id(monkeypatch):
    store = FakeStore()
    phones = FakePhones([FakeSegment('phone-a')], store='cgn-store')

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', lambda *a, **k: None)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    extract_mfcc.extract_phone_mfcc(
        phones, store=store, phraser_source_id='custom-source', verbose=False)

    assert store.attach_phraser_store_calls == [('custom-source', 'cgn-store')]


def test_extract_phone_mfcc_opens_store_when_none_given(tmp_path, monkeypatch):
    phones = FakePhones([FakeSegment('phone-a')], store='cgn-store')
    opened_store = FakeStore()
    store_roots = []

    def fake_store_constructor(root):
        store_roots.append(root)
        return opened_store

    monkeypatch.setattr(extract_mfcc.echoframe, 'Store', fake_store_constructor)
    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', lambda *a, **k: None)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    result = extract_mfcc.extract_phone_mfcc(
        phones, store_root=tmp_path / 'store', verbose=False)

    assert result is opened_store
    assert store_roots == [str(tmp_path / 'store')]


def test_extract_phone_mfcc_batches_30_recordings_by_default(monkeypatch):
    segments = [
        FakeSegment(f'phone-{index}', filename=f'audio-{index}.wav')
        for index in range(31)
    ]
    phones = FakePhones(segments, store='cgn-store')
    store = FakeStore()
    computed_batches = []
    stored_batches = []

    def fake_mfcc_batch(batch, **kwargs):
        computed_batches.append(list(batch))
        for segment in batch:
            segment._mfcc = f'mfcc-{segment.key}'

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(
        extract_mfcc, 'store_mfcc_batch',
        lambda batch, *args, **kwargs: stored_batches.append(list(batch)))

    extract_mfcc.extract_phone_mfcc(phones, store=store, verbose=False)

    assert [len(batch) for batch in computed_batches] == [30, 1]
    assert [len(batch) for batch in stored_batches] == [30, 1]
    assert all(not hasattr(segment, '_mfcc') for segment in segments)


def test_extract_phone_mfcc_computes_only_missing_segments(monkeypatch):
    present = FakeSegment('present')
    missing = FakeSegment('missing')
    present_key = ('acoustic_feature', 'mfcc', present.key)
    store = FakeStore(existing_keys={present_key})
    phones = FakePhones([present, missing], store='cgn-store')
    computed = []

    def fake_mfcc_batch(segments, **kwargs):
        computed.extend(segments)
        for segment in segments:
            segment._mfcc = f'mfcc-{segment.key}'

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    extract_mfcc.extract_phone_mfcc(phones, store=store, verbose=False)

    assert computed == [missing]


def test_extract_phone_mfcc_can_keep_segment_cache(monkeypatch):
    segment = FakeSegment('phone-a')
    phones = FakePhones([segment], store='cgn-store')

    def fake_mfcc_batch(segments, **kwargs):
        for item in segments:
            item._mfcc = f'mfcc-{item.key}'

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    extract_mfcc.extract_phone_mfcc(
        phones, store=FakeStore(), keep_segment_cache=True, verbose=False)

    assert segment._mfcc == 'mfcc-phone-a'


def test_extract_phone_mfcc_reports_pipeline_stages(monkeypatch, capsys):
    present = FakeSegment('present', filename='audio-a.wav')
    missing = FakeSegment('missing', filename='audio-b.wav')
    present_key = ('acoustic_feature', 'mfcc', present.key)
    store = FakeStore(existing_keys={present_key})
    phones = FakePhones([present, missing], store='cgn-store')

    def fake_mfcc_batch(segments, **kwargs):
        for segment in segments:
            segment._mfcc = f'mfcc-{segment.key}'

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    extract_mfcc.extract_phone_mfcc(
        phones, store=store, workers=3, verbose=True)

    output = capsys.readouterr().out
    assert "[mfcc] attaching Phraser source 'cgn-awd'" in output
    assert '[mfcc] loading matched Phraser phones' in output
    assert '[mfcc] loaded 2 matched phones' in output
    assert 'grouped phones from 2 recordings into 1 batch' in output
    assert 'batch 1/1: checking 2 phones from 2 recordings' in output
    assert 'batch 1/1: computing 1 missing MFCC using up to 3 workers' in output
    assert 'batch 1/1: computation finished; preparing and writing 1 MFCC' \
        in output
    assert 'batch 1/1: complete; stored 1 MFCC and cleared temporary ' \
        'segment cache' in output
    assert '[mfcc] complete: stored 1 MFCC; skipped 1 MFCC already present' \
        in output


def test_extract_phone_mfcc_validates_recordings_per_batch():
    phones = FakePhones([], store='cgn-store')

    with pytest.raises(TypeError, match='positive integer'):
        extract_mfcc.extract_phone_mfcc(
            phones, store=FakeStore(), recordings_per_batch=True)
    with pytest.raises(ValueError, match='positive integer'):
        extract_mfcc.extract_phone_mfcc(
            phones, store=FakeStore(), recordings_per_batch=0)
