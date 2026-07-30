import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing import extract_mfcc


class FakeStore:
    def __init__(self):
        self.attach_phraser_store_calls = []

    def attach_phraser_store(self, source_id, phraser_store):
        self.attach_phraser_store_calls.append((source_id, phraser_store))


class FakePhones:
    def __init__(self, phraser_phones, store):
        self.phraser_phones = phraser_phones
        self.store = store


# -- extract_phone_mfcc ------------------------------------------------------

def test_extract_phone_mfcc_attaches_store_and_computes(monkeypatch):
    store = FakeStore()
    phraser_phones = ['phone-a', 'phone-b']
    phones = FakePhones(phraser_phones, store='cgn-store')

    mfcc_batch_calls = []
    store_mfcc_batch_calls = []

    def fake_mfcc_batch(segments, workers=None, cache_on_segment=True):
        mfcc_batch_calls.append(dict(
            segments=segments, workers=workers,
            cache_on_segment=cache_on_segment))

    def fake_store_mfcc_batch(segments, store_arg, tags=None, verbose=True):
        store_mfcc_batch_calls.append(dict(
            segments=segments, store=store_arg, tags=tags, verbose=verbose))

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', fake_mfcc_batch)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', fake_store_mfcc_batch)

    result = extract_mfcc.extract_phone_mfcc(
        phones, store=store, workers=4, cache_on_segment=False,
        tags=['exp-a'], verbose=False)

    assert result is store
    assert store.attach_phraser_store_calls == [
        (extract_mfcc.default_phraser_source_id, 'cgn-store')]
    assert mfcc_batch_calls == [dict(
        segments=phraser_phones, workers=4, cache_on_segment=False)]
    assert store_mfcc_batch_calls == [dict(
        segments=phraser_phones, store=store, tags=['exp-a'], verbose=False)]


def test_extract_phone_mfcc_uses_custom_phraser_source_id(monkeypatch):
    store = FakeStore()
    phones = FakePhones(['phone-a'], store='cgn-store')

    monkeypatch.setattr(extract_mfcc, 'mfcc_batch', lambda *a, **k: None)
    monkeypatch.setattr(extract_mfcc, 'store_mfcc_batch', lambda *a, **k: None)

    extract_mfcc.extract_phone_mfcc(
        phones, store=store, phraser_source_id='custom-source', verbose=False)

    assert store.attach_phraser_store_calls == [('custom-source', 'cgn-store')]


def test_extract_phone_mfcc_opens_store_when_none_given(tmp_path, monkeypatch):
    phones = FakePhones(['phone-a'], store='cgn-store')
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
