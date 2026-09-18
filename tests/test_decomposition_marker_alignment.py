'''Check manifest ordering before assigning per-marker fitting splits.'''

from pathlib import Path
from types import SimpleNamespace

import pytest

from decomposition import load_embeddings as loading


@pytest.fixture
def manifest():
    return {'audio_infos': [
        {'filename': 'a.wav', 'frame_indices': [2, 8]},
        {'filename': 'unused.wav', 'frame_indices': []},
        {'filename': 'b.wav', 'frame_indices': [2]},
    ]}


@pytest.fixture
def markers():
    result = []
    for filename, frame in [('a.wav', 2), ('a.wav', 8), ('b.wav', 2)]:
        audio = SimpleNamespace(filename=Path(filename))
        result.append(SimpleNamespace(audio=audio,
            label=f'decomp_random_frames_{frame}'))
    return result


def test_matching_manifest_accepts_iterator_and_path_filenames(markers, manifest):
    assert loading.check_marker_alignment(iter(markers), manifest)


@pytest.mark.parametrize('change', ['reorder', 'filename', 'label', 'duplicate'])
def test_alignment_rejects_wrong_identity_or_order(markers, manifest, change):
    if change == 'reorder': markers.reverse()
    elif change == 'filename': markers[0].audio.filename = 'wrong.wav'
    elif change == 'label': markers[0].label = 'decomp_random_frames_3'
    else: markers[0] = markers[1]
    with pytest.raises(ValueError, match='marker 0: expected'):
        loading.check_marker_alignment(markers, manifest)


@pytest.mark.parametrize('count', [2, 4])
def test_alignment_rejects_missing_and_extra_markers(markers, manifest, count):
    markers = (markers + markers)[:count]
    with pytest.raises(ValueError, match='expected 3 markers'):
        loading.check_marker_alignment(markers, manifest)


def test_alignment_loads_requested_manifest(markers, manifest, monkeypatch):
    def load(region, components):
        assert region == 'vl'
        assert components == ('comp-o',)
        return manifest
    monkeypatch.setattr(loading, 'load_manifest', load)
    assert loading.check_marker_alignment(markers, region='vl',
        components=('comp-o',))


def test_alignment_supports_custom_labels(markers, manifest):
    for marker in markers:
        marker.label = marker.label.replace('decomp_random_frames', 'custom')
    assert loading.check_marker_alignment(markers, manifest, label='custom')


def test_empty_markers_match_empty_manifest():
    assert loading.check_marker_alignment([], {'audio_infos': []})
