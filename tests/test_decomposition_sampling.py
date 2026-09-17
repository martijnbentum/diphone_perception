import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from decomposition import sampling
from decomposition.sampling import assign_fit_and_eval_splits, audio_to_info
from decomposition.sampling import filter_audios_on_component, iter_marker_info
from decomposition.sampling import load_manifest, make_manifest
from decomposition.sampling import sample_frames, save_manifest


def _make_audio(number, duration=1005, component='comp-k', region='nl'):
    '''Create recording metadata without phone or speaker annotations.'''
    filename = f'/cgn/{component}/{region}/audio-{number:03d}.wav'
    key = f'audio-key-{number}'.encode()
    return SimpleNamespace(filename=filename, duration=duration, key=key)


class FakeStore:
    def __init__(self, audios):
        self.audios = audios
        self.path = Path('/fake/phraser-store')
        self.by_key = {audio.key: audio for audio in audios}

    def load(self, key):
        return self.by_key[key]


class TestDecompositionSampling(unittest.TestCase):
    def test_filters_exact_components_and_region(self):
        '''Exclude paths merely resembling the requested components/region.'''
        audios = [_make_audio(1), _make_audio(2, component='comp-o'),
            _make_audio(3, region='vl'), _make_audio(4, region='nl-extra'),
            _make_audio(5, component='comp-k-extra')]
        self.assertEqual(filter_audios_on_component(audios), audios[:2])
        selected = filter_audios_on_component(audios, components=('comp-o',))
        self.assertEqual(selected, [audios[1]])
        selected = filter_audios_on_component(audios, region='vl')
        self.assertEqual(selected, [audios[2]])

    def test_filters_duration_at_one_second_by_default(self):
        '''Apply an inclusive duration threshold before constructing frames.'''
        audios = [_make_audio(i, duration)
            for i, duration in enumerate((0, 24, 25, 999, 1000, 1005))]
        self.assertEqual(filter_audios_on_component(audios), audios[4:])
        selected = filter_audios_on_component(audios, min_duration_ms=25)
        self.assertEqual(selected, audios[2:])

    def test_audio_info_preserves_identity_and_frame_boundaries(self):
        '''Count complete windows, including one ending at the audio boundary.'''
        audios = [_make_audio(i, duration)
            for i, duration in enumerate((25, 44, 45, 105))]
        rows = [audio_to_info(audio) for audio in audios]
        self.assertEqual([row['n_frames'] for row in rows], [1, 1, 2, 5])
        row = rows[-1]
        self.assertEqual(row['audio_key'], audios[-1].key.hex())
        self.assertEqual(row['filename'], audios[-1].filename)
        self.assertEqual(row['component'], 'comp-k')
        self.assertEqual(row['duration_ms'], 105)
        self.assertIsNone(row['split'])
        self.assertEqual(row['frame_indices'], [])

    def test_sampling_full_population_maps_every_frame_once(self):
        '''Map cumulative indices correctly across recording boundaries.'''
        audios = [_make_audio(1, 45), _make_audio(2, 105)]
        audio_infos = sample_frames(audios, n_samples=7)
        identities = []
        for row in audio_infos:
            identities.extend((row['audio_key'], i) for i in row['frame_indices'])
        expected = {(audios[0].key.hex(), 0), (audios[0].key.hex(), 1)}
        expected.update((audios[1].key.hex(), i) for i in range(5))
        self.assertEqual(set(identities), expected)
        self.assertEqual(len(identities), 7)
        markers = list(iter_marker_info({'audio_infos': audio_infos},
            FakeStore(audios)))
        self.assertEqual(len(markers), 7)
        for marker, (key, index) in zip(markers, identities):
            self.assertEqual(marker['audio'].key.hex(), key)
            self.assertEqual(marker['label'], f'decomp_random_frames_{index}')
            self.assertEqual(marker['start'], index * 20)
            self.assertEqual(marker['end'], index * 20 + 25)

    def test_selection_is_repeatable_and_independent_of_input_order(self):
        '''The fixed seed reproduces selection across store query orders.'''
        audios = [_make_audio(i) for i in range(6)]
        first = sample_frames(audios, n_samples=30)
        reordered = sample_frames(list(reversed(audios)), n_samples=30)
        repeated = sample_frames(audios, n_samples=30)
        self.assertEqual(first, reordered)
        self.assertEqual(first, repeated)
        keys = [row['audio_key'] for row in first]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(sum(len(row['frame_indices']) for row in first), 30)
        for row in first:
            indices = row['frame_indices']
            self.assertEqual(indices, sorted(set(indices)))
            self.assertTrue(all(0 <= i < row['n_frames'] for i in indices))

    def test_sampling_does_not_change_global_random_state(self):
        '''Sampling and split assignment use local random generators.'''
        before = random.getstate()
        sample_frames([_make_audio(1), _make_audio(2)], n_samples=10)
        self.assertEqual(random.getstate(), before)

    def test_recording_splits_do_not_depend_on_sample_size(self):
        '''Keep complete recordings in one split at every sample size.'''
        audios = [_make_audio(i, 105) for i in range(4)]
        audios.extend(_make_audio(i, 105, component='comp-o')
            for i in range(4, 8))
        small = sample_frames(audios, n_samples=5)
        full = sample_frames(audios, n_samples=40)
        first = {row['audio_key']: row['split'] for row in small}
        second = {row['audio_key']: row['split'] for row in full}
        self.assertEqual(first, second)
        for component in ('comp-k', 'comp-o'):
            fitting = [row for row in full if row['component'] == component
                and row['split'] == 'fitting']
            self.assertEqual(len(fitting), 2)
        by_split = {'fitting': set(), 'evaluation': set()}
        for row in full:
            by_split[row['split']].add(row['audio_key'])
        self.assertEqual(len(by_split['fitting']), 4)
        self.assertEqual(len(by_split['evaluation']), 4)
        self.assertFalse(by_split['fitting'] & by_split['evaluation'])

    def test_assign_splits_is_seeded_and_updates_rows_in_place(self):
        '''Split assignment preserves row order and existing frame selections.'''
        rows = [audio_to_info(_make_audio(i)) for i in range(12)]
        for row in rows:
            row['frame_indices'] = [1, 3]
        keys = [row['audio_key'] for row in rows]
        result = assign_fit_and_eval_splits(rows, seed=7)
        self.assertIs(result, rows)
        self.assertEqual([row['audio_key'] for row in rows], keys)
        self.assertTrue(all(row['frame_indices'] == [1, 3] for row in rows))
        first = {row['audio_key']: row['split'] for row in rows}
        reordered = [dict(row) for row in reversed(rows)]
        assign_fit_and_eval_splits(reordered, seed=7)
        self.assertEqual(first,
            {row['audio_key']: row['split'] for row in reordered})
        assign_fit_and_eval_splits(reordered, seed=8)
        self.assertNotEqual(first,
            {row['audio_key']: row['split'] for row in reordered})

    def test_odd_and_single_recording_components_allow_evaluation_extra(self):
        '''An odd remainder and a singleton both go to evaluation.'''
        rows = [audio_to_info(_make_audio(i)) for i in range(3)]
        rows.append(audio_to_info(_make_audio(3, component='comp-o')))
        assign_fit_and_eval_splits(rows)
        splits = [row['split'] for row in rows[:3]]
        self.assertEqual(splits.count('fitting'), 1)
        self.assertEqual(splits.count('evaluation'), 2)
        self.assertEqual(rows[-1]['split'], 'evaluation')
        self.assertEqual(assign_fit_and_eval_splits([]), [])
        singleton = sample_frames([_make_audio(1)], n_samples=1)
        self.assertEqual(singleton[0]['split'], 'evaluation')

    def test_long_recordings_contribute_more_frame_slots(self):
        '''Sample frame slots rather than assigning quotas to recordings.'''
        audios = [_make_audio(1, 25), _make_audio(2, 2005)]
        rows = sample_frames(audios, n_samples=50)
        counts = [len(row['frame_indices']) for row in rows]
        self.assertLessEqual(counts[0], 1)
        self.assertGreaterEqual(counts[1], 49)
        self.assertEqual(sum(counts), 50)

    def test_rejects_impossible_sample_counts(self):
        '''Reject negative counts and requests exceeding the frame population.'''
        audios = [_make_audio(1, 105), _make_audio(2, 105)]
        for count in (-1, 11):
            with self.subTest(n_samples=count):
                with self.assertRaises(ValueError):
                    sample_frames(audios, n_samples=count)

    def test_zero_samples_retains_inventory_and_splits(self):
        '''A zero-sized selection keeps recording metadata without samples.'''
        rows = sample_frames([_make_audio(1), _make_audio(2)], n_samples=0)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['split'] for row in rows},
            {'fitting', 'evaluation'})
        self.assertEqual(list(iter_marker_info(
            {'audio_infos': rows}, FakeStore([]))), [])
        self.assertEqual(sample_frames([], n_samples=0), [])

    def test_marker_times_use_milliseconds_and_preserve_recording_positions(self):
        audio = _make_audio(1, 10005)
        info = audio_to_info(audio)
        info['frame_indices'] = [0, 150, info['n_frames'] - 1]
        markers = list(iter_marker_info({'audio_infos': [info]},
            FakeStore([audio]), label='custom'))
        expected = [(0, 25, 'custom_0'), (3000, 3025, 'custom_150'),
            (9980, 10005, 'custom_499')]
        for marker, (start, end, label) in zip(markers, expected):
            self.assertEqual(marker,
                {'audio': audio, 'start': start, 'end': end, 'label': label})

    def test_marker_milliseconds_round_floating_point_frame_bounds(self):
        audio = _make_audio(1)
        info = audio_to_info(audio)
        info['frame_indices'] = [3]
        marker, = iter_marker_info({'audio_infos': [info]}, FakeStore([audio]))
        self.assertEqual(marker['start'], 60)
        self.assertEqual(marker['end'], 85)

    def test_make_manifest_filters_and_roundtrips_without_annotations(self):
        '''Make and reload a manifest using only audio identity and timing.'''
        audios = [_make_audio(1), _make_audio(2, component='comp-o'),
            _make_audio(3, region='vl'), _make_audio(4, 999)]
        store = FakeStore(audios)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'random_frames'
            with patch.object(sampling.locations,
                'decomposition_random_frames_base', base):
                manifest = make_manifest(store, n_samples=10)
                self.assertTrue((base / 'region-nl_comps-k_o.json').is_file())
                loaded = load_manifest()
                self.assertEqual(loaded['audio_infos'], manifest['audio_infos'])
                self.assertEqual(loaded['components'], ['comp-k', 'comp-o'])
                with self.assertRaises(FileExistsError):
                    make_manifest(store, n_samples=10)
                self.assertEqual(load_manifest(), loaded)
                replacement = make_manifest(store, n_samples=3, overwrite=True)
                replaced = load_manifest()
                self.assertEqual(replaced['n_samples'], 3)
                self.assertEqual(replaced['audio_infos'], replacement['audio_infos'])
                self.assertNotEqual(replaced['audio_infos'], loaded['audio_infos'])
        self.assertEqual(manifest['n_samples'], 10)
        self.assertEqual(manifest['region'], 'nl')
        self.assertEqual(manifest['sampling_unit'], 'uniform_unique_frame')
        self.assertTrue(manifest['include_all_audio'])
        self.assertEqual(manifest['phraser_store_path'], str(store.path))
        keys = [row['audio_key'] for row in manifest['audio_infos']]
        self.assertEqual(keys, [audio.key.hex() for audio in audios[:2]])
        self.assertEqual(sum(len(row['frame_indices']) for row in manifest['audio_infos']), 10)

    def test_make_and_load_manifest_with_custom_region_and_components(self):
        '''Apply region/component overrides to selection and output naming.'''
        audios = [_make_audio(1, region='vl'),
            _make_audio(2, component='comp-o', region='vl'),
            _make_audio(3, component='comp-o')]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(sampling.locations,
                'decomposition_random_frames_base', base):
                manifest = make_manifest(FakeStore(audios),
                    region='vl', components=('comp-o',), n_samples=3)
                loaded = load_manifest(region='vl', components=('comp-o',))
                self.assertTrue((base / 'region-vl_comps-o.json').is_file())
        self.assertEqual(loaded['region'], 'vl')
        self.assertEqual(loaded['components'], ['comp-o'])
        self.assertEqual(len(manifest['audio_infos']), 1)
        self.assertEqual(loaded['audio_infos'][0]['audio_key'],
            audios[1].key.hex())
        self.assertEqual(sum(len(row['frame_indices']) for row in loaded['audio_infos']), 3)

    def test_load_missing_manifest_explains_how_to_create_it(self):
        '''A missing selection provides the requested creation parameters.'''
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sampling.locations,
                'decomposition_random_frames_base', Path(directory)):
                with self.assertRaises(FileNotFoundError) as caught:
                    load_manifest(region='vl', components=('comp-o',))
        self.assertIn('make_manifest', str(caught.exception))
        self.assertIn('region="vl"', str(caught.exception))
        self.assertIn('comp-o', str(caught.exception))

    def test_saved_manifest_roundtrip_and_replacement(self):
        '''The low-level writer saves and replaces manifest contents.'''
        rows = sample_frames([_make_audio(1), _make_audio(2)], n_samples=6)
        manifest = {'audio_infos': rows}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'frames.json'
            save_manifest(manifest, path)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, manifest)
            replacement = {'audio_infos': []}
            save_manifest(replacement, path)
            self.assertEqual(json.loads(path.read_text()), replacement)


if __name__ == '__main__':
    unittest.main()
