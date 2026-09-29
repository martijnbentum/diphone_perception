'''End-to-end checks for the one-time MFCC shard migration.'''

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import echoframe
import numpy as np
from echoframe.acoustic_features import make_acoustic_feature_item
from echoframe.output_storage import Hdf5ShardStore

import migrate_mfcc_shards


class TestMigrateMfccShards(unittest.TestCase):
    def test_resumes_after_interruption_during_source_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'mfcc_store'
            source = echoframe.Store(str(root))
            keys = []
            for number in range(5):
                phraser_key = f'marker-{number}'.encode().ljust(22, b'\0')
                key = source.make_echoframe_key('acoustic_feature',
                    feature_name='mfcc', phraser_key=phraser_key)
                matrix = np.full((1, 39), number, dtype='float32')
                item = make_acoustic_feature_item(key, 'mfcc', matrix,
                    source, phraser_source_id='test')
                source.save_many([item])
                keys.append(key)
            source.close()

            original_load = Hdf5ShardStore.load_many
            reads = 0

            def interrupt_second_source_read(storage, records):
                nonlocal reads
                if storage.root == root.resolve() / 'shards':
                    reads += 1
                    if reads == 2: raise KeyboardInterrupt()
                return original_load(storage, records)

            with mock.patch.object(Hdf5ShardStore, 'load_many',
                    interrupt_second_source_read), redirect_stdout(StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    migrate_mfcc_shards.migrate_mfcc_shards(root,
                        batch_size=2)
            self.assertTrue(root.is_dir())
            self.assertTrue(root.with_name('mfcc_store.resharded').is_dir())
            with mock.patch.object(migrate_mfcc_shards,
                    'MAX_SHARD_ITEMS', 2), redirect_stdout(StringIO()):
                backup = migrate_mfcc_shards.migrate_mfcc_shards(root,
                    batch_size=2, resume=True)

            rebuilt = echoframe.Store(str(root))
            try:
                self.assertEqual(len(rebuilt.index.all_echoframe_keys), 5)
                self.assertTrue(backup.is_dir())
                for number, key in enumerate(keys):
                    expected = np.full((1, 39), number, dtype='float32')
                    np.testing.assert_array_equal(rebuilt.load(key), expected)
            finally:
                rebuilt.close()

    def test_rebuilds_and_keeps_the_original_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'mfcc_store'
            source = echoframe.Store(str(root))
            keys = []
            for number in range(7):
                phraser_key = f'marker-{number}'.encode().ljust(22, b'\0')
                key = source.make_echoframe_key('acoustic_feature',
                    feature_name='mfcc', phraser_key=phraser_key)
                matrix = np.full((1, 39), number, dtype='float32')
                item = make_acoustic_feature_item(key, 'mfcc', matrix,
                    source, phraser_source_id='test')
                source.save_many([item])
                keys.append(key)
            source.close()

            output = StringIO()
            with mock.patch.object(migrate_mfcc_shards,
                    'MAX_SHARD_ITEMS', 2), redirect_stdout(output):
                backup = migrate_mfcc_shards.migrate_mfcc_shards(root,
                    batch_size=2)

            self.assertIn('Read 2/7; saved 0/7;', output.getvalue())
            self.assertIn('rough ETA', output.getvalue())
            self.assertIn('saved 7/7;', output.getvalue())
            rebuilt = echoframe.Store(str(root))
            original = echoframe.Store(str(backup))
            try:
                self.assertEqual(len(rebuilt.index.list_shards()), 4)
                self.assertEqual(len(original.index.list_shards()), 1)
                for number, key in enumerate(keys):
                    expected = np.full((1, 39), number, dtype='float32')
                    np.testing.assert_array_equal(rebuilt.load(key), expected)
                    np.testing.assert_array_equal(original.load(key), expected)
            finally:
                rebuilt.close()
                original.close()


if __name__ == '__main__':
    unittest.main()
