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

import migrate_mfcc_shards


class TestMigrateMfccShards(unittest.TestCase):
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
