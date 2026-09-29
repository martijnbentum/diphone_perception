'''Store marker acoustic measurements in a small SQLite table.'''

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

import locations


COLUMNS = ('intensity_db', 'power_0_500', 'power_500_1000',
    'power_1000_2000', 'power_2000_4000')


def load_marker_acoustics(markers, database=None):
    '''Return named acoustic arrays in marker order from the SQLite file.

    markers:   iterable of saved markers with unique Phraser keys
    database:  SQLite path; None uses the decomposition acoustics database

    A missing marker raises ValueError. Missing intensity within a stored row
    is returned as NaN; zero band power remains zero.
    '''
    markers = list(markers)
    if not markers: raise ValueError('markers must not be empty')
    keys = [bytes(marker.key) for marker in markers]
    if len(set(keys)) != len(keys):
        raise ValueError('markers must have unique Phraser keys')
    if database is None:
        database = locations.decomposition_random_frames_acoustics_db
    database = Path(database)
    if not database.exists(): raise FileNotFoundError(database)
    with closing(sqlite3.connect(database)) as connection:
        found = {}
        for start in range(0, len(keys), 900):
            selected = keys[start:start + 900]
            placeholders = ','.join('?' for _ in selected)
            columns = ', '.join(('marker_key',) + COLUMNS)
            query = f'SELECT {columns} FROM marker_acoustics '
            query += f'WHERE marker_key IN ({placeholders})'
            for row in connection.execute(query, selected):
                found[row[0]] = row[1:]
    missing = [index for index, key in enumerate(keys) if key not in found]
    if missing:
        message = f'missing acoustics for marker {missing[0]} '
        message += f'({len(missing)} missing)'
        raise ValueError(message)
    rows = []
    for key in keys:
        row = []
        for value in found[key]:
            row.append(np.nan if value is None else value)
        rows.append(row)
    values = np.array(rows, dtype=np.float64)
    return {column: values[:, index] for index, column in enumerate(COLUMNS)}


def _prepare_database(connection, settings, overwrite):
    '''Create tables and reset rows when extraction settings change.'''
    connection.execute('''CREATE TABLE IF NOT EXISTS marker_acoustics (
        marker_key BLOB PRIMARY KEY,
        intensity_db REAL,
        power_0_500 REAL NOT NULL,
        power_500_1000 REAL NOT NULL,
        power_1000_2000 REAL NOT NULL,
        power_2000_4000 REAL NOT NULL
    )''')
    connection.execute('''CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )''')
    encoded = json.dumps(settings, sort_keys=True)
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = 'settings'")
    previous = row.fetchone()
    if overwrite or previous is None or previous[0] != encoded:
        connection.execute('DELETE FROM marker_acoustics')
        connection.execute('''INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('settings', ?)''', (encoded,))
