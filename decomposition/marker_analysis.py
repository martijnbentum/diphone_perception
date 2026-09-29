from decomposition import sampling
from decomposition import load_embeddings
from decomposition import load_mfcc as mfcc_loader
from decomposition import svd
from decomposition.audio import database as acoustic_database
import locations

import numpy as np
from progressbar import progressbar

def load_cgn_store():
    '''Open the default CGN Phraser store and return it.'''
    return sampling.load_cgn_store()


class Table:
    def __init__(self, cgn, layer = 9, select = 'eval', region = 'nl',
        components = ('comp-k', 'comp-o'), label = 'decomp_random_frames'):
        self.cgn = cgn
        self.layer = layer
        if select not in ['fit', 'eval', 'all']:
            raise ValueError(f'unknown select value {select!r}')
        self.select = select
        self.region = region
        self.components = components
        self.label = label
        self.manifest = sampling.load_manifest(self.region, self.components)
        self.markers = None

    def __repr__(self):
        comps = ','.join(self.components)
        markers = '?' if self.markers is None else len(self.markers)
        rows = len(self.rows) if hasattr(self, 'rows') else '?'
        return (f'Table({self.select} {self.region} comps=[{comps}] '
            f'layer={self.layer} markers={markers} rows={rows})')

    def load_markers(self):
        print('Loading markers...', flush=True)
        markers = list(self.cgn.markers.filter(label__startswith=self.label))
        load_embeddings.check_marker_alignment(markers, self.manifest,
            self.region, self.components, label=self.label)
        if self.select != 'all':
            split_list = sampling.load_splits(self.region, self.components,
                per_marker=True)
            if len(split_list) != len(markers):
                m = f'# marker {len(markers)} != # split {len(split_list)}'
                raise ValueError('marker and split counts must match' + m)
            selected = []
            bar = progressbar(markers, prefix='Selecting markers: ')
            for index, marker in enumerate(bar):
                if split_list[index] == self.select: selected.append(marker)
            markers = selected
        for name in ('embeddings', 'scores', 'mfcc', 'intensity',
            'frequency_band_power', 'frequency_band_names', 'marker_infos',
            'rows'):
            if hasattr(self, name): delattr(self, name)
        self.markers = markers
        print(f'Loaded {len(markers)} markers.', flush=True)

    def load_mfcc(self, store=None):
        '''Load a 39-column MFCC matrix in marker order.'''
        if self.markers is None: self.load_markers()
        print('Loading MFCCs...', flush=True)
        self.mfcc = mfcc_loader.load_mfcc(self.markers, store=store)
        if hasattr(self, 'rows'): del self.rows
        print('Loaded MFCCs.', flush=True)
        return self.mfcc

    def load_acoustic_features(self, database=None):
        '''Load dB intensity and linear frequency-band power in marker order.'''
        if self.markers is None: self.load_markers()
        print('Loading acoustic features...', flush=True)
        values = acoustic_database.load_marker_acoustic_vector(
            self.markers, 'all', database=database)
        names = acoustic_database.COLUMNS[1:]
        intensity = values['intensity_db']
        bands = []
        for name in names:
            bands.append(values[name])
        frequency_band_power = np.column_stack(bands)
        self.intensity = intensity
        self.frequency_band_power = frequency_band_power
        self.frequency_band_names = names
        if hasattr(self, 'rows'): del self.rows
        print('Loaded acoustic features.', flush=True)
        return intensity, frequency_band_power

    def load_embeddings(self):
        if self.markers is None: self.load_markers()
        print('Loading embeddings...', flush=True)
        embeddings = load_embeddings.load_embeddings(self.markers, self.cgn,
            layer=self.layer)
        self.embeddings = embeddings.embeddings
        if len(self.embeddings) != len(self.markers):
            m = f': {len(self.embeddings)} != {len(self.markers)}'
            m += 'embeddings and markers must have equal length'
            raise ValueError('embeddings and markers must have equal length')
        print('Loaded embeddings.', flush=True)

    def load_decomposition(self):
        print('Loading decomposition...', flush=True)
        p = locations.random_frames_decomposition
        self.decomposition = svd.load_svd(p)
        print('Loaded decomposition.', flush=True)

    def compute_svd_scores(self):
        if not hasattr(self, 'decomposition'): self.load_decomposition()
        if not hasattr(self, 'embeddings'): self.load_embeddings()
        print('Computing SVD scores...', flush=True)
        means = []
        bar = progressbar(self.embeddings, prefix='Embedding means: ')
        for embed in bar:
            means.append(embed.data.mean(axis=0))
        m = np.array(means)
        self.scores = svd.transform(m, self.decomposition)
        if hasattr(self, 'rows'): del self.rows
        print('Computed SVD scores.', flush=True)

    def set_marker_infos(self):
        if self.markers is None: self.load_markers()
        print('Loading marker info...', flush=True)
        marker_infos = []
        bar = progressbar(self.markers, prefix='Marker info: ')
        for marker in bar:
            info = marker_info_dict(marker)
            marker_infos.append(info)
        self.marker_infos = marker_infos
        print('Loaded marker info.', flush=True)

    def make_rows(self):
        '''Build rows from the loaded marker-aligned values.'''
        self._load()
        self.rows = []
        bar = progressbar(self.markers)
        for index, marker in enumerate(bar):
            row = Row(marker, self, index=index)
            self.rows.append(row)
        return self.rows

    def vector_names(self):
        '''Return column names for Row.to_vector in vector order.'''
        self._load()
        names = [f'mode_{index}' for index in range(self.scores.shape[1])]
        for kind in ('mfcc', 'delta_mfcc', 'delta2_mfcc'):
            for index in range(13):
                names.append(f'{kind}_{index}')
        names.append('intensity_db')
        names.extend(self.frequency_band_names)
        return tuple(names)

    def _load(self):
        '''Load every marker-aligned value needed by Row.'''
        if self.markers is None: self.load_markers()
        steps = []
        if not hasattr(self, 'decomposition'):
            steps.append(('decomposition', self.load_decomposition))
        if not hasattr(self, 'embeddings'):
            steps.append(('embeddings', self.load_embeddings))
        if not hasattr(self, 'scores'):
            steps.append(('SVD scores', self.compute_svd_scores))
        if not hasattr(self, 'mfcc'):
            steps.append(('MFCCs', self.load_mfcc))
        if not hasattr(self, 'intensity') or not hasattr(self,
            'frequency_band_power'):
            steps.append(('acoustic features', self.load_acoustic_features))
        if not steps: return
        bar = progressbar(steps, prefix='Table loading: ')
        for _, load in bar:
            load()


class Row:
    def __init__(self, marker, table, index=None):
        self.marker = marker
        if index is None: index = table.markers.index(marker)
        self.index = index
        self.marker_info = marker_info_row(marker)
        self.scores = table.scores[self.index]
        self.mfcc = table.mfcc[self.index]
        self.intensity = table.intensity[self.index]
        self.frequency_band_power = table.frequency_band_power[self.index]
        self.table = table

    def __repr__(self):
        info = self.marker_info
        kind = 'speech' if info['speech'] else 'no-speech'
        comp = info['comp']
        phone = info['phone_label'] or '-'
        if len(phone) > 4: phone = phone[:3] + '.'
        phone = f'[{phone}]'
        age = info['age'] if info['age'] is not None else '-'
        gender = info['gender'] or '-'
        return (f'Row({kind:<9} {comp:<6} p:{phone:<6} '
            f'age={str(age):<5} g:{gender:<6} '
            f'i:{self.intensity:>6.1f}dB)')

    def to_vector(self):
        '''Return scores and acoustic values as a flat numeric NumPy row.'''
        return np.concatenate((self.scores, self.mfcc,
            np.array([self.intensity]), self.frequency_band_power))

def marker_info_row(marker):
    return marker_info_dict(marker)

def marker_info_dict(marker):
    filename = marker.audio.filename
    comp = filename_to_comp(filename)
    d = {'marker_key':marker.key, 'filename': filename,
        'start': marker.start,'overlap': marker.overlap,
        'speech': marker.overlap, 'comp':comp}
    d.update(_marker_to_phone_label(marker))
    d.update(_marker_to_speaker_info(marker))
    return d

def _marker_to_phone_label(marker):
    d = {'phone_label': None}
    if not marker.overlap: return d
    labels = []
    for item in marker.overlap_items:
        if item.object_type == 'Phone':
            labels.append(item.label)
    if labels: d['phone_label'] = ','.join(labels)
    return d

def _marker_to_speaker_info(marker):
    speaker = marker.phrase.speaker if marker.phrase else None
    d = {}
    d['speaker'] = speaker.name if speaker else None
    d['gender'] = speaker.gender() if speaker else None
    d['age'] = speaker.age if speaker else None
    return d

def filename_to_comp(filename):
    parts = filename.split('/')
    for part in parts:
        if part.startswith('comp-'):
            return part
    raise ValueError(f'no comp- prefix found in {filename!r}')
