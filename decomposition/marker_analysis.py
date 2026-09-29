from decomposition import sampling
from decomposition import load_embeddings
from decomposition import load_mfcc as mfcc_loader
from decomposition import svd
from decomposition.audio import database as acoustic_database
import locations

import numpy as np

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

    def load_markers(self):
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
            for marker, split in zip(markers, split_list):
                if split == self.select: selected.append(marker)
            markers = selected
        for name in ('embeddings', 'scores', 'mfcc', 'intensity',
            'frequency_band_power', 'frequency_band_names', 'marker_infos',
            'rows'):
            if hasattr(self, name): delattr(self, name)
        self.markers = markers

    def load_mfcc(self, store=None):
        '''Load a 39-column MFCC matrix in marker order.'''
        if self.markers is None: self.load_markers()
        self.mfcc = mfcc_loader.load_mfcc(self.markers, store=store)
        if hasattr(self, 'rows'): del self.rows
        return self.mfcc

    def load_acoustic_features(self, database=None):
        '''Load dB intensity and linear frequency-band power in marker order.'''
        if self.markers is None: self.load_markers()
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
        return intensity, frequency_band_power

    def load_embeddings(self):
        if self.markers is None: self.load_markers()
        embeddings = load_embeddings.load_embeddings(self.markers, self.cgn,
            layer=self.layer)
        self.embeddings = embeddings.embeddings
        if len(self.embeddings) != len(self.markers):
            m = f': {len(self.embeddings)} != {len(self.markers)}'
            m += 'embeddings and markers must have equal length'
            raise ValueError('embeddings and markers must have equal length')

    def load_decomposition(self):
        p = locations.random_frames_decomposition
        self.decomposition = svd.load_svd(p)

    def compute_svd_scores(self):
        if not hasattr(self, 'decomposition'): self.load_decomposition()
        if not hasattr(self, 'embeddings'): self.load_embeddings()
        m = np.array([embed.data.mean(axis=0) for embed in self.embeddings])
        self.scores = svd.transform(m, self.decomposition)
        if hasattr(self, 'rows'): del self.rows

    def set_marker_infos(self):
        marker_infos = [marker_info_dict(marker) for marker in self.markers]
        self.marker_infos = marker_infos

    def make_rows(self):
        '''Build rows from the loaded marker-aligned values.'''
        self._load()
        self.rows = []
        for index, marker in enumerate(self.markers):
            row = Row(marker, self, index=index)
            self.rows.append(row)
        return self.rows

    def _load(self):
        '''Load every marker-aligned value needed by Row.'''
        if self.markers is None: self.load_markers()
        if not hasattr(self, 'scores'): self.compute_svd_scores()
        if not hasattr(self, 'mfcc'): self.load_mfcc()
        if not hasattr(self, 'intensity') or not hasattr(self,
            'frequency_band_power'):
            self.load_acoustic_features()


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
    d['gender'] = speaker.gender if speaker else None
    d['age'] = speaker.age if speaker else None
    return d

def filename_to_comp(filename):
    parts = filename.split('/')
    for part in parts:
        if part.startswith('comp-'):
            return part
    raise ValueError(f'no comp- prefix found in {filename!r}')
