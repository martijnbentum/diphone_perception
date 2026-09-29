from decomposition import sampling
from decomposition import load_embeddings
from decomposition import svd
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
        self.markers = markers
        if self.select == 'all': return
        split_list = sampling.load_splits(self.region, self.components,
            per_marker=True)
        if len(split_list) != len(self.markers):
            m = f'# marker {len(self.markers)} != # split {len(split_list)}'
            raise ValueError('marker and split counts must match' + m)
        markers = []
        for marker, split in zip(self.markers, split_list):
            if split == self.select: markers.append(marker)
        self.markers = markers

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

    def set_marker_infos(self):
        marker_infos = [marker_info_dict(marker) for marker in self.markers]
        self.marker_infos = marker_infos


class Row:
    def __init__(self, marker, table):
        self.marker = marker
        self.marker_info = marker_info_row(marker)
        self.scores = table.scores[table.markers.index(marker)]
        self.table= table

def marker_info_row(marker):
    d = marker_info_dict(marker)
    return d.values()

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
