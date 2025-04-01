import locations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import phoneme_mapper as pm

pp_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21]

def open_responses(pp_id = 1, gate = 1):
    pp = locations.rawdata / f'pp{pp_id:02d}_gate{gate}'
    with open(pp) as f:
        t = f.read().split('\n')
    return t

class Participants:
    def __init__(self, pp_ids = pp_ids):
        self.pp_ids = pp_ids
        self._add_participants()

    def __repr__(self):
        m = f'(Participants) n: {self.n_participants}'
        m += f', n_trials: {self.n_trials}'
        return m

    def _add_participants(self):
        self.participants = []
        for pp_id in self.pp_ids:
            self.participants.append(Participant(pp_id))
        self.n_participants = len(self.participants)
        self.n_trials = sum([p.n_trials for p in self.participants])
    

class Participant:
    def __init__(self, pp_id = 1):
        self.pp_id = pp_id
        self._add_responses()
    
    def __repr__(self):
        m = f'(Participant) pp_id: {self.pp_id}'
        m += f', n_trials: {self.n_trials}'
        return m

    def _add_responses(self):
        self.responses = []
        for gate in range(1, 7):
            self.responses.append(Response(self.pp_id, gate))
        self.n_trials = sum([r.n_trials for r in self.responses])

class Response:
    def __init__(self, pp_id, gate):
        self.pp_id = pp_id
        self.gate = gate
        self.data = open_responses(pp_id, gate)
        self.n_trials = len(self.data)

    def __repr__(self):
        m = f'(Response) pp_id: {self.pp_id}, gate: {self.gate}'
        m += f', n_trials: {self.n_trials}'
        return m

def _clean_row(row):
    cleaned_row = [char for char in ''.join(row).split()]
    return cleaned_row

def load_matrix(filename = '', diphone_position = 1, gate = 1):
    if not filename:
        phon = diphone_position
        filename = locations.matrices / f'phon{phon}_conf_matrix_gate{gate}.dat'
    with open(filename, 'r') as f:
        m = f.read().split('\n')
    matrix = []    
    row_names = []
    for i, row in enumerate(m):
        if not row: continue
        if i == 0: 
            column_names = _clean_row(row)
            continue
        row = _clean_row(row)
        row_names.append(row[0])
        matrix.append(list(map(int,row[1:])))
    matrix = np.array(matrix)
    return matrix, row_names, column_names, filename

class Matrix:
    def __init__(self,diphone_position= 1, gate = 1, filename = ''):
        self.diphone_position = diphone_position
        self.gate = gate
        self.filename = filename
        self.to_ipa_org = pm.to_ipa_org
        self._add_matrix()
        self._normalise_matrix()
        self.info = {'row': 'responses, row name is the ground truth phoneme', 
            'column': 'phoneme class, column name is the selected phoneme class'}

    def __repr__(self):
        m = f'(Matrix) diphone position: {self.diphone_position}'
        m += f', gate: {self.gate}'
        m += f', {self.n_rows} X {self.n_columns}'
        return m

    def _add_matrix(self):
        matrix, _row_names, _column_names, filename = load_matrix(
            diphone_position = self.diphone_position, gate = self.gate,
            filename = self.filename)
        self.matrix = matrix
        self._row_names = _row_names
        self._column_names = _column_names
        self.filename = filename
        self.n_rows = len(self._row_names)
        self.n_columns = len(self._column_names)
        self.row_names = [self.to_ipa_org[r] for r in self._row_names]
        self.column_names = [self.to_ipa_org[c] for c in self._column_names]

    def _normalise_matrix(self):
        row_norms = np.linalg.norm(self.matrix, axis = 1, keepdims = True)
        self.normalised_matrix = self.matrix / row_norms

    def plot(self, color_scheme = 'Blues', normalise = False, ax = None,
        show_colorbar = True, show = True, return_im = False, 
        show_axis_labels = True):
        if normalise:
            matrix = self.normalised_matrix
        else:
            matrix = self.matrix
            
        name = self.__repr__().replace(', 38 X 38', '').replace('(Matrix) ', '')
        if not ax: fig, ax = plt.subplots(1, 1, figsize = (15, 15))
        im = ax.imshow(matrix, cmap = color_scheme)
        if show_colorbar:
            plt.colorbar()
        ax.set_title(f'{name}')
        ax.set_xticks( range(self.n_columns)) 
        ax.set_xticklabels( self.column_names, rotation = 90)
        ax.set_yticks(range(self.n_rows))
        ax.set_yticklabels( self.row_names )
        if show_axis_labels:
            ax.set_ylabel('ground truth')
            ax.set_xlabel('selected')
        if show: plt.show()
        if return_im: return im


class Matrices:
    def __init__(self, diphone_positions = [1, 2], gates = [1, 2, 3, 4, 5, 6]):
        self.diphone_positions = diphone_positions
        self.gates = gates
        self._add_matrices()

    def __repr__(self):
        m = f'(Matrices) n: {self.n_matrices}'
        m += f', diphone_positions: {self.diphone_positions}'
        m += f', gates: {self.gates}'
        return m

    def _add_matrices(self):
        self.matrices = []
        for diphone_position in self.diphone_positions:
            for gate in self.gates:
                self.matrices.append(Matrix(diphone_position, gate))
        '''
        # unknown format
        for f in ['con_conf_matrix_gates14.dat', 'vow_conf_matrix_gates14.dat']:
            filename = locations.matrices / f
            self.matrices.append(Matrix(filename = filename))
        '''

    def plot(self, color_scheme = 'Blues', figsize= (12,36)):
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize = figsize)
        for matrix in self.matrices:
            col_index = matrix.diphone_position - 1
            row_index = matrix.gate - 1
            show_labels = True if col_index == 0 and row_index == 5 else False
            ax = axes[row_index, col_index]
            im = matrix.plot(color_scheme, ax = ax, show_colorbar = False, 
                show = False, return_im = True, normalise = True,
                show_axis_labels = show_labels)
        cbar_ax = fig.add_axes([0.975, 0.05, 0.009, 0.1])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, cmap = color_scheme)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0', '1'])
        cbar_ax.yaxis.set_ticks_position('right')
        plt.show()
        return cbar_ax, cbar

coding_dict = {'0': 'unstresssed', '1': 'stressed', 
    '2': 'syllable boundary in diphone, stress on second phoneme', 
    '3': 'syllable boundary in diphone, stress on first phoneme',
    's': 'segment contains two strong vowels both stressed',
    'w': 'segment contains two weak vowels both stressed'}

def _handle_second_phoneme(disc2, to_ipa):
    if len(disc2) == 1:
        phoneme2, coding = to_ipa[disc2], None
    elif disc2[1] in '0123sw':
        phoneme2, coding = to_ipa[disc2[0]], disc2[1:]
    elif disc2[:2] in to_ipa:
        phoneme2, coding = to_ipa[disc2[:2]], disc2[2:]
    else:
        raise ValueError(f'Unknown phoneme: {disc2}')
    if len(coding) == 1:
        coding, prefix = coding[0], None
    else:
        coding, prefix = coding[0], coding[1:]
    prefix = to_ipa[prefix] if prefix else None
    return phoneme2, coding, prefix

def handle_label_filename(filename, to_ipa = None):
    if not to_ipa: to_ipa = pm.to_ipa
    filename = Path(filename)
    name = filename.stem.split('.')[0]
    phoneme_type = filename.parent.stem
    disc1 = name.split('_')[0]
    disc2 = name.split('_')[1].split('.')[0]
    phoneme1 = to_ipa[disc1]
    phoneme2, coding, prefix = _handle_second_phoneme(disc2, to_ipa)
    coding_name = coding_dict[coding] if coding else None
    info = {'filename': filename, 'name': name, 'phoneme_type': phoneme_type,
        'disc1': disc1, 'disc2': disc2, 'phoneme1': phoneme1,
        'phoneme2': phoneme2, 'coding': coding, 'coding_name': coding_name,
        'prefix': prefix}
    return info

class Label:
    '''
    Structure of the label file:
    label starts with some metadata, followed by lines of time information
    #                   start of the lines with time information
    0.000       117     diphone_label
    start_time  121     b (begin?) start time of first phoneme
    end_time    121     m (end?) end time of first phoneme
    [optional]  121     1 or 2  if this line is present
                        it is the start time of the second phoneme
                        otherwise it 'm' is the start time of the second phoneme
    end_time    121     m end time of second phoneme
    duration    117     diphone_label duration of audio file
    signal wav_filename
    '''

    def __init__(self, filename):
        self.filename = filename
        self._add_label()
        self.to_ipa = pm.Mapper().to_ipa
            

    def __repr__(self):
        m = f'(Label) phoneme 1: {self.phoneme1}'
        m += f', phoneme 2: {self.phoneme2}'
        return m

    def _add_label(self):
        with open(self.filename, 'r') as f:
            self.raw_data = f.read().split('\n')
        self.disc1 = self.filename.split('_')[0]
        self.disc2 = self.filename.split('_')[1]
        self.phoneme1 = pm.to_ipa_org[self.disc1]
            

