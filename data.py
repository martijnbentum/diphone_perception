import json
import locations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import phoneme_mapper as pm
# from progressbar import progressbar

pp_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21]

def open_responses(pp_id = 1, gate = 1):
    pp = locations.rawdata / f'pp{pp_id:02d}_gate{gate}'
    with open(pp) as f:
        t = f.read().split('\n')
    t = [x for x in t if x]
    return t

def open_gate_timestamps():
    with open(locations.gate_timestamps) as f:
        t = [x.split(',') for x in f.read().split('\n')]
    output = []
    for line in t:
        if len(line) == 2:
            try: line[1] = float(line[1])
            except ValueError: continue
            output.append(line)
    output = {x[0]: x[1] for x in output}
    return output
    

def save_per_participant_confusion_dicts(participants = None, filename = ''):
    if not filename:
        filename = '../per_participant_confusion_dicts.json'
    if not participants:
        participants = Participants()
    with open(filename, 'w') as f:
        json.dump(participants.confusion_dict, f, indent=4)
    print(f'Saved per participant confusion dicts to {filename}')

class Participants:
    def __init__(self, pp_ids = pp_ids, labels = None,):
        self.pp_ids = pp_ids
        if labels is None: self.labels = Labels()
        else: self.labels = labels
        self._add_participants()

    def __repr__(self):
        m = f'(Participants) n: {self.n_participants}'
        m += f', n_trials: {self.n_trials} n_errors: {self.n_errors}'
        return m

    def _add_participants(self):
        self.participants = []
        for pp_id in self.pp_ids:
            self.participants.append(Participant(pp_id, self))
        self.n_participants = len(self.participants)
        self.n_trials = sum([p.n_trials for p in self.participants])
        self.n_errors = sum([p.n_errors for p in self.participants])

    @property
    def confusion_dict(self):
        if hasattr(self, '_confusion_dict'):
            return self._confusion_dict
        d = {}
        for participant in self.participants:
            pp_id = participant.pp_id
            d[f'pp{pp_id}'] = participant.confusion_dict
        self._confusion_dict = d
        return self._confusion_dict
    

class Participant:
    def __init__(self, pp_id = 1, parent = None):
        self.pp_id = pp_id
        self.parent = parent
        self._add_responses()
    
    def __repr__(self):
        m = f'(Participant) pp_id: {self.pp_id}'
        m += f', n_trials: {self.n_trials} n_errors: {self.n_errors}'
        return m

    def _add_responses(self):
        self.responses = []
        for gate in range(1, 7):
            self.responses.append(Response(self.pp_id, gate, self))
        self.n_trials = sum([r.n_trials for r in self.responses])
        self.n_errors = sum([r.n_errors for r in self.responses])

    @property
    def labels(self):
        if self.parent is None:
            return None
        if not hasattr(self.parent, 'labels'):
            return None
        return self.parent.labels

    def get_gate_responses(self, gate):
        if gate < 1 or gate > 6:
            raise ValueError('Gate must be between 1 - 6')
        return self.responses[gate - 1]

    def gate_phoneme_position_to_confusion_dict(self, gate, phoneme_position):
        if gate < 1 or gate > 6:
            raise ValueError('Gate must be between 1 - 6')
        if phoneme_position < 1 or phoneme_position > 2:
            raise ValueError('phoneme_position must be 1 or 2')
        responses =  self.responses[gate - 1]
        return responses.confusion_dict[f'phoneme{phoneme_position}']

    @property
    def confusion_dict(self):
        if hasattr(self, '_confusion_dict'):
            return self._confusion_dict
        d = {}
        for responses in self.responses:
            gate= responses.gate
            d[f'gate{gate}'] = responses.confusion_dict
        self._confusion_dict = d
        return self._confusion_dict

    def matrices(self, diphone_positions = [1, 2], gates = [1, 2, 3, 4, 5, 6]):
        if hasattr(self, '_matrices'):
            return self._matrices
        m = Matrices(diphone_positions = diphone_positions,
            gates = gates, participant = self)
        self._matrices = m
        return self._matrices


    def plot_confusion_matrices(self,save = False):
        matrices = self.matrices()
        matrices.plot()
        plt.show()
        plt.tight_layout()
        plt.show()
        if save:
            p = locations.matrix_plots / f'pp{self.pp_id}_confusion_matrices.pdf'
            filename = str(p)
            plt.savefig(filename)
            print(f'Saved confusion matrices to {filename}')



class Response:
    def __init__(self, pp_id, gate, parent = None):
        self.pp_id = pp_id
        self.gate = gate
        self.parent = parent
        self.data = open_responses(pp_id, gate)
        self.n_trials = len(self.data)
        self._parse_data()

    def __repr__(self):
        m = f'(Response) pp_id: {self.pp_id}, gate: {self.gate}'
        m += f', n_trials: {self.n_trials} n_errors: {self.n_errors}'
        return m

    def _parse_data(self):
        self.responses= []
        self.errors = []
        for line in self.data:
            response = Response_line(line, self)
            if response.ok:
                self.responses.append(response)
            else:
                self.errors.append(Response_line(line, self))
        self.n_errors = len(self.errors)

    @property
    def labels(self):
        if self.parent is None:
            return None
        if not hasattr(self.parent, 'labels'):
            return None
        return self.parent.labels

    @property
    def confusion_dict(self):
        if hasattr(self, '_confusion_dict'):
            return self._confusion_dict
       
        self._confusion_dict = response_lines_to_confusion_dict(self.responses)
        return self._confusion_dict

    @property
    def confusion_dict_phoneme1(self):
        return self.confusion_dict['phoneme1'] 

    @property
    def confusion_dict_phoneme2(self):
        return self.confusion_dict['phoneme2']
        


def response_lines_to_confusion_dict(response_lines):
    output = {'phoneme1': {}, 'phoneme2': {}}
    for response in response_lines:
        for phoneme_position in [1, 2]:
            d = output[f'phoneme{phoneme_position}'] 
            gt, hyp = getattr(response, f'gt_phoneme{phoneme_position}'), \
                getattr(response, f'response_phoneme{phoneme_position}')
            if gt not in d:
                d[gt] = {}
            if hyp not in d[gt]:
                d[gt][hyp] = 0
            d[gt][hyp] += 1
    return output

def parse_response_line(line):
    gt, response = line.split('||')
    gt_phoneme1, gt_phoneme2, coding = gt.split(':')
    gt_phoneme1 = gt_phoneme1.strip()
    gt_phoneme1 = pm.to_ipa_org[gt_phoneme1]
    gt_phoneme2 = pm.to_ipa_org[gt_phoneme2]
    coding = coding.strip()
    coding_label = coding_dict[coding]
    response = response.strip()
    assert len(response) == 2
    response_labels = [pm.to_ipa_org[r] for r in response]
    response_phoneme1, response_phoneme2 = response_labels
    info = {'gt_phoneme1': gt_phoneme1, 'gt_phoneme2': gt_phoneme2,
        'coding': coding, 'coding_label': coding_label, 'response': response,
        'response_phoneme1': response_phoneme1, 
        'response_phoneme2': response_phoneme2}
    return info

            

    
class Response_line:
    def __init__(self, line, parent):
        self.line = line
        self.parent = parent
        try: self._parse_line()
        except ValueError: self.ok = False
        else: self.ok = True
        if self.ok: self.set_label()

    def __repr__(self):
        if not self.ok:
            return f'(Response_line) invalid line: {self.line}'
        m = f'(Response_line) gt: {self.gt}, response: {self.response}'
        return m

    def _parse_line(self):
        self.info = parse_response_line(self.line)
        for k, v in self.info.items():
            setattr(self, k, v)
        self.gt = f'{self.gt_phoneme1} {self.gt_phoneme2}'
        self.response = f'{self.response_phoneme1} {self.response_phoneme2}'

    def set_label(self):
        if self.labels is None:
            self.label = None
            return
        p1, p2, coding = self.gt_phoneme1, self.gt_phoneme2, self.coding
        labels = self.labels.get_labels(p1, p2, coding)
        self.label = labels[0] if labels else None
        if len(labels) > 1:
            print(f'Warning: multiple labels found for {p1} {p2} {coding}')
            self.all_found_labels = labels
        else: self.all_found_labels = []
        self.n_labels = len(labels)
        if self.label:
            self.label.add_response(self)
            

    @property
    def labels(self):
        if self.parent is None:
            return None
        if not hasattr(self.parent, 'labels'):
            return None
        return self.parent.labels
        

    @property
    def gate(self):
        if self.parent is None:
            return None
        if not hasattr(self.parent, 'gate'):
            return None
        return self.parent.gate

    @property
    def pp_id(self):
        if self.parent is None:
            return None
        if not hasattr(self.parent, 'pp_id'):
            return None
        return self.parent.pp_id

    @property
    def info_dict(self):
        d = {}
        d['gt_phoneme1'] = self.gt_phoneme1
        d['gt_phoneme2'] = self.gt_phoneme2
        d['response_phoneme_1'] = self.response_phoneme1
        d['response_phoneme_2'] = self.response_phoneme2
        d['participant'] = self.pp_id
        d['gate'] = self.gate
        return d



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

def _get_row_and_column_names():
    m = Matrix()
    return m.row_names, m.column_names

class Matrix:
    def __init__(self,diphone_position= 1, gate = 1, filename = '', 
        confusion_dict = None, participant = None):
        self.diphone_position = diphone_position
        self.gate = gate
        self.filename = filename
        self.confusion_dict = confusion_dict
        if participant:
            d = participant.gate_phoneme_position_to_confusion_dict(gate, 
                diphone_position)
            self.confusion_dict = d
            self.participant = participant
            self.pp_id = participant.pp_id
        self.to_ipa_org = pm.to_ipa_org
        self._add_matrix()
        self._normalise_matrix()
        self.info = {'row': 'responses, row name is the ground truth phoneme', 
            'column': 'phoneme class, column name is the selected phoneme class'}

    def __repr__(self):
        m = f'(Matrix) diphone position: {self.diphone_position}'
        m += f', gate: {self.gate}'
        m += f', {self.n_rows} X {self.n_columns}'
        if hasattr(self, 'pp_id'):
            m += f', particpant id: {self.pp_id}'
        return m

    def _add_matrix(self):
        if self.confusion_dict:
            self._add_matrix_from_confusion_dict()
            return
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


    def _add_matrix_from_confusion_dict(self):
        self.row_names, self.column_names = _get_row_and_column_names()
        self.n_rows = len(self.row_names)
        self.n_columns = len(self.column_names)
        self.matrix = np.zeros((self.n_rows, self.n_columns))
        for gt in self.confusion_dict.keys():
            for  hyp in self.confusion_dict[gt].keys():
                count = self.confusion_dict[gt][hyp]
                gt_index = self.row_names.index(gt)
                hyp_index = self.column_names.index(hyp)
                self.matrix[gt_index, hyp_index] = count
        

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
            try:plt.colorbar()
            except RuntimeError:
                fig.colorbar(im, ax=ax)
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
    def __init__(self, diphone_positions = [1, 2], gates = [1, 2, 3, 4, 5, 6],
        participant = None): 
        self.diphone_positions = diphone_positions
        self.gates = gates
        self.participant = participant
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
                self.matrices.append(Matrix(diphone_position, gate, 
                    participant = self.participant))
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
    'w': 'segment contains two weak vowels both stressed',
    'b': 'leading environment'}

def _handle_second_phoneme(disc2, to_ipa):
    if len(disc2) == 1:
        phoneme2, coding = to_ipa[disc2], None
    elif disc2[1] in '0123swb':
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
    f = str(filename)
    info = {'filename': f, 'name': name, 'phoneme_type': phoneme_type,
        'disc1': disc1, 'disc2': disc2, 'phoneme1': phoneme1,
        'phoneme2': phoneme2, 'coding': coding, 'coding_name': coding_name,
        'prefix': prefix}
    return info

class Labels:
    def __init__(self, directory = locations.labels):
        self.to_ipa = pm.to_ipa
        self.directory = directory
        self.filenames = directory.glob('*/*.lab')
        self.gate_timestamps = open_gate_timestamps()
        self.add_labels()

    def add_labels(self):
        self.labels = []
        # for filename in progressbar(self.filenames):
        for filename in self.filenames:
            self.labels.append(Label(filename, self))

    def get_labels(self, phoneme_1, phoneme_2, coding):
        '''
        phoneme_1: str, phoneme 1
        phoneme_2: str, phoneme 2
        coding: str, coding of the second phoneme
        '''
        labels = []
        for label in self.labels:
            if label.phoneme1 == phoneme_1 and label.phoneme2 == phoneme_2:
                if label.coding == coding:
                    labels.append(label)
        return labels
            
    def to_json(self, filename = 'info.json', save = False):
        labels = [x for x in self.labels if x.n_gated_audio_files >= 4]
        sorted_labels = sorted(labels, key = lambda x: x.name)
        d = {}
        for label in sorted_labels:
            name = '_'.join(label.name)
            d[name] = label.info_dict
        if save:
            with open(filename, 'w') as f:
                json.dump(d, f, indent=4)
            print(f'Saved labels to {filename}')
        return d
    
    

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
    end_time    121     e end time of second phoneme
    duration    117     diphone_label duration of audio file
    signal wav_filename
    '''

    def __init__(self, filename, parent = None):
        self.filename = filename
        self.parent = parent
        self.gate_timestamps = parent.gate_timestamps if parent else None
        if self.gate_timestamps is None:
            self.gate_timestamps = open_gate_timestamps()
        if self.parent == None: self.to_ipa = pm.to_ipa
        else: self.to_ipa = self.parent.to_ipa
        self._add_info()
        with open(self.filename, 'r') as f:
            self.raw_data = f.read().split('\n')
        self._parse_label_data()
        self._set_times()
        self._find_audio_files()
        self.responses = []
        self.name = (self.phoneme1, self.phoneme2, self.coding)
            
    def __repr__(self):
        m = f'(Label) phoneme 1: {self.phoneme1}'
        m += f' ({self.phoneme1_start_time:.2f} - {self.phoneme1_end_time:.2f})'
        m += f', phoneme 2: {self.phoneme2}'
        m += f' ({self.phoneme2_start_time:.2f} - {self.phoneme2_end_time:.2f})'
        return m

    def _add_info(self):
        info = handle_label_filename(self.filename, to_ipa = self.to_ipa)
        for k, v in info.items():
            setattr(self, k, v)

    def _parse_label_data(self):
        label_data = self.raw_data
        label_lines = label_data[label_data.index('#') + 1:]
        label_lines = [x for x in label_lines if x]
        self.label_lines= []
        for label_line in label_lines:
            self.label_lines.append(Label_line(label_line, self))

    def _set_times(self):
        self.timing_ok = True
        self.start_time = self.label_lines[0].time
        self.end_time = self.label_lines[-1].time
        self.duration = self.end_time - self.start_time
        if 'b' in self.label_to_label_line:
            self.phoneme1_start_time = self.label_to_label_line['b'].time
        else:self.phoneme1_start_time = None
        if 'm' in self.label_to_label_line:
            self.phoneme1_end_time = self.label_to_label_line['m'].time
        else: self.phoneme1_end_time = None
        if '1' in self.label_to_label_line:
            self.phoneme2_start_time = self.label_to_label_line['1'].time
        elif '2' in self.label_to_label_line:
            self.phoneme2_start_time = self.label_to_label_line['2'].time
        elif 'm' in self.label_to_label_line:
            self.phoneme2_start_time = self.label_to_label_line['m'].time
        else: self.phoneme2_start_time = None
        if 'e' in self.label_to_label_line:
            self.phoneme2_end_time = self.label_to_label_line['e'].time
        else: self.phoneme2_end_time = None

    def _find_audio_files(self):
        p = locations.original / self.phoneme_type
        name = Path(self.filename).stem.split('.')[0]
        original_audio = p / f'{name}.wav'
        self.audio_filenames_ok = True
        if original_audio.exists():
            self.original_audio_filename = str(original_audio)
        else:
            print(f'Original audio file not found: {original_audio}')
            self.original_audio_filename = '' 
            self.audio_filenames_ok = False
        for gate in range(1, 7):
            p = locations.gated / self.phoneme_type
            gated_audio = p / f'{name}{gate}.wav'
            if gated_audio.exists():
                setattr(self, f'gated_audio_filename_{gate}', str(gated_audio))
            else:
                setattr(self, f'gated_audio_filename_{gate}', '')
                self.audio_filenames_ok = False
        self.n_gated_audio_files = len(self.all_gated_audio_filenames)

    def add_response(self, response):
        self.responses.append(response)

    @property
    def label_to_label_line(self):
        d = {}
        for line in self.label_lines:
            d[line.label] = line
        return d

    def get_gated_audio_filename(self, gate):
        if gate < 1 or gate > 6:
            raise ValueError('Gate must be between 1 - 6')
        if hasattr(self, f'gated_audio_filename_{gate}'):
            return getattr(self, f'gated_audio_filename_{gate}')
        else:
            return None

    def get_gate_timestamp(self, gate):
        if gate < 1 or gate > 6:
            raise ValueError('Gate must be between 1 - 6')
        f = self.get_gated_audio_filename(gate)
        filename = Path(f).name
        if filename in self.gate_timestamps:
            return self.gate_timestamps[filename]
        else:
            return None

    @property
    def all_gated_audio_filenames(self):
        filenames = []
        for gate in range(1, 7):
            filename = self.get_gated_audio_filename(gate)
            if filename:
                filenames.append(filename)
        return filenames

    @property
    def all_gated_audio_filenames_with_timestamps(self):
        filenames = []
        for gate in range(1, 7):
            filename = self.get_gated_audio_filename(gate)
            if filename:
                timestamp = self.get_gate_timestamp(gate)
                filenames.append((filename, timestamp))
        return filenames

    @property
    def response_dicts(self):
        output = []
        for response in self.responses:
            output.append(response.info_dict)
        return output
        

    @property
    def timestamp_dict(self):
        d = {}
        d['start_time'] = self.start_time
        d['end_time'] = self.end_time
        d['duration'] = self.duration
        d['phoneme_1_start_time'] = self.phoneme1_start_time
        d['phoneme_1_end_time'] = self.phoneme1_end_time
        d['phoneme_2_start_time'] = self.phoneme2_start_time
        d['phoneme_2_end_time'] = self.phoneme2_end_time
        for i in range(1,7):
            name = f'gate_{i}_timestamp'
            timestamp = self.get_gate_timestamp(i)
            if timestamp:d[name] = timestamp
        return d

    @property
    def filename_dict(self):
        d = {}
        d['original_audio_filename'] = self.original_audio_filename
        for i in range(1, 7):
            name = f'gate_{i}_audio_filename'
            filename = self.get_gated_audio_filename(i)
            if filename:d[name] = filename
        d['label_filename'] = self.filename
        for k, v in d.items():
            d[k] = v.replace(str(locations.base) + '/', '')
        return d
                
    @property
    def info_dict(self):
        d = {}
        d['diphone'] = f'{self.phoneme1} {self.phoneme2}'
        d['phoneme_1'] = self.phoneme1
        d['phoneme_2'] = self.phoneme2
        d['disc_1'] = self.disc1
        d['disc_2'] = self.disc2
        d['phoneme_type'] = self.phoneme_type
        d['coding'] = self.coding
        d['coding_name'] = self.coding_name
        d['prefix'] = self.prefix
        d.update(self.filename_dict)
        d.update(self.timestamp_dict)
        d['responses'] = self.response_dicts
        return d
        
        

        
        
class Label_line:
    def __init__(self, label_line, parent = None):
        self.label_line = label_line
        time, code, label = [x for x in label_line.split(' ') if x]
        self.time = float(time)
        self.code = int(code)
        self.label = label

    def __repr__(self):
        m = f'(Label_line) time: {self.time}'
        m += f', code: {self.code}'
        m += f', label: {self.label}'
        return m
            

