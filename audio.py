import librosa
import math

def load_audio(filename, start = 0.0, end=None, output_sr = 16000):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = output_sr, offset=start, 
        duration=duration)
	return audio

class Targets:
    def __init__(self, targets = [], contexts = []):
        self.targets = list(set(targets))
        self.contexts = contexts 
        self.context_dict = {c.identifier: c for c in contexts}
        self._connect_targets_to_contexts()

    def _connect_targets_to_contexts(self):
        self.errors= []
        for target in self.targets:
            if target.context_id not in self.context_dict:
                self.errors.append(target)
            context = self.context_dict[target.context_id]
            context.add_target(target)
    

class Target:
    def __init__(self, label, start_time, end_time, duration, context_id,
        context = None, **kwargs):
        self.label = label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.context_id = context_id
        self.target_id = f'{self.label}_{self.start_time}_{self.end_time}'
        self.target_id += f'__{self.context_id}'
        self.context = context
        self.info = {}
        for k, v in kwargs.items():
            self.info[k] = v

    def __repr__(self):
        m = (f'Target({self.label}, {self.start_time} - {self.end_time}, '
             f'context_id={self.context_id})')
        return m

    def __eq__(self, other):
        return (self.label == other.label and
                math.isclose( self.start_time, other.start_time) and
                math.isclose(self.end_time, other.end_time) and
                self.context_id == other.context_id)

    def __hash__(self):
        return hash((self.label, self.start_time, self.end_time, 
            self.context_id))
        
    def set_context(self, context):
        context.add_target(self)

    @property
    def audio_array(self):
        if self.context is None:
            raise ValueError(f'Target {self} has no context set.')
        return self.context.audio_array


class Context:
    def __init__(self, audio_filename, start_time, end_time, identifier,
        **kwargs):
        self.audio_filename = audio_filename
        self.start_time = start_time
        self.end_time = end_time
        self.identifier = identifier
        self.info = {}
        for k, v in kwargs.items():
            self.info[k] = v
        self.targets = []

    def __repr__(self):
        m = f'Context({self.identifier}, {self.start_time} - {self.end_time})'
        return m

    def __hash__(self):
        return hash((self.start_time, self.end_time, 
            self.context_id))

    def add_target(self, target):
        if target in self.targets:
            m = f'Target {target} already exists in context {self}.'
            m += ' Not adding it again.'
            print(m)
            return
        if target.context_id != self.identifier:
            m = f'Target {target} has context_id {target.context_id}, '
            m += f'but context {self} has identifier {self.identifier}.'
            m += ' Not adding target.'
            print(m)
            return
        if target.context is not None and target.context != self:
            m = f'Target {target} already has context {target.context}, '
            m += f'but trying to add to context {self}.'
            raise ValueError(m)
        self.targets.append(target)
        target.context = self

    @property
    def audio_array(self) :
        return load_audio(self.audio_filename, self.start_time, 
            self.end_time)
