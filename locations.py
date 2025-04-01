from pathlib import Path

base = Path(__file__).resolve().parent.parent

original = base / 'original'
gated = base / 'gated'
labels = base / 'labels'
responses = base / 'responses'

matrices = responses / 'matrices'
rawdata = responses / 'rawdata'
