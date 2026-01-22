from collections import Counter
import data
import locations
import math
from pathlib import Path
from progressbar import progressbar
from textgrid import TextGrid, IntervalTier, Interval


def make_textgrids_from_labels(labels = None, overwrite=False):
    if labels is None:
        p = data.Participants()
        labels = p.labels
    tgs, errors = [], []
    for label in progressbar(labels.labels):
        try:tg = label_to_textgrid(label, overwrite=overwrite)
        except Exception as e: errors.append((label, e))
    print(f'Created {len(tgs)} TextGrids with {len(errors)} errors.')
    return tgs, errors

def label_to_textgrid(label, overwrite=False):
    if not validate_label(label): return None
    start = label.timestamp_dict['start_time']
    end = label.timestamp_dict['end_time']
    gate_intervals = [gate_to_interval(label, i) for i in range(1, 7)]
    gate_tier = make_tier('gates', gate_intervals, start=start, end=end)
    phoneme_intervals = [phoneme_to_interval(label, i) for i in range(1, 3)]
    phoneme_tier_1 = make_tier('phoneme 1', [phoneme_intervals[0]], start=start, end=end)
    phoneme_tier_2 = make_tier('phoneme 2', [phoneme_intervals[1]], start=start, end=end)
    rtier_p1 = response_to_tier(label, 1) 
    rtier_p2 = response_to_tier(label, 2) 
    filename = Path(label.all_gated_audio_filenames[0]).stem[:-1] 
    filename = locations.textgrid_directory / (filename + '.TextGrid')
    tiers = [gate_tier, phoneme_tier_1,phoneme_tier_2, rtier_p1, rtier_p2]
    tg = make_textgrid(tiers, filename = filename, overwrite=overwrite)
    return tg

def validate_label(label):
    for gate in range(1, 7):
        if f'gate_{gate}_timestamp' not in label.timestamp_dict:
            return False
    return True

def phoneme_to_interval(label, phoneme_number):
    time_dict = label.timestamp_dict
    start = time_dict[f'phoneme_{phoneme_number}_start_time']
    end = time_dict[f'phoneme_{phoneme_number}_end_time']
    phoneme = getattr(label, f'phoneme{phoneme_number}')
    return make_interval(start, end, phoneme)

def gate_to_interval(label, gate_number):
    time_dict = label.timestamp_dict
    if gate_number == 1:
        start = time_dict['start_time']
    else:
        start = time_dict[f'gate_{gate_number - 1}_timestamp']
    end = time_dict[f'gate_{gate_number}_timestamp']
    return make_interval(start, end, f'{gate_number}')

def response_to_tier(label, phoneme_number):
    time_dict = label.timestamp_dict
    start = time_dict['start_time']
    end = time_dict['end_time']
    responses= label.responses
    pd = responses_to_confusion_dict(responses, phoneme_number)
    label = counter_to_pretty_string(pd)
    interval =  make_interval(start, end, label)
    gate = 1 if phoneme_number == 1 else 4
    tier = make_tier(f'resp p-{phoneme_number} g-{gate}', [interval])
    return tier

def responses_to_confusion_dict(responses, phoneme_number):
    gate = 1 if phoneme_number == 1 else 4
    pn = phoneme_number
    p = [getattr(x,f'response_phoneme{pn}') for x in responses if x.gate==gate]
    return Counter(p)

def make_interval(start, end, label):
    return Interval(minTime=float(start), maxTime=float(end), mark=str(label))

def make_tier(name, intervals, start = None, end = None):
    start = intervals[0].minTime if start is None else start
    end = intervals[-1].maxTime if end is None else end
    tier = IntervalTier(name=name, minTime=float(start), maxTime=float(end)) 
    for itv in intervals:
        tier.addInterval(itv)
    # set tier end if not given
    if end is None and intervals:
        tier.maxTime = max(i.maxTime for i in intervals)
    return tier

def make_textgrid(tiers, start=None, end=None, filename = None, 
    overwrite = False):
    if filename and not overwrite:
        p = Path(filename)
        if p.exists():
            raise FileExistsError(f"File '{filename}' already exists.")
    start = find_lowest(tiers) if start is None else start
    end = find_highest(tiers) if end is None else end
    tg = TextGrid(minTime=float(start), maxTime=float(end))
    for tier in tiers:
        tg.append(tier)
        tg.maxTime = max(tg.maxTime, tier.maxTime)
    if end is not None:
        tg.maxTime = float(end)
        for tier in tg:
            tier.maxTime = float(end)
    if filename: save_textgrid(tg, filename, overwrite=overwrite)
    return tg

def save_textgrid(tg, filename, overwrite=False):
    p = Path(filename)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File '{filename}' already exists.")
    tg.write(filename)

def find_lowest(items):
    lowest = math.inf
    for item in items:
        if item.minTime < lowest:
            lowest = item.minTime
    return lowest

def find_highest(items):
    highest = -math.inf
    for item in items:
        if item.maxTime > highest:
            highest = item.maxTime
    return highest

def counter_to_pretty_string(counter, sep='   |   ', sort=True):
    items = counter.items()
    if sort:
        items = sorted(items, key=lambda x: (-x[1], x[0]))
    return sep.join(f'{k} {v}' for k, v in items)
