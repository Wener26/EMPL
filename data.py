import mne
import numpy as np
from moabb.datasets import BNCI2014_001

mne.set_log_level("ERROR")

WINDOW = (1.5, 6.0)
CUE = 2.0
SFREQ = 250


def load_subject(subject):
    raws = BNCI2014_001().get_data(subjects=[subject])[subject]
    out = {}
    for session in sorted(raws):
        xs, ys = [], []
        for run in sorted(raws[session]):
            raw = raws[session][run].copy().pick(picks="eeg")
            events, event_id = mne.events_from_annotations(raw)
            keep = {k: v for k, v in event_id.items() if k in ("left_hand", "right_hand", "feet", "tongue")}
            epochs = mne.Epochs(raw, events, keep, tmin=WINDOW[0] - CUE, tmax=WINDOW[1] - CUE - 1.0 / SFREQ,
                                baseline=None, preload=True)
            labels = {v: i for i, v in enumerate(sorted(keep.values()))}
            xs.append(epochs.get_data(copy=True) * 1e6)
            ys.append(np.array([labels[e] for e in epochs.events[:, 2]]))
        out[session] = (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64))
    return out


def z_scale(train, *others):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return [(a - mean) / std for a in (train,) + others]


def within_subject(subject):
    sessions = load_subject(subject)
    (x_tr, y_tr), (x_te, y_te) = [sessions[k] for k in sorted(sessions)]
    x_tr, x_te = z_scale(x_tr, x_te)
    return x_tr, y_tr, x_te, y_te


def leave_one_subject_out(target, subjects=range(1, 10)):
    x_tr, y_tr, x_va, y_va = [], [], [], []
    for s in subjects:
        sessions = load_subject(s)
        (x0, y0), (x1, y1) = [sessions[k] for k in sorted(sessions)]
        if s == target:
            x_te, y_te = x1, y1
        else:
            x_tr.append(x0); y_tr.append(y0); x_va.append(x1); y_va.append(y1)
    x_tr, y_tr = np.concatenate(x_tr), np.concatenate(y_tr)
    x_va, y_va = np.concatenate(x_va), np.concatenate(y_va)
    x_tr, x_va, x_te = z_scale(x_tr, x_va, x_te)
    return x_tr, y_tr, x_va, y_va, x_te, y_te
