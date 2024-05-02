import numpy as np
from scipy.signal import find_peaks
from mhealth_activity import Recording

def centered_moving_average(data, window_size):
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def predict_stepcount_acc(ax, ay, az):
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    cen_mov_avg = centered_moving_average(acc_mag, 62)
    peaks, _ = find_peaks(cen_mov_avg, height=1.1, distance=80)
    return len(peaks)

def predict_stepcount(recording:Recording):
    ax = recording.data['ax'].values
    ay = recording.data['ay'].values
    az = recording.data['az'].values
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    cen_mov_avg = centered_moving_average(acc_mag, 62)
    peaks, _ = find_peaks(cen_mov_avg, height=1.1, distance=80)
    return len(peaks)