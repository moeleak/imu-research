from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .common import SequenceData


def evaluate_baseline(sequence_data: SequenceData) -> tuple[np.ndarray, float]:
    acc_mag_f = savgol_filter(sequence_data.acc_mag, 11, 3)
    peaks, _ = find_peaks(acc_mag_f, height=0.6, distance=14)
    yaws_smooth = np.arctan2(
        savgol_filter(np.sin(sequence_data.yaws), 31, 3),
        savgol_filter(np.cos(sequence_data.yaws), 31, 3),
    )

    pred_pos = np.zeros_like(sequence_data.gt_pos)
    curr_x = 0.0
    curr_y = 0.0
    step_ptr = 0
    for index in range(len(sequence_data.gt_pos)):
        if step_ptr < len(peaks) and index == peaks[step_ptr]:
            curr_x += 0.7 * np.cos(yaws_smooth[index])
            curr_y += 0.7 * np.sin(yaws_smooth[index])
            step_ptr += 1
        pred_pos[index] = [curr_x, curr_y]

    best_rmse = float("inf")
    best_pred = pred_pos
    for angle in np.linspace(0, 2 * np.pi, 360):
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        candidate = (rotation @ pred_pos.T).T
        rmse = np.sqrt(np.mean(np.sum((candidate - sequence_data.gt_pos) ** 2, axis=1)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_pred = candidate
    return best_pred, best_rmse
