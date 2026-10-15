"""Binary stress detector. Thresholds use only dates before the decision window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from study.protocol import DETECTOR_FEATURE_MONTHS, DETECTOR_MIN_CALIBRATION, DETECTOR_QUANTILE


def _features(window: pd.DataFrame) -> tuple[float, float]:
    vol = float(window.std(ddof=1).mean() * np.sqrt(12.0))
    corr = window.corr().to_numpy()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return vol, float(np.mean(np.abs(corr[mask])))


def detect_stress(train: pd.DataFrame) -> str:
    """Return 'stress' or 'calm'. Default is calm when calibration is short.

    The decision feature is the last 12 months. Quantiles are computed from
    earlier 12-month windows that end at or before the start of that window.
    """
    width = DETECTOR_FEATURE_MONTHS
    t = len(train)
    last_cal_end = t - width
    if last_cal_end < width + DETECTOR_MIN_CALIBRATION - 1:
        return "calm"
    vols = []
    corrs = []
    for end in range(width, last_cal_end + 1):
        v, c = _features(train.iloc[end - width : end])
        vols.append(v)
        corrs.append(c)
    if len(vols) < DETECTOR_MIN_CALIBRATION:
        return "calm"
    v_thr = float(np.quantile(vols, DETECTOR_QUANTILE))
    c_thr = float(np.quantile(corrs, DETECTOR_QUANTILE))
    v_now, c_now = _features(train.iloc[-width:])
    if v_now > v_thr or c_now > c_thr:
        return "stress"
    return "calm"
