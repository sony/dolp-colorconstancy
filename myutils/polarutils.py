"""
polarutils.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""
import numpy as np

AOLPMAX_DEG = 180.


def calc_s0s1s2_from_fourPolar(i000, i045, i090, i135):
    """
    Return s0, s1, and s2 from four-directional polarization images.
    Args:
        i000: ndarray
        i045: ndarray
        i090: ndarray
        i135: ndarray
    Returns: ndarray
    """
    s0 = (i000 + i045 + i090 + i135) / 2.
    s1 = i000 - i090
    s2 = i045 - i135

    return s0, s1, s2


def calc_dolp_from_s0s1s2(s0, s1, s2):
    """ 
    Return DoLP (Degree of Linear Polarization) from s0, s1, and s2.
    Args:
        s0: ndarray
        s1: ndarray
        s2: ndarray
    Returns: ndarray
    """
    s0_ = np.clip(s0, 1e-06, None)
    dolp = np.sqrt(s1 ** 2 + s2 ** 2) / s0_
    dolp = np.clip(dolp, 0, 1)

    return dolp


def calc_aolp_from_s1s2(s1, s2):
    """ 
    Return AoLP (Angle of Linear Polarization) from s1 and s2.
    Args:
        s1: ndarray
        s2: ndarray
    Returns: ndarray
        AoLP (Angle of Linear Polarization) (degree), [0, 180).
    """
    s1_ = np.copy(s1)
    s1_[s1 == 0] += 1e-06
    phase = np.rad2deg(np.arctan2(s2, s1_))
    mask = (phase < 0)
    phase[mask] = phase[mask] + AOLPMAX_DEG * 2
    phase = phase / 2.
    phase = np.clip(phase, 0, AOLPMAX_DEG)

    return phase
