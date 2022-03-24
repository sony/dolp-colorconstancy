"""
wbutils.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np


def polarAWB_achromatic(imean, weight):
    pixels_r = np.copy(imean[..., 0])
    pixels_g = np.copy(imean[..., 1])
    pixels_b = np.copy(imean[..., 2])

    pixels_g = np.clip(pixels_g, 1e-06, None)

    illum_r = np.sum(pixels_r * weight / pixels_g) / np.sum(weight)
    illum_b = np.sum(pixels_b * weight / pixels_g) / np.sum(weight)
    return np.array([illum_r, 1, illum_b])


def polarAWB_chromatic(dolp, imean, weight):
    weight_zero_mask = (weight > 0)

    weight_masked = weight[weight_zero_mask]
    dop_valid_R = dolp[..., 0][weight_zero_mask]
    dop_valid_G = dolp[..., 1][weight_zero_mask]
    dop_valid_B = dolp[..., 2][weight_zero_mask]
    imean_valid_R = imean[..., 0][weight_zero_mask]
    imean_valid_G = imean[..., 1][weight_zero_mask]
    imean_valid_B = imean[..., 2][weight_zero_mask]

    ys = (dop_valid_R - dop_valid_B) * imean_valid_G * weight_masked / np.sum(weight_masked)
    A = np.zeros((np.cumsum(weight_zero_mask)[-1], 2), dtype=np.float32)
    A[:, 0] = (dop_valid_G - dop_valid_B) * imean_valid_R * weight_masked / np.sum(weight_masked)
    A[:, 1] = (dop_valid_R - dop_valid_G) * imean_valid_B * weight_masked / np.sum(weight_masked)

    A_inv = np.linalg.pinv(A)
    r_gain, b_gain = A_inv.dot(ys)

    return np.array([1 / r_gain, 1, 1 / b_gain])


def polarAWB(dolp, imean, weight_ach, weight_ch, achromatic_ratio_default):
    if np.sum(weight_ach) > 0:
        illum_achromatic = polarAWB_achromatic(imean, weight_ach)
        achromatic_ratio = achromatic_ratio_default
    else:
        illum_achromatic = np.array([1, 1, 1])
        achromatic_ratio = 0

    if np.sum(weight_ch) > 0:
        illum_chromatic = polarAWB_chromatic(dolp, imean, weight_ch)
        chromatic_ratio = 1 - achromatic_ratio
    else:
        illum_chromatic = np.array([1, 1, 1])
        chromatic_ratio = 0

    if achromatic_ratio + chromatic_ratio == 0:
        print('Your image does not have available pixels.')
        return np.array([1, 1, 1])

    return achromatic_ratio * illum_achromatic + chromatic_ratio * illum_chromatic
