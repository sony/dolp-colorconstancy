"""
weightutils.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""
import numpy as np
from .polarutils import AOLPMAX_DEG


def sigmoid(xs, alpha, center):
    """
    Return the sigmoid of an array, element-wise.
    Args:
        xs: ndarray
            The values whose sigmoid is required.
        alpha: float
            A value that defines the gradient of the sigmoid.
        center: float
            A value that defines the shift along x-axis.
    Returns: ndarray
        1 / (1 + np.exp(-alpha * (xs - center)))
    --------
    Raises:
        When alpha is negative.
    """
    if alpha <= 0:
        raise ValueError("Alpha must be larger than 0.")

    sigmoid_range = 34.538776394910684

    x = np.clip(-alpha * (xs - center), -sigmoid_range, sigmoid_range)
    return 1 / (1 + np.exp(x))


def valid_weight(img, th):
    """
    Return an array for excluding too-bright or too-dark areas in an image.
    Args:
        img: ndarray
            The image you want to exclude too-bright or too-dark areas.
            Must be normalized into (0,1). Must contain RGB channels.
        th: float
            The threshold defining too-bright or too-dark areas.
            Must be between (0,1).
    Returns: ndarray
        img[i, j, c] > th ---> return[i, j, c] = 0
        img[i, j, c] < th ---> return[i, j, c] = 0
        th <= img[i, j, c] <= th ---> return[i, j, c] = 1
    --------
    Raises:
        When img doesn't contain RGB channels.
        When img aren't normalized into (0,1).
        When th aren't between 0 and 1.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise TypeError("Your input image doesn't contain color channels.")
    if np.max(img) > 1 or np.min(img) < 0:
        raise ValueError("Input image must be normalized into (0, 1).")
    if th > 1 or th < 0:
        raise ValueError("Threshold must be between 0 and 1. Your input is {}".format(th))

    min_th = th
    max_th = 1 - th

    mask_r = (img[..., 0] > min_th) * (img[..., 0] < max_th)
    mask_g = (img[..., 1] > min_th) * (img[..., 1] < max_th)
    mask_b = (img[..., 2] > min_th) * (img[..., 2] < max_th)

    return (mask_r * mask_g * mask_b).astype(np.float32)


def valid_weight_fourPolar(i000, i045, i090, i135, th):
    i000_mask = valid_weight(i000, th=th)
    i045_mask = valid_weight(i045, th=th)
    i090_mask = valid_weight(i090, th=th)
    i135_mask = valid_weight(i135, th=th)

    return i000_mask * i045_mask * i090_mask * i135_mask


def calc_rg_bg_diff(img, normalize):
    """
    Return the differences between (r, g) and (b, g) channels normalized by rgb-average.
    Args:
        img: ndarray
            An image whose differences between (r, g) and (b, g) channels are required.
        normalize: bool
            Whether or not normalize the computed differences by RGB average.
    Returns: ndarray
        |r - g|/(r + g + b)/3, |b - g|/(r + g + b)/3.
    --------
    Raises:
        When img doesn't contain RGB channels.
        When normalize isn't a bool.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise TypeError("Your input image doesn't contain color channels.")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be bool.")

    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    diff_rg = np.abs(r - g)
    diff_bg = np.abs(b - g)

    if normalize:
        rgb_mean = (r + g + b) / 3.
        rgb_mean = np.clip(rgb_mean, 1e-06, None)

        diff_rg /= rgb_mean
        diff_bg /= rgb_mean

    return diff_rg, diff_bg


def rg_bg_sigmoid_weight_achromatic(img, alpha, center, normalize):
    """
    This function returns larger values when an image's RGB differences are smaller.
    Args:
        img: ndarray
            The image whose weight you want to compute.
        alpha: float
            See the description of sigmoid().
        center: float
            See the description of sigmoid().
        normalize: bool
            See the description of calc_rg_bg_diff().
    Returns: ndarray
        Computed pixel-wise weights.
    """
    diff_rg, diff_bg = calc_rg_bg_diff(img, normalize=normalize)

    weight_rg = np.clip(sigmoid(diff_rg, alpha=alpha, center=center), 0, 1)
    weight_bg = np.clip(sigmoid(diff_bg, alpha=alpha, center=center), 0, 1)

    return (1 - weight_rg) * (1 - weight_bg)


def rg_bg_sigmoid_weight_chromatic(img, alpha, center, normalize):
    """
    This function returns larger values when an image's RGB differences are larger.
    Args:
        img: ndarray
            The image whose weight you want to compute.
        alpha: float
            See the description of sigmoid().
        center: float
            See the description of sigmoid().
        normalize: bool
            See the description of calc_rg_bg_diff().
    Returns: ndarray
        Computed pixel-wise weights.
    """
    diff_rg, diff_bg = calc_rg_bg_diff(img, normalize=normalize)

    weight_rg = np.clip(sigmoid(diff_rg, alpha=alpha, center=center), 0, 1)
    weight_bg = np.clip(sigmoid(diff_bg, alpha=alpha, center=center), 0, 1)

    return weight_rg * weight_bg


def calc_rg_bg_diff_phase(phase):
    """
    Return the polarization phase differences considering 180deg ambiguity.
    Args:
        phase: ndarray
            polarization phase.
    Returns: ndarray
        Polarization phase differences considering 180deg ambiguity.
    """
    if phase.ndim != 3 or phase.shape[2] != 3:
        raise TypeError("Your input phase doesn't contain color channels.")
    if np.min(phase) < 0 or np.max(phase) > AOLPMAX_DEG:
        raise ValueError("Your phase has wrong range.")

    flip_deg = 90.

    r, g, b = phase[..., 0], phase[..., 1], phase[..., 2]

    diff_rg = np.abs(r - g)
    diff_bg = np.abs(b - g)

    flip_mask_rg = (diff_rg > flip_deg)
    flip_mask_bg = (diff_bg > flip_deg)

    diff_rg[flip_mask_rg] -= AOLPMAX_DEG
    diff_bg[flip_mask_bg] -= AOLPMAX_DEG

    diff_rg[flip_mask_rg] *= -1
    diff_bg[flip_mask_bg] *= -1

    diff_rg = np.clip(diff_rg, 0, AOLPMAX_DEG)
    diff_bg = np.clip(diff_bg, 0, AOLPMAX_DEG)

    return diff_rg, diff_bg


def rg_bg_sigmoid_weight_achromatic_phase(phase, alpha, center):
    """
    This function returns larger values when an image's RGB differences are smaller.
    Args:
        phase: ndarray
            The polarization phase whose weight you want to compute.
        alpha: float
            See the description of sigmoid().
        center: float
            See the description of sigmoid().
    Returns: ndarray
        Computed pixel-wise weights.
    """
    diff_rg, diff_bg = calc_rg_bg_diff_phase(phase=phase)

    weight_rg = np.clip(sigmoid(diff_rg, alpha=alpha, center=center), 0, 1)
    weight_bg = np.clip(sigmoid(diff_bg, alpha=alpha, center=center), 0, 1)

    return (1 - weight_rg) * (1 - weight_bg)
