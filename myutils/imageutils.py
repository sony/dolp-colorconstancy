"""
imageutils.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""
import pathlib
import numpy as np
import cv2

MAX_8BIT = 255.
MAX_16BIT = 65535.


def my_read_image(img_path):
    """
    Return a loaded image according to the input path.
    Note that the color channels are sorted as RGB order.

    Args:
        img_path: pathlib.Path
            File path of an image you want to load.
    Returns: ndarray
        Loaded image sorted as RGB order. Astype is numpy.float32.
    -------
    Raises:
        TypeError: When your input path is not the pathlib.Path object.
        FileNotFoundError: When your input path does not include any images.
    """
    if not isinstance(img_path, pathlib.Path):
        raise TypeError("Input type must be pathlib.Path object.")
    else:
        img = cv2.imread(str(img_path), -1)
        if img is None:
            raise FileNotFoundError("{} not found.".format(str(img_path)))
        else:
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img


def my_write_image(img_path, img):
    """
    This function saves the input image as 16bit.png according to the input path.
    Note that the color channels muat be sorted as RGB order.

    Args:
        img_path: pathlib.Path
            File path where you want to save the image.
        img: numpy.ndarray
            Image you want to save. Must be sorted as RGB order.
    -------
    Raises:
        TypeError: When your input path is not the pathlib.Path object.
        ValueError: When your input array includes values which is smaller than 0 or larger than 65535.
    """
    if not isinstance(img_path, pathlib.Path):
        raise TypeError("Input type must be pathlib.Path object.")
    else:
        if np.min(img) < 0 or np.max(img) > MAX_16BIT:
            raise ValueError("Your input array's range doesn't match to 16bit.")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img.astype(np.uint16))


def rgb_to_srgb(img):
    """
    Convert linear RGB images to sRGB images.
    Args:
        img: ndarray
            The linear RGB image whose sRGB you want.
    Returns: ndarray
    """
    if np.max(img) > 1:
        raise ValueError("Input image must be normalized into (0, 1).")

    a = 0.055

    high_mask = (img > 0.0031308)

    low_c = 12.92 * img
    high_c = (1 + a) * np.power(img, 1.0 / 2.4) - a

    low_c[high_mask] = high_c[high_mask]

    return high_c
