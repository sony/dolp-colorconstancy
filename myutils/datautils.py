"""
datautils.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np


def macbeth_position_txt_parse(line):
    scene_name = line.split(" ")[0]
    x = int(line.split(" ")[1].replace("x:", "").replace(",", ""))
    y = int(line.split(" ")[2].replace("y:", "").replace(",", ""))
    w = int(line.split(" ")[3].replace("w:", "").replace(",", ""))
    h = int(line.split(" ")[4].replace("h:", "").replace(",", ""))
    return scene_name, x, y, w, h


def compute_gt_illum(macbeth, x, y, w, h):
    macbeth_crop = macbeth[y: y + h, x: x + w, :]
    r_val = np.mean(macbeth_crop[..., 0])
    g_val = np.mean(macbeth_crop[..., 1])
    b_val = np.mean(macbeth_crop[..., 2])

    return np.array([r_val, g_val, b_val])


def calc_ang_error(a, b):
    dot_ab = np.sum(a * b) / np.sqrt(np.sum(a * a)) / np.sqrt(np.sum(b * b))

    return np.rad2deg(np.arccos(dot_ab))


def scene_err_list(path, sky_names):
    """
    Args:
        path: pathlib.Path
        sky_names: list
    Returns: list
    """
    with open(path.joinpath("error.txt"), "r")as f:
        lines = f.readlines()

    scene_names, errs, errs_noSky = [], [], []
    for line in lines:
        scene_name = line.split("'")[0]
        err = float(line.split(":")[1].replace(" ", ""))

        scene_names.append(scene_name)
        errs.append(err)

        if scene_name in sky_names:
            pass
        else:
            errs_noSky.append(err)

    return scene_names, errs, errs_noSky


def scene_name_list_check(arg1, *args):
    scene_name_list_is_same = True

    scene_num = len(arg1)
    for arg in args:
        if scene_num != len(arg):
            return False
        else:
            for i in range(scene_num):
                if arg1[i] != arg[i]:
                    return False
                else:
                    pass

    return scene_name_list_is_same


def calc_various_metrics(err_array, method_name):
    """
    Return mean, median, trimean, mean of good/bad 25%.
    Args:
        err_array: ndarray
            Err arrays whose various metrics you want.
        method_name: str
    Returns: float
    """
    mean = np.mean(err_array)
    median = np.median(err_array)

    err_array_sort = np.sort(err_array)

    # Compute trimean
    q1, q3 = np.percentile(err_array_sort, q=[25, 75])
    tri_mean = (q1 + 2 * median + q3) / 4

    # Compute good 25% and Bad 25%.
    scene_num = err_array_sort.shape[0]
    scene_num_q = int(scene_num * 0.25)

    mean_goodq = np.mean(err_array_sort[:scene_num_q])
    mean_badq = np.mean(err_array_sort[scene_num-scene_num_q:])

    print("{:<15} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
        method_name, mean, median, tri_mean, mean_goodq, mean_badq))
