"""
polarAWB.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import json
from pathlib import Path
import shutil

import numpy as np

from myutils.imageutils import MAX_16BIT, my_read_image, my_write_image
from myutils.datautils import macbeth_position_txt_parse, compute_gt_illum, calc_ang_error
import myutils.polarutils as plutil
import myutils.weighturils as weutil
import myutils.wbutils as wbutil


if __name__ == "__main__":
    params = json.load(open("parameters.json", "r"))

    input_path = Path("images").joinpath(params["input_folder"])

    result_path = Path("results").joinpath(input_path.name)
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copy("parameters.json", result_path)

    with open(input_path.joinpath("macbeth_position.txt"), "r") as f:
        lines = f.readlines()

    for line in lines:
        scene_name, x, y, w, h = macbeth_position_txt_parse(line)

        imean_path = input_path.joinpath("{}_imean.png".format(scene_name))
        i000_path = input_path.joinpath("{}_i000.png".format(scene_name))
        i045_path = input_path.joinpath("{}_i045.png".format(scene_name))
        i090_path = input_path.joinpath("{}_i090.png".format(scene_name))
        i135_path = input_path.joinpath("{}_i135.png".format(scene_name))
        macbeth_path = input_path.joinpath("{}_macbeth.png".format(scene_name))

        imean = my_read_image(imean_path) / MAX_16BIT
        i000 = my_read_image(i000_path) / MAX_16BIT
        i045 = my_read_image(i045_path) / MAX_16BIT
        i090 = my_read_image(i090_path) / MAX_16BIT
        i135 = my_read_image(i135_path) / MAX_16BIT
        macbeth = my_read_image(macbeth_path)

        s0, s1, s2 = plutil.calc_s0s1s2_from_fourPolar(i000, i045, i090, i135)
        dolp = plutil.calc_dolp_from_s0s1s2(s0, s1, s2)
        aolp = plutil.calc_aolp_from_s1s2(s1, s2)

        # Weights
        w_valid = weutil.valid_weight_fourPolar(i000, i045, i090, i135, th=params["valid_th"])
        w_dolp = weutil.sigmoid(
            np.mean(dolp, axis=2), alpha=params["w_dolp_a"], center=params["w_dolp_b"])
        w_dolp_ach = weutil.rg_bg_sigmoid_weight_achromatic(
            dolp, alpha=params["w_dolp_ach_a"], center=params["w_dolp_ach_b"], normalize=True)
        w_aolp_ach = weutil.rg_bg_sigmoid_weight_achromatic_phase(
            aolp, alpha=params["w_aolp_ach_a"], center=params["w_aolp_ach_b"])

        w_dolp_ch = weutil.rg_bg_sigmoid_weight_chromatic(
            dolp, alpha=params["w_dolp_ch_a"], center=params["w_dolp_ch_b"], normalize=True)
        w_aolp_ch = weutil.rg_bg_sigmoid_weight_achromatic_phase(
            aolp, alpha=params["w_aolp_ch_a"], center=params["w_aolp_ch_b"])

        weight_achromatic = w_valid * w_dolp * w_dolp_ach * w_aolp_ach
        weight_chromatic = w_valid * w_dolp * w_dolp_ch * w_aolp_ch

        # WB.
        illum_est = wbutil.polarAWB(dolp, imean, weight_achromatic, weight_chromatic, params["alpha"])

        # Compute Error.
        illum_gt = compute_gt_illum(macbeth, x, y, w, h)
        err_deg = calc_ang_error(illum_est, illum_gt)
        with open(result_path.joinpath("error.txt"), "a") as f2:
            f2.write("{}'s Error: {:.3f}\n".format(scene_name, err_deg))

        # Save White-balanced Images.
        macbeth_wb = np.copy(imean)
        polar_wb = np.copy(imean)

        polar_wb[..., 0] /= illum_est[..., 0]
        polar_wb[..., 2] /= illum_est[..., 2]
        polar_wb = np.clip(polar_wb, 0, 1) * MAX_16BIT
        my_write_image(result_path.joinpath("{}_PolarWB.png".format(scene_name)), polar_wb)

        r_gain = illum_gt[1] / illum_gt[0]
        b_gain = illum_gt[1] / illum_gt[2]
        macbeth_wb[..., 0] *= r_gain
        macbeth_wb[..., 2] *= b_gain
        macbeth_wb = np.clip(macbeth_wb, 0, 1) * MAX_16BIT
        my_write_image(result_path.joinpath("{}_MacbethWB.png".format(scene_name)), macbeth_wb)
