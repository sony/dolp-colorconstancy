"""
polarAWB_noGT.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import json
from pathlib import Path
import shutil

import numpy as np

from myutils.imageutils import MAX_16BIT, my_read_image, my_write_image, rgb_to_srgb
import myutils.polarutils as plutil
import myutils.weighturils as weutil
import myutils.wbutils as wbutil


if __name__ == "__main__":
    params = json.load(open("parameters.json", "r"))

    input_path = Path("images").joinpath(params["input_folder"])

    result_path = Path("results").joinpath(input_path.name)
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copy("parameters.json", result_path)

    imean_paths = input_path.glob("*_imean.png")

    for imean_path in imean_paths:
        i000_path = Path(str(imean_path).replace("imean", "i000"))
        i045_path = Path(str(imean_path).replace("imean", "i045"))
        i090_path = Path(str(imean_path).replace("imean", "i090"))
        i135_path = Path(str(imean_path).replace("imean", "i135"))

        imean = my_read_image(imean_path) / MAX_16BIT
        i000 = my_read_image(i000_path) / MAX_16BIT
        i045 = my_read_image(i045_path) / MAX_16BIT
        i090 = my_read_image(i090_path) / MAX_16BIT
        i135 = my_read_image(i135_path) / MAX_16BIT

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

        # Save White-balanced Images.
        imean[..., 0] /= illum_est[..., 0]
        imean[..., 2] /= illum_est[..., 2]
        imean = np.clip(imean, 0, 1)

        imean_sRGB = rgb_to_srgb(imean)
        imean_sRGB = np.clip(imean_sRGB, 0, 1)

        scene_name = str(imean_path.name).replace("_imean", "")
        my_write_image(result_path.joinpath("{}.png".format(scene_name)), imean * MAX_16BIT)
        my_write_image(result_path.joinpath("{}_sRGB.png".format(scene_name)), imean_sRGB * MAX_16BIT)
