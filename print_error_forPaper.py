"""
print_error_forPaper.py
Copyright (c) 2022 Sony Group Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path
import numpy as np
from myutils.datautils import scene_err_list, calc_various_metrics

blueSky_names = ["scene014", "scene017", "scene024"]

if __name__ == "__main__":
    result_path = Path("results/")

    scene_names, errs, errNoSky = scene_err_list(result_path, blueSky_names)

    calc_various_metrics(np.array(errs), "Ours w/ blue-sky scenes")
    calc_various_metrics(np.array(errNoSky), "Ours w/o blue-sky scenes")
