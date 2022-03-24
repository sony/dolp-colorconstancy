# Degree-of-linear-polarization-based Color Constancy (CVPR 2022)
Taishi Ono, Yuhi Kondo, Legong Sun, Teppei Kurita, and Yusuke Moriuchi

Sony Group Corporation, Tokyo, Japan

[paper](hogehogehoge) |
[data](https://sonyjpn.sharepoint.com/sites/S168-DOLPCC)

## Introduction
This repository is the implementation of "Degree-of-linear-polarization-based Color Constancy," presented in CVPR 2022.

This paper proves that the degree of linear polarization (DoLP) dramatically solves the color constancy problem.
The main idea is quite simple: **the DoLP values of achromatic pixels are also achromatic regardless of the illumination colors.**
Additionally, we widely analyzed how to estimate illuminations from chromatic pixels and examined a solution for multi-illumination scenes.

![macbeth](/uploads/ec970edd037189041fbecec9853bbe35/macbeth.gif)

Raw-RGB(top) and DoLP(bottom).

## Dependency
We use the following version of libraries. Subtle version differences are acceptable.
- Python 3.7.3
- opencv-python 4.1.2.30
- numpy 1.17.4

## Usage
1. Set the parameters in `parameters.json`. 
2. Run `python polarAWB.py` or `python polarAWB_noGT.py` according to your folder includes gt illuminations or not.

## Reproduce the results of our paper
1. Copy our evaluation data from [data](https://sonyjpn.sharepoint.com/sites/S168-DOLPCC) to `images/`.
2. Set the parameters in `parameters.json` according to our paper. The preset values are the same as the parameters used in our paper.
3. Run `python polarAWB.py`.
4. Set the file path in `print_error_forPaper.py` and run `python print_error_forPaper.py`.

## Folder structure
Please see also `images/sample_images/`.
```
./Sony_PolarizationAWB_CVPR2022/
    |- polarAWB.py
    |- polarAWB_noGT.py
    |- parameters.json
    |- myutils/
    |- images/
        |- folder_1/    # w/ ground-truth illuminations
            |- macbeth_position.txt
            |- scene000_i000.png
            |- scene000_i045.png
            |- scene000_i090.png
            |- scene000_i135.png
            |- scene000_imean.png
            |- scene000_mecbeth.png
            |- ...
        |- folder_2/    # w/o ground-truth illuminations
            |- scene000_i000.png
            |- scene000_i045.png
            |- scene000_i090.png
            |- scene000_i135.png
            |- scene000_imean.png
            |- ...
    |- results/
```
`macbeth_position.txt` must include scene names and their corresponding positions of color charts as follows:
```
scene000 x:123, y:456, w:123, h:456
scene001 x:111, y:222, w:333, h:444
...
``` 

`parameters.json` includes the following parameters:
```
"input_folder"      : the name of your folder.
"valid_th"          : percentile for excluding the high and low intensity.
"w_dolp_a"          : "a" in Equation 12.
"w_dolp_b"          : "b" in Equation 12.
"w_dolp_ach_a"      : "a" in Equation 9.
"w_dolp_ach_b"      : "b" in Equation 9.
"w_aolp_ach_a"      : "a" in differences between RGB values of AoLP.
"w_aolp_ach_b"      : "b" in differences between RGB values of AoLP.
"w_dolp_ch_a"       : "a" in Equation 10.
"w_dolp_ch_b"       : "b" in Equation 10.
"w_aolp_ch_a"       : "a" in AoLP differences for exlcluding edges.
"w_aolp_ch_b"       : "b" in AoLP differences for exlcluding edges.
"alpha"             : "alpha" to blend the results of chromatic and achromatic pixels, described in Fig. 3 in supplementary material.
```

## License
This software is released under the MIT License. See [LICENSE](/LICENSE) for details.

## Citation
```
@InProceedings{Ono_2022_CVPR,
    author    = {Ono, Taishi and Kondo, Yuhi and Sun, Legong and Kurita, Teppei and Moriuchi, Yusuke},
    title     = {Degree-of-linear-polarization-based Color Constancy},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {xxxxx-xxxxx}
}
```
