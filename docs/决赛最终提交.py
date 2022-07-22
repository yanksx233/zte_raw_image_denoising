import os
import cv2
import time
import numpy as np
import rawpy
import glob
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def normalization(input_data, black_level, white_level):
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
    return output_data


def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


####### please copy the code between 'my start' and 'my end' #######
################## my start ##################
def func(p, mu):
    a, b = p
    return a*mu + b  # sigma^2 = a * mu + b
def error(p, mu, var):
    return func(p, mu) - var


def patch_based(img, patch_size, ratio):
    H, W, C = img.shape
    N = patch_size * patch_size
    stride = patch_size
    min_value = img.min()
    max_value = img.max()

    mu_list = []
    var_list = []
    var_mean_list = []
    for h in range(0, H-patch_size+1, stride):
        for w in range(0, W-patch_size+1, stride):
            patch = img[h:h+patch_size, w:w+patch_size]

            min_count = (patch == min_value).sum()  # ignore dark patch
            if min_count > int(N*0.1):
                continue
            max_count = (patch == max_value).sum()  # ignore light patch
            if max_count > int(N*2):
                continue

            token = np.reshape(patch, (N, C))

            mean_patch = np.mean(token, 0)  # shape of (C,)
            var_patch = np.var(token, 0, ddof=1)
            
            mu_list.append(mean_patch)
            var_list.append(var_patch)
            var_mean_list.append(var_patch.mean())

    number = int(len(var_mean_list) * ratio)  # filter patches with larger variance
    indices = np.argpartition(var_mean_list, number)[:number]

    mu_arr = np.concatenate(np.array(mu_list)[indices])
    var_arr = np.concatenate(np.array(var_list)[indices])
    var_arr = (1 + 0.001 * (var_arr - 40)) * var_arr  # corrected variance (optional)
    return mu_arr, var_arr


def cal_noise_profile(test_dir, black_level, white_level):
    """
    your code should be given here
    """
    img, h, w = read_image(test_dir)
    img = normalization(img, black_level, white_level)

    patch_para = {'patch_size': 8, 'ratio': 0.4}
    mu_arr, var_arr = patch_based(img, **patch_para)

    p0 = [1e-3, 1e-4]
    a, b = leastsq(error, p0, args=(mu_arr, var_arr))[0]
    return a, b
################## my end ##################