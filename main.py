# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 23:56:16 2017
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import os
import shutil
import time
import cv2
import numpy as np
from ifs_simple import ifs_function
from functions import *


iteration_num = 500  # 画像生成の試行回数(各バリエーションごとの回数)
numof_point = 100000  # IFSのイテレーション数
save_dir = "./test"
image_size = 512
remake_dataset_flag = True  # True時、既存のデータセットが削除される


def cal_pix(gray):
    height, width = gray.shape
    num_pixels = np.count_nonzero(gray) / float(height * width)
    return num_pixels


def generator(params, func_V):
    generators = ifs_function(prev_x=0.0, prev_y=0.0)
    for param in params:
        generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]))
    generators.calculate(numof_point, func_V)  # class
    img = generators.draw_point(
        image_size, image_size, 6, 6, 'gray', 0)  # image (x,y pad x,y)
    return img  # return by cv2


func_collection = [[linear, 0.2],
                   [sinusoidal, 0.2],
                   [spherical, 0.04],
                   [swirl, 0.06],
                   [polar, 0.1],
                   [hand_kerchief, 0.1],
                   [heart, 0.1],
                   [disc, 0.1],
                   [spiral, 0.1],
                   [hyperbolic, 0.1]]
func_collection = [[hyperbolic, 0.04], [spiral, 0.05]]


if __name__ == "__main__":
    np.random.seed(1)
    pixels_csv_dir = os.path.join(save_dir, "pixels_csv")
    img_root_dir = os.path.join(save_dir, "fractal_image")
    if os.path.exists(pixels_csv_dir) == False:
        os.makedirs(pixels_csv_dir)
        os.makedirs(img_root_dir)
    elif remake_dataset_flag:
        # フォルダの作り直し
        shutil.rmtree(pixels_csv_dir)
        shutil.rmtree(img_root_dir)
        os.makedirs(pixels_csv_dir)
        os.makedirs(img_root_dir)
    else:
        print("fractal data exits")
        exit()

    for i, x in enumerate(func_collection):
        func, threshold = tuple(x)
        class_num = 0
        img_dir = os.path.join(img_root_dir, func.__name__)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        elif remake_dataset_flag:
            shutil.rmtree(img_dir)
            os.makedirs(img_dir)
        else:
            print("fractal data exists")
            exit()
        pixels_list = []
        while(class_num < iteration_num):  # class
            a, b, c, d, e, f, prob = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            param_size = np.random.randint(2, 8)
            params = np.zeros((param_size, 7), dtype=float)
            sum_proba = 0.0
            for i in range(param_size):
                a, b, c, d, e, f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                param_rand = np.random.uniform(-1.0, 1.0, 6)
                a, b, c, d, e, f = param_rand[0:6]
                prob = abs(a*d-b*c)
                sum_proba += prob
                params[i, 0:7] = a, b, c, d, e, f, prob
            for i in range(param_size):
                params[i, 6] /= sum_proba
            class_str = '%05d' % class_num
            fractal_img = generator(params, func)
            pixels = cal_pix(fractal_img[:, :, 0].copy())
            pixels_list.append(pixels)
            if pixels >= threshold:
                print('save: '+func.__name__+class_str)
                cv2.imwrite(os.path.join(img_dir, class_str + '.png'), fractal_img)
            class_num += 1
        np.savetxt(os.path.join(pixels_csv_dir, func.__name__+'.csv'), pixels_list, delimiter=',')
