# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2017

@author: Kazushige Okayasu, Hirokatsu Kataoka

MIT License

Copyright (c) 2020 National Institute of Advanced Industrial Science and Technology (AIST)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import time
import argparse
import numpy as np

from ifs_function import ifs_function


def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_root', default='./data/csv',
                        type=str, help='load csv root')
    parser.add_argument(
        '--save_root', default='./data/FractalDB', type=str, help='save png root')
    parser.add_argument('--image_size_x', default=224,
                        type=int, help='image size x')
    parser.add_argument('--image_size_y', default=224,
                        type=int, help='image size y')
    parser.add_argument('--pad_size_x', default=6,
                        type=int, help='padding size x')
    parser.add_argument('--pad_size_y', default=6,
                        type=int, help='padding size y')
    parser.add_argument('--iteration', default=100000,
                        type=int, help='iteration')
    parser.add_argument('--draw_type', default='patch_gray', type=str,
                        help='{point, patch}_{gray, color}_{point_gray_filter}')
    parser.add_argument(
        '--weight_csv', default='./fractal_renderer/weights/weights_0.1.csv', type=str, help='weight parameter')
    parser.add_argument('--instance', default=10, type=int,
                        help='#instance, 10 => 1000 instance, 100 => 10,000 instance per category')
    parser.add_argument('-f', '--force', action="store_true")
    args = parser.parse_args()
    return args


def make_directory(save_root, name):
    if not os.path.exists(os.path.join(save_root, name)):
        os.mkdir(os.path.join(save_root, name))


if __name__ == "__main__":
    starttime = time.time()
    args = conf()
    csv_names = os.listdir(args.load_root)
    csv_names.sort()
    weights = np.genfromtxt(args.weight_csv, dtype=np.str, delimiter=',')

    if not os.path.exists(os.path.join(args.save_root)):
        os.mkdir(os.path.join(args.save_root))
    else:
        print("folder exists")
        if args.force:
            print("remake fractalDB")
        else:
            exit()

    for csv_name in csv_names:
        name, ext = os.path.splitext(csv_name)
        print(name, ext)

        if ext != '.csv':  # Skip except for csv file
            continue
        print(name)

        make_directory(args.save_root, name)  # Make directory
        fractal_weight = 0
        for weight in weights:
            padded_fractal_weight = '%02d' % fractal_weight
            if args.draw_type == 'point_gray':
                generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                          fractal_name=name, fractal_weight_count=padded_fractal_weight)
                params = np.genfromtxt(os.path.join(
                    args.load_root, csv_name), dtype=np.str, delimiter=',')
                for param in params:
                    generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                                         weight_a=float(weight[0]), weight_b=float(weight[1]), weight_c=float(weight[2]), weight_d=float(weight[3]), weight_e=float(weight[4]), weight_f=float(weight[5]))
                generators.calculate(args.iteration)
                generators.draw_point(
                    args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y, 'gray', 0)

            elif args.draw_type == 'point_color':
                generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                          fractal_name=name, fractal_weight_count=padded_fractal_weight)
                params = np.genfromtxt(os.path.join(
                    args.load_root, csv_name), dtype=np.str, delimiter=',')
                for param in params:
                    generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                                         weight_a=float(weight[0]), weight_b=float(weight[1]), weight_c=float(weight[2]), weight_d=float(weight[3]), weight_e=float(weight[4]), weight_f=float(weight[5]))
                generators.calculate(args.iteration)
                generators.draw_point(
                    args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y, 'color', 0)

            elif args.draw_type == 'patch_gray':
                for count in range(args.instance):
                    generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                              fractal_name=name, fractal_weight_count=padded_fractal_weight)
                    params = np.genfromtxt(os.path.join(
                        args.load_root, csv_name), dtype=np.str, delimiter=',')
                    for param in params:
                        generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                                             weight_a=float(weight[0]), weight_b=float(weight[1]), weight_c=float(weight[2]), weight_d=float(weight[3]), weight_e=float(weight[4]), weight_f=float(weight[5]))
                    generators.calculate(args.iteration)
                    generators.draw_patch(
                        args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y, 'gray', count)

            elif args.draw_type == 'patch_color':
                for count in range(args.instance):
                    generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                              fractal_name=name, fractal_weight_count=padded_fractal_weight)
                    params = np.genfromtxt(os.path.join(
                        args.load_root, csv_name), dtype=np.str, delimiter=",")
                    for param in params:
                        generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                                             weight_a=float(weight[0]), weight_b=float(weight[1]), weight_c=float(weight[2]), weight_d=float(weight[3]), weight_e=float(weight[4]), weight_f=float(weight[5]))
                    generators.calculate(args.iteration)
                    generators.draw_patch(
                        args.image_size_x, args.image_size_y, args.pad_size_x, args.pad_size_y, 'color', count)
            else:  # ぼかしによるデータ拡張
                for count in range(args.instance):
                    generators = ifs_function(prev_x=0.0, prev_y=0.0, save_root=args.save_root,
                                              fractal_name=name, fractal_weight_count=padded_fractal_weight)
                    params = np.genfromtxt(os.path.join(
                        args.load_root, csv_name), dtype=np.str, delimiter=',')
                    for param in params:
                        generators.set_param(float(param[0]), float(param[1]), float(param[2]), float(param[3]), float(param[4]), float(param[5]), float(param[6]),
                                             weight_a=float(weight[0]), weight_b=float(weight[1]), weight_c=float(weight[2]), weight_d=float(weight[3]), weight_e=float(weight[4]), weight_f=float(weight[5]))
                    generators.calculate(args.iteration)
                    generators.draw_point(args.image_size_x, args.image_size_y,
                                          args.pad_size_x, args.pad_size_y, 'gray', count, args.draw_type)
            fractal_weight += 1

    endtime = time.time()
    interval = endtime - starttime
    print("passed time = %dh %dm %ds" % (int(interval/3600),
          int((interval % 3600)/60), int((interval % 3600) % 60)))
