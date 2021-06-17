#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:24:58 2021

@author: marina
"""

import argparse
import multiprocessing
from joblib import Parallel, delayed
import numpy as np

import functions


def do_one_channel(ch, n, m, img, W, w, img_means, blocks_list, num_blocks,
                   samples_per_bin, T_mask):
    all_blocks = np.zeros(
        (int(img.shape[0] / W) + 1, int(img.shape[1] / W) + 1))
    red_blocks = np.zeros(
        (int(img.shape[0] / W) + 1, int(img.shape[1] / W) + 1))
    DCTS = functions.DCT_all_blocks(img[:, :, ch], w)
    means = functions.means_list(img_means, blocks_list, ch, w)
    blocks_list_aux = blocks_list[np.argsort(means)]
    num_bins = int(round(num_blocks / samples_per_bin))
    if num_bins == 0: num_bins = 1
    muestras_por_bin = int(num_blocks / num_bins)
    if muestras_por_bin == 0: muestras_por_bin = 1

    for b in range(num_bins):
        bin_blocks = functions.bin_block_list(b, blocks_list_aux,
                                              muestras_por_bin)
        VL = functions.compute_low_freq_var(DCTS, bin_blocks, T_mask,
                                            img.shape[1], w)
        N = int(samples_per_bin * n)
        sorted_blocks = bin_blocks[np.argsort(VL)][0:N]
        stds_sorted_blocks = functions.std_blocks(img, w, sorted_blocks, ch)
        M = int(N * m)
        if len(np.where(stds_sorted_blocks == 0)[0]) < M:
            best_blocks_pos = np.array(sorted_blocks)[np.argsort(
                stds_sorted_blocks)]

            for k, pos in enumerate(best_blocks_pos):
                all_blocks[int(pos[0] / W), int(pos[1] / W)] += 1
                if k < M: red_blocks[int(pos[0] / W), int(pos[1] / W)] += 1
    return all_blocks, red_blocks


def do_one_image(f):

    filename = f
    w = 5
    W = 256
    samples_per_bin = 20000
    n = 0.05
    m = 0.3

    res_directory = functions.create_res_directory(filename)
    img = functions.read_image(filename)

    blocks_list = functions.valid_blocks(img, w)
    num_blocks = len(blocks_list)
    img_means = functions.all_image_means(img, w)
    num_channels = img.shape[2]
    T_mask = functions.get_T_mask(w)

    with multiprocessing.Pool(processes=num_channels) as pool:

        results = Parallel(n_jobs=num_channels)(
            delayed(do_one_channel)(i, n, m, img, W, w, img_means, blocks_list,
                                    num_blocks, samples_per_bin, T_mask)
            for i in range(num_channels))
        all_blocks = results[0][0] + results[1][0] + results[2][0]
        red_blocks = results[0][1] + results[1][1] + results[2][1]

    functions.compute_save_NFA(img, w, W, n, m, samples_per_bin, all_blocks,
                               red_blocks, res_directory)
    functions.do_mask(res_directory, img, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.parse_args()
    args = parser.parse_args()

    f = args.filename
    do_one_image(f)
