#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:24:58 2021

@author: marina
"""

import argparse
import multiprocessing
from joblib import Parallel, delayed
from functions import *

def do_one_channel(ch, n, m, I, W, w, I_means, blocks_list, num_blocks, samples_per_bin, T_mask):
    all_blocks = np.zeros((int(I.shape[0]/W) +1, int(I.shape[1]/W)+1))
    red_blocks = np.zeros((int(I.shape[0]/W) +1, int(I.shape[1]/W)+1))
    DCTS = DCT_all_blocks(I[:,:,ch], w)
    means = means_list(I_means, blocks_list, ch, w)
    blocks_list_aux = blocks_list[np.argsort(means)]
    num_bins = int(round(num_blocks/samples_per_bin))
    if num_bins == 0: num_bins = 1
    muestras_por_bin = int(num_blocks/num_bins) 
    if muestras_por_bin == 0: muestras_por_bin = 1

    for b in range(num_bins):
        bin_blocks = bin_block_list(b, blocks_list_aux, muestras_por_bin)
        VL = compute_low_freq_var(DCTS, bin_blocks, T_mask, I.shape[1], w)
        N = int(samples_per_bin*n)
        sorted_blocks = bin_blocks[np.argsort(VL)] [0:N]
        stds_sorted_blocks = std_blocks(I, w, sorted_blocks,ch)
        M = int(N*m)
        if len(np.where(stds_sorted_blocks==0)[0])<M:             
            best_blocks_pos = np.array(sorted_blocks)[np.argsort(stds_sorted_blocks)]   

            for k,pos in enumerate(best_blocks_pos):
                all_blocks[int(pos[0]/W), int(pos[1]/W)] +=1
                if k < M: red_blocks[int(pos[0]/W), int(pos[1]/W)] +=1
    return all_blocks, red_blocks

def do_one_image(f):
    
    filename = f
    w=5
    W = 256
    samples_per_bin = 20000
    n = 0.05
    m = 0.3
    
    res_directory = create_res_directory(filename)
    I = read_image(filename)        

    blocks_list = valid_blocks(I,w)
    num_blocks = len(blocks_list)
    I_means = all_image_means(I,w)
    num_channels = I.shape[2]    
    T_mask = get_T_mask(w)


    with multiprocessing.Pool(processes=num_channels) as pool:

        results = Parallel(n_jobs=num_channels)(delayed(do_one_channel)
        (i, n, m, I, W, w, I_means, blocks_list, num_blocks, samples_per_bin, T_mask) 
        for i in range(num_channels))
        all_blocks = results[0][0] + results[1] [0] + results[2] [0]
        red_blocks = results[0][1] + results[1] [1] + results[2] [1]

    compute_save_NFA(I, w, W, n, m, samples_per_bin, all_blocks, red_blocks, res_directory)
    do_mask(res_directory, I, 1)
    
parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.parse_args()
args = parser.parse_args()

f = args.filename
do_one_image(f)