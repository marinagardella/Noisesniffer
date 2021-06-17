#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:18:51 2021

@author: marina
"""

#import matplotlib
#matplotlib.use('agg')
from pathlib import Path
import os
import imageio
import numpy as np
from scipy.fftpack import dct
from scipy.stats import binom
#import matplotlib.pyplot as plt
import cv2
from skimage.util import view_as_windows


def conv(img, kernel):
    '''
    computes 2D convolution.
    '''
    C = cv2.filter2D(img, -1, kernel)
    H = np.floor(np.array(kernel.shape) / 2).astype(np.int)
    C = C[H[0]:-H[0] + 1,
          H[1]:-H[1] + 1] if kernel.shape[0] % 2 == 0 else C[H[0]:-H[0],
                                                             H[1]:-H[1]]
    return C


def read_image(filename):
    '''
    reads an image with len(shape) = 3.
    '''
    img = imageio.imread(filename).astype(float)
    if img.ndim == 2:
        img = img[:, :, None]
    img = img[:, :, :3]
    return img


def create_res_directory(filename):
    '''
    creates the 'results' directory in case it does not exists
    and within it, creates a subdirectory for the given filename.
    '''
    image = Path(filename).stem
    out_path = Path('results') / image 
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def valid_blocks(img, w):
    '''
    computes all valid (not containing saturations) wxw blocks in image img.
    '''
    C = img.shape[2]  
    img_not_saturated = np.ones(img.shape[:2]) 
    for ch in range(C):
        maximum = img[:, :, ch].max()
        minimum = img[:, :, ch].min()
        img_aux = np.ones((img.shape[0], img.shape[1]))
        img_aux[np.where(img[:, :, ch] == maximum)] = 0
        img_aux[np.where(img[:, :, ch] == minimum)] = 0
        img_not_saturated *= img_aux
    kernel = np.ones((w, w))
    img_int = conv(img_not_saturated, kernel)
    indices = np.where(img_int > w**2 - .5)
    blocks_list = np.zeros((len(indices[0]), 2), dtype=int)
    blocks_list[:, 0] = indices[0]
    blocks_list[:, 1] = indices[1]
    return blocks_list


def all_image_means(img, w):
    '''
    computes the means for all the wxw blocks in image img.
    '''
    kernel = (1 / w**2) * np.ones((w, w))
    img_means = conv(img, kernel)
    if img.shape[2] == 1:
        return img_means.reshape(img_means.shape[0], img_means.shape[1], 1)
    else:
        return img_means


def means_list(img_means, blocks_list, ch, w):
    '''
    creates a list with the means in img_means in channel ch for the blocks in blocks_list.
    '''
    means = [img_means[pos[0], pos[1], ch] for pos in blocks_list]
    means = np.array(means)
    return np.round(means, decimals=2)


def get_T(w):
    '''
    returns the threshold to define low-med frequencies 
    according the block size w.
    '''
    if w == 3:
        return 3
    if w == 5:
        return 5
    if w == 7:
        return 8
    if w == 8:
        return 9
    if w == 11:
        return 13
    if w == 16:
        return 18
    if w == 21:
        return 24
    else:
        print(f'unknown block side {w}')


def get_T_mask(w):
    '''
    computes a mask that corresponds to the low-med 
    frequencies according to the block size w.
    '''
    mask = np.zeros((w, w))
    for i in range(w):
        for j in range(w):
            if (0 != i + j) and (i + j < get_T(w)):
                mask[i, j] = 1
    return mask



def DCT_all_blocks(img,w):
    '''
    computes the DCT II of all the wxw overlapping blocks in img.
    '''
    Q_aux = view_as_windows(img, w).reshape(-1, w, w)
    return dct(dct(Q_aux, axis=1, norm='ortho'), axis=2, norm='ortho')


def compute_low_freq_var(DCTS, blocks, mask, shape, w):
    '''
    computes the variance of the DCT coefficients given by mask,
    on each wxw block given by blocks.
    '''
    VL = [
        np.sum((DCTS[pos[0] * (shape - w + 1) + pos[1]] * mask)**2)
        for pos in blocks
    ]
    VL = np.array(VL)
    return VL


def bin_block_list(b, blocks_list_aux, muestras_por_bin):
    '''
    computes the list of blocks corresponding to bin b, having muestras_por_bin elements
    '''
    num_blocks = len(blocks_list_aux)
    num_bins = int(round(num_blocks / muestras_por_bin))
    if num_bins == 0: num_bins = 1
    bin_block_list = blocks_list_aux[
        int(num_bins - 1) *
        muestras_por_bin:num_blocks] if b == num_bins - 1 else blocks_list_aux[
            b * muestras_por_bin:(b + 1) * muestras_por_bin]
    return bin_block_list


def std_blocks(img, w, sorted_blocks, ch):
    '''
    computes the std in channel ch of the wxw blocks in the list sorted_blocks.
    '''
    stds = [
        np.std(img[pos[0]:pos[0] + w, pos[1]:pos[1] + w, ch])
        for pos in sorted_blocks
    ]
    stds = np.array(stds)
    return stds


def compute_save_NFA(img, w, W, n, m, b, all_blocks, red_blocks,
                     res_directory):
    '''
    computes the NFA on WxW macroblocks and saves the the result as a txt file.
    '''
    macro_x = int(img.shape[0] / W)
    if macro_x * W < img.shape[0]: macro_x += 1
    macro_y = int(img.shape[1] / W)
    if macro_y * W < img.shape[1]: macro_y += 1
    num_macroblocks = macro_x * macro_y
    v = w**2
    with open(f"{res_directory}/NFA_w{w}_W{W}_n{n}_m{m}_b{b}.txt",
              "w") as NFA_file:
        NFA_file.write('macroblock_origin_x macroblock_origin_y NFA \n')
    with open(f"{res_directory}/NFA_w{w}_W{W}_n{n}_m{m}_b{b}.txt",
              "a") as NFA_file:
        [
            NFA_file.write(
                f'{i*W} {j*W} {v*num_macroblocks*(1 - binom.cdf((int(red_blocks[i, j]/v))-1, int(all_blocks[i,j]/v)+ 1, m))} \n'
            ) for i in range(macro_x) for j in range(macro_y)
        ]


#    plt.imshow(NFA, vmin = 0, vmax=1)
#    plt.title(f'NFA W{W} n{n} m{m} b{b}')
#    plt.colorbar()
#    plt.savefig(f"{res_directory}/NFA_w{w}_W{W}_n{n}_m{m}_b{b}_s.png")
#    plt.close()


def do_mask(res_directory, img, thresh):
    '''
    computes the forgery mask according to the threshols thresh.
    '''
    mask = np.zeros((img.shape[0], img.shape[1]))
    NFA_file = np.genfromtxt(res_directory /
                             'NFA_w5_W256_n0.05_m0.3_b20000.txt',
                             usecols=(0, 1, 2),
                             skip_header=1,
                             dtype=float)
    block_origin = np.array(NFA_file[:, 0:2]).astype(np.int)
    NFA = np.array(NFA_file[:, 2]).astype(np.float)
    for origin in block_origin[np.where(NFA <= thresh)]:
        mask[origin[0]:origin[0] + 256, origin[1]:origin[1] + 256] = 255
    imageio.imsave(res_directory / f'mask_thresh{thresh}.png',
                   mask.astype(np.uint8))
