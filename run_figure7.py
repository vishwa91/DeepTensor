#!/usr/bin/env python

import os
import sys
import glob
import tqdm
import importlib

import numpy as np
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib.pyplot as plt
plt.gray()
import cv2

sys.path.append('modules')

import models
import utils
import losses
import deep_prior
import deep_decoder

if __name__ == '__main__':
    # Set your simulation constants here
    nrows = 64                  # Size of the matrix
    ncols = 64
    rank = 10                   # Rank
    noise_type = 'gaussian'     # Type of noise
    signal_type = 'gaussian'    # Type of signal
    noise_snr = 0.2             # Std. dev for gaussian noise
    tau = 1000                  # Max. lambda for photon noise
    
    # Network parameters
    nettype = 'dip'
    n_inputs = 10
    init_nconv = 128
    num_channels_up = 5
    
    # Learning constants
    learning_rate = 1e-4
    
    # The number of epochs should be set according to noise level, and number
    # of data points. As a general rule of thumb, higher noise or fewer samples
    # requires fewer epochs
    epochs = 500
    
    # Generate data
    mat, mat_gt, basis = utils.get_pca(nrows, ncols, rank,
                                       noise_type, signal_type,
                                       noise_snr, tau)
    
    # Compute covariance matrix
    mat_centered = mat - mat.mean(0).reshape(1, ncols)
    mat_gt_centered = mat_gt - mat_gt.mean(0).reshape(1, ncols)
    covmat = mat_centered.dot(mat_centered.T)
    covmat_gt = mat_gt_centered.dot(mat_gt_centered.T)
    
    minval = covmat_gt.min()
    maxval = covmat_gt.max()
    
    covmat = (covmat - minval)/(maxval - minval)
    covmat_gt = (covmat_gt - minval)/(maxval - minval)    
        
    # Move them to device
    covmat_ten = torch.tensor(covmat)[None, ...].cuda()
    covmat_gt_ten = torch.tensor(covmat_gt)[None, ...].cuda()
    
    # Since we are doing PCA, we need only one network
    u_inp = utils.get_inp([1, n_inputs, nrows])
    
    # Create networks
    if nettype == 'unet':
        u_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
    elif nettype == 'dip':
        u_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection', 'linear',
                                   rank).cuda()
    elif nettype == 'dd':
        u_net = deep_decoder.decodernw1d(rank,
                                         [init_nconv]*num_channels_up).cuda()
        
        # Deep decoder requires smaller inputs
        u_inp = utils.get_inp([1, init_nconv, nrows // pow(2, num_channels_up)])
    
    # Create optimizer
    net_params = list(u_net.parameters()) 
    inp_params = [u_inp]
    
    # You can either optimize both net and inputs, or just net
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    # Create loss functions -- loses.L1Norm() or losses.L2Norm()
    criterion = losses.L2Norm()
    
    mse_array = np.zeros(epochs)
    
    # Now start iterations
    best_mse = float('inf')
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        u_output = u_net(u_inp)
        
        covmat_estim = torch.bmm(u_output.permute(0, 2, 1), u_output)
        
        loss = criterion(covmat_estim - covmat_ten)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Visualize the reconstruction
        diff = abs(covmat_gt_ten - covmat_estim).squeeze().detach().cpu()
        mat_cpu = covmat_estim.squeeze().detach().cpu().numpy()
        
        cv2.imshow('Diff x10', diff.numpy().reshape(nrows, nrows)*10)
        cv2.imshow('Rec', np.hstack((covmat_gt, mat_cpu)))
        cv2.waitKey(1)
        
        mse_array[idx] = ((covmat_estim - covmat_gt_ten)**2).mean().item()
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        if loss.item() < best_mse:
            best_epoch = idx
            best_mat = mat_cpu
            best_mse = loss.item()
        
    # Now compute accuracy
    psnr1 = utils.psnr(covmat_gt, best_mat)
    psnr2 = utils.psnr(covmat_gt, utils.lr_decompose(covmat, rank))
    
    print('PCA with DeepTensor: %.2fdB'%psnr1)
    print('PCA with SVD: %.2fdB'%psnr2)
    
        