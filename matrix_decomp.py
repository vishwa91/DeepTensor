#!/usr/bin/env python

'''
    This script replicates instances of results in figure 4 in the main paper.
    To obtain the plots, please sweep the relevant parameters.
'''

import os
import sys
import glob
import tqdm
import importlib
import argparse

import numpy as np
from scipy import io
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
    nrows = 64                 # Size of the matrix
    ncols = 64
    rank = 32                  # Rank
    noise_type = 'gaussian'    # Type of noise
    signal_type = 'gaussian'   # Type of signal
    noise_snr = 0.2              # Std. dev for gaussian noise
    tau = 1000                  # Max. lambda for photon noise
    
    # Network parameters
    nettype = 'dip'
    n_inputs = rank
    init_nconv = 128
    num_channels_up = 5
    
    sched_args = argparse.Namespace()
    # Learning constants
    # Important: The number of epochs is decided by noise levels. As a general
    # rule of thumb, higher the noise, fewer the epochs.
    scheduler_type = 'none'
    learning_rate = 1e-4
    epochs = 1000
    sched_args.step_size = 2000
    sched_args.gamma = pow(10, -1/epochs)
    sched_args.max_lr = learning_rate
    sched_args.min_lr = 1e-6
    sched_args.epochs = epochs
    
    # Generate data
    mat, mat_gt = utils.get_matrix(nrows, ncols, rank, noise_type, signal_type,
                                   noise_snr, tau)
        
    # Move them to device
    mat_ten = torch.tensor(mat)[None, ...].cuda()
    mat_gt_ten = torch.tensor(mat_gt)[None, ...].cuda()
    
    u_inp = utils.get_inp([1, n_inputs, nrows])
    v_inp = utils.get_inp([1, n_inputs, ncols])
    
    # Create networks
    if nettype == 'unet':
        u_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
        v_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
    elif nettype == 'dip':
        u_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                        upsample_mode='linear',
                                        skip_n33d=init_nconv,
                                        skip_n33u=init_nconv,
                                        num_scales=5,
                                        n_channels=rank).cuda()
        v_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=5,
                                    n_channels=rank).cuda()
    elif nettype == 'dd':
        u_net = deep_decoder.decodernw1d(rank,
                                         [init_nconv]*num_channels_up).cuda()
        v_net = deep_decoder.decodernw1d(rank,
                                         [init_nconv]*num_channels_up).cuda()
        
        # Deep decoder requires smaller inputs
        u_inp = utils.get_inp([1, init_nconv, nrows // pow(2, num_channels_up)])
        v_inp = utils.get_inp([1, init_nconv, ncols // pow(2, num_channels_up)])
    
    # Extract training parameters
    net_params = list(u_net.parameters()) + list(v_net.parameters())
    inp_params = [u_inp] + [v_inp]
    
    # You can either optimize both net and inputs, or just net
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    # Create a learning scheduler
    scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)
    
    # Create loss functions -- loses.L1Norm() or losses.L2Norm()
    criterion = losses.L2Norm()
    
    mse_array = np.zeros(epochs)
    
    # Now start iterations
    best_mse = float('inf')
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        u_output = u_net(u_inp).permute(0, 2, 1)
        v_output = v_net(v_inp)
        
        mat_estim = torch.bmm(u_output, v_output)
        
        loss = criterion(mat_estim - mat_ten)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Visualize the reconstruction
        diff = abs(mat_gt_ten - mat_estim).squeeze().detach().cpu()
        mat_cpu = mat_estim.squeeze().detach().cpu().numpy()
        
        cv2.imshow('Diff x10', diff.numpy().reshape(nrows, ncols)*10)
        cv2.imshow('Rec', np.hstack((mat_gt, mat_cpu)))
        cv2.waitKey(1)
        
        mse_array[idx] = ((mat_estim - mat_gt_ten)**2).mean().item()
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        if loss.item() < best_mse:
            best_epoch = idx
            best_mat = mat_cpu
            best_mse = loss.item()
        
    # Now compute accuracy
    psnr1 = utils.psnr(mat_gt, best_mat)
    psnr2 = utils.psnr(mat_gt, utils.lr_decompose(mat, rank))
    
    print('DeepTensor: %.2fdB'%psnr1)
    print('SVD: %.2fdB'%psnr2)
    
        
