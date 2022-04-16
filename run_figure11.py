#!/usr/bin/env python

'''
    This script replicates the DeepTensor results in Figure 7. The accuracy is
    different from the paper as the video has been downsampled and clipped to
    fit supplementary material requirements.
'''

import os
import sys
import tqdm
import copy
import time

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib.pyplot as plt
plt.gray()
import cv2

sys.path.append('../modules')

import models
import utils
import losses
import deep_prior
import deep_decoder
import lin_inverse

if __name__ == '__main__':
    expname = 'cat_video'
    nframes = 8
    rank = 32
        
    # Network parameters
    nettype = 'dip'
    n_inputs = rank
    init_nconv = 64
    num_channels_up = 3
    
    # Noise parameters
    scaling = 1
    tau = 100
    noise_snr = 2
    
    # Learning constants
    learning_rate = 1e-3
    epochs = 3000
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99  
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, totalframes = cube.shape

    if rank == 'max':
        rank = min(H, W, totalframes)
        n_inputs = rank
    
    cube_noisy = utils.measure(cube, noise_snr, tau)
    
    # Get masks
    masks = lin_inverse.get_video_coding_frames(cube.shape, nframes)
    
    # Get video measurements
    measurements = np.zeros((H, W, totalframes//nframes+1), dtype=np.float32)
    
    for idx in range(0, totalframes, nframes):
        cube_chunk = cube_noisy[..., idx:idx+nframes]
        masks_chunk = masks[..., idx:idx+nframes]
        measurements[..., idx//nframes] = (cube_chunk*masks_chunk).sum(2) 
        
    if idx < totalframes:
        cube_chunk = cube_noisy[..., idx:]
        masks_chunk = masks[..., idx:]
        measurements[..., idx//nframes + 1] = (cube_chunk*masks_chunk).sum(2) 
        
    # Send data to device
    measurements_ten = torch.tensor(measurements)[None, ...]
    measurements_ten = measurements_ten.permute(0, 3, 1, 2).cuda()
    cube_ten = torch.tensor(cube)[None, ...].permute(0, 3, 1, 2).cuda()
    masks_ten = torch.tensor(masks)[None, ...].permute(0, 3, 1, 2).cuda()
    
    if nettype == 'unet':
        im_net = models.UNetND(n_inputs, rank, 2, init_nconv).cuda()    
        spec_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
        
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, totalframes])
    elif nettype == 'dip':
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, totalframes])
    elif nettype == 'dd':
        nchans = [init_nconv]*num_channels_up
        im_net = deep_decoder.decodernw(rank, nchans).cuda()
        spec_net = deep_decoder.decodernw1d(rank, nchans).cuda()
        
        H1 = H // pow(2, num_channels_up)
        W1 = W // pow(2, num_channels_up)
        nwvl1 = totalframes // pow(2, num_channels_up)
        im_inp = utils.get_inp([1, init_nconv, H1, W1])
        spec_inp = utils.get_inp([1, init_nconv, nwvl1])
    
    # Switch to trairning mode
    im_net.train()
    spec_net.train()
    
    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    cube_estim_avg = None
    
    params = net_params + inp_params

    criterion_l1 = losses.L2Norm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)    
    time_array = np.zeros(epochs)
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    best_loss = float('inf')
    best_epoch = 0
    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        # Perturb inputs
        im_inp_perturbed = im_inp + im_inp_per.normal_()*reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_()*reg_noise_std
        
        U_img = im_net(im_inp_perturbed)
        V = spec_net(spec_inp_perturbed)
        
        U = U_img.reshape(-1, rank, H*W).permute(0, 2, 1)
        mat_estim = torch.bmm(U, V)
        
        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)
        
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)
        
        loss = criterion_l1(measurements_ten - meas_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_item = loss.item()
        mse_array[idx] = ((cube_ten - cube_estim)**2).mean().item()
        time_array[idx] = time.time() - tic
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        if loss_item < best_loss:
            best_loss = loss_item
            best_epoch = idx
            best_cube_estim = copy.deepcopy(cube_estim.detach())
            
            # Averaging as per original code    
            if cube_estim_avg is None:
                cube_estim_avg = cube_estim.detach()
            else:
                cube_estim_avg = exp_weight*cube_estim_avg +\
                            (1 - exp_weight)*cube_estim.detach()
        
        with torch.no_grad():
            img_idx = idx%totalframes
            ref = cube[..., img_idx]
            diff = abs(cube_ten - cube_estim).mean(2).squeeze().detach().cpu()
            img = cube_estim[0, img_idx, ...].detach().cpu().numpy()

            if sys.platform == 'win32':
                cv2.imshow('Diff x10', diff.numpy()*10)
                cv2.imshow('Avg', np.hstack((ref, img)))
                cv2.waitKey(1)
            
    cube_estim = best_cube_estim.cpu().squeeze().permute(1, 2, 0).numpy()
    psnrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    
    print('PSNR: %.2f'%psnrval)
    print('SSIM: %.2f'%ssimval)
