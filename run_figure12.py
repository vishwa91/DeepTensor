#!/usr/bin/env python

'''
    This script replicates the DeepTensor results in Figure 8. The accuracy is
    different from the paper as PET scan was downsampled.
'''

import sys
import tqdm
import copy
import time
import argparse

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch
import tensorly
tensorly.set_backend('pytorch')

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
    expname = 'pet'
    nangles = 40
    rank = 1000
    nettype = 'dip'
        
    # Network parameters
    n_inputs = rank
    init_nconv = 128
    num_channels_up = 5
    
    # Noise parameters
    scaling = 1
    tau = 1000
    noise_snr = 2
    
    # Learning constants
    sched_args = argparse.Namespace()
    scheduler_type = 'none'
    learning_rate = 1e-3
    lambda_tv = 1e1
    epochs = 5000
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99  
    sched_args.step_size = 2000
    sched_args.gamma = pow(10, -1/epochs)
    sched_args.max_lr = learning_rate
    sched_args.min_lr = 1e-4
    sched_args.epochs = epochs
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, T = cube.shape
    
    cube_noisy = utils.measure(cube, noise_snr, tau).astype(np.float32)
    
    # Send data to device
    cube_gt_ten = torch.tensor(cube).cuda()
    cube_ten = torch.tensor(cube).cuda().permute(2, 0, 1)[None, ...]
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32)).cuda()
    
    # Generate sinogram
    measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()
    maxval = measurements.max()
    measurements = utils.measure(measurements, noise_snr, tau)
    measurements = torch.tensor(measurements).cuda()
    
    # Generate inputs -- positionally encoded inputs results in smoother reconstruction
    #u_inp = utils.get_inp([1, rank, H])
    #v_inp = utils.get_inp([1, rank, W])
    #w_inp = utils.get_inp([1, rank, T])
    
    u_inp = utils.get_1d_posencode_inp(H, rank//2)
    v_inp = utils.get_1d_posencode_inp(W, rank//2)
    w_inp = utils.get_1d_posencode_inp(T, rank//2)
    
    H1 = rank
    W1 = rank
    T1 = rank
    factor = rank
   
    # Generate networks
    if nettype == 'unet':
        u_net = models.UNetND(H1, H1, 1, 16).cuda()    
        v_net = models.UNetND(W1, W1, 1, 16).cuda()
        w_net = models.UNetND(T1, T1, 1, 16).cuda()
    elif nettype == 'dip':
        u_net = deep_prior.get_net(H1, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=H1).cuda()
        v_net = deep_prior.get_net(W1, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=W1).cuda()
        w_net = deep_prior.get_net(T1, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=init_nconv,
                                    skip_n33u=init_nconv,
                                    num_scales=num_channels_up,
                                    n_channels=T1).cuda()
    else:
        u_net = deep_decoder.decodernw1d(H1, [H//8, 64, 64],
                                         filter_size_up=3).cuda()
        v_net = deep_decoder.decodernw1d(W1, [W//8, 64, 64],
                                         filter_size_up=3).cuda()
        w_net = deep_decoder.decodernw1d(T1, [T//8, 64, 64],
                                         filter_size_up=3).cuda()
        # Deep decoder requires smaller inputs
        u_inp = utils.get_inp([1, H//8, H//8])
        v_inp = utils.get_inp([1, W//8, W//8])
        w_inp = utils.get_inp([1, T//8, T//8])
        
    # Switch to trairning mode
    u_net.train()
    v_net.train()
    w_net.train()
    
    net_params = list(u_net.parameters()) + list(v_net.parameters()) +\
                 list(w_net.parameters()) 
    inp_params = [u_inp] + [v_inp] + [w_inp]
    
    params = net_params + inp_params
             
    core = utils.get_inp(rank)    
    with torch.no_grad():
        core[...] = 1/rank
    params += [core]
    
    criterion_l1 = losses.L2Norm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)    
    time_array = np.zeros(epochs)
    
    X, Y = np.mgrid[:H, :W]
    mask = (np.hypot((X-H/2), (Y - W/2)) < min(H, W)/2).astype(np.float32)
    maskten = torch.tensor(mask)[..., None].cuda()
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    # Create a learning scheduler
    scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)
    
    u_inp_per = u_inp.detach().clone()
    v_inp_per = v_inp.detach().clone()
    w_inp_per = w_inp.detach().clone()
    
    best_loss = float('inf')
    best_epoch = 0
    tic = time.time()
    
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        # Perturbe inputs
        u_inp_perturbed = u_inp + u_inp_per.normal_()*reg_noise_std
        v_inp_perturbed = v_inp + v_inp_per.normal_()*reg_noise_std
        w_inp_perturbed = w_inp + w_inp_per.normal_()*reg_noise_std
        
        Uk = u_net(u_inp_perturbed)
        Vk = v_net(v_inp_perturbed)
        Wk = w_net(w_inp_perturbed)
        
        factors = [Uk.T[..., 0], Vk.T[..., 0], Wk.T[..., 0]]
        
        tensor_estim = tensorly.cp_to_tensor((core, factors))
        
        measurements_estim = lin_inverse.radon(
            tensor_estim[None, ...].permute(0, 3, 1, 2),
                                               angles)
        
        loss = criterion_l1(measurements - measurements_estim)
        
        mse_array[idx] = ((tensor_estim - cube_gt_ten)**2).mean().item()
        loss_array[idx] = loss.item()
        time_array[idx] = time.time() - tic
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        if loss_array[idx] < best_loss:
            best_loss = loss_array[idx]
            best_epoch = idx
            best_cube_estim = copy.deepcopy(tensor_estim.detach().squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            img_idx = idx%T
            ref = cube[..., img_idx]
            diff = abs(cube_gt_ten - tensor_estim).mean(2)
            meas_diff = abs(measurements - measurements_estim).mean(0)
            diff = diff.squeeze().detach().cpu()
            meas_diff = meas_diff.detach().cpu()
            img = tensor_estim[..., img_idx].detach().cpu().numpy()

            if sys.platform == 'win32':
                cv2.imshow('Rec Diff x10', diff.numpy()*10)
                cv2.imshow('Meas Diff x10', 10*meas_diff.numpy()/maxval)
                cv2.imshow('Rec', np.hstack((ref, img)))
                cv2.waitKey(1)

    cube_estim = best_cube_estim.detach().squeeze().cpu().numpy()
    
    psnrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    
    print('PSNR: %.2f'%psnrval)
    print('SSIM: %.2f'%ssimval)
    
