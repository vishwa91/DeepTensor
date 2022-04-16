#!/usr/bin/env python

'''
    This script replicates the TV results in Figure 8. The accuracy is
    different from the paper as PET scan was downsampled.
'''

import sys
import tqdm

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch
import tensorly
tensorly.set_backend('pytorch')

import matplotlib.pyplot as plt
plt.gray()
import cv2

sys.path.append('modules')

import utils
import losses
import lin_inverse

if __name__ == '__main__':
    expname = 'pet'
    nangles = 40            # Number of measurementes per z-slice
    rank = 1000
    
    # Network parameters
    uv_decompose = False     # Set True to decompose the 3D volume into PARAFAC tensor
        
    # Noise parameters
    scaling = 1
    tau = 1000
    noise_snr = 2
    
    # Learning constants
    learning_rate = 1e-1
    epochs = 100
    lambda_tv = 1e0
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)[:, :, :288]
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, T = cube.shape

    if rank == 'max':
        rank = min(H, W, T)
        n_inputs = rank
    
    cube_noisy = utils.measure(cube, noise_snr, tau).astype(np.float32)
    
    # Send data to device
    cube_gt_ten = torch.tensor(cube).cuda()
    cube_ten = torch.tensor(cube).cuda().permute(2, 0, 1)[None, ...]
    angles = torch.tensor(np.linspace(0, 180, nangles).astype(np.float32)).cuda()
    
    # Generate sinogram
    measurements = lin_inverse.radon(cube_ten, angles).detach().cpu().numpy()
    measurements = utils.measure(measurements, noise_snr, tau)
    measurements = torch.tensor(measurements).cuda()
    
    if uv_decompose:
        U = utils.get_inp([H, rank])
        V = utils.get_inp([W, rank])
        W = utils.get_inp([T, rank])
        core = utils.get_inp(rank)
        
        params = [U] + [V] + [W] + [core]
    else:
        tensor_estim = utils.get_inp([H, W, T])
        params = [tensor_estim]    
        
    criterion_l1 = losses.L2Norm()
    criterion_tv = losses.TVNorm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)    
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:
        if uv_decompose:            
            factors = [U, V, W]
            tensor_estim = tensorly.cp_to_tensor((core, factors))
            tensor_estim = tensor_estim
        
        measurements_estim = lin_inverse.radon(
            tensor_estim[None, ...].permute(0, 3, 1, 2),
                                               angles)
        
        loss1 = criterion_l1(measurements - measurements_estim)
        loss2 = criterion_tv(tensor_estim)
        
        loss = loss1 + lambda_tv*loss2
        
        mse_array[idx] = ((tensor_estim - cube_gt_ten)**2).mean().item()
        
        tbar.set_description('%.4e'%mse_array[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            img_idx = idx%T
            ref = cube[..., img_idx]
            diff = abs(cube_gt_ten - tensor_estim).mean(2).squeeze().detach().cpu()
            img = tensor_estim[..., img_idx].detach().cpu().numpy()

            if sys.platform == 'win32':
                cv2.imshow('Diff x10', diff.numpy()*10)
                cv2.imshow('Avg', np.hstack((ref, img)))
                cv2.waitKey(1)
            
    cube_estim = tensor_estim.detach().squeeze().cpu().numpy()
    
    psnrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    
    print('PSNR: %.2f'%psnrval)
    print('SSIM: %.2f'%ssimval)
    
