#!/usr/bin/env python
#


'''
    This script replicates the TV results in Figure 7. The accuracy is
    different from the paper as the video has been downsampled and clipped to
    fit supplementary material requirements.
'''


import sys
import tqdm

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import torch

import matplotlib.pyplot as plt
plt.gray()
import cv2

sys.path.append('modules')

import utils
import losses
import lin_inverse

if __name__ == '__main__':
    expname = 'cat_video'
    nframes = 8
    rank = 200
    
    # Network parameters
    uv_decompose = False        # Set this to true to optimize U, V instead
    
    # Noise parameters
    scaling = 1
    tau = 100
    noise_snr = 2
    
    # Learning constants
    learning_rate = 1e-1
    epochs = 500
    lambda_tv = 1e-2
    
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
    
    # Generate variables
    if uv_decompose:
        U = utils.get_inp([1, H*W, rank])
        V = utils.get_inp([1, rank, totalframes])
        params = [U] + [V]
    else:
        mat_estim = utils.get_inp([1, H*W, totalframes])
        params = [mat_estim]
    
    criterion_l1 = losses.L2Norm()
    criterion_tv = losses.TVNorm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)    
    
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    for idx in tqdm.tqdm(range(epochs)):
        if uv_decompose:
            U_img = U.reshape(1, H, W, rank).permute(3, 0, 1, 2)
            
            mat_estim = torch.bmm(U, V)
        else:
            U_img = mat_estim
        cube_estim = mat_estim.reshape(1, H, W, totalframes).permute(0, 3, 1, 2)
        
        meas_estim = lin_inverse.video2codedvideo(cube_estim, masks_ten, nframes)
        
        loss1 = criterion_l1(measurements_ten - meas_estim)
        loss2 = criterion_tv(U_img)
        
        loss = loss1 + lambda_tv*loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            img_idx = idx%totalframes
            ref = cube[..., img_idx]
            diff = abs(cube_ten - cube_estim).mean(2).squeeze().detach().cpu()
            img = cube_estim[0, img_idx, ...].detach().cpu().numpy()
            cv2.imshow('Diff x10', diff.numpy()*10)
            cv2.imshow('Avg', np.hstack((ref, img)))
            cv2.waitKey(1)
            
    cube_estim = cube_estim.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    
    psnrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    
    print('PSNR: %.2f'%psnrval)
    print('SSIM: %.2f'%ssimval)