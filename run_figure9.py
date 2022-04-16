#!/usr/bin/env python
#


'''
    This script replicates the DeepTensor results in Figure 6. The accuracy is
    different from the paper as hyperspectral image was downsampled.
'''

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

sys.path.append('modules')

import models
import utils
import losses
import spectral
import deep_prior
import deep_decoder

if __name__ == '__main__':
    expname = 'icvl'
    rank = 20
    n_inputs = rank
    init_nconv = 128
    num_channels_up = 5
    nettype = 'dip'
    
    scaling = 1
    tau = 100
    noise_snr = 2
    
    # Learning constants
    learning_rate = 1e-3
    epochs = 5000
    reg_noise_std = 1.0/30.0
    exp_weight = 0.99
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    wavelengths = data['wavelengths'].astype(np.float32).ravel()
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, nwvl = cube.shape

    if rank == 'max':
        rank = nwvl - 1
        n_inputs = rank
    
    hsmat = cube.reshape(H*W, nwvl)
    
    cube_noisy = utils.measure(cube, noise_snr, tau)
    
    hsten = torch.tensor(hsmat)[None, ...].cuda()
    hsten_noisy = torch.tensor(cube_noisy.reshape(H*W, nwvl))[None, ...].cuda()
    
    criterion_l1 = losses.L1Norm()
    
    loss_array = np.zeros(epochs)
    mse_array = np.zeros(epochs)

    # Generate networks
    if nettype == 'unet':
        im_net = models.UNetND(n_inputs, rank, 2, init_nconv).cuda()    
        spec_net = models.UNetND(n_inputs, rank, 1, init_nconv).cuda()
        
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, nwvl])
    elif nettype == 'dip':
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                    upsample_mode='linear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=rank).cuda()
        im_inp = utils.get_inp([1, n_inputs, H, W])
        spec_inp = utils.get_inp([1, n_inputs, nwvl])
    elif nettype == 'dd':
        nchans = [init_nconv]*num_channels_up
        im_net = deep_decoder.decodernw(rank).cuda()
        spec_net = deep_decoder.decodernw1d(rank).cuda()
        
        H1 = H // pow(2, num_channels_up)
        W1 = W // pow(2, num_channels_up)
        nwvl1 = nwvl // pow(2, num_channels_up)
        im_inp = utils.get_inp([1, init_nconv, H1, W1])
        spec_inp = utils.get_inp([1, init_nconv, nwvl1])
    
    # Switch to training mode
    im_net.train()
    spec_net.train()
    
    net_params = list(im_net.parameters()) + list(spec_net.parameters())
    inp_params = [im_inp] + [spec_inp]
    
    im_inp_per = im_inp.detach().clone()
    spec_inp_per = spec_inp.detach().clone()
    
    hs_estim_avg = None
    best_loss = float('inf')
    best_epoch = 0
    
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    tic = time.time()
    tbar = tqdm.tqdm(range(epochs))
    for idx in tbar:        
        # Perturb inputs
        im_inp_perturbed = im_inp + im_inp_per.normal_()*reg_noise_std
        spec_inp_perturbed = spec_inp + spec_inp_per.normal_()*reg_noise_std
        
        U_img = im_net(im_inp_perturbed)
        V = spec_net(spec_inp_perturbed)
        
        U = U_img.reshape(-1, rank, H*W).permute(0, 2, 1)
        
        hs_estim = torch.bmm(U, V)
        
        loss = criterion_l1(hsten_noisy - hs_estim)
        loss_array[idx] = loss.item()
        mse_array[idx] = ((hs_estim - hsten)**2).mean().item()
        
        tbar.set_description('%.4e'%mse_array[idx])
        tbar.refresh()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx
            best_hs_estim = copy.deepcopy(hs_estim.detach().squeeze())
        
            # Averaging as per original code    
            if hs_estim_avg is None:
                hs_estim_avg = hs_estim.detach()
            else:
                hs_estim_avg = exp_weight*hs_estim_avg +\
                            (1 - exp_weight)*hs_estim.detach()
        
        if sys.platform == 'win32':
            im_idx = idx%nwvl
            with torch.no_grad():
                diff = abs(hs_estim - hsten).mean(2).squeeze().detach().cpu()
                avg = hs_estim.mean(2).squeeze().detach().cpu().numpy()
                img = hs_estim[0, :, im_idx].detach().cpu().numpy().reshape(H, W)
                cv2.imshow('Diff x10', diff.numpy().reshape(H, W)*10)
                cv2.imshow('Band',
                        np.sqrt(abs(np.hstack((cube[..., im_idx], img)))))
                cv2.waitKey(1)
            
    toc = time.time()
    
    cube_estim = best_hs_estim.cpu().numpy().reshape(H, W, nwvl)
    cube_lr = spectral.lr_decompose(cube_noisy, rank)
    diff = abs(hs_estim - hsten).mean(2).squeeze().detach().cpu()
   
    wvl_idx = nwvl // 2
    
    plt.subplot(2, 2, 1)
    plt.imshow(cube[..., wvl_idx]); plt.title('Ground truth')
    
    plt.subplot(2, 2, 2)
    snrval = utils.psnr(cube, cube_lr)
    ssimval = ssim_func(cube, cube_lr, multichannel=True)
    plt.imshow(cube_lr[..., wvl_idx])
    plt.title('SVD | %.1f dB | %.2f'%(snrval, ssimval))
    
    plt.subplot(2, 2, 3)    
    snrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    plt.imshow(cube_estim[..., wvl_idx])
    plt.title('DeepTensor | %.1f dB | %.2f'%(snrval, ssimval))
    
    plt.subplot(2, 2, 4)
    plt.imshow(diff.numpy().reshape(H, W))
    plt.title('Diff image')
    plt.colorbar()
    
    plt.show()
