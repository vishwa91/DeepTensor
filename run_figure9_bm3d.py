#!/usr/bin/env python


'''
    This script replicates the BM3D results in Figure 6. The accuracy is
    different from the paper as the hyperspectral image has been downsampled.
'''


import sys
import tqdm

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
import bm3d

import matplotlib.pyplot as plt
plt.gray()

sys.path.append('modules')
import utils

if __name__ == '__main__':
    expname = 'icvl'
    
    scaling = 1
    tau = 100
    noise_snr = 2
    
    sigma_psd = int(np.sqrt(tau))/255
    
    # Load data
    data = io.loadmat('data/%s.mat'%expname)
    cube = data['hypercube'].astype(np.float32)
    cube = utils.resize(cube/cube.max(), scaling)
    
    H, W, nwvl = cube.shape    
    cube_noisy = utils.measure(cube, noise_snr, tau)
    
    cube_estim = np.zeros_like(cube_noisy)
    
    for idx in tqdm.tqdm(range(nwvl)):
        denoised_img = bm3d.bm3d(cube_noisy[..., idx],
                                 sigma_psd=sigma_psd,
                                 stage_arg=bm3d.BM3DStages.ALL_STAGES)
        cube_estim[..., idx] = denoised_img    
    
    snrval = utils.psnr(cube, cube_estim)
    ssimval = ssim_func(cube, cube_estim, multichannel=True)
    
    plt.subplot(2, 2, 1)
    plt.imshow(cube[..., nwvl//2])
    plt.title('Ground truth')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cube_noisy[..., nwvl//2])
    plt.title('Noisy')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cube_estim[..., nwvl//2])
    plt.title('BM3D denoised output')
    
    plt.subplot(2, 2, 4)
    plt.imshow(abs(cube - cube_estim).mean(2), cmap='jet')
    plt.colorbar()
    plt.title('Absolute error')
    
    print(snrval, ssimval)
    plt.show()