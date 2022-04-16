#!/usr/bin/env python

'''
    Miscellaneous utilities that are required in all experiments.
'''

# System imports
import pickle

import torch

# Scientific computing
import numpy as np
import scipy.linalg as lin
from scipy import io
from scipy import signal
from scipy.sparse import linalg

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
except ModuleNotFoundError:
    from .cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1)) + 1e-12
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2)) + 1e-12

    snrval = 10*np.log10(np.max(x)/denom)

    return snrval

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def rician(sig, noise_snr):
    '''
        Add Rician noise
        
        Inputs:
            sig: N dimensional signal
            noise_snr: std. dev of input noise
            
        Outputs:
            sig_rician: Rician corrupted signal
    '''
    n1 = np.random.randn(*sig.shape)*noise_snr
    n2 = np.random.randn(*sig.shape)*noise_snr

    return np.sqrt((sig + n1)**2 + n2**2)

def resize(cube, scale):
    '''
        Resize a multi-channel image
        
        Inputs:
            cube: (H, W, nchan) image stack
            scale: Scaling 
    '''
    H, W, nchan = cube.shape
    
    im0_lr = cv2.resize(cube[..., 0], None, fx=scale, fy=scale)
    Hl, Wl = im0_lr.shape
    
    cube_lr = np.zeros((Hl, Wl, nchan), dtype=cube.dtype)
    
    for idx in range(nchan):
        cube_lr[..., idx] = cv2.resize(cube[..., idx], None,
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
    return cube_lr

def moduloclip(cube, mulsize):
    '''
        Clip a cube to have multiples of mulsize
        
        Inputs:
            cube: (H, W, T) sized cube
            mulsize: (h, w) tuple having smallest stride size
            
        Outputs:
            cube_clipped: Clipped cube with size equal to multiples of h, w
    '''
    H, W, T = cube.shape
    
    H1 = mulsize[0]*(H // mulsize[0])
    W1 = mulsize[1]*(W // mulsize[1])
    
    cube_clipped = cube[:H1, :W1, :]
    
    return cube_clipped

def implay(cube, delay=20):
    '''
        Play hyperspectral image as a video
    '''
    if cube.dtype != np.uint8:
        cube = (255*cube/cube.max()).astype(np.uint8)
    
    T = cube.shape[-1]
    
    for idx in range(T):
        cv2.imshow('Video', cube[..., idx])
        cv2.waitKey(delay)
        
def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = images[cnt, ...]
            cnt += 1
    
    return montage_im
        
def get_matrix(nrows, ncols, rank, noise_type, signal_type,
               noise_snr=5, tau=1000):
    '''
        Get a matrix for simulations
        
        Inputs:
            nrows, ncols: Size of the matrix
            rank: Rank of the matrix
            noise_type: Type of the noise to add. Currently None, gaussian,
                and poisson
            signal_type: Type of the signal itself. Currently gaussian and
                piecewise constant
            noise_snr: Amount of noise to add in terms of 
    '''
    if signal_type == 'gaussian':
        U = np.random.randn(nrows, rank)
        V = np.random.randn(rank, ncols)
    elif signal_type == 'piecewise':
        nlocs = 10
        
        U = np.zeros((nrows, rank))
        V = np.zeros((rank, ncols))
        
        for idx in range(rank):
            u_locs = np.random.randint(0, nrows, nlocs)
            v_locs = np.random.randint(0, ncols, nlocs)
            
            U[u_locs, idx] = np.random.randn(nlocs)
            V[idx, v_locs] = np.random.randn(nlocs)
        
        U = np.cumsum(U, 0)
        V = np.cumsum(V, 1)
    else:
        raise AttributeError('Signal type not implemented')
    
    mat = normalize(U.dot(V), True)
    
    if noise_type == 'gaussian':
        mat_noisy = measure(mat, noise_snr, float('inf'))
    elif noise_type == 'poisson':
        mat_noisy = measure(mat, noise_snr, tau)
    elif noise_type == 'rician':
        noise1 = np.random.randn(nrows, ncols)*noise_snr
        noise2 = np.random.randn(nrows, ncols)*noise_snr

        mat_noisy = np.sqrt((mat + noise1)**2 + noise2**2)
    else:
        raise AttributeError('Noise type not implemented')
    
    return mat_noisy, mat

def get_pca(nrows, ndata, rank, noise_type, signal_type,
            noise_snr=5, tau=1000):
    '''
        Get PCA data
        
        Inputs:
            nrows: Number of rows in data
            ndata: Number of data points
            rank: Intrinsic dimension
            noise_type: Type of the noise to add. Currently None, gaussian,
                and poisson
            signal_type: Type of the signal itself. Currently gaussian and
                piecewise constant
            noise_snr: Amount of noise to add in terms of 
    '''
    # Generate normalized coefficients
    coefs = np.random.randn(rank, ndata)
    coefs_norm = np.sqrt((coefs*coefs).sum(0)).reshape(1, ndata)
    coefs = coefs/coefs_norm

    if signal_type == 'gaussian':
        basis = np.random.randn(nrows, rank)
    elif signal_type == 'piecewise':
        nlocs = 10
        
        basis = np.zeros((nrows, rank))
        
        for idx in range(rank):
            u_locs = np.random.randint(0, nrows, nlocs)            
            basis[u_locs, idx] = np.random.randn(nlocs)
        
        basis = np.cumsum(basis, 0)
    else:
        raise AttributeError('Signal type not implemented')
    
    # Compute orthogonal basis with QR decomposition
    basis, _ = np.linalg.qr(basis)
    mat = basis.dot(coefs)
    
    if noise_type == 'gaussian':
        mat_noisy = measure(mat, noise_snr, float('inf'))
    elif noise_type == 'poisson':
        mat_noisy = measure(mat, noise_snr, tau)
    elif noise_type == 'rician':
        noise1 = np.random.randn(nrows, ndata)*noise_snr
        noise2 = np.random.randn(nrows, ndata)*noise_snr

        mat_noisy = np.sqrt((mat + noise1)**2 + noise2**2)
    else:
        raise AttributeError('Noise type not implemented')
    
    return mat_noisy, mat, basis

def get_inp(tensize, const=10.0):
    '''
        Wrapper to get a variable on graph
    '''
    inp = torch.rand(tensize).cuda()/const
    inp = torch.autograd.Variable(inp, requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp)
    
    return inp

def get_2d_posencode_inp(H, W, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X, Y = np.mgrid[:H, :W]
    coords = np.hstack(((10*X/H).reshape(-1, 1), (10*Y/W).reshape(-1, 1)))
    
    freqs = np.random.randn(2, n_inputs)
    
    angles = coords.dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    
    inp = posencode_vals.reshape(H, W, 2*n_inputs)
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp.permute(2, 0, 1)[None, ...])
    
    return inp

def get_1d_posencode_inp(H, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X = 10*np.arange(H).reshape(-1, 1)/H
    
    freqs = np.random.randn(1, n_inputs)
    
    angles = X.dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    
    inp = posencode_vals.reshape(H, 2*n_inputs)
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp.permute(1, 0)[None, ...])
    
    return inp  

def lr_decompose(mat, rank=6):
    '''
        Low rank decomposition
    '''
    u, s, vt = linalg.svds(mat, k=rank)
    mat_lr = u.dot(np.diag(s)).dot(vt)
    
    return mat_lr

def get_scheduler(scheduler_type, optimizer, args):
    '''
        Get a scheduler 
        
        Inputs:
            scheduler_type: 'none', 'step', 'exponential', 'cosine'
            optimizer: One of torch.optim optimizers
            args: Namspace containing arguments relevant to each optimizer
            
        Outputs:
            scheduler: A torch learning rate scheduler
    '''
    if scheduler_type == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.epochs)
    elif scheduler_type == 'step':
        # Compute gamma 
        gamma = pow(10, -1/(args.epochs/args.step_size))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.step_size,
                                                    gamma=gamma)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=args.gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=200,
                                                cycle_mult=1.0,
                                                max_lr=args.max_lr,
                                                min_lr=args.min_lr,
                                                warmup_steps=50,
                                                gamma=args.gamma)
        
    return scheduler