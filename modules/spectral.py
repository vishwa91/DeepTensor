#!/usr/bin/env python

'''
    This module implements functions required for hyperspectral image experiments
    in section 4.
'''

import numpy as np
import torch

# We will use tensorly to compute outer product
import tensorly
tensorly.set_backend('pytorch')
from tensorly import decomposition

from scipy.sparse import linalg

def lr_decompose(cube, rank):
    '''
        Perform a truncated SVD
        
        Inputs:
            cube: (H, W, nwvl) hyperspectral cube
            rank: Rank to decompose the cube
            
        Outputs:
            cube_lr: Low rank decomposition
    '''
    H, W, nwvl = cube.shape
    hsmat = cube.reshape(H*W, nwvl)
    
    u, s, vt = linalg.svds(hsmat, k=rank)
    
    hsmat_lr = u.dot(np.diag(s)).dot(vt)
    cube_lr = hsmat_lr.reshape(H, W, nwvl)
    
    return cube_lr

def tucker_decompose(cube, rank, max_iters=1000):
    '''
        Perform a truncated tucker decomposition
        
        Inputs:
            cube: (H, W, T) numpy array
            rank: Rank for tucker decomposition
            max_iters: Maximum iterations for tucker decomposition
            
        Outputs:
            cube_lr: Low tucker rank decompositoin
    '''
    cube_ten = torch.tensor(cube).cuda()
    tucker_core, tucker_factors = decomposition.tucker(cube_ten,
                                                       (rank, rank, rank), 
                                                       n_iter_max=max_iters)

    cube_approx_tucker = tensorly.tucker_to_tensor((tucker_core,
                                                    tucker_factors))
    cube_approx_tucker = cube_approx_tucker.cpu().numpy()
    
    return cube_approx_tucker
    