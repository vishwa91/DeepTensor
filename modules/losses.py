#!/usr/bin/env python

'''
    This script implements all loss functions utilized in our experiments 
    including L1 loss, L2 loss, and TV norm.
'''

import torch


class TVNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            return torch.sqrt(grad_x**2 + grad_y**2).mean()
        elif self.mode == 'l1':
            return abs(grad_x).mean() + abs(grad_y).mean()
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).mean()     
    
class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return abs(x).mean()        

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return (x.pow(2)).mean()    
