#!/usr/bin/env python

'''
This script replicates results in Figure 2 of our main paper.
'''

import os
import sys
import tqdm
import importlib
import copy
import argparse

import numpy as np
from scipy import io
from scipy.sparse import linalg
from skimage.metrics import structural_similarity as ssim_func

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score as apfunc
from sklearn.decomposition import PCA, FastICA
from sklearn.svm import SVC

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

def do_pca_and_svm(X_train, y_train, X_test, n_components, param_grid):
    '''
        Just a combined wrapper for PCA + SVM
    '''
    
    X_train_centered = X_train - X_train.mean(1).reshape(-1, 1)
    X_train_centered /= X_train_centered.std(1).reshape(-1, 1)
    
    covmat = X_train_centered.T.dot(X_train_centered)
    _, eigenfaces = linalg.eigsh(covmat, k=n_components)
    eigenfaces = eigenfaces.T   
    
    y_pred = do_svm(eigenfaces, X_train, y_train, X_test, param_grid)
    
    return eigenfaces, y_pred

def do_svm(eigenfaces, X_train, y_train, X_test, param_grid):
    '''
        Just do SVM
    '''
    
    X_train_centered = X_train - X_train.mean(1).reshape(-1, 1)
    
    X_test_centered = X_test - X_test.mean(1).reshape(-1, 1)
    
    X_train_proj = X_train_centered.dot(eigenfaces.T)
    X_test_proj = X_test_centered.dot(eigenfaces.T)
    
    # Train a SVM classification model for pca
    clf = GridSearchCV(
        SVC(kernel='linear', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train_proj, y_train)

    # Quantitative evaluation of the model quality on the test set
    y_pred = clf.predict(X_test_proj)
    
    return y_pred

def do_ica_and_svm(X_train, y_train, X_test, n_components, param_grid):
    '''
        Independent component analysis
    '''
    ica = FastICA(n_components=n_components, random_state=0).fit(X_train)
    eigenfaces = ica.components_
    
    y_pred = do_svm(eigenfaces, X_train, y_train, X_test, param_grid)
    
    return eigenfaces, y_pred

def average_precision(y_test, y_pred, n_classes):
    return apfunc(np.eye(n_classes)[y_test], np.eye(n_classes)[y_pred])

if __name__ == '__main__':
    expname = 'weizzman'
    n_components = 84
    train_size = 0.25
    scale = 1
    
    # Noise constants
    tau = 20
    noise_snr = 2
    
    # Network constants
    n_inputs = n_components
    init_nconv = 128
    num_channels_up = 5
    nettype = 'dip'    
    
    # SVM constants
    param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0, 10.0], }
    
    # Learning constants
    sched_args = argparse.Namespace()
    scheduler_type = 'none'
    learning_rate = 1e-3
    epochs = 100
    reg_noise_std = 1.0/30.0
    exp_weight = 0.9 
    sched_args.step_size = 2000
    sched_args.gamma = pow(10, -2/epochs)
    sched_args.max_lr = learning_rate
    sched_args.min_lr = 1e-6
    sched_args.epochs = epochs 
    
    # Load data
    if expname == 'lfw':
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
        n_samples, h, w = lfw_people.images.shape
        
        X = lfw_people.data/255.0
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]
    elif expname == 'olivetti':
        X, y = fetch_olivetti_faces(return_X_y=True)
        h = 64
        w = 64
        n_classes = np.unique(y).size
        n_samples = X.shape[0]
        
    else:
        data = io.loadmat('data/weizzman.mat')
        faces = data['faces'].astype(np.float32)/255
        y = data['labels'].ravel()
        n_classes = np.unique(y).size
        
        names = data['names'].ravel()
        target_names = [name[0] for name in names]
        
        # Resize to avoid flooding GPU
        data = np.transpose(utils.resize(np.transpose(faces, [1, 2, 0]), scale),
                            [2, 0, 1])
        n_samples, h, w = data.shape
        X = data.reshape(n_samples, h*w)
            
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42)
    
    # Corrupt data after splitting so that we can keep track of clean PCA
    X_train_noisy = utils.measure(X_train, noise_snr, tau)
    X_test_noisy = utils.measure(X_test, noise_snr, tau)
    
    # Generate networks
    if nettype == 'unet':
        eigen_net = models.UNetND(n_inputs, n_components, 2,
                                  init_nconv).cuda()            
        net_inp = utils.get_inp([1, n_inputs, h, w])
    elif nettype == 'dip':
        eigen_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=n_components).cuda()
        net_inp = utils.get_inp([1, n_inputs, h, w])
    elif nettype == 'dd':
        nchans = [init_nconv]*num_channels_up
        eigen_net = deep_decoder.decodernw(n_components).cuda()
        
        H1 = h // pow(2, num_channels_up)
        W1 = w // pow(2, num_channels_up)
        net_inp = utils.get_inp([1, init_nconv, H1, W1])
    
    # Clean data
    print('Generating eigenfaces on clean data')
    eigenfaces_gt, y_gt = do_ica_and_svm(X_train, y_train, X_test, n_components,
                                         param_grid)
    
    print('Generating eigenfaces on noisy data')
    eigenfaces_ica, y_ica = do_ica_and_svm(X_train_noisy, y_train, X_test_noisy,
                                           n_components, param_grid)
    
    print('Computing ICA components')
    eigenfaces_pca, y_pca = do_pca_and_svm(X_train_noisy, y_train, X_test_noisy,
                                           n_components, param_grid)
    
    ## Part two DeepTensor
    # Compute covariance matrix
    print('Now starting DeepTensor')
    X_train_noisy_centered = X_train_noisy - X_train_noisy.mean(1).reshape(-1, 1)
    
    covmat = X_train_noisy_centered.T.dot(X_train_noisy_centered)
    covmat_ten = torch.tensor(covmat)[None, ...].cuda()
    
    net_params = list(eigen_net.parameters())
    inp_params = [net_inp]
    
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    # Create a learning scheduler
    scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)
    
    net_inp_delta = net_inp.detach().clone()
    
    criterion_l1 = losses.L2Norm()
    loss_array = np.zeros(epochs)
    best_loss = float('inf')
    best_epoch = 0
    
    eigenimg_array = np.zeros((h, w, n_components, epochs), dtype=np.float32)
    
    for idx in tqdm.tqdm(range(epochs)):
        # Perturb inputs
        net_inp_perturbed = net_inp + net_inp_delta.normal_()*reg_noise_std
            
        eigenimg = eigen_net(net_inp_perturbed)
        eigenvec = eigenimg.reshape(1, n_components, h*w)
                
        covmat_estim = torch.bmm(eigenvec.permute(0, 2, 1), eigenvec)
        
        loss = criterion_l1(covmat_ten - covmat_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_array[idx] = loss.item()
        
        tmp = eigenimg[0, ...].permute(1, 2, 0).detach().cpu().numpy()
        eigenimg_array[:, :, :, idx] = tmp
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx

            best_eigenvec = copy.deepcopy(eigenvec.detach())
            best_covmat = copy.deepcopy(covmat_estim.detach())
            
        if sys.platform == 'win32':            
            img = utils.build_montage(eigenimg.squeeze().detach().cpu().numpy())
            
            cv2.imshow('Eigenvector', utils.normalize(img, True))
            cv2.waitKey(1)
    
    covmat_dlrf = best_covmat.squeeze().cpu().numpy()
    _, eigenface_dlrf = linalg.eigsh(covmat_dlrf, k=n_components)
    eigenface_dlrf = eigenface_dlrf.T
    
    # Now predict
    y_dlrf = do_svm(eigenface_dlrf, X_train_noisy, y_train, X_test_noisy,
                    param_grid)
        
    accuracy_gt = (y_test == y_gt).sum()/y_test.size
    accuracy_pca = (y_test == y_pca).sum()/y_test.size
    accuracy_ica = (y_test == y_ica).sum()/y_test.size
    accuracy_dlrf = (y_test == y_dlrf).sum()/y_test.size
        
    print('Noiseless PCA accuracy: %.2f'%((y_test == y_gt).sum()/y_test.size))
    print('Noiseless PCA mAP: %.2f'%average_precision(y_test, y_gt, n_classes))
    print('')
    
    print('PCA accuracy: %.2f'%((y_test == y_pca).sum()/y_test.size))
    print('PCA mAP: %.2f'%average_precision(y_test, y_pca, n_classes))
    print('')
    
    print('ICA accuracy: %.2f'%((y_test == y_ica).sum()/y_test.size))
    print('ICA mAP: %.2f'%average_precision(y_test, y_ica, n_classes))
    print('')
    
    print('DeepTensor accuracy: %.2f'%((y_test == y_dlrf).sum()/y_test.size))
    print('DeepTensor mAP: %.2f'%average_precision(y_test, y_dlrf, n_classes))