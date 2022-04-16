#!/usr/bin/env python

'''
    This script replicates results in table 1 in the main paper.
    
    Note: Running this script downloads datasets that can be up to 2.5GB in size.
    Make sure you have enough disk space before running this script.
'''

import sys
import tqdm
import argparse
import torchaudio
import numpy as np


import torch

import matplotlib.pyplot as plt

sys.path.append("modules")
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import NMF
import models
import utils
import losses
import deep_prior
import deep_decoder

import numpy as np
import logging
import logging.config
import scipy.sparse
from numpy.linalg import eigh


def eighk(M, k=0):
    """Returns ordered eigenvectors of a squared matrix. Too low eigenvectors
    are ignored. Optionally only the first k vectors/values are returned.
    Arguments
    ---------
    M - squared matrix
    k - (default 0): number of eigenvectors/values to return
    Returns
    -------
    w : [:k] eigenvalues
    v : [:k] eigenvectors
    """
    values, vectors = eigh(M)

    # get rid of too low eigenvalues
    s = np.where(values > _EPS)[0]
    vectors = vectors[:, s]
    values = values[s]

    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:, :k]

    return values, vectors


def cmdet(d):
    """Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.
    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)
    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0] + 1, d.shape[0] + 1))
    D[0, 0] = 0.0
    D[1:, 1:] = d ** 2
    j = np.float32(D.shape[0] - 2)
    f1 = (-1.0) ** (j + 1) / ((2 ** j) * ((factorial(j)) ** 2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))


def simplex(d):
    """Computed the volume of a simplex S given by a coordinate matrix D.
    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)
    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0] + 1, d.shape[1]))
    D[1:, :] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from providing some basic methods.
    """

    # some small value

    _EPS = 1e-10

    def __init__(self, data, num_bases=4, **kwargs):
        """ """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases

        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """Returns the residual in % of the total amount of data
        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.
        Returns:
        -------
        frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self, "H") and hasattr(self, "W"):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """Initalize W to random values [0,1]."""
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10 ** -4

    def _init_h(self):
        """Initalize H to random values [0,1]."""
        self.H = np.random.random((self._num_bases, self._num_samples)) + 10 ** -4

    def _update_h(self):
        """Overwrite for updating H."""
        pass

    def _update_w(self):
        """Overwrite for updating W."""
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.
        Parameters
        ----------
            i   : index of the update step
        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(
        self,
        niter=100,
        show_progress=False,
        compute_w=True,
        compute_h=True,
        compute_err=True,
        epoch_hook=None,
    ):
        """Factorize s.t. WH = data
        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].
        epoch_hook : function
                If this exists, evaluate it every iteration
        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, "W") and compute_w:
            self._init_w()

        if not hasattr(self, "H") and compute_h:
            self._init_h()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                self._logger.info("FN: %s (%s/%s)" % (self.ferr[i], i + 1, niter))
            else:
                self._logger.info("Iteration: (%s/%s)" % (i + 1, niter))

            if epoch_hook is not None:
                epoch_hook(self)

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class SNMF(PyMFBase):
    """
    SNMF(data, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying Semi-NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.factorize(niter=10)

    The basis vectors are now stored in snmf_mdl.W, the coefficients in snmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to snmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.W = W
    >>> snmf_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        self.W = np.dot(W1, np.linalg.inv(W2))

    def _update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10 ** -9

        self.H *= np.sqrt(H1 / H2)


DATASET = "TFR"


def get_nmf(data, noisy_data, n_clusters):
    flat_data = data.reshape((data.shape[0], -1))
    flat_noisy_data = noisy_data.reshape((noisy_data.shape[0], -1))
    scores = []
    reconstructions = []
    criterion = losses.L2Norm()

    print('Running NMF')
    nmf = NMF(n_components=n_clusters, alpha=0.0, l1_ratio=0.0)
    nmf.fit(flat_noisy_data)
    reconstructions = [nmf.inverse_transform(nmf.transform(flat_noisy_data))]
    scores = [criterion(torch.from_numpy(flat_data - reconstructions[-1]))]

    print('Running semi-NMF')
    snmf_mdl = SNMF(flat_noisy_data, num_bases=n_clusters)
    snmf_mdl.factorize(niter=500)
    reconstructions.append(np.dot(snmf_mdl.W, snmf_mdl.H))
    scores.append(criterion(torch.from_numpy(flat_data - reconstructions[-1])))

    return scores, reconstructions


if __name__ == "__main__":

    # Generate data
    if DATASET == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        data = datasets.MNIST(
            root="./", train=True, download=True, transform=transform
        )
        train_loader = DataLoader(data, batch_size=len(data))
        data = next(iter(train_loader))[0].numpy()[:2048]
    # Generate data
    elif DATASET == "CIFAR":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        data = datasets.CIFAR10(
            root="./", train=True, download=True, transform=transform
        )
        train_loader = DataLoader(data, batch_size=len(data))
        data = next(iter(train_loader))[0].numpy()[:2048]
    elif DATASET == "TFR":
        data = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)
        spectro = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            win_length=512,
            hop_length=32,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        data = [data[i][0] for i in range(40)]
        data = torch.stack(data)
        data = spectro(data)[:, :, :500, :500]

        data = data.numpy()

    data = data.mean(1, keepdims=True)
    data -= data.min((1, 2, 3), keepdims=True)
    data /= data.max((1, 2, 3), keepdims=True)
    noisy_data = data + (
        np.random.randn(*data.shape) * 0.3 + (np.random.randn(*data.shape) ** 2) * 0.3
    )
    noisy_data = np.clip(noisy_data, 0, 10)

    print("Input PSNR: %.2f dB"%utils.psnr(data, noisy_data))

    # Set your simulation constants here
    n_samples = data.shape[0]  # Size of the matrix
    n_channels = data.shape[1]
    image_size = data.shape[2:]
    h, w = image_size
    n_clusters = 128

    # Network parameters
    nettype = "DD"
    reg_noise_std = 1 / 30.0

    def runner(
        activation_u,
        activation_v,
        nettype=nettype,
        data=data,
        noisy_data=noisy_data,
        n_clusters=n_clusters,
        image_size=image_size,
        n_channels=n_channels,
        n_samples=n_samples,
    ):
        sched_args = argparse.Namespace()
        # Learning constants
        scheduler_type = "none"
        learning_rate = 1e-2
        epochs = 1000
        sched_args.step_size = 2000
        sched_args.gamma = 0.9999
        sched_args.max_lr = learning_rate
        sched_args.min_lr = 1e-6
        sched_args.epochs = epochs

        u_inp = utils.get_inp([n_samples, n_clusters])
        v_inp = utils.get_inp([n_clusters, n_channels, h, w])

        # Create networks
        if nettype == "MLP":
            u_net = models.SimpleForwardMLP(n_samples, [32, 32, n_samples]).cuda()
            v_net = models.SimpleForwardMLP(n_clusters, [32, 32, n_clusters]).cuda()
        elif nettype == "CNN":
            v_net = models.SimpleForward2D(
                n_channels, n_channels, [512, 512, 512]
            ).cuda()
        elif nettype == "DIP":
            v_net = deep_prior.get_net(
                n_clusters,
                "skip",
                "reflection",
                upsample_mode="bilinear",
                skip_n33d=128,
                skip_n33u=128,
                num_scales=4,
                n_channels=n_clusters,
            ).cuda()
            u_net = deep_prior.get_net(
                n_clusters,
                "skip1d",
                "reflection",
                upsample_mode="linear",
                skip_n33d=128,
                skip_n33u=128,
                num_scales=5,
                n_channels=n_clusters,
            ).cuda()
            v_inp = utils.get_inp([1, n_clusters, h, w])
            u_inp = utils.get_inp([1, n_clusters, n_samples])
        elif nettype == "DD":
            # This is multichannel version
            v_inp = utils.get_inp([n_channels, n_clusters, h // 4, w // 4])
            u_inp = utils.get_inp([n_channels, n_clusters, n_samples // 4])
            v_net = deep_decoder.decodernw(
                num_output_channels=n_clusters,
                num_channels_up=[n_clusters]
                + [128] * int(np.log2(image_size[0] // v_inp.shape[2]) - 1),
            ).cuda()

            # probability membership network
            u_net = deep_decoder.decodernw1d(
                num_output_channels=n_clusters, num_channels_up=[n_clusters, 128]
            )
            u_net = u_net.cuda()

        # Extract training parameters
        net_params = list(v_net.parameters()) + list(u_net.parameters())
        inp_params = [u_inp] + [v_inp]

        # You can either optimize both net and inputs, or just net
        params = net_params + inp_params
        optimizer = torch.optim.Adam(lr=learning_rate, params=params)

        # Create a learning scheduler
        scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)

        # Create loss functions -- loses.L1Norm() or losses.L2Norm()
        criterion = losses.L2Norm()

        mse_array = np.zeros(epochs)
        tmse_array = np.zeros(epochs)

        # Now start iterations
        best_mse = float("inf")
        best_epoch = 0

        # Move them to device
        scores, reconstructions = get_nmf(data, noisy_data, n_clusters)
        print(scores)
        psnr2 = utils.psnr(data, reconstructions[0].reshape(data.shape))
        print("NMF: %.2f dB"%psnr2)
        psnr2 = utils.psnr(data, reconstructions[1].reshape(data.shape))
        print("sNMF: %.2f dB"%psnr2)

        data = torch.tensor(data).cuda()
        noisy_data = torch.tensor(noisy_data).cuda()

        tbar = tqdm.tqdm(range(epochs))
        for idx in tbar:
            u_inp_per = u_inp
            v_inp_per = v_inp

            u_output = activation_u(u_net(u_inp_per))
            centroids = activation_v(v_net(v_inp_per))

            centroids_mat = centroids.reshape(1, n_clusters, h * w)
            reconstruction = torch.bmm(u_output.permute(0, 2, 1), centroids_mat)
            reconstruction = reconstruction.reshape(1, n_samples, h, w).permute(
                1, 0, 2, 3
            )

            loss = criterion(reconstruction - noisy_data)

            loss_l1 = centroids.abs().mean() + u_output.abs().mean()
            loss_l2 = criterion(centroids) + criterion(u_output)

            loss = loss + 0.1 * 0.5 ** loss_l1 + 0.5 * 0.1 * 0.5 * loss_l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # # Visualize the reconstruction
            mat_cpu = reconstruction.detach().cpu().numpy()
            centroids_cpu = centroids.detach().cpu().numpy()

            mseval = criterion(reconstruction - data).item()

            mse_array[idx] = mseval
            tmse_array[idx] = loss.item()
            
            tbar.set_description('%.4e'%mseval)
            tbar.refresh()

            if tmse_array[idx] < best_mse:
                best_mse = tmse_array[idx]
                best_epoch = idx
                best_mat = reconstruction.detach().cpu().numpy()

        # Now compute accuracy
        data = data.cpu().numpy()
        psnr1 = utils.psnr(data, best_mat)

        print("DeepTensor NMF: %.2fdB" % psnr1)
    
    print('NMF, softplus')
    runner(torch.nn.Softplus(), torch.nn.Softplus())
    
    print('NMF, abs')
    runner(torch.abs, torch.abs)
    
    print('NMF, relu')
    runner(torch.relu, torch.relu)

    print('semi-NMF, softplus')
    runner(torch.nn.Softplus(), lambda x: x)
    
    print('semi-NMF, abs')
    runner(torch.abs, lambda x: x)
    
    print('semi-NMF, relu')
    runner(torch.relu, lambda x: x)
