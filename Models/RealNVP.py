# ------------------------------------------------------------------------------
# PyTorch implementation of a convolutional Real NVP (2017 L. Dinh
# "Density estimation using Real NVP" in https://arxiv.org/abs/1605.08803)
# ------------------------------------------------------------------------------

import os
import torch

import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch import distributions

import sys
from copy import deepcopy

sys.path.append('./')
from Models.base import LinBlock


class RealNVP(nn.Module):
    def __init__(self, tfm_layers=8, latent_dim=2, internal_dim=256):
        super(RealNVP, self).__init__()

        self.tfm_layers = tfm_layers
        self.latent_dim = latent_dim

        # For now, checkerboard masking
        self.mask = np.zeros((tfm_layers, latent_dim), dtype=np.float32)
        self.mask[1::2, ::2] = 1.
        self.mask[::2, 1::2] = 1.
        self.mask = nn.Parameter(torch.from_numpy(self.mask), requires_grad=False)

        self.prior = distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

        self.block_scale = nn.Sequential(nn.Linear(latent_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, latent_dim), nn.Tanh())
        self.block_trans = nn.Sequential(nn.Linear(latent_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, latent_dim))
        self.net_trans = nn.ModuleList([deepcopy(self.block_trans) for _ in range(tfm_layers)])
        self.net_scale = nn.ModuleList([deepcopy(self.block_scale) for _ in range(tfm_layers)])

    def forward(self, x):
        return self.log_prob(x)

    def to_latent(self, x):
        log_det_j, z = torch.zeros(x.shape[0]), x
        for i in reversed(range(len(self.net_scale))):
            mask = self.mask[i]
            z_masked = mask * z
            scale = self.net_scale[i](z_masked) * (1 - mask)
            trans = self.net_trans[i](z_masked) * (1 - mask)
            z = (1 - mask) * (z - trans) * torch.exp(-scale) + z_masked
            log_det_j -= scale.sum(dim=1)

        return z, log_det_j

    def to_image(self, z):
        x = z
        for i in range(len(self.net_scale)):
            mask = self.mask[i]
            x_masked = mask * x
            scale = self.net_scale[i](x_masked) * (1 - mask)
            trans = self.net_trans[i](x_masked) * (1 - mask)
            x = x_masked + (1 - mask) * (x * torch.exp(scale) + trans)
        return x

    def log_prob(self, x):
        z, log_p = self.to_latent(x)
        return self.prior.log_prob(z) + log_p

    def sample(self, num_samples):
        z = self.prior.sample((num_samples, 1))
        x = self.to_image(z)
        return x
