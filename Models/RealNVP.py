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
from torchsummary import summary

import sys
from copy import deepcopy

from tqdm import tqdm

sys.path.append('./')


class RealNVP(nn.Module):
    def __init__(self, tfm_layers=6, latent_dim=2, internal_dim=256, device=None):
        super(RealNVP, self).__init__()

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        log_det_j, z = torch.zeros(x.shape[0]).to(self.device), x
        for i in reversed(range(len(self.net_scale))):
            mask = self.mask[i]
            z_masked = mask * z
            scale = self.net_scale[i](z_masked) * (1 - mask)
            trans = self.net_trans[i](z_masked) * (1 - mask)
            z = (1 - mask) * (z - trans) * torch.exp(-scale) + z_masked
            log_det_j -= scale.sum(dim=1)

        return z, log_det_j

    def to_image(self, z, flag_condition=False):
        x = z
        log_det_j = torch.zeros(x.shape[0]).to(self.device)
        if flag_condition:
            diag_j = torch.ones(x.shape).to(self.device)
        for i in range(len(self.net_scale)):
            mask = self.mask[i]
            x_masked = mask * x
            scale = self.net_scale[i](x_masked) * (1 - mask)
            scale_exp = torch.exp(scale)
            trans = self.net_trans[i](x_masked) * (1 - mask)
            x = x_masked + (1 - mask) * (x * scale_exp + trans)
            print(scale.shape)
            log_det_j += torch.einsum("ijk->i", scale)

            if flag_condition:
                diag_j *= scale_exp
                eig_max, _ = torch.max(diag_j, axis=-1)
                eig_min, _ = torch.min(diag_j, axis=-1)
                cond_num = (eig_max / eig_min).view(-1)

        if flag_condition:
            return x, cond_num
        else:
            return x, log_det_j

    def log_prob(self, x):
        z, log_p = self.to_latent(x)
        return self.prior.log_prob(z.to("cpu")).to(self.device) + log_p

    def sample(self, num_samples, flag_condition=False):
        z = self.prior.sample((num_samples, 1)).to(self.device)
        x, _ = self.to_image(z, flag_condition=flag_condition)
        return x

    def print_summary(self):
        # TODO: Doesn't work for some reason, investigate if required
        summary(self.block_scale, (64, 1, self.latent_dim))
        summary(self.net_trans, (64, 1, self.latent_dim))


def train(dataloader, latent_dim=2, lr=1e-4, max_epochs=1000, tfm_layers=6, device=None):
    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")
    if not os.path.exists("TrainedModels"):
        os.makedirs("TrainedModels")

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    flow = RealNVP(tfm_layers=tfm_layers, latent_dim=latent_dim, device=device).to(device)
    flow.train()

    # Show the network architectures
    # flow.print_summary()

    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=lr)

    ## TODO: No scheduler yet (check later if it helps)

    ### Loop over epochs
    loss_history = []
    # Setting up tqdm bar using the walrus operation (available from python 3.8)
    for _ in (pbar := tqdm(range(max_epochs))):
        ## TODO: For now dataloader is just the two moons dataset, change later when shifting to images
        noisy_moons = dataloader.make_moons(n_samples=100, noise=.05)[0].astype(np.float32)
        loss = -flow(torch.from_numpy(noisy_moons).to(device)).mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # house keeping stuff
        loss_history.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        del noisy_moons

    ### See how the loss evolves
    fig = plt.figure(figsize=(12,9))
    plt.plot(loss_history, label='Loss History')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, max_epochs+1)
    plt.legend()
    plt.grid(True)
    fig.savefig("Outputs/LossHistory.png", bbox_inches='tight')
    plt.close(fig)

    torch.save(flow.state_dict(), "TrainedModels/flow.pth")

    return flow


