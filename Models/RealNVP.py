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
from torch.autograd.functional import jacobian

import sys
from copy import deepcopy

from tqdm import tqdm

sys.path.append('./')


class RealNVP(nn.Module):
    """
    The RealNVP model based on: https://arxiv.org/pdf/1605.08803.pdf
    Code inspired and expanded from: https://colab.research.google.com/github/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
    RealNVP is a generative model that utilises real-valued non-volume preserving (real NVP) transformations for density estimation.
    The model can perform efficient and exact inference, sampling and log-density estimation of data points. (https://paperswithcode.com/method/realnvp)
    """
    def __init__(self, tfm_layers=6, latent_dim=2, internal_dim=256, distribution="normal", num_modes=1,
                 masking="checkerboard",device=None):
        """
        The constructor of the RealNVP class

        Args:
            tfm_layers      (int)                   : (default: 6) Number of transformation layers.
            latent_dim      (int)                   : (default: 2) Number of latent dimensions.
            internal_dim    (int)                   : (default: 256) Number of internal dimensions/nodes in the linear
                                                      layers of the scale and translate neural networks.
            distribution    (string)                : (default: "normal")(options: ["normal", "mixture"])
                                                      The distribution of the prior. "normal" is a standard gaussian and
                                                      "mixture" is a mixture of gaussians.
            num_modes       (int)                   : (default: int) The number of modes in the prior distribution.
                                                      (Only relevant if prior is "mixture").
            masking         (string)                : (default: "checkerboard")(options: ["checkerboard", "half"])
                                                      Describes the type of masking for affine coupling layers.
            device          (torch.device/string)   : (default: None) If device is None then the first GPU is chosen if
                                                      available, else the CPU is chosen.

        """
        super(RealNVP, self).__init__()

        # All the checks here please
        assert distribution in ["normal", "mixture"], f"Only 'normal' and 'mixture' available, got: '{distribution}'"
        assert masking in ["checkerboard", "half"], f"Only 'half' and 'checkerboard' available, got: '{masking}'"
        #

        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tfm_layers = tfm_layers
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.num_modes = num_modes

        self.mask = np.zeros((tfm_layers, latent_dim), dtype=np.float32)
        if masking == "checkerboard":
            self.mask[1::2, ::2] = 1.
            self.mask[::2, 1::2] = 1.
        if masking == "half":
            self.mask[:, latent_dim//2:] = 1.
        self.mask = nn.Parameter(torch.from_numpy(self.mask), requires_grad=False)

        self.prior = distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
        if self.distribution == "mixture":
            mean_latent = 5 * torch.cat([torch.eye(num_modes),
                                         torch.zeros(size=(num_modes, self.latent_dim - num_modes))],
                                        dim=1)
            mix = distributions.Categorical(torch.ones(num_modes, ))
            comp = distributions.Independent(distributions.Normal(mean_latent, torch.ones_like(mean_latent)), 1)
            self.prior = distributions.MixtureSameFamily(mix, comp)

        self.block_scale = nn.Sequential(nn.Linear(latent_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, latent_dim), nn.Tanh())
        self.block_trans = nn.Sequential(nn.Linear(latent_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, internal_dim), nn.LeakyReLU(),
                                         nn.Linear(internal_dim, latent_dim))
        self.net_trans = nn.ModuleList([deepcopy(self.block_trans) for _ in range(tfm_layers)])
        self.net_scale = nn.ModuleList([deepcopy(self.block_scale) for _ in range(tfm_layers)])

    def forward(self, x):
        """
        The forward pass of the RealNVP model. Flow from image space to latent space.
        Args:
            x   (torch.float32) :   [#batch_size, #latent_dims] The batch of "images"

        Returns:
            The log probability of the batch of inputs. (torch.float32) [#batch_size].
        """
        return self.log_prob(x)

    def to_latent(self, x):
        """
        Flow from image space to latent space.
        Args:
            x           (torch.float32)     :   [#batch_size, #latent_dims] The batch of "images"

        Returns:
            z           (torch.float32)     :   [#batch_size, #latent_dims] The batch of corresponding latent vectors.
            log_det_j   (torch.float32)     :   [#batch_size] The log determinant of the jacobian of each input.
        """
        log_det_j, z = torch.zeros(x.shape[0]).to(self.device), x
        for i in reversed(range(len(self.net_scale))):
            mask = self.mask[i]
            z_masked = mask * z
            scale = self.net_scale[i](z_masked) * (1 - mask)
            trans = self.net_trans[i](z_masked) * (1 - mask)
            z = (1 - mask) * (z - trans) * torch.exp(-scale) + z_masked
            log_det_j -= scale.sum(dim=1)

        return z, log_det_j

    def to_image(self, z):
        """
        Flow from latent space to image space
        Args:
            z       (torch.float32)     :   TODO: Remove this 1 [#batch_size, 1, #latent_dims] The batch of latent vectors.

        Returns:
            x           (torch.float32)     :   [#batch_size, #latent_dims] The batch of corresponding "images"
            log_det_j   (torch.float32)     :   [#batch_size] The log determinant of the jacobian of each input.
        """
        x = z
        log_det_j = torch.zeros(x.shape[0]).to(self.device)
        for i in range(len(self.net_scale)):
            mask = self.mask[i]
            x_masked = mask * x
            scale = self.net_scale[i](x_masked) * (1 - mask)
            scale_exp = torch.exp(scale)
            trans = self.net_trans[i](x_masked) * (1 - mask)
            x = x_masked + (1 - mask) * (x * scale_exp + trans)
            log_det_j += torch.einsum("ijk->i", scale)
        return x, log_det_j

    def log_prob(self, x):
        """
        Returns the log of the probability of the generated latent vector based on its prior
        Args:
            x           (torch.float32)     :   [#batch_size, #latent_dims] The batch of "images"

        Returns:
            The log probability of the batch of inputs. (torch.float32) [#batch_size].
        """
        z, log_p = self.to_latent(x)
        return self.prior.log_prob(z.to("cpu")).to(self.device) + log_p

    def sample(self, num_samples):
        """

        Args:
            num_samples     (int)       :

        Returns:

        """
        z = self.prior.sample((num_samples, 1)).to(self.device)
        x, _ = self.to_image(z)
        return x

    def get_condition_number_to_image(self, z):
        """

        Args:
            z:

        Returns:

        """
        jacob = self.get_jacobian_to_image(z)
        return jacob

    def get_jacobian_to_image(self, z):
        """

        Args:
            z:

        Returns:

        """
        def func(z):
            x = z
            for i in range(len(self.net_scale)):
                mask = self.mask[i]
                x_masked = mask * x
                scale = self.net_scale[i](x_masked) * (1 - mask)
                scale_exp = torch.exp(scale)
                trans = self.net_trans[i](x_masked) * (1 - mask)
                x = x_masked + (1 - mask) * (x * scale_exp + trans)
            return x

        # TODO: May send this to utils
        def batch_jacobian(func, x, vectorize=True):
            """
            Ref: https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/4
            """

            # x in shape (Batch, Length)
            def _func_sum(x):
                return func(x).sum(dim=0)

            return jacobian(_func_sum, x, vectorize=vectorize).permute(1, 0, 2)

        return batch_jacobian(func, z)

    def energy_in_image_space(self, gamma, ts, flag_grad=False, length=False):
        """

        Args:
            gamma: Trainable Curve
            ts: [#N_t, #latent_dim]

        Returns:

        """
        if not flag_grad:
            with torch.set_grad_enabled(False):
                curve_pts = gamma(ts.to(self.device))[0]
                image_pts, _ = self.to_image(curve_pts[:, None, :])
                image_pts = torch.squeeze(image_pts, axis=1)
                if length:
                    return (torch.linalg.norm(image_pts[:-1, :] - image_pts[1:, :],dim=1)).sum().detach().cpu().numpy()
                else:
                    # TODO: Sum of squares
                    return (torch.linalg.norm(image_pts[:-1, :] - image_pts[1:, :],
                                              dim=1) ** 2).sum().detach().cpu().numpy()
        else:
            # Redundant, but still specifying this explicitly
            with torch.set_grad_enabled(True):
                curve_pts = gamma(ts.to(self.device))[0]
                image_pts, _ = self.to_image(curve_pts[:, None, :])
                image_pts = torch.squeeze(image_pts, axis=1)
                if length:
                    return (torch.linalg.norm(image_pts[:-1, :] - image_pts[1:, :], dim=1)).sum()
                else:
                    return (torch.linalg.norm(image_pts[:-1, :] - image_pts[1:, :], dim=1) ** 2).sum()

    def sesl_loss(self, gamma, ts, flag_grad=False):
        if not flag_grad:
            with torch.set_grad_enabled(False):
                curve_pts = gamma(ts.to(self.device))[0]
                image_pts, _ = self.to_image(curve_pts[:, None, :])
                image_pts = torch.squeeze(image_pts, axis=1)

                # Geodesic in image space (euclidean space) is just the straight line
                image_gds = tensor_linspace(image_pts[0, :], image_pts[-1, :], image_pts.shape[0])

                return (torch.linalg.norm(image_pts - image_gds, dim=1) ** 2).sum().detach().cpu().numpy()
                # return (torch.linalg.norm(image_pts - image_gds, dim=1)).sum().detach().cpu().numpy()
        else:
            # Redundant, but still specifying this explicitly
            with torch.set_grad_enabled(True):
                curve_pts = gamma(ts.to(self.device))[0]
                image_pts, _ = self.to_image(curve_pts[:, None, :])
                image_pts = torch.squeeze(image_pts, axis=1)

                # Geodesic in image space (euclidean space) is just the straight line
                image_gds = tensor_linspace(image_pts[0, :], image_pts[-1, :], image_pts.shape[0])

                return (torch.linalg.norm(image_pts - image_gds, dim=1) ** 2).sum()
                # return (torch.linalg.norm(image_pts - image_gds, dim=1)).sum()

def train(dataloader, latent_dim=2, lr=1e-4, max_epochs=1000, tfm_layers=6, device=None, save_name=None,
          distribution="normal", masking="checkerboard", num_modes=1):
    """

    Args:
        dataloader:
        latent_dim:
        lr:
        max_epochs:
        tfm_layers:
        device:
        save_name:
        distribution:
        masking:
        num_modes:

    Returns:

    """
    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")
    if not os.path.exists("TrainedModels"):
        os.makedirs("TrainedModels")

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    flow = RealNVP(tfm_layers=tfm_layers, latent_dim=latent_dim, device=device,
                   distribution=distribution, num_modes=num_modes, masking=masking).to(device)
    flow.train()

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
    if save_name is None:
        fig.savefig("Outputs/flow_loss_history.png", bbox_inches='tight')
    else:
        fig.savefig(f"Outputs/{save_name}_loss_history.png", bbox_inches='tight')
    plt.close(fig)

    if save_name is None:
        torch.save(flow.state_dict(), "TrainedModels/flow.pth")
    else:
        torch.save(flow.state_dict(), f"TrainedModels/{save_name}.pth")

    return flow

# TODO: Move to utils
def tensor_linspace(start, end, steps=10):
    """
    ref: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out.T