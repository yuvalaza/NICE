"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform, SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer.
"""


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        self.mask_config = mask_config
        self.in_net = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())

        self.mid_net = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
                for _ in range(hidden - 1)
            ]
        )

        self.out_net = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        
        if self.mask_config:
            x1, x2 = x[:, ::2], x[:, 1::2]
        else:
            x2, x1 = x[:, ::2], x[:, 1::2]

        x1_ = self.in_net(x1)
        for layer in self.mid_net:
            x1_ = layer(x1_)

        shift = self.out_net(x1_)

        if reverse:
            x2 = x2 - shift  
        else:
            x2 = x2 + shift  

        x_transformed = torch.empty_like(x)
        if self.mask_config:
            x_transformed[:, ::2] = x1
            x_transformed[:, 1::2] = x2
        else:
            x_transformed[:, ::2] = x2
            x_transformed[:, 1::2] = x1

        return x_transformed, log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        self.mask_config = mask_config
        self.in_net = nn.Sequential(nn.Linear(in_out_dim//2, mid_dim), nn.ReLU())
        self.mid_net = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
                for _ in range(hidden - 1)
            ]
        )
        self.out_net = nn.Linear(mid_dim, in_out_dim)

    def forward(self, x, log_det_J, reverse=False):
        """
        Forward pass.
    
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
     
        if self.mask_config:
            x1, x2 = x[:, ::2], x[:, 1::2]
        else:
            x2, x1 = x[:, ::2], x[:, 1::2]
    
        x1_transformed = self.in_net(x1)
        for layer in self.mid_net:
            x1_transformed = layer(x1_transformed)
        out_net = self.out_net(x1_transformed)

        
        log_s, t = out_net[:, ::2], out_net[:, 1::2]
        
        log_s = torch.tanh(log_s)  

        if reverse:
            x2 = (x2 - t) / torch.exp(log_s)
            log_det_J -= torch.sum(log_s, dim=1)
        else:
            x2 = x2 * torch.exp(log_s) + t
            log_det_J += torch.sum(log_s, dim=1)
    
        x_transformed = torch.empty_like(x)
        if self.mask_config:
            x_transformed[:, ::2] = x1
            x_transformed[:, 1::2] = x2
        else:
            x_transformed[:, ::2] = x2
            x_transformed[:, 1::2] = x1

        return x_transformed, log_det_J


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        print("Value of dim:", dim)
        print("Type of dim:", type(dim))
        self.scale = nn.Parameter(torch.zeros((1,dim)), requires_grad=True)
        self.eps = 1e-10

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)+self.eps
        
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)

        return x, log_det_J


"""Standard logistic distribution.
"""
logistic = TransformedDistribution(
    Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0.0, scale=1.0)]
)
def get_logistic_distribution(device):
    base_dist = Uniform(
        torch.tensor(0., device=device),
        torch.tensor(1., device=device)
    )
    transforms = [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)]
    return TransformedDistribution(base_dist, transforms)

"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(
        self,
        prior,
        coupling,
        coupling_type,
        in_out_dim,
        mid_dim,
        hidden,
        mask_config,
        device,
    ):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == "gaussian":
            self.prior = torch.distributions.Normal(
                torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
            )
        elif prior == "logistic":
            self.prior = get_logistic_distribution(device)
        else:
            raise ValueError("Prior not implemented.")
        self.in_out_dim = in_out_dim
        self.scaling = Scaling(in_out_dim)
        self.coupling = coupling
        self.coupling_type = coupling_type
        if self.coupling_type == "additive":
            self.coupling = nn.ModuleList(
                [
                    AdditiveCoupling(
                        in_out_dim=in_out_dim,
                        mid_dim=mid_dim,
                        hidden=hidden,
                        mask_config=(mask_config + i) % 2,
                    )
                    for i in range(coupling)
                ]
            )
        else:
            self.coupling = nn.ModuleList(
                [
                    AffineCoupling(
                        in_out_dim=in_out_dim,
                        mid_dim=mid_dim,
                        hidden=hidden,
                        mask_config=(mask_config + i) % 2,
                    )
                    for i in range(coupling)
                ]
            )

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for coupling_layer in reversed(self.coupling):
            x,_=coupling_layer(x,0,reverse='True')
        
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_j=0
        for coupling_layer in self.coupling:
            x,log_det_j=coupling_layer(x,log_det_j)
        z,log_det_j_scaled=self.scaling(x)
        log_det_j += log_det_j_scaled
        
        return z,log_det_j
        
        

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= (
            np.log(256) * self.in_out_dim
        )  # log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
