import torch
from torch import nn
from .layers import GaussianFourierFeatures

class Landfill(nn.Module):
    def __init__(self, sigma, n_fourier_features, n_layers, hidden_width, coord_dims, other_dims=0,):
        super(Landfill, self).__init__()
        self.sigma = sigma
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.gff = GaussianFourierFeatures(coord_dims,n_fourier_features, sigma)
        self.fourier_linear = nn.Linear(2*n_fourier_features+other_dims, hidden_width)
        self.mlp = nn.Sequential(* n_layers*[nn.Linear(hidden_width, hidden_width), nn.LeakyReLU()])
        self.out = nn.Linear(hidden_width,1)

    def forward(self, x, u=None):
        z = self.gff(x)
        if u is not None:
            z = torch.cat([z,u], dim=-1)

        z = self.relu(self.fourier_linear(z))
        z = self.mlp(z)
        return self.softplus(self.out(z)).squeeze()
