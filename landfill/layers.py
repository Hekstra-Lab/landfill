import torch
from torch import nn

class GaussianFourierFeatures(nn.Module):
    def __init__(self, n_dims, n_features, sigma):
        super().__init__()
        self.sigma = sigma
        if not isinstance(self.sigma, Number):
            assert self.sigma.shape[0]==n_dims,\
            f"Vector sigma must have n_dims dimensions. Found {sigma.shape=} and {n_dims=}"
            if not isinstance(self.sigma, torch.Tensor):
                self.sigma = torch.tensor(self.sigma)
        self.n_features = n_features
        self.n_dims = n_dims
        self.B = (sigma*torch.randn(self.n_features, n_dims)).t()
    def forward(self, x):
        y = x.mm(self.B)
        c = torch.cos(y)
        s = torch.sin(y)
        return torch.cat((c,s),dim=-1)