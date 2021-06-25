from numbers import Number
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

class PositionalEncoding(nn.Module):
    def __init__(self, n_dims, n_features, log_min, log_max, rand=False):
        super().__init__()
        self.n_features = n_features
        self.n_dims =n_dims
        if rand:
            B = 2**((log_max-log_min)*torch.rand(self.n_features)+log_min)
        else:
            B = 2**torch.linspace(log_min, log_max, self.n_features)
           
        self.B = nn.Parameter(B.unsqueeze(-1).expand(-1,n_dims).t(), requires_grad=False)
    
    def forward(self, x):
        y =x[:,:,None]*self.B[None, None, :]
        y = y.reshape(x.shape[0], -1)
        c = torch.cos(2*np.pi*y)
        s = torch.sin(2*np.pi*y)
        return torch.cat((c,s),dim=-1)
