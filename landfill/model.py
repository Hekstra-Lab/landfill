import torch
from torch import nn
from .layers import GaussianFourierFeatures

class Landfill(nn.Module):
    def __init__(self, sigma, n_fourier_features, n_layers, hidden_width, coord_dims, other_dims=0,leaky_relu_slope=0.25):
        super(Landfill, self).__init__()
        self.sigma = sigma
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.ff = GaussianFourierFeatures(coord_dims,n_fourier_features, sigma)
        self.fourier_linear = nn.Linear(2*n_fourier_features+other_dims, hidden_width)
        with torch.no_grad():
            self.fourier_linear.bias = nn.Parameter(torch.zeros(hidden_width))
        
        dense_layers = []
        for i in range(n_layers):
            linear = nn.Linear(hidden_width,hidden_width)
            with torch.no_grad():
                linear.bias = nn.Parameter(torch.zeros(hidden_width))
            dense_layers.append(linear)
            dense_layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        
        self.mlp = nn.Sequential(*dense_layers)#[nn.Linear(hidden_width, hidden_width), nn.LeakyReLU()])
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Linear(hidden_width,1)
                                                                                                                                                                                                                                    
    def forward(self, x, u=None):
        #         print(x.dtype)
        z = self.ff(x)
        if u is not None:
            if u.ndim==1:
                z = torch.cat([z,u[:,None]], dim=-1)
            else:
                z = torch.cat([z,u], dim=-1)

        z = self.relu(self.fourier_linear(z))
        z = self.mlp(z)
        z = self.dropout(z)
        return self.softplus(self.out(z)).squeeze()
        #return self.sigmoid(self.out(z)).squeeze()
"""
        elif fourier_layer=='positional':
            self.ff = PositionalEncoding(coord_dims, n_fourier_features, log_min=log_min, log_max=log_max)
            self.fourier_linear = nn.Linear(4*n_fourier_features+other_dims, hidden_width)
"""