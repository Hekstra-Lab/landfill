import torch
from pyro.contrib import gp

import cupy as cp
from cupyx.scipy.ndimage.interpolation import zoom as cupy_zoom
from scipy.ndimage.interpolation import zoom as scipy_zoom


class LandfillGP:
    def __init__(self, X, y, img_shape=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.img_shape = img_shape

        self.gpr = self._init_gp(X, y)

        self.losses = None
        self.trained = False

    def train(self, lr=0.1, num_steps=100):
        opt = torch.optim.Adam(self.gpr.parameters(), lr=lr)
        self.losses = []
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        self.losses = []
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())

        self.trained = True

    def get_scaler(self, fullsize=False, downsample_factor=4):
        """
        If fullsize=True predict full img_shape rescaling field.
        Otherwise predict a downsampled version and interpolate back up.
        """
        shape = (
            self.img_shape
            if fullsize
            else tuple((x // downsample_factor for x in self.img_shape))
        )
        if self.device == "cuda":
            zoom_fn = cupy_zoom
            Xpred = torch.tensor(
                cp.indices(shape).reshape(2, -1).T,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            zoom_fn = scipy_zoom
            Xpred = torch.tensor(
                np.indices(shape).reshape(2, -1).T,
                dtype=torch.float32,
                device=self.device,
            )

        mean, var = self.gpr(Xpred)
        if fullsize:
            return mean.reshape(shape).detach()
        else:
            return zoom_fn(mean.reshape(shape), downsample_factor, order=1)

    def _init_gp(self, X, y):
        K = gp.kernels.Sum(
            gp.kernels.WhiteNoise(input_dim=2),
            gp.kernels.RBF(
                input_dim=2,
                variance=torch.tensor(1.0),
                lengthscale=torch.tensor(
                    [self.img_shape[0] / 4, self.img_shape[1] / 4]
                ),
            ),
        )
        gpr = gp.models.GPRegression(X, y, K).to(self.device)
        return gpr
