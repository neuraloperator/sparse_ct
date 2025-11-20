import torch
from torch import nn 
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    SSIM module.
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    @property
    def name(self):
        return "ssim_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"

    def _ssim(self, X, Y, data_range):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        dims = tuple(range(1, len(X.shape)))
        
        return S.mean(dim=dims)

    def forward(self, Xs, Ys):
        data_range = [Y.max() for Y in Ys]
        data_range = torch.stack(data_range, dim=0)

        return 1 - self._ssim(Xs, Ys, data_range=data_range.detach())

class SSIM(SSIMLoss):
    def forward(self, Xs, Ys):
        return 1 - super().forward(Xs, Ys)

    @property
    def name(self):
        return "ssim"

    @property
    def mode(self):
        return "max"
