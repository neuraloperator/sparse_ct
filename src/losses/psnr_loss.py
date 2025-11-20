import torch
import torch.nn as nn

class PSNRLoss(nn.Module):
    """
    PSNR module.
    """

    def __init__(self):
        super().__init__()
    
    def _mse(self, Xs, Ys):
        dims = tuple(range(1, len(Xs.shape)))
        return ((Ys - Xs) ** 2).mean(dim=dims)
    
    @property
    def name(self):
        return "psnr_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"

    def forward(self, Xs, Ys):
        mse = self._mse(Xs, Ys)
        data_range = [Y.max() for Y in Ys]
        data_range = torch.stack(data_range, dim=0)
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        if torch.any(torch.isnan(psnr)):
            print(mse)
            print(torch.isnan(psnr))
            print(Xs.shape, Ys.shape)
            import matplotlib.pyplot as plt
            for i in range(len(Xs)):
                plt.figure()
                plt.imshow(Xs[i,0].detach().cpu())
                plt.colorbar()
                plt.savefig(f'image{i}.png')
                plt.figure()
                plt.imshow(Ys[i,0].detach().cpu())
                plt.colorbar()
                plt.savefig(f'gt{i}.png')
            raise
        return -torch.mean(psnr)
    
class PSNR(PSNRLoss):
    def forward(self, Xs, Ys):
        return - super().forward(Xs, Ys)
    
    @property
    def name(self):
        return "psnr"

    @property
    def mode(self):
        return "max"