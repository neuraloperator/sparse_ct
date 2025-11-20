import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input:    predictions tensor
            target:   ground-truth tensor (same shape as input)
        Returns:
            loss tensor, either a scalar (mean or sum) or same-shape (none)
        """
        diff2 = (input - target).pow(2)
        if self.reduction == 'mean':
            return diff2.mean()
        elif self.reduction == 'sum':
            return diff2.sum()
        else:  # 'none'
            return diff2
    
    @property
    def name(self):
        return "mse_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "recon"


class MSE (MSELoss):
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__(reduction)
    
    def forward(self, Xs, Ys):
        return - super().forward(Xs, Ys)
    
    @property
    def name(self):
        return "mse"

    @property
    def mode(self):
        return "max"