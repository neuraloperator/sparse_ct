import torch.nn as nn
import torch


class SinoIdentityReconstructor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        return sino