import torch.nn as nn
from typing import Tuple, Literal
import torch

class ImageIdentityReconstructor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        curr_rspace: torch.Tensor,
        ref_rspace: torch.Tensor,
        sampled_indices: torch.Tensor,
        angles: torch.Tensor,
        radon_transform
    ):
        return curr_rspace