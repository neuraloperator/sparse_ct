from typing import Tuple, Literal
import torch
import torch.nn as nn
from src.networks import NOVarnet



class ImageNOVarnetReconstructor(nn.Module):
    
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        num_cascades: int = 4,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        in_shape: Tuple[int, int] = (320, 320),
        kernel_shape: Tuple[int, int] = (3, 4),
        basis_type: Literal["piecewise_linear",
                            "morlet", "zernike"] = "piecewise_linear",   
        use_dc_term: bool = True,     
    ):
        super().__init__()
        self.varnet = NOVarnet(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            num_cascades=num_cascades,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=in_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            use_dc_term=use_dc_term,
        )
        
    def forward(
        self,
        curr_rspace: torch.Tensor,
        ref_rspace: torch.Tensor,
        sampled_indices: torch.Tensor,
        angles: torch.Tensor,
        radon_transform
    ):
        return self.varnet(
            curr_rspace=curr_rspace,
            ref_rspace=ref_rspace,
            sampled_indices=sampled_indices,
            angles=angles,
            radon_transform=radon_transform
        )
