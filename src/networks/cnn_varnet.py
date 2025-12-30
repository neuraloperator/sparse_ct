from typing import Literal, Tuple
import torch
import torch.nn as nn
from .unet import NormUnet

class VarNetBlock(nn.Module):
    
    def __init__(self, model: nn.Module, use_dc_term: bool = True):
        super().__init__()
        
        self.model = model
        if use_dc_term:
            self.dc_weight = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        current_rspace: torch.Tensor,
        ref_rspace: torch.Tensor,
        angles: torch.Tensor,
        sampled_indices: torch.Tensor,
        radon_transform,
        use_dc_term: bool = True,
    ):
        
        current_image = radon_transform.iradon(current_rspace)
        refined_image = self.model(
            current_image
        )
        model_term, _ = radon_transform.radon(refined_image)
        
        if not use_dc_term:
            return current_rspace - model_term
        
        zero = torch.zeros_like(current_rspace).to(current_rspace)
        zero[:, :, sampled_indices, :] = current_rspace[:, :, sampled_indices, :] - ref_rspace[:, :, sampled_indices, :]
        
        soft_dc = self.dc_weight * zero
        
        return current_rspace - soft_dc - model_term


class CNNVarnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_cascades: int = 4,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        use_dc_term: bool = True,
    ):
        super().__init__()
        
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet(
                        in_chans=in_chans,
                        out_chans=out_chans,
                        chans=chans,
                        num_pool_layers=num_pool_layers,
                        drop_prob=drop_prob,
                    ),
                    use_dc_term=use_dc_term
                ) for i in range(num_cascades)
            ]
        )
        
        self.use_dc_term = use_dc_term
        
        
    def forward(
        self,
        curr_rspace: torch.Tensor,
        ref_rspace: torch.Tensor,
        sampled_indices: torch.Tensor,
        angles: torch.Tensor,
        radon_transform
    ):
        current_rspace = curr_rspace
        
        
        for cascade in self.cascades:
            current_rspace = cascade(
                current_rspace=current_rspace,
                ref_rspace=ref_rspace,
                angles=angles,
                sampled_indices=sampled_indices,
                radon_transform=radon_transform,
                use_dc_term=self.use_dc_term
            )
            
        
        return current_rspace