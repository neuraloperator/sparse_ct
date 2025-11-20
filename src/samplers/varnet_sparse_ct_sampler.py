from typing import List
import torch
import torch.nn as nn
import numpy as np

class SparseCTSamplerKeepDim(nn.Module):
    def __init__(self, accn_rate: float):
        super().__init__()
        self.accn_rate = accn_rate
        
    def forward(self, sino: torch.Tensor, theta: torch.Tensor):
        # sino: [N, 1, 720, 512]
        assert sino.ndim == 4, "expected shape (B, num_channels, num_angles, num_detectors)"
        num_angles = sino.shape[-2]
        num_angles_sampled = int(num_angles/self.accn_rate)
        indices = torch.linspace(0, num_angles-1, num_angles_sampled, dtype=torch.long)
        #rest should be zero
        sampled_sino = torch.zeros_like(sino)
        sampled_sino[:, :, indices, :] = sino[:, :, indices, :]
        # sampled_sino = sino[:, :, indices, :]
        sampled_theta = theta
        return sampled_sino, sampled_theta
    

class VarnetSparseCTSampler(nn.Module):
    def __init__(self, accn_rate: List[float]):
        super().__init__()
        self.accn_rate = accn_rate
        
    def forward(self, sino: torch.Tensor, theta: torch.Tensor):
        # sino: [N, 1, 720, 512]
        assert sino.ndim == 4, "expected shape (B, num_channels, num_angles, num_detectors)"
        num_angles = sino.shape[-2]
        sample_accn_rate = np.random.choice(self.accn_rate)
        num_angles_sampled = int(num_angles/sample_accn_rate)
        indices = torch.linspace(0, num_angles-1, num_angles_sampled, dtype=torch.long)
        sampled_sino = torch.zeros_like(sino)
        sampled_sino[:, :, indices, :] = sino[:, :, indices, :]
        sampled_theta = theta
        return sampled_sino, sampled_theta, indices