import torch
from .base_radon import BaseRadonTransform
from torch_radon import Radon
import numpy as np

class KitsRadonTransform(BaseRadonTransform):
    def __init__(self, num_projections):
        super().__init__()
        self.resolution = 256
        self.det_count = 300
        self.thetas = torch.tensor(np.linspace(0.0, np.pi, num_projections, endpoint=False).astype(np.float32))

        self.rf = Radon(resolution=self.resolution, angles=self.thetas, det_count=self.det_count)


    def radon(self, image):
                
        if image.ndim == 2:
            image = image[None, None, ...]  # add batch dim
        elif image.ndim == 3:
            image = image[None, ...]
        
        assert image.shape[-1] == self.resolution and image.shape[-2] == self.resolution, "Input image resolution does not match radon transform resolution"        

        sino = self.rf.forward(image)
        return sino, self.thetas 


    def iradon(self, sinogram):
        
        assert sinogram.ndim in (2, 3, 4), "sinogram must be 2D or 3D (batch)"
        if sinogram.ndim == 3:
            sinogram = sinogram[None, ...]
        if sinogram.ndim == 2:
            sinogram = sinogram[None, None, ...]  # add batch dim

        filtered_sinogram = self.rf.filter_sinogram(sinogram, "ram-lak")
        image = self.rf.backprojection(filtered_sinogram)

        return image