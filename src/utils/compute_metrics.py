import torch
from src.losses import PSNR, MSELoss, SSIM
import numpy as np

class CTTools:
    def __init__(self, mu_water: float = 0.192):
        self.mu_water = mu_water

    def HU2mu(self, hu_img: np.ndarray) -> np.ndarray:
        # μ = μ_water * (HU/1000 + 1)
        return hu_img / 1000.0 * self.mu_water + self.mu_water

    def mu2HU(self, mu_img: np.ndarray) -> np.ndarray:
        return (mu_img - self.mu_water) / self.mu_water * 1000.0
    
    def window_transform(self, hu_img, width=3000, center=500, norm=False):
        # HU -> 0-1 normalized 
        min_window = float(center) - 0.5 * float(width)
        win_img = (hu_img - min_window) / float(width)
        win_img[win_img < 0] = 0
        win_img[win_img > 1] = 1
        if norm:
            print('normalize to 0-255')
            win_img = (win_img * 255).astype('float')
        return win_img

def compute_metrics(image, image_gt):

    psnr_class = PSNR().to(image.device)
    ssim_class = SSIM().to(image.device)
    mse_class = MSELoss().to(image.device)
    ct_tools = CTTools()
    
    psnr = psnr_class(image, image_gt)
    ssim = ssim_class(image, image_gt).mean()

    image_hu = ct_tools.mu2HU(image)
    image_gt_hu = ct_tools.mu2HU(image_gt)
    
    rmse = torch.sqrt(mse_class(image_hu, image_gt_hu))
    
    return rmse, psnr, ssim
