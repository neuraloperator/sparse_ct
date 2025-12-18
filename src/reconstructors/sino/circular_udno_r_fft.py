from typing import Tuple, Literal
import torch
import torch.nn as nn
from src.networks import CircularNormUDNO


class SinoCircularUDNORFFTReconstructor(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, drop_prob: float = 0.0, num_pool_layers: int = 4, radius_cutoff: float = 0.02, in_shape: Tuple[int, int] = (320, 320), kernel_shape: Tuple[int, int] = (3, 4), basis_type: Literal[
        "piecewise_linear", "morlet", "zernike"
    ] = "piecewise_linear", pad_size: int = 0):
        super().__init__()
        self.spatial_udno = CircularNormUDNO(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=in_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            pad_size=pad_size,
        )

        self.freq_udno = CircularNormUDNO(
            in_chans=2*in_chans,
            out_chans=2*out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=in_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            pad_size=pad_size,
        )

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        # FFT along r (detector) → centered frequency
        spec_r = torch.fft.fftshift(
            torch.fft.fft(sino, dim=-1, norm="ortho"),
            dim=-1
        )

        # Pack complex to real channels
        packed = torch.cat([spec_r.real, spec_r.imag], dim=1)  # [B, 2*C, θ, r]

        corrected = self.freq_udno(packed)
        C2 = corrected.shape[1] // 2
        real_corr = corrected[:, :C2, :, :]
        imag_corr = corrected[:, C2:, :, :]
        spec_r_corr = torch.complex(real_corr, imag_corr)

        # iFFT back to spatial along r (from centered spectrum)
        freq_sino_recon = torch.fft.ifft(
            torch.fft.ifftshift(spec_r_corr, dim=-1),
            dim=-1, norm="ortho"
        ).real

        # Residuals (ensure channel equality)
        spatial_sino_udno = sino + self.spatial_udno(sino)     # -> [B, C, θ, r]
        freq_sino_udno    = sino + freq_sino_recon             # -> [B, C, θ, r]

        combined = torch.cat([spatial_sino_udno, freq_sino_udno], dim=1)
        return combined.mean(dim=1, keepdim=True)
