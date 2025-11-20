"""
U-shaped DISCO Neural Operator
"""

from typing import Literal, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from neuralop.layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d as DISCO2d,
)


def _match_hw(
    x: torch.Tensor,
    ref_or_shape: Union[torch.Tensor, torch.Size, Tuple[int, ...], Tuple[int, int]],
    mode: str = "reflect",
) -> torch.Tensor:
    """
    Center-crop or pad x to match target H,W using a single pad mode.
    x:           (N, C, H, W)
    ref_or_shape: tensor with shape (..., Ht, Wt) OR a shape/size with Ht,Wt at the end
    """
    if isinstance(ref_or_shape, torch.Tensor):
        target_h, target_w = ref_or_shape.shape[-2:]
    else:
        # torch.Size or tuple: allow (..., H, W) or (H,W)
        if len(ref_or_shape) >= 2:
            target_h, target_w = ref_or_shape[-2], ref_or_shape[-1]
        else:
            raise ValueError("ref_or_shape must carry target H and W.")

    _, _, h, w = x.shape

    # --- Width ---
    if w != target_w:
        if w > target_w:
            dw = w - target_w
            l = dw // 2
            r = dw - l
            x = x[:, :, :, l:w - r]
        else:
            dw = target_w - w
            l = dw // 2
            r = dw - l
            if l or r:
                x = F.pad(x, (l, r, 0, 0), mode=mode)

    # --- Height ---
    _, _, h, w = x.shape
    if h != target_h:
        if h > target_h:
            dh = h - target_h
            t = dh // 2
            b = dh - t
            x = x[:, :, t:h - b, :]
        else:
            dh = target_h - h
            t = dh // 2
            b = dh - t
            if t or b:
                x = F.pad(x, (0, 0, t, b), mode=mode)

    return x


class NormUDNO(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        in_shape: Tuple[int, int] = (320, 320),
        kernel_shape: Tuple[int, int] = (3, 4),
        basis_type: Literal["piecewise_linear",
                            "morlet", "zernike"] = "piecewise_linear",
    ):
        super().__init__()
        self.udno = UDNO(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=in_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
        )

    def norm(self, x: torch.Tensor, eps: float = 1e-6):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        mean = x_flat.mean(dim=2).view(b, c, 1, 1)
        std = x_flat.std(dim=2, unbiased=False).view(b, c, 1, 1)
        return (x - mean) / (std + eps), mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            (N, in_chans, H, W)
        Returns
        -------
        torch.Tensor
            (N, out_chans, H, W)
        """
        x, mean, std = self.norm(image)
        output = self.udno(x)
        # Anchor to the original input H,W to avoid residual/consumer mismatches.
        output = _match_hw(output, image, mode="reflect")
        output = self.unnorm(output, mean, std)
        return output


class UDNO(nn.Module):
    """
    U-shaped DISCO Neural Operator in PyTorch
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        in_shape: Tuple[int, int] = (320, 320),
        kernel_shape: Tuple[int, int] = (3, 4),
        basis_type: Literal["piecewise_linear",
                            "morlet", "zernike"] = "piecewise_linear",
    ):
        super().__init__()
        assert len(in_shape) == 2, "Input shape must be 2D"

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.in_shape = in_shape
        self.kernel_shape = kernel_shape

        self.down_sample_layers = nn.ModuleList(
            [
                DISCOBlock(
                    in_chans,
                    chans,
                    radius_cutoff,
                    drop_prob,
                    in_shape,
                    kernel_shape,
                    basis_type=basis_type,
                )
            ]
        )
        ch = chans
        shape = (in_shape[0] // 2, in_shape[1] // 2)
        rcut = radius_cutoff * 2
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                DISCOBlock(
                    ch,
                    ch * 2,
                    rcut,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                )
            )
            ch *= 2
            shape = (shape[0] // 2, shape[1] // 2)
            rcut *= 2

        self.bottleneck = DISCOBlock(
            ch,
            ch * 2,
            rcut,
            drop_prob,
            in_shape=shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
        )

        self.up = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose.append(
                TransposeDISCOBlock(
                    ch * 2,
                    ch,
                    rcut,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                )
            )
            shape = (shape[0] * 2, shape[1] * 2)
            rcut /= 2
            self.up.append(
                DISCOBlock(
                    ch * 2,
                    ch,
                    rcut,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                )
            )
            ch //= 2

        self.up_transpose.append(
            TransposeDISCOBlock(
                ch * 2,
                ch,
                rcut,
                in_shape=shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
            )
        )
        shape = (shape[0] * 2, shape[1] * 2)
        rcut /= 2
        self.up.append(
            nn.Sequential(
                DISCOBlock(
                    ch * 2,
                    ch,
                    rcut,
                    drop_prob,
                    in_shape=shape,
                    kernel_shape=kernel_shape,
                    basis_type=basis_type,
                ),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (N, in_chans, H, W)
        return: (N, out_chans, H, W) [after final anchoring in NormUDNO]
        """
        stack = []
        output = image

        # Down path: after each block, match back to pre-block H,W before pooling.
        for layer in self.down_sample_layers:
            pre_shape = output.shape
            output = layer(output)
            output = _match_hw(output, pre_shape, mode="reflect")
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        # Bottleneck: keep spatial size fixed across it.
        pre_shape = output.shape
        output = self.bottleneck(output)
        output = _match_hw(output, pre_shape, mode="reflect")

        # Up path: after upsample, match to doubled size; after block, keep that size.
        for transpose, disco in zip(self.up_transpose, self.up):
            skip = stack.pop()

            # target HW after upsample should match the skip's HW
            output = transpose(output)
            output = _match_hw(output, skip, mode="reflect")

            output = torch.cat([output, skip], dim=1)
            pre_shape = output.shape  # shape after concat, before block
            output = disco(output)
            output = _match_hw(output, pre_shape, mode="reflect")

        return output


class DISCOBlock(nn.Module):
    """
    Two DISCO layers each followed by InstanceNorm2d, LeakyReLU, and Dropout2d.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        drop_prob: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int] = (3, 4),
        basis_type: Literal["piecewise_linear",
                            "morlet", "zernike"] = "piecewise_linear",
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            DISCO2d(
                in_chans,
                out_chans,
                in_shape=in_shape,
                out_shape=in_shape,
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                basis_type=basis_type,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            DISCO2d(
                out_chans,
                out_chans,
                in_shape=in_shape,
                out_shape=in_shape,
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                basis_type=basis_type,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeDISCOBlock(nn.Module):
    """
    Upsample (bilinear, align_corners=False) -> DISCO -> InstanceNorm2d -> LeakyReLU
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        in_shape: Tuple[int, int],
        kernel_shape: Tuple[int, int] = (3, 4),
        basis_type: Literal["piecewise_linear",
                            "morlet", "zernike"] = "piecewise_linear",
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DISCO2d(
                in_chans,
                out_chans,
                in_shape=(2 * in_shape[0], 2 * in_shape[1]),
                out_shape=(2 * in_shape[0], 2 * in_shape[1]),
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=(radius_cutoff / 2),
                basis_type=basis_type,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


# """
# U-shaped DISCO Neural Operator
# """

# from typing import Literal, Tuple

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from neuralop.layers.discrete_continuous_convolution import (
#     EquidistantDiscreteContinuousConv2d as DISCO2d,
# )

# # Toggle to sanity-check shapes during development
# DEBUG_SHAPES = False


# def _match_hw(x: torch.Tensor, ref: torch.Tensor, mode: str = "reflect") -> torch.Tensor:
#     """
#     Center-crop or pad x to match ref's H,W using a single pad mode.
#     x:   (N, C, H, W)
#     ref: (N, C?, H_ref, W_ref) -- only H,W are used.
#     """
#     target_h, target_w = ref.shape[-2:]
#     _, _, h, w = x.shape

#     # --- Width ---
#     if w != target_w:
#         if w > target_w:
#             dw = w - target_w
#             l = dw // 2
#             r = dw - l
#             x = x[:, :, :, l:w - r]
#         else:
#             dw = target_w - w
#             l = dw // 2
#             r = dw - l
#             if l or r:
#                 x = F.pad(x, (l, r, 0, 0), mode=mode)

#     # --- Height ---
#     _, _, h, w = x.shape  # update after width adjustment
#     if h != target_h:
#         if h > target_h:
#             dh = h - target_h
#             t = dh // 2
#             b = dh - t
#             x = x[:, :, t:h - b, :]
#         else:
#             dh = target_h - h
#             t = dh // 2
#             b = dh - t
#             if t or b:
#                 x = F.pad(x, (0, 0, t, b), mode=mode)

#     return x


# class NormUDNO(nn.Module):
#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         radius_cutoff: float,
#         chans: int = 32,
#         num_pool_layers: int = 4,
#         drop_prob: float = 0.0,
#         in_shape: Tuple[int, int] = (320, 320),
#         kernel_shape: Tuple[int, int] = (3, 4),
#         basis_type: Literal["piecewise_linear",
#                             "morlet", "zernike"] = "piecewise_linear",
#     ):
#         super().__init__()
#         self.udno = UDNO(
#             in_chans=in_chans,
#             out_chans=out_chans,
#             radius_cutoff=radius_cutoff,
#             chans=chans,
#             num_pool_layers=num_pool_layers,
#             drop_prob=drop_prob,
#             in_shape=in_shape,
#             kernel_shape=kernel_shape,
#             basis_type=basis_type,
#         )

#     def norm(self, x: torch.Tensor, eps: float = 1e-6):
#         b, c, h, w = x.shape
#         x_flat = x.view(b, c, -1)
#         mean = x_flat.mean(dim=2).view(b, c, 1, 1)
#         std = x_flat.std(dim=2, unbiased=False).view(b, c, 1, 1)
#         return (x - mean) / (std + eps), mean, std

#     def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
#         return x * std + mean

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         image : torch.Tensor
#             Input 4D tensor of shape `(N, in_chans, H, W)`.

#         Returns
#         -------
#         torch.Tensor
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         x, mean, std = self.norm(image)
#         output = self.udno(x)
#         output = self.unnorm(output, mean, std)
#         return output


# class UDNO(nn.Module):
#     """
#     U-shaped DISCO Neural Operator in PyTorch
#     """

#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         radius_cutoff: float,
#         chans: int = 32,
#         num_pool_layers: int = 4,
#         drop_prob: float = 0.0,
#         in_shape: Tuple[int, int] = (320, 320),
#         kernel_shape: Tuple[int, int] = (3, 4),
#         basis_type: Literal["piecewise_linear",
#                             "morlet", "zernike"] = "piecewise_linear",
#     ):
#         """
#         Parameters
#         ----------
#         in_chans : int
#             Number of channels in the input to the U-Net model.
#         out_chans : int
#             Number of channels in the output to the U-Net model.
#         radius_cutoff : float
#             Effective radius proportion of the DISCO kernel (0..1).
#         chans : int
#             Base channel count.
#         num_pool_layers : int
#             Number of down/up stages.
#         drop_prob : float
#             Dropout probability.
#         in_shape : (H, W)
#             Compile-time spatial shape for DISCO (resolution invariance).
#         kernel_shape : (rings, anisotropic_bases)
#             DISCO kernel parameters (not a spatial kernel size).
#         """
#         super().__init__()
#         assert len(in_shape) == 2, "Input shape must be 2D"

#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.chans = chans
#         self.num_pool_layers = num_pool_layers
#         self.drop_prob = drop_prob
#         self.in_shape = in_shape
#         self.kernel_shape = kernel_shape

#         self.down_sample_layers = nn.ModuleList(
#             [
#                 DISCOBlock(
#                     in_chans,
#                     chans,
#                     radius_cutoff,
#                     drop_prob,
#                     in_shape,
#                     kernel_shape,
#                     basis_type=basis_type,
#                 )
#             ]
#         )
#         ch = chans
#         shape = (in_shape[0] // 2, in_shape[1] // 2)
#         rcut = radius_cutoff * 2
#         for _ in range(num_pool_layers - 1):
#             self.down_sample_layers.append(
#                 DISCOBlock(
#                     ch,
#                     ch * 2,
#                     rcut,
#                     drop_prob,
#                     in_shape=shape,
#                     kernel_shape=kernel_shape,
#                     basis_type=basis_type,
#                 )
#             )
#             ch *= 2
#             shape = (shape[0] // 2, shape[1] // 2)
#             rcut *= 2

#         self.bottleneck = DISCOBlock(
#             ch,
#             ch * 2,
#             rcut,
#             drop_prob,
#             in_shape=shape,
#             kernel_shape=kernel_shape,
#             basis_type=basis_type,
#         )

#         self.up = nn.ModuleList()
#         self.up_transpose = nn.ModuleList()
#         for _ in range(num_pool_layers - 1):
#             self.up_transpose.append(
#                 TransposeDISCOBlock(
#                     ch * 2,
#                     ch,
#                     rcut,
#                     in_shape=shape,
#                     kernel_shape=kernel_shape,
#                     basis_type=basis_type,
#                 )
#             )
#             shape = (shape[0] * 2, shape[1] * 2)
#             rcut /= 2
#             self.up.append(
#                 DISCOBlock(
#                     ch * 2,
#                     ch,
#                     rcut,
#                     drop_prob,
#                     in_shape=shape,
#                     kernel_shape=kernel_shape,
#                     basis_type=basis_type,
#                 )
#             )
#             ch //= 2

#         self.up_transpose.append(
#             TransposeDISCOBlock(
#                 ch * 2,
#                 ch,
#                 rcut,
#                 in_shape=shape,
#                 kernel_shape=kernel_shape,
#                 basis_type=basis_type,
#             )
#         )
#         shape = (shape[0] * 2, shape[1] * 2)
#         rcut /= 2
#         self.up.append(
#             nn.Sequential(
#                 DISCOBlock(
#                     ch * 2,
#                     ch,
#                     rcut,
#                     drop_prob,
#                     in_shape=shape,
#                     kernel_shape=kernel_shape,
#                     basis_type=basis_type,
#                 ),
#                 # 1x1 conv is pixel-wise, resolution invariant
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )
#         )

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         image : torch.Tensor
#             Input 4D tensor of shape `(N, in_chans, H, W)`.

#         Returns
#         -------
#         torch.Tensor
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         stack = []
#         output = image

#         # Down path
#         for layer in self.down_sample_layers:
#             output = layer(output)
#             stack.append(output)
#             output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

#         # Bottleneck
#         output = self.bottleneck(output)

#         # Up path
#         for transpose, disco in zip(self.up_transpose, self.up):
#             skip = stack.pop()
#             output = transpose(output)
#             output = _match_hw(output, skip, mode="reflect")

#             if DEBUG_SHAPES:
#                 assert output.shape[-2:] == skip.shape[-2:
#                                                        ], (output.shape, skip.shape)

#             output = torch.cat([output, skip], dim=1)
#             output = disco(output)

#             # Guard against any tiny drift introduced by convolutions
#             output = _match_hw(output, skip, mode="reflect")

#             if DEBUG_SHAPES:
#                 assert output.shape[-2:] == skip.shape[-2:
#                                                        ], (output.shape, skip.shape)

#         return output


# class DISCOBlock(nn.Module):
#     """
#     Two DISCO layers each followed by InstanceNorm2d, LeakyReLU, and Dropout2d.
#     """

#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         radius_cutoff: float,
#         drop_prob: float,
#         in_shape: Tuple[int, int],
#         kernel_shape: Tuple[int, int] = (3, 4),
#         basis_type: Literal["piecewise_linear",
#                             "morlet", "zernike"] = "piecewise_linear",
#     ):
#         super().__init__()

#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.drop_prob = drop_prob

#         self.layers = nn.Sequential(
#             DISCO2d(
#                 in_chans,
#                 out_chans,
#                 in_shape=in_shape,
#                 out_shape=in_shape,
#                 kernel_shape=kernel_shape,
#                 bias=False,
#                 radius_cutoff=radius_cutoff,
#                 basis_type=basis_type,
#             ),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#             DISCO2d(
#                 out_chans,
#                 out_chans,
#                 in_shape=in_shape,
#                 out_shape=in_shape,
#                 kernel_shape=kernel_shape,
#                 bias=False,
#                 radius_cutoff=radius_cutoff,
#                 basis_type=basis_type,
#             ),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#         )

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         return self.layers(image)


# class TransposeDISCOBlock(nn.Module):
#     """
#     Upsample (bilinear) -> DISCO -> InstanceNorm2d -> LeakyReLU
#     """

#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         radius_cutoff: float,
#         in_shape: Tuple[int, int],
#         kernel_shape: Tuple[int, int] = (3, 4),
#         basis_type: Literal["piecewise_linear",
#                             "morlet", "zernike"] = "piecewise_linear",
#     ):
#         super().__init__()

#         self.in_chans = in_chans
#         self.out_chans = out_chans

#         self.layers = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             DISCO2d(
#                 in_chans,
#                 out_chans,
#                 in_shape=(2 * in_shape[0], 2 * in_shape[1]),
#                 out_shape=(2 * in_shape[0], 2 * in_shape[1]),
#                 kernel_shape=kernel_shape,
#                 bias=False,
#                 radius_cutoff=(radius_cutoff / 2),
#                 basis_type=basis_type,
#             ),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         return self.layers(image)
