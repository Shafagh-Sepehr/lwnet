import torch
import torch.nn as nn

from .fr_u2 import (
    DilatedResidualBlock,
    DownsampleTransition,
    MultiResolutionFusionStage,
    UpsampleTransition,
)
from .fr_initialization import initialize_fr_weights


class FullResolutionMultiResU2Lite(nn.Module):
    """Fixed-width, three-stage full-resolution multi-resolution U2."""

    base_channels = 4
    dilations = (1, 2, 1)

    def __init__(self, in_c, n_classes):
        super().__init__()
        c = self.base_channels
        self.in_c = in_c
        self.n_classes = n_classes

        self.stem = DilatedResidualBlock(in_c, c, dilation=1)
        self.down_0_to_1 = DownsampleTransition(c, 2 * c)
        self.half_init = DilatedResidualBlock(2 * c, 2 * c, dilation=1)
        self.down_1_to_2 = DownsampleTransition(2 * c, 4 * c)
        self.quarter_init = DilatedResidualBlock(4 * c, 4 * c, dilation=1)

        self.fusion_stages = nn.ModuleList(
            MultiResolutionFusionStage(c, dilation)
            for dilation in self.dilations
        )

        self.out_up_2_to_1 = UpsampleTransition(4 * c, 2 * c)
        self.half_fusion = DilatedResidualBlock(4 * c, 2 * c, dilation=1)
        self.out_up_1_to_0 = UpsampleTransition(2 * c, c)
        self.full_fusion = DilatedResidualBlock(2 * c, c, dilation=1)
        self.final = nn.Conv2d(c, n_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        initialize_fr_weights(self)

    def forward(self, u2_input):
        f0 = self.stem(u2_input)
        f1 = self.half_init(
            self.down_0_to_1(
                f0,
                target_size=(f0.shape[-2] // 2, f0.shape[-1] // 2),
            )
        )
        f2 = self.quarter_init(
            self.down_1_to_2(
                f1,
                target_size=(f1.shape[-2] // 2, f1.shape[-1] // 2),
            )
        )

        for stage in self.fusion_stages:
            f0, f1, f2 = stage(f0, f1, f2)

        p1 = self.out_up_2_to_1(f2, target_size=f1.shape[-2:])
        p1 = self.half_fusion(torch.cat([f1, p1], dim=1))
        p0 = self.out_up_1_to_0(p1, target_size=f0.shape[-2:])
        p0 = self.full_fusion(torch.cat([f0, p0], dim=1))
        return self.final(p0)
