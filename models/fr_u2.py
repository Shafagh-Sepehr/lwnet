import torch
import torch.nn as nn
import torch.nn.functional as F

from .fr_initialization import initialize_fr_weights


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        return out + residual


class DownsampleTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, target_size=None):
        out = self.bn(self.relu(self.conv(x)))
        if target_size is not None and out.shape[-2:] != tuple(target_size):
            out = F.interpolate(
                out, size=target_size, mode="bilinear", align_corners=False
            )
        return out


class UpsampleTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, target_size=None):
        out = self.bn(self.relu(self.conv(x)))
        if target_size is not None and out.shape[-2:] != tuple(target_size):
            out = F.interpolate(
                out, size=target_size, mode="bilinear", align_corners=False
            )
        return out


class MultiResolutionFusionStage(nn.Module):
    """Exchange features between adjacent resolutions in parallel."""

    def __init__(self, base_channels, dilation):
        super().__init__()
        half_channels = 2 * base_channels
        quarter_channels = 4 * base_channels

        self.up_1_to_0 = UpsampleTransition(half_channels, base_channels)
        self.down_0_to_1 = DownsampleTransition(base_channels, half_channels)
        self.up_2_to_1 = UpsampleTransition(quarter_channels, half_channels)
        self.down_1_to_2 = DownsampleTransition(half_channels, quarter_channels)

        self.block0 = DilatedResidualBlock(2 * base_channels, base_channels, dilation)
        self.block1 = DilatedResidualBlock(
            3 * half_channels, half_channels, dilation
        )
        self.block2 = DilatedResidualBlock(
            2 * quarter_channels, quarter_channels, dilation
        )

    def forward(self, f0, f1, f2):
        # Every branch is computed from the same previous-stage features.
        up_f1 = self.up_1_to_0(f1, target_size=f0.shape[-2:])
        down_f0 = self.down_0_to_1(f0, target_size=f1.shape[-2:])
        up_f2 = self.up_2_to_1(f2, target_size=f1.shape[-2:])
        down_f1 = self.down_1_to_2(f1, target_size=f2.shape[-2:])

        new_f0 = self.block0(torch.cat([f0, up_f1], dim=1))
        new_f1 = self.block1(torch.cat([f1, down_f0, up_f2], dim=1))
        new_f2 = self.block2(torch.cat([f2, down_f1], dim=1))
        return new_f0, new_f1, new_f2


class FullResolutionMultiResU2(nn.Module):
    def __init__(
        self,
        in_c,
        n_classes,
        base_channels=8,
        dilations=(1, 2, 4, 2, 1),
    ):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if len(dilations) != 5:
            raise ValueError("dilations must contain exactly 5 values")
        if any(d <= 0 for d in dilations):
            raise ValueError("dilations must contain positive integers")

        self.base_channels = int(base_channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.n_classes = n_classes
        self.in_c = in_c
        c = self.base_channels

        self.stem = DilatedResidualBlock(in_c, c, dilation=1)
        self.down_0_to_1 = DownsampleTransition(c, 2 * c)
        self.half_init = DilatedResidualBlock(2 * c, 2 * c, dilation=1)
        self.down_1_to_2 = DownsampleTransition(2 * c, 4 * c)
        self.quarter_init = DilatedResidualBlock(4 * c, 4 * c, dilation=1)

        self.fusion_stages = nn.ModuleList(
            MultiResolutionFusionStage(c, dilation)
            for dilation in self.dilations
        )

        self.up_2_to_1 = UpsampleTransition(4 * c, 2 * c)
        self.half_fusion_block = DilatedResidualBlock(4 * c, 2 * c, dilation=1)
        self.up_1_to_0 = UpsampleTransition(2 * c, c)
        self.full_fusion_block = DilatedResidualBlock(2 * c, c, dilation=1)

        self.detail = nn.Sequential(
            nn.Conv2d(c + in_c, c, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(c),
        )
        self.detail_dilation1 = self._head_branch(c, dilation=1)
        self.detail_dilation2 = self._head_branch(c, dilation=2)
        self.detail_dilation3 = self._head_branch(c, dilation=3)
        self.head_fusion = nn.Sequential(
            nn.Conv2d(3 * c, c, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(c),
        )
        self.final = nn.Conv2d(c, n_classes, kernel_size=1)

        self._initialize_weights()

    @staticmethod
    def _head_branch(channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )

    def _initialize_weights(self):
        initialize_fr_weights(self)

    def forward(self, u2_input):
        f0 = self.stem(u2_input)
        f1 = self.half_init(
            self.down_0_to_1(f0, target_size=(f0.shape[-2] // 2, f0.shape[-1] // 2))
        )
        f2 = self.quarter_init(
            self.down_1_to_2(
                f1,
                target_size=(f1.shape[-2] // 2, f1.shape[-1] // 2),
            )
        )

        for stage in self.fusion_stages:
            f0, f1, f2 = stage(f0, f1, f2)

        p1 = self.up_2_to_1(f2, target_size=f1.shape[-2:])
        p1 = self.half_fusion_block(torch.cat([f1, p1], dim=1))
        p0 = self.up_1_to_0(p1, target_size=f0.shape[-2:])
        p0 = self.full_fusion_block(torch.cat([f0, p0], dim=1))

        base_detail = self.detail(torch.cat([p0, u2_input], dim=1))
        multi = torch.cat(
            [
                self.detail_dilation1(base_detail),
                self.detail_dilation2(base_detail),
                self.detail_dilation3(base_detail),
            ],
            dim=1,
        )
        final_feature = torch.relu(self.head_fusion(multi) + base_detail)
        return self.final(final_feature)
