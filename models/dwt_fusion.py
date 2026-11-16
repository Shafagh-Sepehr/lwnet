"""Opt-in lightweight Haar-DWT split/fuse at the U2 encoder bottleneck.

Label: FDPR pillar-2 main idea in risk-controlled form -- one 1-level Haar
split at one feature stage. This is NOT FDPR's dual-branch encoder and is not
a FDPR reproduction.

The transform is parameter-free. A fixed orthonormal 2x2 Haar bank is stored
as non-persistent buffers and applied per channel (no cross-channel mixing):

    LL = (x00 + x01 + x10 + x11) / 2
    LH = (x00 + x01 - x10 - x11) / 2
    HL = (x00 - x01 + x10 - x11) / 2
    HH = (x00 - x01 - x10 + x11) / 2

The low group is LL (C channels); the high group is the tuple
{LH, HL, HH} (3C channels). Both groups live at half resolution.

Variants (both zero-initialized residuals on the bottleneck feature):

    dwt: low/high 1x1 projections, depthwise 3x3 sigmoid cross-gates
        (CFF-prose form, same family as ``CrossGatedFusion``), then one 1x1
        projection of Cat(low * g_high, high * g_low) to C. That fused map
        becomes the learned LL band, the original high bands are preserved,
        and the fixed Haar synthesis reconstructs a full-resolution feature.
        The output gate is a scalar initialized to exactly 0.0.
    plain: matched-parameter control using the same Haar split, the same
        low/high projections, depthwise 3x3 convolutions without sigmoid
        (same shapes/count as the gates), plain concat, and the same 1x1
        projection and synthesis. It tests whether the cross-gated sigmoid
        fusion helps beyond the same-capacity plain convolution path.

The control deliberately uses the same Haar split (not the plan's avg-pool
split) so the comparison isolates the fusion mechanism at an exactly matched
parameter budget. An avg-pool split would change both the decomposition and
the fusion; it is left as a separate follow-up if S4 shows signal.

Synthesis is a labeled adaptation: FDPR projects each sub-band to C and feeds
a dual-branch encoder; it never inverts the transform. Here a single fused
low-band map cannot recover three detail bands, so the original Haar detail
bands are retained unchanged and the exact inverse transform recombines them
with the learned low band. The transform itself (split then reconstruct) is
exact.

Odd spatial sizes are padded on the bottom/right with replicated border rows
and columns before the 2x2 non-overlapping transform, then cropped back after
synthesis. Because synthesis reconstructs the padded tensor exactly and the
padding is a copy of an edge row/column, the crop restores the original size
and the round trip remains exact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DWT_FUSION_TYPES = ['none', 'dwt', 'plain']
DWT_STAGE_NAME = 'u2_bottleneck'


def _haar_bank():
    """Fixed orthonormal 1-level 2D Haar analysis/synthesis bank, (4,1,2,2)."""
    return torch.tensor([
        [[0.5, 0.5], [0.5, 0.5]],      # LL
        [[0.5, 0.5], [-0.5, -0.5]],    # LH
        [[0.5, -0.5], [0.5, -0.5]],    # HL
        [[0.5, -0.5], [-0.5, 0.5]],    # HH
    ], dtype=torch.float32).unsqueeze(1)


class HaarTransform(nn.Module):
    """Parameter-free 1-level 2D Haar split and exact synthesis per channel."""

    def __init__(self, channels):
        super(HaarTransform, self).__init__()
        if int(channels) <= 0:
            raise ValueError('HaarTransform channels must be > 0')
        self.channels = int(channels)
        bank = _haar_bank().repeat(self.channels, 1, 1, 1)
        # Non-persistent: deterministic constants, never checkpoint state.
        self.register_buffer('analysis_weight', bank.clone(), persistent=False)
        self.register_buffer('synthesis_weight', bank.clone(), persistent=False)

    def split(self, x):
        """Return (LL, Cat(LH,HL,HH), (original_height, original_width)).

        The two returned feature groups are at floor(padded_size / 2).
        """
        if x.dim() != 4:
            raise ValueError('HaarTransform expects a 4D tensor, got {!r}'.format(
                tuple(x.shape)))
        channels = x.shape[1]
        if channels != self.channels:
            raise ValueError('HaarTransform built for {} channels, got {}'.format(
                self.channels, channels))
        height, width = x.shape[-2:]
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        bands = F.conv2d(x, self.analysis_weight, stride=2, groups=channels)
        batch, _, half_h, half_w = bands.shape
        bands = bands.view(batch, channels, 4, half_h, half_w)
        low = bands[:, :, 0]
        high = bands[:, :, 1:].reshape(batch, 3 * channels, half_h, half_w)
        return low, high, (height, width)

    def reconstruct(self, low, high):
        """Exact inverse Haar; returns the full (possibly padded) resolution."""
        if low.dim() != 4 or high.dim() != 4:
            raise ValueError('HaarTransform.reconstruct expects 4D tensors')
        batch, channels, half_h, half_w = low.shape
        if channels != self.channels:
            raise ValueError('HaarTransform built for {} channels, got {}'.format(
                self.channels, channels))
        if high.shape[0] != batch or high.shape[1] != 3 * channels \
                or high.shape[-2:] != low.shape[-2:]:
            raise ValueError(
                'high group must be {} channels at the low group spatial size, '
                'got {}'.format(3 * channels, tuple(high.shape)))
        high = high.view(batch, channels, 3, half_h, half_w)
        bands = torch.cat([low.unsqueeze(2), high], dim=2)
        bands = bands.reshape(batch, 4 * channels, half_h, half_w)
        return F.conv_transpose2d(
            bands, self.synthesis_weight, stride=2, groups=channels)


class HaarDWTFusion(nn.Module):
    """Headline DWT variant: Haar split + cross-gated fusion + synthesis."""

    def __init__(self, channels=32):
        super(HaarDWTFusion, self).__init__()
        self.channels = int(channels)
        self.haar = HaarTransform(self.channels)
        self.low_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.high_proj = nn.Conv2d(3 * self.channels, self.channels, kernel_size=1)
        self.gate_low = nn.Conv2d(self.channels, self.channels, kernel_size=3,
                                  padding=1, groups=self.channels)
        self.gate_high = nn.Conv2d(self.channels, self.channels, kernel_size=3,
                                   padding=1, groups=self.channels)
        self.project = nn.Conv2d(2 * self.channels, self.channels, kernel_size=1)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        low, high, (height, width) = self.haar.split(x)
        low_p = self.low_proj(low)
        high_p = self.high_proj(high)
        gate_low = torch.sigmoid(self.gate_low(low_p))
        gate_high = torch.sigmoid(self.gate_high(high_p))
        fused = self.project(torch.cat(
            [low_p * gate_high, high_p * gate_low], dim=1))
        reconstructed = self.haar.reconstruct(fused, high)
        reconstructed = reconstructed[..., :height, :width]
        return x + self.output_gate * (reconstructed - x)


class HaarPlainControl(nn.Module):
    """Matched-parameter plain concat control on the same Haar split."""

    def __init__(self, channels=32):
        super(HaarPlainControl, self).__init__()
        self.channels = int(channels)
        self.haar = HaarTransform(self.channels)
        self.low_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.high_proj = nn.Conv2d(3 * self.channels, self.channels, kernel_size=1)
        self.low_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3,
                                  padding=1, groups=self.channels)
        self.high_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3,
                                   padding=1, groups=self.channels)
        self.project = nn.Conv2d(2 * self.channels, self.channels, kernel_size=1)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        low, high, (height, width) = self.haar.split(x)
        low_p = self.low_proj(low)
        high_p = self.high_proj(high)
        fused = self.project(torch.cat(
            [self.low_conv(low_p), self.high_conv(high_p)], dim=1))
        reconstructed = self.haar.reconstruct(fused, high)
        reconstructed = reconstructed[..., :height, :width]
        return x + self.output_gate * (reconstructed - x)


def build_dwt_fusion(dwt_fusion, channels):
    if dwt_fusion == 'dwt':
        return HaarDWTFusion(channels)
    if dwt_fusion == 'plain':
        return HaarPlainControl(channels)
    raise ValueError('unsupported dwt_fusion type {!r}'.format(dwt_fusion))
