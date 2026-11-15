"""Opt-in cross-gated fusion variants at the U1-to-U2 half-scale bridge.

Every variant is a zero-initialized residual on the U2 decoder feature:

    y = B + output_gate * project(fused(A, B))

with ``output_gate`` initialized to exactly 0.0, so an enabled instance is
feature-off-equivalent at initialization.

Labels:
    corrected: C3-corrected (CFF prose form),
        Cat(A * g_B, B * g_A) with g_X = sigmoid(DWConv3x3(X)).
    plain: matched-budget control, project(Cat(A, B)) with an ordinary 3x3
        convolution and no gates.
    literal: CFF-literal (FDPR Eqs. 13-15 as printed), a degenerate
        comparator only. The printed product W_low * W_high appears twice
        because element-wise multiplication commutes, so the fused tensor is
        the same map channel-duplicated. It is never the headline mechanism.

The 3x3 output projection is shared by all three variants: it is the seam
adaptation from the published 2C-wide fusion tensor back to the 16-channel
U2 decoder feature, and it keeps the plain control capacity-matched. It is
not part of FDPR's published Eq. 15 fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_TYPES = ['none', 'corrected', 'plain', 'literal']


class CrossGatedFusion(nn.Module):
    """C3-corrected (CFF prose form) cross-gated fusion module."""

    def __init__(self, channels=16):
        super(CrossGatedFusion, self).__init__()
        self.gate_a = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                groups=channels)
        self.gate_b = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                groups=channels)
        self.project = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, b, a):
        gate_b = torch.sigmoid(self.gate_b(b))
        gate_a = torch.sigmoid(self.gate_a(a))
        fused = torch.cat([a * gate_b, b * gate_a], dim=1)
        return b + self.output_gate * self.project(fused)


class PlainConcatFusion(nn.Module):
    """Matched-budget plain control: concat plus one ordinary 3x3 conv."""

    def __init__(self, channels=16):
        super(PlainConcatFusion, self).__init__()
        self.project = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, b, a):
        fused = torch.cat([a, b], dim=1)
        return b + self.output_gate * self.project(fused)


class CFFLiteralFusion(nn.Module):
    """CFF-literal degenerate comparator (FDPR Eqs. 13-15 as printed)."""

    def __init__(self, channels=16):
        super(CFFLiteralFusion, self).__init__()
        self.gate_a = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                groups=channels)
        self.gate_b = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                groups=channels)
        self.project = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, b, a):
        w_low = torch.sigmoid(F.relu(self.gate_a(a)))
        w_high = torch.sigmoid(F.gelu(self.gate_b(b)))
        fused = torch.cat([w_low * w_high, w_high * w_low], dim=1)
        return b + self.output_gate * self.project(fused)


def build_cross_gate_fusion(fusion_type, channels):
    if fusion_type == 'corrected':
        return CrossGatedFusion(channels)
    if fusion_type == 'plain':
        return PlainConcatFusion(channels)
    if fusion_type == 'literal':
        return CFFLiteralFusion(channels)
    raise ValueError('unsupported cross_gate_fusion type {!r}'.format(fusion_type))
