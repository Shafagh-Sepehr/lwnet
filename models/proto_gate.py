"""Opt-in prototype channel gate at U2 decoder stages.

Label: C2-corrected -- FDPR prose intent (recalibration incl. suppression);
FDPR-literal MPCR is amplification-only and not implemented.

All variants are identity at initialization, so an enabled model starts
feature-off-equivalent. The prototype descriptor is a spatially constant
per-channel vector, as in the FDPR module:

    p = GAP(f) -> Conv1x1 -> g -> out = f * g

Variants:
    corrected: g = 2 * sigmoid(p) in (0, 2). Zero-initialized 1x1 convolution
        makes g = 1 exactly at initialization; suppression (g < 1) is
        available during training. This is the headline C2-corrected form.
    se: g = 1 + clamp(scale, 0, 1) * sigmoid(p) in (1, 2). Amplification-only
        control-grade proxy for the closest honest MPCR-literal behavior.
        It is not a faithful MPCR implementation.
    plain: out = f + Conv1x1(f), no gate. Parameter-matched plain-capacity
        control (same 1x1 parameter count as corrected), zero-initialized.
"""

import torch
import torch.nn as nn

PROTO_GATE_TYPES = ['none', 'corrected', 'se', 'plain']


def _global_average(x):
    return x.mean(dim=(2, 3), keepdim=True)


class CorrectedProtoGate(nn.Module):
    """C2-corrected prototype gate: g = 2 * sigmoid(conv(GAP(f))) in (0, 2)."""

    def __init__(self, channels):
        super(CorrectedProtoGate, self).__init__()
        self.prototype = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.prototype.weight)
        nn.init.zeros_(self.prototype.bias)

    def channel_gain(self, x):
        return 2.0 * torch.sigmoid(self.prototype(_global_average(x)))

    def forward(self, x):
        return x * self.channel_gain(x)


class SEProtoGate(nn.Module):
    """Amplification-only control: g = 1 + s * sigmoid(p), s = clamp(scale, 0, 1)."""

    def __init__(self, channels):
        super(SEProtoGate, self).__init__()
        self.prototype = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.prototype.weight)
        nn.init.zeros_(self.prototype.bias)
        self.gate_scale = nn.Parameter(torch.zeros(()))

    def channel_gain(self, x):
        scale = self.gate_scale.clamp(0.0, 1.0)
        return 1.0 + scale * torch.sigmoid(self.prototype(_global_average(x)))

    def forward(self, x):
        return x * self.channel_gain(x)


class PlainProtoControl(nn.Module):
    """Matched-budget control: f + Conv1x1(f), no gating."""

    def __init__(self, channels):
        super(PlainProtoControl, self).__init__()
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        return x + self.project(x)


def build_proto_gate(proto_gate, channels):
    if proto_gate == 'corrected':
        return CorrectedProtoGate(channels)
    if proto_gate == 'se':
        return SEProtoGate(channels)
    if proto_gate == 'plain':
        return PlainProtoControl(channels)
    raise ValueError('unsupported proto_gate type {!r}'.format(proto_gate))
