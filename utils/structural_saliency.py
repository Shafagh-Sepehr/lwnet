"""Structural-saliency target generation for LwNet self-supervision.

Implements the Gaussian high-frequency-content (HFC) structural target used by
RaffeSDG's structural-saliency pretext task, adapted to LwNet's [0,1] image
range.  The module is self-contained: it knows nothing about FreeSDG, Raffe,
augmentation modes, or the training script.  It receives an RGB image and a
binary FOV mask and returns a three-channel reconstruction target in [-1, +1].

Target definition:
    * median-fill the background (median over ALL pixels + 0.2),
    * replication-padded depthwise Gaussian blur,
    * residual = ratio * (filled - blurred), clamped to [-1, +1],
    * outside the FOV the target is exactly -1.

The module contains no trainable parameters; the Gaussian kernel is registered
as a buffer.  The target is supervision and must be produced under
``torch.no_grad()`` by the caller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gaussian_kernel2d(kernel_size, sigma):
    """Build a normalized 2D Gaussian kernel of shape [1, 1, K, K].

    Constructed in float64, normalized, then cast to float32. The
    1D Gaussian is ``g(x) = exp(-x^2 / (2 sigma^2))`` over the integer
    coordinate range ``-(K//2) .. +(K//2)``, normalized to sum 1, and the 2D
    kernel is the outer product, normalized once more.
    """
    if kernel_size < 1:
        raise ValueError('kernel_size must be >= 1')
    if sigma <= 0:
        raise ValueError('sigma must be > 0')
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float64)
    g = torch.exp(-((coords - center) ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel2d = g[:, None] * g[None, :]
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.to(torch.float32).unsqueeze(0).unsqueeze(0)


class StructuralSaliencyTarget(nn.Module):
    """Gaussian HFC structural-saliency reconstruction target.

    ``forward(image01, mask01)`` returns a ``[B, 3, H, W]`` tensor in
    ``[-1, +1]``: the amplified high-frequency residual inside the FOV and
    exactly ``-1`` outside it.  No trainable parameters; the Gaussian kernel is
    a registered buffer.
    """

    def __init__(self, kernel_size=27, sigma=9.0, ratio=4.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.ratio = float(ratio)
        self.register_buffer(
            'kernel', build_gaussian_kernel2d(self.kernel_size, self.sigma))

    def forward(self, image01, mask01):
        if image01.ndim != 4:
            raise ValueError('image01 must be [B, C, H, W], got ndim {}'
                             .format(image01.ndim))
        if mask01.ndim != 4:
            raise ValueError('mask01 must be [B, 1, H, W], got ndim {}'
                             .format(mask01.ndim))
        if image01.shape[1] != 3:
            raise ValueError('image01 must have 3 channels, got {}'
                             .format(image01.shape[1]))
        if mask01.shape[1] != 1:
            raise ValueError('mask01 must have 1 channel, got {}'
                             .format(mask01.shape[1]))
        if image01.shape[0] != mask01.shape[0]:
            raise ValueError('image01 and mask01 batch dims differ: {} vs {}'
                             .format(image01.shape[0], mask01.shape[0]))
        if image01.shape[2:] != mask01.shape[2:]:
            raise ValueError('image01 and mask01 spatial dims differ: {} vs {}'
                             .format(image01.shape[2:], mask01.shape[2:]))
        if not torch.isfinite(image01).all():
            raise ValueError('image01 contains non-finite values')
        if not torch.isfinite(mask01).all():
            raise ValueError('mask01 contains non-finite values')

        mask01 = (mask01 > 0.5).to(image01.dtype)

        b, c, h, w = image01.shape

        # Median fill outside the FOV: median over ALL pixels,
        # + 0.2, then replace background pixels. The +0.2 offset is
        # intentionally unclamped, so a bright image (median near 1.0) yields
        # a background fill slightly above the nominal [0,1] range; this
        # matches the original RaffeSDG HFC implementation and is harmless
        # because the final target is clamped to [-1, +1].
        median = image01.flatten(2).median(dim=2).values
        median = median.view(b, c, 1, 1)
        median = median + 0.2
        x_filled = image01 * mask01 + median * (1.0 - mask01)

        # Replication-padded depthwise Gaussian convolution.
        padding = self.kernel_size // 2
        padded = F.pad(x_filled, (padding, padding, padding, padding),
                       mode='replicate')
        kernel = self.kernel.to(device=x_filled.device, dtype=x_filled.dtype)
        blurred = F.conv2d(padded, kernel.repeat(c, 1, 1, 1), groups=c)

        # HFC residual.
        residual = self.ratio * (x_filled - blurred)
        residual = torch.clamp(residual, -1.0, 1.0)
        target = (residual + 1.0) * mask01 - 1.0
        return target
