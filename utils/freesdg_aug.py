"""FreeSDG-style Frequency-Mixed Augmentation (FMAug) for LwNet.

Reference: FreeSDG (arXiv:2307.09005, MICCAI'23). This module reimplements the
released repository's HFC filter bank, median background padding, Gaussian
mixing, and fixed "anchor" HFC preprocessing, adapted to LwNet's dataset-side,
per-sample augmentation pipeline.

Range contract (plan §1.2/§1.3):
    * Low-level filters (``median_padding``, ``HFCFilter``, ``GaussianMixUp``)
      accept and return tensors in [-1, 1] only.
    * Only ``FreeSDGAugmentor`` performs the [0,1] <-> [-1,1] conversions, so
      LwNet's W-Net keeps receiving [0,1] inputs.

Randomness (plan §6):
    * FMAug never consumes the global Python/NumPy/PyTorch RNG used by LwNet's
      existing paired transforms. ``FreeSDGAugmentor`` owns a dedicated
      ``random.Random(freesdg_seed)`` stream.
    * Under ``num_workers > 0`` the dedicated stream is (re)derived lazily per
      DataLoader worker from the FreeSDG seed + the worker's torch seed, so
      workers get independent streams (see ``fmaug_seed_for_worker``).

Documented deviations from the released FreeSDG implementation (full log in
docs/freesdg_deviations.md; plan §19):
    * §19.1  W-Net receives [0,1], not FreeSDG's [-1,1] HFC representation.
    * §19.2  FMAug is applied per sample in ``TrainDataset``; the released code
              selects filters/rectangles once per batch. The dedicated-RNG draw
              order is therefore ours: filter i, filter j, then rectangle
              (upstream drew rectangle first).
    * §19.4  ``repo`` and ``paper`` rectangle policies are kept strictly
              separate; they are not reconciled.
    * §19.5  ``paper`` policy interprets the rebuttal's singular patch "size"
              as an s x s square, and requires 512x512 input (asserted).
    * §19.6  ``FreeSDGAugmentor.to_pil_uint8`` introduces ~1/255 quantization
              on the *train* path only, because LwNet's paired transforms are
              PIL-based. Evaluation anchor processing stays float.
    * §19.8  The filter bank contains exactly 20 filters: zip(range(5, 50, 2),
              range(2, 22)) truncates at the shorter iterable.
    * §29.1  With ``num_workers > 0`` and ``persistent_workers=False`` the
              worker-local FMAug RNG restarts its sequence every epoch
              (workers are recreated per iterator). Exact replay is primarily
              validated under the repository default ``num_workers=0``.
"""

import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Released FreeSDG filter bank: zip(range(5, 50, 2), range(2, 22)).
# Python's zip terminates at the shorter iterable -> exactly 20 (width, sigma)
# pairs: (5,2), (7,3), ..., (43,21). NOT 23 (plan §3.1/§19.8).
FREESDG_FILTER_BANK = tuple(zip(range(5, 50, 2), range(2, 22)))

MIX_POLICIES = ("repo", "paper")

AUG_MODES = ("fmaug", "fixed_hfc", "random_hfc", "raffe_filter",
             "raffe_smooth_mix")

# --- Official RaffeSDG random-frequency-filtering constants ---------------
# Transcribed from the official repository (Non-IID_Medical_Image_Segmentation,
# utils/augmentations/fmaug.py, hfc_type='butterworth_per_channel'):
RAFFE_D0_NUM = 12        # d0_num
RAFFE_D0_MAX = 0.1       # d0_max
RAFFE_REMAP_RATIO = 4.1  # remap_ratio
RAFFE_N_LIST = (1, 2, 3)  # n_list
RAFFE_PCTL = 3           # normalization_percentile_threshold
# Test-time filter of the official RaffeSDG model (raffesdg_segmentation_model):
RAFFE_TEST_D0_RATIO = 0.005
RAFFE_TEST_N = 1


def raffe_d0_list():
    """Official d0_list: exponential remap of linspace(0, d0_max, 12)."""
    lin = [i / (RAFFE_D0_NUM - 1) for i in range(RAFFE_D0_NUM)]
    return [RAFFE_D0_MAX * (math.exp(RAFFE_REMAP_RATIO * l) - 1) /
            (math.exp(RAFFE_REMAP_RATIO) - 1) for l in lin]


def build_corner_butterworth_map(height, width, d0_ratio, n,
                                 dtype=torch.float32):
    """Official FourierButterworthHFCFilter.filter_map (vectorized).

    Semantics transcribed literally from the reference implementation,
    including its index transposition (``filter_map[j, i]``):

        d0 = int(max((d0_ratio * min(H, W)) // 2, 1))
        for i in rows, j in cols:
            d = min euclidean distance of (i, j) to the four image corners
            map[j, i] = 1 / (1 + (d0 / (d + 1)) ** (2 * n))

    With no fftshift, torch.fft.fft2 keeps low frequencies at the spectrum
    corners, so this map is a corner-centered Butterworth high-pass. The
    double loop is vectorized for speed; results are numerically identical
    (verified against the literal loop in test_freesdg_smoke.py).
    """
    d0 = int(max((d0_ratio * min(height, width)) // 2, 1))
    if height != width:
        # the reference implementation's transposed assignment
        # (filter_map[j, i]) overflows for non-square sizes — square input
        # is the only officially defined case (it uses 512x512)
        raise ValueError(
            "FourierButterworth corner map requires square input, "
            "got {}x{}".format(height, width))
    rows = torch.arange(height, dtype=torch.float64)
    cols = torch.arange(width, dtype=torch.float64)
    corners = ((0.0, 0.0), (0.0, float(width - 1)),
               (float(height - 1), float(width - 1)), (float(height - 1), 0.0))
    dist = None
    for (cx, cy) in corners:
        # reference: sqrt((i - x)^2 + (j - y)^2) for row i, col j
        d = (rows - cx).unsqueeze(1) ** 2 + (cols - cy).unsqueeze(0) ** 2
        d = d.sqrt()
        dist = d if dist is None else torch.minimum(dist, d)
    fmap = 1.0 / (1.0 + (d0 / (dist + 1.0)) ** (2 * n))
    return fmap.t().to(dtype)  # reference assigns filter_map[j, i]


class FourierButterworthView:
    """One official FourierButterworthHFCFilter, operating in [0,1].

    Faithful port of the official train/test behavior used by RaffeSDG
    (do_median_padding=False):  fft2 -> corner-map multiply -> abs(ifft2) ->
    per-sample/per-channel [3rd, 97th] percentile normalization (percentiles
    computed on the 1/256-quantized copy, applied to the unquantized tensor,
    exactly as the reference) -> FOV masking (res * mask).

    Deviation from the official pipeline (documented): the official network
    consumed unclamped values after a [0,1]->[-1,1] remap; LwNet requires
    [0,1] inputs, so the normalized output is clipped to [0,1] here
    (percentiles are still computed on the unclamped plane, exactly like
    the reference). Background (mask == 0) is 0, consistent with the
    reference's res * mask.
    """

    def __init__(self, height, width, d0_ratio, n):
        self.d0_ratio = float(d0_ratio)
        self.n = int(n)
        self.fmap = build_corner_butterworth_map(height, width, self.d0_ratio,
                                                 self.n)

    def __call__(self, x01, mask01):
        if x01.dim() != 4:
            raise ValueError("x01 must be [B, C, H, W] in [0,1]")
        fmap = self.fmap.to(device=x01.device, dtype=x01.dtype)
        spec = torch.fft.fft2(x01) * fmap
        res = torch.abs(torch.fft.ifft2(spec))  # >= 0 by construction
        # Fastest percentile path is chosen automatically: numpy on CPU (its
        # O(n) partition beats torch.quantile's full sort), torch.quantile on
        # CUDA (avoids a per-plane GPU->CPU synchronization).
        if res.is_cuda:
            out = self._percentile_vectorized(res)
        else:
            out = self._percentile_legacy(res)
        return out * mask01.to(dtype=out.dtype)

    @staticmethod
    def _percentile_legacy(res):
        # official percentile normalization, per sample and channel
        out = torch.empty_like(res)
        for b in range(res.shape[0]):
            for c in range(res.shape[1]):
                plane = res[b, c].detach().cpu().numpy()
                # reference temp: (res*256).int().float()/256 (truncating cast)
                temp = np.trunc(plane * 256) / 256.0
                lo = float(np.percentile(temp, RAFFE_PCTL))
                hi = float(np.percentile(temp, 100 - RAFFE_PCTL))
                denom = hi - lo if hi > lo else 1e-8
                out[b, c] = torch.from_numpy(
                    ((plane - lo) / denom).clip(0.0, 1.0))
        return out

    @staticmethod
    def _percentile_vectorized(res):
        # Vectorized equivalent of _percentile_legacy: per-sample/per-channel
        # [3rd, 97th] percentile computed with torch.quantile (linear
        # interpolation, matching np.percentile) on the 1/256-quantized copy,
        # applied to the unquantized tensor. No .cpu().numpy() round-trip and
        # no Python per-plane loop; runs on the tensor's own device.
        #
        # Used only by the 'cuda' backend, where a .cpu().numpy() round-trip
        # would force a GPU->CPU synchronization per plane. On CPU this is
        # *slower* than numpy's np.percentile (torch.quantile sorts the whole
        # plane while np.percentile uses an O(n) partition), so both CPU
        # backends keep the numpy path (see benchmark_freesdg.py).
        b, c, h, w = res.shape
        flat = res.reshape(b * c, h * w)
        temp = torch.trunc(flat * 256) / 256.0
        lo = torch.quantile(temp, RAFFE_PCTL / 100.0, dim=1)
        hi = torch.quantile(temp, (100 - RAFFE_PCTL) / 100.0, dim=1)
        denom = hi - lo
        denom = torch.where(denom > 0, denom, torch.full_like(denom, 1e-8))
        out = ((flat - lo.unsqueeze(1)) / denom.unsqueeze(1)).clamp(0.0, 1.0)
        return out.reshape(b, c, h, w)


class RaffeFilterBank:
    """Official 36-filter bank: 12 remapped d0 ratios x n in {1,2,3}.

    Bank index i (official loop): d0_ratio = d0_list[i // 3], n = n_list[i % 3].
    Views are built lazily per spatial size (official uses fixed 512x512).
    """

    def __init__(self):
        d0s = raffe_d0_list()
        self.params = [(d0s[i // 3], RAFFE_N_LIST[i % 3])
                       for i in range(len(RAFFE_N_LIST) * RAFFE_D0_NUM)]
        self._views = {}  # (H, W) -> list of FourierButterworthView

    def views(self, height, width):
        key = (height, width)
        if key not in self._views:
            self._views[key] = [
                FourierButterworthView(height, width, d0, n)
                for d0, n in self.params]
        return self._views[key]

    def sample_indices(self, rng, channels=3):
        """Official per-channel independent choice (butterworth_per_channel)."""
        return [rng.randrange(0, len(self.params)) for _ in range(channels)]

    def apply(self, x01, mask01, indices):
        views = self.views(x01.shape[-2], x01.shape[-1])
        return torch.cat([views[indices[c]](x01[:, c:c + 1], mask01)
                          for c in range(x01.shape[1])], dim=1)


# Bounded cache of integer coordinate grids for the direct mask calculation.
# Keyed by (height, width, device, dtype); cleared when it grows too large so
# a long-lived process cannot accumulate one grid per (size, device) pair.
_COORD_CACHE = {}
_COORD_CACHE_MAX = 64


def _coord_grids(height, width, device=None, dtype=torch.int64):
    key = (int(height), int(width), str(device), str(dtype))
    grids = _COORD_CACHE.get(key)
    if grids is None:
        yy = torch.arange(height, dtype=dtype, device=device).view(-1, 1)
        xx = torch.arange(width, dtype=dtype, device=device).view(1, -1)
        grids = (yy, xx)
        if len(_COORD_CACHE) >= _COORD_CACHE_MAX:
            _COORD_CACHE.clear()
        _COORD_CACHE[key] = grids
    return grids


def direct_smooth_mask(points, height, width, expansion_step,
                       device=None, dtype=torch.float32):
    """Direct Chebyshev-distance equivalent of ``DT2SmoothMask.generate``.

    For binary point seeds and stride-1 5x5 max pooling with zero padding,
    repeated dilation admits a closed form. After iteration ``t`` a pixel is
    active iff its Chebyshev distance to the nearest seed ``d <= 2*t``, so the
    number of contributions is ``count = clamp(T - max(1, ceil(d/2)) + 1,
    0, T)`` and ``acc = step * count`` (plan §2). Integer distance arithmetic
    keeps the count exact; only the final ``step * count`` and min-max
    normalization are floating point, matching the reference's float32
    accumulation to within ~1e-7 (repeated ``+= step`` vs a single multiply).

    ``points`` is the same list of ``(row, col)`` seed coordinates the
    iterative reference consumes, so duplicate and boundary seeds are
    preserved exactly. ``expansion_step`` is the reference's ``T`` (100).
    """
    T = int(expansion_step)
    yy, xx = _coord_grids(height, width, device=device)
    d = None
    for (r, c) in points:
        dist = torch.maximum(torch.abs(yy - r), torch.abs(xx - c))
        d = dist if d is None else torch.minimum(d, dist)
    # integer ceil(d / 2) for d >= 0; seeds (d == 0) still enter at t == 1
    t_enter = (d + 1) // 2
    t_enter = torch.maximum(t_enter, torch.ones_like(t_enter))
    count = torch.clamp(T - t_enter + 1, 0, T)
    acc = (1.0 / T) * count.to(dtype)
    lo, hi = acc.min(), acc.max()
    if float(hi - lo) <= 0.0:
        return torch.zeros_like(acc)
    return (acc - lo) / (hi - lo)


class DT2SmoothMask:
    """Official DT2MASKGenerator (utils/mixup_mask.py), verbatim port.

    The official 'smooth' blending masks are OFFLINE files not distributed
    with the repository (OfflineMaskGenerator reads them from
    /data/lihaojin/random_mask), so the official in-repo continuous
    generator DT2MASKGenerator is used instead: 5 random seed points,
    100 steps of MaxPool2d(5, 1, 2) expansion accumulated with weight 1/100,
    then min-max normalized (MaskGenerator.generate_mask). Draw order per
    mask: 5 centers, each randint(0, size-1) (official hardcodes 511 for
    512; generalized to the actual size here — documented adaptation).
    All randomness comes from the CALLER's dedicated RNG.

    ``generate`` uses the direct Chebyshev-distance calculation (~300x
    faster than the reference iterative max-pool expansion, mathematically
    equivalent). The iterative reference is kept as ``_generate_iterative``
    for parity tests.
    """

    def __init__(self, center_num=5, expansion_step=100):
        self.center_num = center_num
        self.expansion_step = expansion_step
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def sample_points(self, rng, height, width):
        return [(rng.randint(0, height - 1), rng.randint(0, width - 1))
                for _ in range(self.center_num)]

    def generate(self, rng, height, width, points=None, device=None):
        if points is None:
            points = self.sample_points(rng, height, width)
        return direct_smooth_mask(points, height, width,
                                  self.expansion_step, device=device)

    def _generate_iterative(self, points, height, width):
        seeds = torch.zeros(1, height, width)
        for (r, c) in points:
            seeds[0, r, c] = 1.0
        acc = torch.zeros(1, height, width)
        step = 1.0 / self.expansion_step
        for _ in range(self.expansion_step):
            seeds = self.max_pool(seeds)
            acc += step * seeds
        m = acc.squeeze(0)
        # official generate_mask(): min-max normalize
        lo, hi = m.min(), m.max()
        if float(hi - lo) <= 0.0:
            return torch.zeros_like(m)
        return (m - lo) / (hi - lo)


def build_gaussian_kernel2d(k_sz, sigma, dtype=torch.float32, device=None):
    """Build a normalized 2D Gaussian kernel of shape [1, 1, k_sz, k_sz].

    Uses the explicit Gaussian formula and an outer product; each 1D vector is
    normalized so the 2D kernel sums to ~1 (this matches cv2.getGaussianKernel
    semantics used by the released FreeSDG implementation, where the Gaussian
    prefactor cancels under normalization). float32 by default; transfer to
    any dtype/device via the arguments (plan §4.1).
    """
    if k_sz < 1:
        raise ValueError("kernel size must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    center = (k_sz - 1) / 2.0
    coords = torch.arange(k_sz, dtype=torch.float64)
    g1d = torch.exp(-((coords - center) ** 2) / (2.0 * sigma * sigma))
    g1d = g1d / g1d.sum()
    kernel = torch.outer(g1d, g1d)
    kernel = kernel / kernel.sum()  # exact normalization of the outer product
    return kernel.to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)


def median_padding(x_pm1, mask):
    """FreeSDG median background padding (plan §3.2).

    Per sample and per channel:
      1. median across ALL spatial pixels (background included),
      2. + 0.2,
      3. replace only pixels where mask == 0 with that value.

    Args:
        x_pm1: [B, C, H, W] tensor in [-1, 1].
        mask:  [B, 1, H, W] tensor in {0, 1}.
    Returns:
        [B, C, H, W] padded tensor.
    """
    if x_pm1.dim() != 4:
        raise ValueError("x_pm1 must be [B, C, H, W]")
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B, 1, H, W]")
    if x_pm1.shape[0] != mask.shape[0] or x_pm1.shape[2:] != mask.shape[2:]:
        raise ValueError("x_pm1 and mask must share batch and spatial dims")
    b, c = x_pm1.shape[0], x_pm1.shape[1]
    # Median over all H*W pixels per sample/channel -> [B, C], matching the
    # reference implementation (torch.median lower-median for even counts).
    flat = x_pm1.reshape(b, c, -1)
    med = flat.median(dim=2).values + 0.2
    med = med.view(b, c, 1, 1)
    return mask * x_pm1 + (1.0 - mask) * med


class HFCFilter(nn.Module):
    """FreeSDG High-Frequency-Content filter (plan §3.3, §4).

    forward(x_pm1, mask):
        x_pad = median_padding(x_pm1, mask)
        blur  = replication-padded depthwise Gaussian convolution of x_pad
        res   = clamp(ratio * (x_pad - blur), -1, 1)
        out   = (res + 1) * mask - 1

    All inputs/outputs are [-1, 1] tensors; no hidden [0,1] conversion.
    Kernel sizes are expected odd (all bank widths are odd); even sizes
    reproduce the upstream ReplicationPad2d(k // 2) spatial-size quirk and are
    not used by this repository's configurations.
    """

    def __init__(self, kernel_size, sigma, ratio=4.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.ratio = float(ratio)
        self.register_buffer(
            "kernel", build_gaussian_kernel2d(self.kernel_size, self.sigma)
        )
        self.pad = nn.ReplicationPad2d(self.kernel_size // 2)

    def gaussian_blur(self, x_pad):
        """Depthwise Gaussian blur with replication padding (plan §4.2).

        Expands the [1,1,k,k] kernel to [C,1,k,k] for groups=C convolution;
        never passes the [1,1,k,k] kernel directly to the grouped conv.
        """
        c = x_pad.shape[1]
        k = self.kernel_size
        weight = self.kernel.to(device=x_pad.device, dtype=x_pad.dtype)
        weight = weight.expand(c, 1, k, k)
        return F.conv2d(self.pad(x_pad), weight, groups=c)

    def forward_padded(self, x_pad, mask):
        """HFC on an already median-padded input (mask is the original FOV)."""
        res = self.ratio * (x_pad - self.gaussian_blur(x_pad))
        res = torch.clamp(res, -1.0, 1.0)
        return (res + 1.0) * mask - 1.0

    def forward(self, x_pm1, mask):
        x_pad = median_padding(x_pm1, mask)
        return self.forward_padded(x_pad, mask)


def _sample_rectangle_repo(rng, height, width, mixup_size):
    """Released-repository rectangle policy, per-sample adaptation (§5.1).

    mixup_size == -1 (random size):
        w = rng.randrange(0, W); h = rng.randrange(0, H)
        x0 = rng.randrange(0, W - w); y0 = rng.randrange(0, H - h)
    The upstream repository contains a TODO questioning this exact range; it
    is reproduced as-is and must not be silently altered.

    mixup_size > 0 (fixed square): safe positioning guard replaces the
    upstream randrange(0, 0) crash — an intentional, documented robustness
    correction (plan §5.2).

    Returns (y0, x0, h, w).
    """
    if mixup_size == -1:
        w = rng.randrange(0, width)
        h = rng.randrange(0, height)
    else:  # mixup_size > 0 (validated upstream of this call)
        s = int(mixup_size)
        if not (0 < s < width and 0 < s < height):
            raise ValueError(
                "freesdg mixup_size={} cannot be positioned inside {}x{}; "
                "upstream would crash with randrange(0, 0)".format(
                    s, height, width
                )
            )
        w = s
        h = s
    x0 = rng.randrange(0, width - w)
    y0 = rng.randrange(0, height - h)
    return y0, x0, h, w


def _sample_rectangle_paper(rng, height, width):
    """MICCAI author-response rectangle policy (plan §5.3).

    cx, cy ~ randint[128, 384]; s ~ randint[32, 256]; s x s square centered
    at (cx, cy). Requires 512x512 input (asserted; no invented scaling rule).
    The square interpretation of the singular "size" is an explicit
    implementation decision (plan §19.5).

    Returns (y0, x0, h, w).
    """
    if (height, width) != (512, 512):
        raise ValueError(
            "freesdg mix_policy='paper' requires 512x512 input, got {}x{}".format(
                height, width
            )
        )
    cx = rng.randint(128, 384)
    cy = rng.randint(128, 384)
    s = rng.randint(32, 256)
    x0 = cx - s // 2
    y0 = cy - s // 2
    return y0, x0, s, s


class GaussianMixUp(nn.Module):
    """FreeSDG Frequency-Mixed Augmentation core (plan §9).

    Per sample: choose filter i, choose filter j, build two HFC views, sample
    one rectangle, paste view 2's rectangle into view 1. This is per-sample by
    design — a documented adaptation of the released code's batch-level
    behavior (plan §19.2). The median-padded source is computed once per call
    and shared between both views (the padding does not depend on the filter);
    this optimization is verified against the straightforward two-view
    reference in test_freesdg_smoke.py.

    Deterministic hooks (plan §9 "Deterministic hooks"): ``filter_idx_1``,
    ``filter_idx_2`` and ``rectangle`` can be injected explicitly so tests
    never rely on probabilistic assertions. When any parameter must be
    sampled, a dedicated ``rng`` (never the global stream) must be provided.
    """

    def __init__(self, filter_bank=None, ratio=4.0, mixup_size=-1,
                 mix_policy="repo"):
        super().__init__()
        if mix_policy not in MIX_POLICIES:
            raise ValueError(
                "mix_policy must be one of {}, got {!r}".format(
                    MIX_POLICIES, mix_policy
                )
            )
        if mixup_size == 0 or mixup_size < -1:
            raise ValueError(
                "mixup_size must be -1 (random size) or > 0 (fixed square); "
                "0 is not a hidden HFC-only mode"
            )
        bank = tuple(filter_bank) if filter_bank is not None \
            else FREESDG_FILTER_BANK
        self.ratio = float(ratio)
        self.mixup_size = int(mixup_size)
        self.mix_policy = mix_policy
        self.filters = nn.ModuleList(
            [HFCFilter(k, s, ratio=self.ratio) for k, s in bank]
        )

    def sample_rectangle(self, rng, height, width):
        """Sample the mixing rectangle with the configured policy."""
        if self.mix_policy == "repo":
            return _sample_rectangle_repo(rng, height, width, self.mixup_size)
        return _sample_rectangle_paper(rng, height, width)

    def forward(self, x_pm1, mask, rng=None, filter_idx_1=None,
                filter_idx_2=None, rectangle=None):
        if x_pm1.dim() != 4:
            raise ValueError("x_pm1 must be [B, C, H, W] in [-1, 1]")
        needs_rng = (
            filter_idx_1 is None or filter_idx_2 is None or rectangle is None
        )
        if needs_rng and rng is None:
            raise ValueError(
                "a dedicated rng (random.Random) is required whenever "
                "filter indices or the rectangle are not injected"
            )
        # Draw order (dedicated RNG): filter i, filter j, then rectangle
        # (plan §9; upstream drew the rectangle first — see §19.2 deviation).
        if filter_idx_1 is None:
            filter_idx_1 = rng.randrange(0, len(self.filters))
        if filter_idx_2 is None:
            filter_idx_2 = rng.randrange(0, len(self.filters))
        height, width = x_pm1.shape[-2:]
        if rectangle is None:
            rectangle = self.sample_rectangle(rng, height, width)
        y0, x0, h, w = rectangle
        # Shared median padding: independent of (kernel_size, sigma).
        x_pad = median_padding(x_pm1, mask)
        view_1 = self.filters[filter_idx_1].forward_padded(x_pad, mask)
        view_2 = self.filters[filter_idx_2].forward_padded(x_pad, mask)
        result_1 = view_1.clone()
        result_1[:, :, y0:y0 + h, x0:x0 + w] = \
            view_2[:, :, y0:y0 + h, x0:x0 + w]
        return result_1


def fmaug_seed_for_worker(base_seed, worker_info):
    """Derive the FMAug RNG seed for the current DataLoader context (§6.3).

    * Main process (``worker_info is None``): the plain FreeSDG seed.
    * Worker process: deterministically combined with the worker's torch seed
      (``worker_info.seed``, which already includes the worker id), so
      workers get independent streams without touching the global Python RNG.

    Caveat (§29.1): with ``num_workers > 0`` and ``persistent_workers=False``
    workers are recreated per epoch, so each worker's FMAug RNG sequence
    restarts every epoch (shuffled sample order usually still changes the
    per-sample transform). Exact replay is primarily validated under
    ``num_workers=0``, the repository default.
    """
    base_seed = int(base_seed)
    if worker_info is None:
        return base_seed
    return base_seed + 1_000_003 * (int(worker_info.seed) % (2 ** 31))


class FreeSDGAugmentor:
    """Owns the filter bank, anchor filter, dedicated RNG, and range
    conversions (plan §9). The only place where [0,1] <-> [-1,1] happens.

    ``augment_train(img01, mask01, raw_prob)``:
        dedicated-RNG raw/FMAug coin; raw branch returns img01 unchanged,
        FMAug branch maps [0,1] -> [-1,1] -> GaussianMixUp -> [0,1].
    ``anchor(img01, mask01)``:
        fixed HFC(anchor_w, anchor_sigma) applied in [-1,1], mapped back to
        [0,1]; stays float (no uint8 round-trip on evaluation paths).
    """

    def __init__(self, seed=0, ratio=4.0, mixup_size=-1, mix_policy="repo",
                 anchor_w=27, anchor_sigma=9, aug_mode="fmaug"):
        if aug_mode not in AUG_MODES:
            raise ValueError(
                "aug_mode must be one of {}, got {!r}".format(
                    AUG_MODES, aug_mode))
        self.seed = int(seed)
        self.aug_mode = aug_mode
        self.mixup = GaussianMixUp(
            ratio=ratio, mixup_size=mixup_size, mix_policy=mix_policy
        )
        self.anchor_filter = HFCFilter(
            int(anchor_w), float(anchor_sigma), ratio=float(ratio)
        )
        self.raffe_bank = RaffeFilterBank()
        self.raffe_smooth = DT2SmoothMask()
        self._rng = None

    @property
    def rng(self):
        """Lazily created dedicated RNG stream (never the global one)."""
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            self._rng = random.Random(
                fmaug_seed_for_worker(self.seed, worker_info)
            )
        return self._rng

    @staticmethod
    def _as_batch(img01, mask01):
        """Normalize img [C,H,W]|[B,C,H,W] and mask [H,W]|[1,H,W]|[B,1,H,W]."""
        if not torch.is_tensor(img01) or not torch.is_tensor(mask01):
            raise TypeError("img01 and mask01 must be torch tensors")
        squeeze = img01.dim() == 3
        x = img01.unsqueeze(0) if squeeze else img01
        m = mask01
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        elif m.dim() != 4:
            raise ValueError("mask01 must have 2, 3 or 4 dims")
        if m.shape[1] != 1:
            raise ValueError("mask01 channel dim must be 1")
        if m.shape[0] == 1 and x.shape[0] != 1:
            m = m.expand(x.shape[0], -1, -1, -1)
        return x, m, squeeze

    def augment_train(self, img01, mask01, raw_prob=0.0, filter_idx=None,
                      raffe_indices_1=None, raffe_indices_2=None,
                      blend_mask=None, blend_points=None):
        """Training-time augmentation with the configured aug_mode.

        Draw order (dedicated RNG, uniform across modes): raw/FMAug coin
        first, then the mode's own draws:
          * fmaug:          filter i, filter j, rectangle   (plan §9)
          * fixed_hfc:      none (deterministic anchor filter)
          * random_hfc:     single Gaussian-bank filter index
          * raffe_filter:   per-channel bank indices (ch0, ch1, ch2)
          * raffe_smooth_mix: view-1 per-channel indices, view-2 per-channel
            indices, then 5 DT2 blend-mask centers

        The optional hook arguments are deterministic test hooks only
        (filter_idx for random_hfc, raffe_indices_*/blend_mask/blend_points
        for the raffe modes); production code never passes them. ``blend_points``
        injects the DT2 seed coordinates so the iterative and direct mask
        implementations can be compared on identical seeds (plan §5/§6.1).
        """
        with torch.no_grad():
            # Raw/FMAug probability comes from the dedicated RNG (§6.2).
            if self.rng.random() < raw_prob:
                return img01
            x, m, squeeze = self._as_batch(img01, mask01)
            m = m.to(dtype=x.dtype)
            if self.aug_mode == "fmaug":
                x_pm1 = 2.0 * x - 1.0
                y = self.mixup(x_pm1, m, rng=self.rng)
                y01 = (y + 1.0) / 2.0
            elif self.aug_mode == "fixed_hfc":
                x_pm1 = 2.0 * x - 1.0
                y01 = (self.anchor_filter(x_pm1, m) + 1.0) / 2.0
            elif self.aug_mode == "random_hfc":
                if filter_idx is None:
                    filter_idx = self.rng.randrange(0, len(self.mixup.filters))
                x_pm1 = 2.0 * x - 1.0
                view = self.mixup.filters[filter_idx].forward(x_pm1, m)
                y01 = (view + 1.0) / 2.0
            elif self.aug_mode == "raffe_filter":
                if raffe_indices_1 is None:
                    raffe_indices_1 = self.raffe_bank.sample_indices(
                        self.rng, channels=x.shape[1])
                y01 = self.raffe_bank.apply(x, m, raffe_indices_1)
            else:  # raffe_smooth_mix
                h, w = x.shape[-2:]
                if raffe_indices_1 is None:
                    raffe_indices_1 = self.raffe_bank.sample_indices(
                        self.rng, channels=x.shape[1])
                if raffe_indices_2 is None:
                    raffe_indices_2 = self.raffe_bank.sample_indices(
                        self.rng, channels=x.shape[1])
                if blend_mask is None:
                    blend_mask = self.raffe_smooth.generate(
                        self.rng, h, w, points=blend_points, device=x.device)
                v1 = self.raffe_bank.apply(x, m, raffe_indices_1)
                v2 = self.raffe_bank.apply(x, m, raffe_indices_2)
                bm = blend_mask.to(device=x.device, dtype=x.dtype)
                bm = bm.view(1, 1, h, w)
                y01 = bm * v1 + (1.0 - bm) * v2
                y01 = y01.clamp(0.0, 1.0) * m
            return y01.squeeze(0) if squeeze else y01

    def anchor(self, img01, mask01):
        """Fixed-anchor HFC preprocessing for evaluation (plan §1.6/§7)."""
        with torch.no_grad():
            x, m, squeeze = self._as_batch(img01, mask01)
            x_pm1 = 2.0 * x - 1.0
            y_pm1 = self.anchor_filter(x_pm1, m.to(dtype=x_pm1.dtype))
            y01 = (y_pm1 + 1.0) / 2.0
            return y01.squeeze(0) if squeeze else y01

    @staticmethod
    def to_pil_uint8(img01):
        """Convert a [C,H,W] [0,1] float tensor to a uint8 PIL image.

        Train-path only; introduces ~1/255 quantization because LwNet's paired
        transforms are PIL-based (plan §8/§19.6). Never used on evaluation
        anchor paths, which stay float.
        """
        from PIL import Image

        if img01.dim() != 3:
            raise ValueError("to_pil_uint8 expects [C, H, W]")
        arr = (
            img01.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        return Image.fromarray(arr)
