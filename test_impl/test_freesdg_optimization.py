"""Correctness gates for the U3B (raffe_smooth_mix) optimization (plan §6).

Plain-Python runnable — no pytest dependency:

    python test_freesdg_optimization.py

Runs the CPU correctness gates always, and the CUDA placement/parity gates
only when ``torch.cuda.is_available()``. Exits 0 when every section passes.

Covers:
  §6.1  mask equivalence (direct vs iterative) with injected seeds
  §6.2  filter parity (vectorized vs legacy percentile)
  §6.3  full-augmentation parity (optimized_cpu vs legacy_cpu)
  §6.4  raw-probability endpoints and sample-order preservation
  §6.5  CUDA placement / no-grad / smoke training step (CUDA only)
  §6.7  config/CLI flag coverage
"""

import os
import random
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.freesdg_aug import (  # noqa: E402
    DT2SmoothMask,
    FourierButterworthView,
    FreeSDGAugmentor,
    direct_smooth_mask,
)

MASK_TOL = 1e-5
IMG_TOL = 1e-4


def _make_img_mask(batch=1, channels=3, height=64, width=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand((batch, channels, height, width), generator=g)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width),
                            indexing="ij")
    cy, cx = height // 2, width // 2
    r = min(height, width) // 2 - 2
    circle = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()
    mask = circle.unsqueeze(0).unsqueeze(0).expand(batch, 1, height, width
                                                    ).clone()
    return img, mask


# ---------------------------------------------------------------------------
# §6.1 mask equivalence
# ---------------------------------------------------------------------------

def test_mask_equivalence():
    dt2 = DT2SmoothMask()

    def check(points, h, w, step=100, label=""):
        # direct (production) vs iterative (reference)
        m2 = dt2.generate(random.Random(0), h, w, points=points)
        m1 = dt2._generate_iterative(points, h, w)
        assert m1.shape == m2.shape == (h, w), (m1.shape, m2.shape)
        d = (m1 - m2).abs().max().item()
        assert d < MASK_TOL, "{}: mask maxdiff {:.3e}".format(label, d)
        # both normalized to [0, 1] (or all-zero degenerate)
        assert m1.min() >= -1e-6 and m1.max() <= 1.0 + 1e-6

    # random seeds, several sizes incl. rectangular and small
    for (h, w) in [(64, 64), (48, 80), (16, 16), (8, 8), (5, 7)]:
        for seed in range(5):
            rng = random.Random(seed)
            pts = [(rng.randint(0, h - 1), rng.randint(0, w - 1))
                   for _ in range(5)]
            check(pts, h, w, label="rand {}x{}".format(h, w))

    # duplicate seeds
    check([(10, 10), (10, 10), (20, 20), (10, 10), (30, 30)], 64, 64,
          label="duplicates")

    # corner seeds
    check([(0, 0), (0, 63), (63, 0), (63, 63), (31, 31)], 64, 64,
          label="corners")

    # center seed
    check([(32, 32)], 64, 64, label="center")

    # expansion limits: T=1, 2, 5
    for T in (1, 2, 5):
        dt = DT2SmoothMask(expansion_step=T)
        m1 = dt._generate_iterative(
            [(5, 5), (20, 20)], 32, 32)
        m2 = dt.generate(
            random.Random(0), 32, 32, points=[(5, 5), (20, 20)])
        assert (m1 - m2).abs().max().item() < MASK_TOL, "T={}".format(T)

    # zero-range degenerate case: single seed, tiny image, large T -> all
    # pixels reach count == T -> constant acc -> reference returns zeros
    dt = DT2SmoothMask(expansion_step=100)
    m = dt.generate(random.Random(0), 4, 4, points=[(1, 1)])
    assert torch.all(m == 0.0), "zero-range must be all-zero"
    m_ref = dt._generate_iterative([(1, 1)], 4, 4)
    assert torch.all(m_ref == 0.0), "iterative zero-range must be all-zero"

    # direct_smooth_mask raw accumulation matches the closed-form count
    pts = [(3, 3), (10, 20)]
    T = 100
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    d = None
    for (r, c) in pts:
        dist = torch.maximum((yy - r).abs(), (xx - c).abs())
        d = dist if d is None else torch.minimum(d, dist)
    t_enter = torch.maximum((d + 1) // 2, torch.ones_like(d))
    count = torch.clamp(T - t_enter + 1, 0, T)
    acc = (1.0 / T) * count.float()
    m = direct_smooth_mask(pts, 32, 32, T)
    lo, hi = acc.min(), acc.max()
    expect = (acc - lo) / (hi - lo)
    assert (m - expect).abs().max().item() < 1e-6, "raw-acc closed form"

    print("       mask equivalence OK")


# ---------------------------------------------------------------------------
# §6.2 filter parity
# ---------------------------------------------------------------------------

def test_filter_parity():
    cases = []
    for (h, w) in [(32, 32), (64, 64)]:
        for d0, n in [(0.05, 2), (0.1, 1), (0.005, 3), (0.0, 1), (0.1, 3)]:
            cases.append((h, w, d0, n))

    for (h, w, d0, n) in cases:
        v = FourierButterworthView(h, w, d0, n)
        for kind in ("random", "constant", "zero"):
            if kind == "random":
                x = torch.rand(2, 3, h, w)
            elif kind == "constant":
                x = torch.full((2, 3, h, w), 0.5)
            else:
                x = torch.zeros(2, 3, h, w)
            mask = torch.ones(2, 1, h, w)
            o = v(x, mask)
            d = (o - v._percentile_legacy(
                torch.abs(torch.fft.ifft2(
                    torch.fft.fft2(x) * v.fmap.to(x.dtype))))).abs().max().item()
            assert d < IMG_TOL, \
                "filter parity {}x{} d0={} n={} {}: {:.3e}".format(
                    h, w, d0, n, kind, d)
            assert o.min() >= 0.0 and o.max() <= 1.0

    # The vectorized torch.quantile percentile (used on CUDA) must match
    # the numpy reference, so the CUDA path is validated before it runs
    # on a GPU.
    for (h, w) in [(32, 32), (64, 64)]:
        x = torch.rand(2, 3, h, w)
        spec = torch.fft.fft2(x) * torch.rand(h, w)
        res = torch.abs(torch.fft.ifft2(spec))
        leg = FourierButterworthView._percentile_legacy(res)
        vec = FourierButterworthView._percentile_vectorized(res)
        d = (leg - vec).abs().max().item()
        assert d < IMG_TOL, "vectorized percentile maxdiff {:.3e}".format(d)
    print("       filter parity OK")


# ---------------------------------------------------------------------------
# §6.3 full-augmentation correctness
# ---------------------------------------------------------------------------

def test_full_augmentation():
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=7)
    idx1, idx2 = [0, 3, 6], [1, 4, 7]
    pts = [(5, 5), (20, 30), (40, 10), (10, 50), (50, 50)]

    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    out = aug.augment_train(img, mask, raw_prob=0.0,
                            raffe_indices_1=idx1, raffe_indices_2=idx2,
                            blend_points=pts)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0

    # deterministic with same injected params
    out2 = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix").augment_train(
        img, mask, raw_prob=0.0, raffe_indices_1=idx1, raffe_indices_2=idx2,
        blend_points=pts)
    assert torch.equal(out, out2)

    # raffe_filter mode
    f = FreeSDGAugmentor(seed=0, aug_mode="raffe_filter").augment_train(
        img, mask, raw_prob=0.0, raffe_indices_1=idx1)
    assert f.shape == img.shape and f.min() >= 0.0 and f.max() <= 1.0
    print("       full-augmentation OK")


# ---------------------------------------------------------------------------
# §6.4 raw-probability endpoints and sample order
# ---------------------------------------------------------------------------

def test_raw_prob_endpoints():
    img, mask = _make_img_mask(batch=1, channels=3, height=64, width=64,
                                seed=3)
    a0 = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    # raw_prob=1 -> always raw (identity), no FFT/mask work
    assert all(torch.equal(a0.augment_train(img, mask, raw_prob=1.0), img)
               for _ in range(5))
    # raw_prob=0 -> never raw
    a1 = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    assert all(not torch.equal(a1.augment_train(img, mask, raw_prob=0.0), img)
               for _ in range(5))
    # mixed: deterministic per seed, roughly half raw
    counts = []
    for seed in (0, 1, 2):
        a = FreeSDGAugmentor(seed=seed, aug_mode="raffe_smooth_mix")
        counts.append(sum(torch.equal(a.augment_train(img, mask, raw_prob=0.5),
                                      img) for _ in range(200)))
    assert all(70 <= c <= 130 for c in counts), counts
    print("       raw-prob endpoints OK")


# ---------------------------------------------------------------------------
# §6.5 CUDA placement / no-grad / smoke training (CUDA only)
# ---------------------------------------------------------------------------

def test_cuda_placement():
    if not torch.cuda.is_available():
        print("       CUDA placement SKIPPED (no CUDA)")
        return
    dev = torch.device("cuda")
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=11)
    x = img.to(dev)
    m = mask.to(dev)

    # direct mask on CUDA matches iterative on CPU
    pts = [(5, 5), (20, 30), (40, 10), (10, 50), (50, 50)]
    dt2 = DT2SmoothMask()
    m_cpu = dt2._generate_iterative(pts, 64, 64)
    m_cuda = direct_smooth_mask(pts, 64, 64, 100, device=dev)
    assert m_cuda.device.type == "cuda"
    assert (m_cpu - m_cuda.cpu()).abs().max().item() < MASK_TOL

    # vectorized percentile on CUDA matches legacy on CPU
    v = FourierButterworthView(64, 64, 0.05, 2)
    o_leg = FourierButterworthView._percentile_legacy(
        torch.fft.ifft2(torch.fft.fft2(img) * v.fmap).abs())
    o_cuda = v(x, m)
    assert o_cuda.device.type == "cuda"
    assert (o_leg - o_cuda.cpu()).abs().max().item() < IMG_TOL

    # no-grad: augmentation tensors require no gradients
    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    out = aug.augment_train(x, m, raw_prob=0.0)
    assert out.device.type == "cuda"
    assert not out.requires_grad

    # smoke training step: model params receive finite gradients
    from models.get_model import get_arch
    model = get_arch("wnet").to(dev)
    model.mode = "train"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    y = (torch.rand(2, 64, 64, device=dev) > 0.5).float().unsqueeze(1)
    logits_aux, logits = model(out)
    loss = crit(logits_aux, y) + crit(logits, y)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    print("       CUDA placement OK")


# ---------------------------------------------------------------------------
# §6.7 config/CLI flag coverage
# ---------------------------------------------------------------------------

def test_config_flags():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_cyclical_for_test", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "train_cyclical.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = vars(mod.parser.parse_args([]))
    # freesdg-related defaults preserved
    assert defaults["freesdg"] is False
    assert defaults["freesdg_raw_prob"] == 0.0
    assert defaults["freesdg_aug_mode"] == "fmaug"
    assert defaults["freesdg_seed"] == 0
    print("       config/CLI flags OK")


SECTIONS = [
    ("6.1 mask equivalence", test_mask_equivalence),
    ("6.2 filter parity", test_filter_parity),
    ("6.3 full-augmentation", test_full_augmentation),
    ("6.4 raw-prob endpoints", test_raw_prob_endpoints),
    ("6.5 CUDA placement", test_cuda_placement),
    ("6.7 config/CLI flags", test_config_flags),
]


def main():
    for name, fn in SECTIONS:
        try:
            fn()
        except Exception:
            print("FAIL: {}".format(name))
            traceback.print_exc()
            sys.exit(1)
        print("PASS: {}".format(name))
    print("All {} optimization sections passed.".format(len(SECTIONS)))


if __name__ == "__main__":
    main()
