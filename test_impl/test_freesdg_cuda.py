"""CUDA validation for the U3B (raffe_smooth_mix) optimization.

Run on a CUDA machine:

    python test_freesdg_cuda.py

Fails immediately if CUDA is unavailable. Validates the automatic CUDA path
(torch.quantile percentile on GPU, direct Chebyshev mask) against the
reference CPU path (numpy percentile, iterative mask), then benchmarks.

The augmentor picks the fastest path automatically: numpy on CPU tensors,
torch.quantile on CUDA tensors.  No flags needed.
"""

import os
import random
import statistics
import sys
import time
import traceback

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


def test_mask_cuda(dev):
    pts = [(5, 5), (20, 30), (40, 10), (10, 50), (50, 50)]
    dt2 = DT2SmoothMask()
    m_cpu = dt2._generate_iterative(pts, 64, 64)
    m_cuda = direct_smooth_mask(pts, 64, 64, 100, device=dev)
    assert m_cuda.device.type == "cuda"
    d = (m_cpu - m_cuda.cpu()).abs().max().item()
    assert d < MASK_TOL, "mask cuda maxdiff {:.3e}".format(d)
    return d


def test_filter_cuda(dev):
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=11)
    x = img.to(dev)
    m = mask.to(dev)
    v = FourierButterworthView(64, 64, 0.05, 2)
    # legacy numpy on CPU (reference)
    o_cpu = v(img, mask)
    # auto-selects torch.quantile on CUDA
    o_cuda = v(x, m)
    assert o_cuda.device.type == "cuda"
    d = (o_cpu - o_cuda.cpu()).abs().max().item()
    assert d < IMG_TOL, "filter cuda maxdiff {:.3e}".format(d)
    return d


def test_raffe_filter_cuda(dev):
    """Fixed-index parity for the single-view Raffe training mode."""
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=19)
    indices = [0, 17, 35]
    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_filter")
    o_cpu = aug.augment_train(img, mask, raw_prob=0.0,
                              raffe_indices_1=indices)
    o_cuda = aug.augment_train(img.to(dev), mask.to(dev), raw_prob=0.0,
                               raffe_indices_1=indices)
    assert o_cuda.device.type == "cuda"
    d = (o_cpu - o_cuda.cpu()).abs().max().item()
    assert d < IMG_TOL, "single-view cuda maxdiff {:.3e}".format(d)
    return d


def test_full_aug_cuda(dev):
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=7)
    idx1, idx2 = [0, 3, 6], [1, 4, 7]
    pts = [(5, 5), (20, 30), (40, 10), (10, 50), (50, 50)]
    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    # CPU reference
    o_cpu = aug.augment_train(img, mask, raw_prob=0.0,
                              raffe_indices_1=idx1, raffe_indices_2=idx2,
                              blend_points=pts)
    # CUDA (auto-selects torch.quantile)
    o_cuda = aug.augment_train(img.to(dev), mask.to(dev), raw_prob=0.0,
                               raffe_indices_1=idx1, raffe_indices_2=idx2,
                               blend_points=pts)
    assert o_cuda.device.type == "cuda"
    assert not o_cuda.requires_grad
    d = (o_cpu - o_cuda.cpu()).abs().max().item()
    assert d < IMG_TOL, "full-aug cuda maxdiff {:.3e}".format(d)
    return d


def test_raffe_device_observed(dev):
    """Verify the Raffe FFT path observes the explicitly selected device."""
    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_filter")
    img, mask = _make_img_mask(batch=1, channels=3, height=64, width=64,
                               seed=23)
    out = aug.augment_train(img.to(dev), mask.to(dev), raw_prob=0.0,
                            raffe_indices_1=[0, 17, 35])
    expected_index = (torch.cuda.current_device()
                      if dev.index is None else dev.index)
    assert out.device.type == dev.type
    assert (out.device.index if out.device.index is not None else
            torch.cuda.current_device()) == expected_index
    observed = [view.last_device for view in aug.raffe_bank.views(64, 64)
                if view.last_device is not None]
    assert observed
    assert all(device.type == dev.type and
               (device.index if device.index is not None else
                torch.cuda.current_device()) == expected_index
               for device in observed)
    # The runner formats every test result as a float.  Return a numeric
    # device index while the assertion above carries the actual guarantee.
    return float(expected_index)


def test_smoke_training(dev):
    from models.get_model import get_arch
    img, mask = _make_img_mask(batch=2, channels=3, height=64, width=64,
                               seed=13)
    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    out = aug.augment_train(img.to(dev), mask.to(dev), raw_prob=0.0)
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
    return float(loss.item())


def _time_cuda(fn, runs):
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(times)


def _time_cpu(fn, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark(dev, size=512, batch=4, runs=5):
    H = W = size
    img = torch.rand(batch, 3, H, W, device=dev)
    mask = torch.ones(batch, 1, H, W, device=dev)

    dt2 = DT2SmoothMask()
    pts = dt2.sample_points(random.Random(0), H, W)
    t_iter = _time_cuda(lambda: dt2._generate_iterative(pts, H, W), runs)
    t_direct = _time_cuda(lambda: dt2.generate(random.Random(0), H, W), runs)

    idx1, idx2 = [0, 17, 35], [1, 18, 34]
    img_cpu = img.cpu()
    mask_cpu = mask.cpu()
    aug_filter = FreeSDGAugmentor(seed=0, aug_mode="raffe_filter")
    t_filter_cpu = _time_cpu(
        lambda: aug_filter.augment_train(img_cpu, mask_cpu, raw_prob=0.0,
                                         raffe_indices_1=idx1), runs)
    t_filter_cuda = _time_cuda(
        lambda: aug_filter.augment_train(img, mask, raw_prob=0.0,
                                         raffe_indices_1=idx1), runs)
    aug_mix = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    t_mix_cpu = _time_cpu(
        lambda: aug_mix.augment_train(img_cpu, mask_cpu, raw_prob=0.0,
                                      raffe_indices_1=idx1,
                                      raffe_indices_2=idx2), runs)
    t_mix_cuda = _time_cuda(
        lambda: aug_mix.augment_train(img, mask, raw_prob=0.0,
                                      raffe_indices_1=idx1,
                                      raffe_indices_2=idx2), runs)
    return (t_iter, t_direct, t_filter_cpu, t_filter_cuda,
            t_mix_cpu, t_mix_cuda)


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available on this machine.")
        sys.exit(1)
    dev = torch.device("cuda")
    print("CUDA device: {} ({})".format(torch.cuda.get_device_name(0),
                                        torch.__version__))

    results = {}
    for name, fn in [
        ("mask_cuda", lambda: test_mask_cuda(dev)),
        ("filter_cuda", lambda: test_filter_cuda(dev)),
        ("raffe_filter_cuda", lambda: test_raffe_filter_cuda(dev)),
        ("raffe_device_observed", lambda: test_raffe_device_observed(dev)),
        ("full_aug_cuda", lambda: test_full_aug_cuda(dev)),
        ("smoke_training", lambda: test_smoke_training(dev)),
    ]:
        try:
            results[name] = fn()
            print("PASS: {} -> {:.3e}".format(name, results[name]))
        except Exception:
            print("FAIL: {}".format(name))
            traceback.print_exc()
            sys.exit(1)

    print("\n--- benchmark (512x512, batch 4) ---")
    (t_iter, t_direct, t_filter_cpu, t_filter_cuda,
     t_mix_cpu, t_mix_cuda) = benchmark(dev)
    print("mask iterative: {:.4f}s   direct: {:.4f}s   speedup {:.1f}x".format(
        t_iter, t_direct, t_iter / t_direct))
    print("Raffe filter CPU: {:.4f}s   CUDA: {:.4f}s   speedup {:.1f}x".format(
        t_filter_cpu, t_filter_cuda, t_filter_cpu / t_filter_cuda))
    print("Raffe smooth CPU: {:.4f}s   CUDA: {:.4f}s   speedup {:.1f}x".format(
        t_mix_cpu, t_mix_cuda, t_mix_cpu / t_mix_cuda))

    print("\n=== REPORT (paste back) ===")
    print("torch={} cuda={}".format(torch.__version__, torch.version.cuda))
    print("mask_cuda_maxdiff={:.3e}".format(results["mask_cuda"]))
    print("filter_cuda_maxdiff={:.3e}".format(results["filter_cuda"]))
    print("full_aug_cuda_maxdiff={:.3e}".format(results["full_aug_cuda"]))
    print("smoke_loss={:.6f}".format(results["smoke_training"]))
    print("bench_mask_iter={:.4f} bench_mask_direct={:.4f}".format(
        t_iter, t_direct))
    print("bench_raffe_filter_cpu={:.4f} bench_raffe_filter_cuda={:.4f}".format(
        t_filter_cpu, t_filter_cuda))
    print("bench_raffe_smooth_cpu={:.4f} bench_raffe_smooth_cuda={:.4f}".format(
        t_mix_cpu, t_mix_cuda))
    print("All CUDA Raffe sections passed; CUDA timing path exercised.")


if __name__ == "__main__":
    main()
