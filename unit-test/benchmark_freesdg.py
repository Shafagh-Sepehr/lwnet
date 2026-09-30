"""Benchmark the U3B (raffe_smooth_mix) optimization.

Measures the augmentation at 512x512 RGB, batch 4, comparing the original
iterative mask (100 max-pool steps) against the optimized direct mask
(Chebyshev distance).  Reports median wall-clock time.  Device-aware:
CUDA timings use torch.cuda.Event when a GPU is present; otherwise CPU.

Usage:
    python benchmark_freesdg.py [--size 512] [--batch 4] [--runs 5]
"""

import argparse
import random
import statistics
import time

import torch

from utils.freesdg_aug import (
    DT2SmoothMask,
    FreeSDGAugmentor,
)


def _time_cpu(fn, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    H = W = args.size
    B = args.batch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = _time_cuda if dev.type == "cuda" else _time_cpu

    print("device={} size={}x{} batch={} runs={}".format(
        dev, H, W, B, args.runs))

    # warm up
    img = torch.rand(B, 3, H, W, device=dev)
    mask = torch.ones(B, 1, H, W, device=dev)

    # --- mask generation: iterative (old) vs direct (new default) ---
    dt2 = DT2SmoothMask()
    pts = dt2.sample_points(random.Random(0), H, W)

    t_iter = timer(lambda: dt2._generate_iterative(pts, H, W), args.runs)
    t_direct = timer(lambda: dt2.generate(random.Random(0), H, W), args.runs)
    print("mask iterative: {:.4f}s   direct: {:.4f}s   speedup {:.1f}x".format(
        t_iter, t_direct, t_iter / t_direct))

    # --- full augment_train (raffe_smooth_mix): old vs new ---
    img_cpu = img.cpu()
    mask_cpu = mask.cpu()

    aug = FreeSDGAugmentor(seed=0, aug_mode="raffe_smooth_mix")
    t_new = timer(lambda: aug.augment_train(img, mask, raw_prob=0.0),
                  args.runs)
    print("full optimized:  {:.4f}s".format(t_new))


if __name__ == "__main__":
    main()
