import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.get_model import get_arch


def _num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _benchmark(model, device, amp, warmup=20, iters=100, size=512):
    model.eval()
    model.mode = 'eval'
    x = torch.randn(1, 3, size, size, device=device)

    def run():
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast(enabled=True):
                    model(x)
            else:
                model(x)

    for _ in range(warmup):
        run()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = sorted(times)
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return mean, median, p95


def _peak_memory(model, device, amp, size=512):
    if device.type != 'cuda':
        return None
    model.eval()
    model.mode = 'eval'
    x = torch.randn(1, 3, size, size, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if amp:
            with torch.cuda.amp.autocast(enabled=True):
                model(x)
        else:
            model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_supported = device.type == 'cuda'

    configs = [
        ('baseline', {}),
        ('scalar/all', {'cross_stage_bridge': 'scalar'}),
        ('channel/all', {'cross_stage_bridge': 'channel'}),
    ]

    print('device={} size={} warmup={} iters={}'.format(device, args.size, args.warmup, args.iters))
    print('{:<14} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}'.format(
        'config', 'params', 'amp', 'mean_ms', 'median_ms', 'p95_ms', 'peak_MiB'))

    for name, kw in configs:
        model = get_arch('wnet', **kw).to(device)
        params = _num_params(model)
        for amp in ([False, True] if amp_supported else [False]):
            mean, median, p95 = _benchmark(model, device, amp, warmup=args.warmup, iters=args.iters, size=args.size)
            peak = _peak_memory(model, device, amp, size=args.size)
            print('{:<14} {:>12} {:>12} {:>12.3f} {:>12.3f} {:>12.3f} {:>12}'.format(
                name, params, str(amp), mean * 1000, median * 1000, p95 * 1000,
                '{:.1f}'.format(peak) if peak is not None else 'n/a'))


if __name__ == '__main__':
    main()
