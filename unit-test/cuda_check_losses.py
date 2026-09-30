"""CUDA verification + overhead measurement for the composable segmentation losses.

Run on a CUDA machine from the repo root:

    python cuda_check_losses.py [--device cuda:0] [--iters 10]

Covers (plan sections 9.8/11, GPU side):
  1. GPU numerical checks: BCE-only bitwise equivalence vs BCEWithLogitsLoss,
     total == explicit weighted sum, per-term finite non-zero gradients.
  2. Training-step time and peak memory at training scale (512x512, bs=4) for
     unet and wnet under each loss configuration.
  3. Finite gradients under AMP (autocast + GradScaler), all four terms.
  4. CPU cost of distance-map (SciPy EDT) preparation at 512x512.
  5. End-to-end run_one_epoch smoke on GPU with distance maps.

Exits 0 when everything passes. Paste the full stdout back.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation_losses import (  # noqa: E402
    SegmentationLoss, signed_distance_map, boundary_loss,
    soft_cldice_loss, soft_dice_loss)
from models.get_model import get_arch  # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print('[{}] {}{}'.format('PASS' if ok else 'FAIL', name,
                             (' -- ' + detail) if detail else ''))
    if not ok:
        FAILED.append(name)


def make_vessels(b, s=512, device='cpu', seed=0):
    """Synthetic multi-width vessel labels [B, 1, S, S]."""
    g = torch.Generator().manual_seed(seed)
    y = torch.zeros(b, s, s)
    for i in range(b):
        r0 = 100 + 60 * i
        y[i, r0:r0 + 3, 40:s - 40] = 1.0            # 3px horizontal
        y[i, 40:s - 40, 100 + 40 * i:105 + 40 * i] = 1.0  # 5px vertical
        d = torch.arange(s)
        y[i] = torch.maximum(y[i], (torch.abs(d - 300 - 20 * i) < 2).float()[:, None]
                             * (torch.abs(d[None, :] - 350) < 160).float())  # 3px bar
    return y.to(device)


def numerical_checks(device):
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 128, 128, device=device) * 3
    target = (torch.rand(4, 128, 128, device=device) > 0.8).float()

    la = logits.clone().requires_grad_(True)
    lb = logits.clone().requires_grad_(True)
    total, comps = SegmentationLoss()(la, target)
    ref = torch.nn.BCEWithLogitsLoss()(logits, target.unsqueeze(1))
    check('bce-only total == BCEWithLogitsLoss (bitwise)', torch.equal(total, ref))
    total.backward()
    torch.nn.BCEWithLogitsLoss()(lb, target.unsqueeze(1)).backward()
    check('bce-only gradients bitwise equal', torch.equal(la.grad, lb.grad))

    y = make_vessels(2, 128, device).unsqueeze(1)  # [B, 1, H, W]
    dist = signed_distance_map(y).to(device)
    w = dict(bce=1.0, dice=1.0, cldice=0.2, boundary=0.01)
    lg = (3.0 * (2 * y - 1)).requires_grad_(True)
    crit = SegmentationLoss(bce_weight=w['bce'], dice_weight=w['dice'],
                            cldice_weight=w['cldice'], boundary_weight=w['boundary'])
    total, comps = crit(lg, y, distance_map=dist)
    p = torch.sigmoid(lg.detach())
    import torch.nn.functional as F
    manual = (w['bce'] * F.binary_cross_entropy_with_logits(lg.detach(), y)
              + w['dice'] * soft_dice_loss(p, y)
              + w['cldice'] * soft_cldice_loss(p, y, 10)
              + w['boundary'] * boundary_loss(p, dist))
    check('total == weighted sum of components',
          torch.allclose(total, manual, atol=1e-6),
          'max dev {:.2e}'.format((total - manual).abs().item()))
    total.backward()
    check('all-term grad finite', bool(torch.isfinite(lg.grad).all()))
    check('all-term grad nonzero', lg.grad.abs().sum().item() > 0)


def bench_step(model_name, crit, imgs, y, dist, iters, device):
    torch.manual_seed(0)
    model = get_arch(model_name).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def one_step():
        logits = model(imgs)
        if isinstance(logits, tuple):
            (aux, main) = logits
            loss = crit(aux, y, distance_map=dist)[0] + crit(main, y, distance_map=dist)[0]
        else:
            loss = crit(logits, y, distance_map=dist)[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):  # warmup
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        one_step()
    torch.cuda.synchronize()
    sec = (time.perf_counter() - start) / iters
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del model, opt
    torch.cuda.empty_cache()
    return sec, peak


def benchmarks(device, iters):
    s, b = 512, 4
    imgs = torch.rand(b, 3, s, s, device=device)
    y = make_vessels(b, s, device)
    dist = signed_distance_map(y).to(device)
    configs = [
        ('bce-only          ', SegmentationLoss()),
        ('bce+dice          ', SegmentationLoss(dice_weight=1.0)),
        ('bce+dice+cldice   ', SegmentationLoss(dice_weight=1.0, cldice_weight=0.2)),
        ('bce+dice+boundary ', SegmentationLoss(dice_weight=1.0, boundary_weight=0.01)),
        ('all four terms    ', SegmentationLoss(dice_weight=1.0, cldice_weight=0.2,
                                                boundary_weight=0.01)),
    ]
    print('\n== training-step benchmark (bs={}, {}x{}, fwd+bwd+step, {} iters) =='
          .format(b, s, s, iters))
    for model_name in ('unet', 'wnet'):
        print('-- {} --'.format(model_name))
        base_sec = None
        for label, crit in configs:
            sec, peak = bench_step(model_name, crit, imgs, y, dist, iters, device)
            if base_sec is None:
                base_sec = sec
            print('   {} : {:.1f} ms/step  (+{:.0f}% vs bce)  peak {:.0f} MB'
                  .format(label, sec * 1e3, 100 * (sec / base_sec - 1), peak))
    # NaN/inf sanity of the loss values themselves at scale
    model = get_arch('wnet').to(device).train()
    logits_aux, logits_main = model(imgs)
    crit = configs[-1][1]
    tot, comps = crit(logits_main, y, distance_map=dist)
    ok = bool(torch.isfinite(tot)) and all(torch.isfinite(v) for v in comps.values())
    check('512x512 wnet main-head loss finite (all terms)', ok,
          ' '.join('{}={:.4f}'.format(k, v.item()) for k, v in comps.items()))


def amp_check(device):
    s, b = 256, 2
    imgs = torch.rand(b, 3, s, s, device=device)
    y = make_vessels(b, s, device)
    dist = signed_distance_map(y).to(device)
    torch.manual_seed(0)
    model = get_arch('wnet').to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    ok_finite, ok_nonzero = True, False
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            aux, main = model(imgs)
            loss = (SegmentationLoss()(aux, y, distance_map=dist)[0]
                    + SegmentationLoss(dice_weight=1.0, cldice_weight=0.2,
                                       boundary_weight=0.01)(main, y, distance_map=dist)[0])
        assert torch.isfinite(loss), 'nonfinite AMP loss'
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                     for p in model.parameters())
        ok_finite &= finite
        ok_nonzero |= any(p.grad is not None and p.grad.abs().sum().item() > 0
                          for p in model.parameters())
        scaler.step(opt)
        scaler.update()
    check('AMP: finite grads (all terms, unscaled)', ok_finite)
    check('AMP: nonzero grads (all terms)', ok_nonzero)


def edt_timing():
    y = make_vessels(1, 512, 'cpu')
    signed_distance_map(y)  # warm import
    start = time.perf_counter()
    for _ in range(20):
        signed_distance_map(y)
    ms = (time.perf_counter() - start) / 20 * 1e3
    print('\n== distance-map preparation (SciPy EDT, CPU, 512x512) == {:.2f} ms/sample'
          .format(ms))


def run_one_epoch_smoke(device):
    import train_cyclical
    from torch.utils.data import DataLoader, TensorDataset
    s, b = 128, 4
    imgs = torch.rand(b, 3, s, s)
    y = make_vessels(b, s, 'cpu')
    dist = signed_distance_map(y.unsqueeze(1))
    loader = DataLoader(TensorDataset(imgs, y, dist), batch_size=2)
    for model_name in ('unet', 'wnet'):
        model = get_arch(model_name).to(device)
        crit = SegmentationLoss(dice_weight=1.0, cldice_weight=0.2, boundary_weight=0.01)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        _, _, run_loss, _, comps = train_cyclical.run_one_epoch(
            loader, model, crit, optimizer=opt, scheduler=sch, assess=True)
        ok = np.isfinite(run_loss) and all(np.isfinite(v) for v in comps.values())
        check('run_one_epoch on {} (all terms)'.format(device), ok,
              'loss={:.4f}'.format(run_loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--iters', type=int, default=10)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print('CUDA is not available on this machine; nothing to do.')
        sys.exit(2)
    if args.device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':', 1)[1]
    device = torch.device('cuda')
    print('== environment ==')
    print('torch {} | cuda {} | {}'.format(torch.__version__,
          torch.version.cuda, torch.cuda.get_device_name(0)))
    numerical_checks(device)
    benchmarks(device, args.iters)
    amp_check(device)
    edt_timing()
    run_one_epoch_smoke(device)
    print('\n{}'.format('ALL CHECKS PASSED' if not FAILED
                         else 'FAILED: ' + ', '.join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
