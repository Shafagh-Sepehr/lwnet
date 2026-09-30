"""Focused verification of the composable segmentation losses (plan section 9).

Plain-Python runnable -- no pytest dependency:

    python test_segmentation_losses.py

Exits 0 when every section passes, non-zero on the first failure.
Uses small synthetic tensors only; no dataset run is required.
"""

import os
import subprocess
import sys
import tempfile
import time
import types
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import segmentation_losses as sl  # noqa: E402
from segmentation_losses import (  # noqa: E402
    SegmentationLoss,
    build_segmentation_loss,
    soft_dice_loss,
    soft_cldice_loss,
    soft_skeleton,
    signed_distance_map,
    boundary_loss,
)


def _vessel(h=64, w=64, rows=(30, 33), cols=(8, 56), thick_image=None):
    """Horizontal 3px-thick binary vessel tensor [H, W]."""
    y = torch.zeros(h, w)
    y[rows[0]:rows[1], cols[0]:cols[1]] = 1.0
    return y


# ---------------------------------------------------------------------------
# 9.1 Compatibility: BCE-only total and gradients match the original criterion
# ---------------------------------------------------------------------------

def test_9_1_bce_only_compatibility():
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 16, 16) * 3
    target = (torch.rand(4, 16, 16) > 0.7).float()  # [B, H, W] on purpose

    ref = torch.nn.BCEWithLogitsLoss()(logits, target.unsqueeze(1))
    la = logits.clone().requires_grad_(True)
    lb = logits.clone().requires_grad_(True)
    total, comps = SegmentationLoss()(la, target)  # defaults = BCE-only
    assert torch.equal(total, ref), (total.item(), ref.item())
    assert list(comps.keys()) == ['bce']
    (total).backward()
    torch.nn.BCEWithLogitsLoss()(lb, target.unsqueeze(1)).backward()
    assert torch.equal(la.grad, lb.grad)

    # wnet-style double call reproduces the original aux+main aggregation
    aux = torch.randn(2, 1, 16, 16)
    main = torch.randn(2, 1, 16, 16)
    t2 = (torch.rand(2, 16, 16) > 0.5).float()
    crit = SegmentationLoss()
    tot_aux, _ = crit(aux, t2)
    tot_main, _ = crit(main, t2)
    ref = (torch.nn.BCEWithLogitsLoss()(aux, t2.unsqueeze(1))
           + torch.nn.BCEWithLogitsLoss()(main, t2.unsqueeze(1)))
    assert torch.allclose(tot_aux + tot_main, ref), 'wnet double-call mismatch'

    # shape contract: wrong channel / spatial mismatch rejected
    for bad in (torch.zeros(2, 2, 8, 8), torch.zeros(2, 8, 8)):
        try:
            SegmentationLoss()(torch.zeros(2, 2, 8, 8), bad)
            raise AssertionError('channel mismatch not rejected')
        except ValueError:
            pass
    try:
        SegmentationLoss()(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 9, 9))
        raise AssertionError('spatial mismatch not rejected')
    except ValueError:
        pass
    # debug-mode nonfinite rejection
    bad_logit = torch.zeros(2, 1, 8, 8)
    bad_logit[0, 0, 0, 0] = float('nan')
    try:
        SegmentationLoss(debug=True)(bad_logit, torch.zeros(2, 1, 8, 8))
        raise AssertionError('nonfinite logits not rejected in debug mode')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 9.2 Composition: total == explicit weighted sum; per-term gradients
# ---------------------------------------------------------------------------

def test_9_2_composition():
    torch.manual_seed(1)
    y = _vessel()
    y4 = y[None, None]
    dist = signed_distance_map(y4)
    logits = (3.0 * (2 * y - 1))[None, None].requires_grad_(True)

    w = dict(bce=1.0, dice=1.0, cldice=0.2, boundary=0.01)
    crit = SegmentationLoss(bce_weight=w['bce'], dice_weight=w['dice'],
                            cldice_weight=w['cldice'], boundary_weight=w['boundary'],
                            cldice_iters=10)
    total, comps = crit(logits, y4, distance_map=dist)
    assert set(comps) == {'bce', 'dice', 'cldice', 'boundary'}

    p = torch.sigmoid(logits.detach())
    manual = (w['bce'] * F.binary_cross_entropy_with_logits(logits.detach(), y4)
              + w['dice'] * soft_dice_loss(p, y4)
              + w['cldice'] * soft_cldice_loss(p, y4, 10)
              + w['boundary'] * boundary_loss(p, dist))
    assert torch.allclose(total, manual, atol=1e-7), (total.item(), manual.item())
    assert torch.allclose(comps['dice'], soft_dice_loss(p, y4))
    assert torch.allclose(comps['cldice'], soft_cldice_loss(p, y4, 10))
    assert torch.allclose(comps['boundary'], boundary_loss(p, dist))

    # backpropagate each component separately: the difference between the
    # bce-only gradient and the bce+geometric gradient must be nonzero, so a
    # detached geometric branch cannot hide behind BCE
    def grad_of(crit):
        lg = (3.0 * (2 * y - 1))[None, None].requires_grad_(True)
        crit(lg, y4, distance_map=dist)[0].backward()
        return lg.grad.clone()

    g_bce = grad_of(SegmentationLoss())
    g_cldice = grad_of(SegmentationLoss(cldice_weight=0.2))
    g_boundary = grad_of(SegmentationLoss(boundary_weight=0.01))
    g_dice = grad_of(SegmentationLoss(bce_weight=0.0, dice_weight=1.0))
    assert not torch.allclose(g_cldice, g_bce), 'clDice branch looks detached'
    assert not torch.allclose(g_boundary, g_bce), 'boundary branch looks detached'
    assert g_dice.abs().sum() > 0, 'dice-only gradient is zero'
    for g in (g_cldice, g_boundary, g_dice):
        assert torch.isfinite(g).all()

    # boundary term without a map is rejected
    try:
        SegmentationLoss(boundary_weight=0.01)(logits.detach(), y4)
        raise AssertionError('missing distance map not rejected')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 9.3 Disabled work: zero weight -> never called; no SciPy for BCE-only
# ---------------------------------------------------------------------------

def test_9_3_disabled_work_and_validation():
    calls = {'dice': 0, 'cldice': 0, 'boundary': 0}
    originals = {'dice': sl.soft_dice_loss, 'cldice': sl.soft_cldice_loss,
                 'boundary': sl.boundary_loss}

    def spy(name):
        def wrapper(*a, **k):
            calls[name] += 1
            return originals[name](*a, **k)
        return wrapper

    sl.soft_dice_loss = spy('dice')
    sl.soft_cldice_loss = spy('cldice')
    sl.boundary_loss = spy('boundary')
    try:
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 8, 8) > 0.5).float()
        total, comps = SegmentationLoss()(logits, target)
        assert calls == {'dice': 0, 'cldice': 0, 'boundary': 0}, calls
        assert list(comps.keys()) == ['bce']
        # a zero weight also disables target preparation requirements
        crit = build_segmentation_loss(types.SimpleNamespace())
        assert crit.need_distance_map is False
        assert crit.weights == {'bce': 1.0, 'dice': 0.0, 'cldice': 0.0, 'boundary': 0.0}
        # partial enablement: only the enabled term runs
        total, comps = SegmentationLoss(dice_weight=1.0)(logits, target)
        assert calls == {'dice': 1, 'cldice': 0, 'boundary': 0}, calls
        assert list(comps.keys()) == ['bce', 'dice']
    finally:
        sl.soft_dice_loss = originals['dice']
        sl.soft_cldice_loss = originals['cldice']
        sl.boundary_loss = originals['boundary']

    # BCE-only forward/backward works without importing SciPy (subprocess)
    code = (
        "import sys, torch;"
        "sys.path.insert(0, r'{}');"
        "import segmentation_losses as m;"
        "x = torch.randn(2, 1, 8, 8, requires_grad=True);"
        "y = (torch.rand(2, 8, 8) > 0.5).float();"
        "t, c = m.SegmentationLoss()(x, y); t.backward();"
        "sys.exit(0 if all('scipy' not in k for k in sys.modules) else 1)"
    ).format(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()

    # builder validation
    def expect_error(**kw):
        try:
            SegmentationLoss(**kw)
            raise AssertionError('invalid config accepted: {}'.format(kw))
        except ValueError:
            pass

    expect_error(bce_weight=-1.0)
    expect_error(bce_weight=0.0, dice_weight=0.0, cldice_weight=0.0, boundary_weight=0.0)
    expect_error(bce_weight=0.0, dice_weight=0.0, cldice_weight=0.2)   # topology-only
    expect_error(bce_weight=0.0, dice_weight=0.0, boundary_weight=0.01)  # boundary-only
    expect_error(cldice_weight=0.2, cldice_iters=0)
    SegmentationLoss(cldice_iters=0)  # iters irrelevant when clDice disabled
    SegmentationLoss(bce_weight=0.0, dice_weight=1.0)  # dice-only is legal
    try:
        SegmentationLoss(bce_weight=float('nan'))
        raise AssertionError('nan weight accepted')
    except ValueError:
        pass

    # fractional labels are rejected on the geometric path only
    frac = torch.rand(2, 1, 8, 8)
    logits = torch.randn(2, 1, 8, 8)
    SegmentationLoss()(logits, frac)  # BCE-only keeps accepting them
    try:
        SegmentationLoss(dice_weight=1.0)(logits, frac)
        raise AssertionError('fractional labels accepted for geometric term')
    except ValueError as e:
        assert 'pseudo-label' in str(e)


# ---------------------------------------------------------------------------
# 9.4 Dice properties
# ---------------------------------------------------------------------------

def test_9_4_dice():
    y = _vessel()
    y4 = y[None, None]
    good = torch.sigmoid(4 * (2 * y - 1))[None, None]
    bad = torch.sigmoid(-4 * (2 * y - 1))[None, None]
    assert soft_dice_loss(good, y4) < soft_dice_loss(bad, y4)

    # empty targets stay in the term and are finite
    empty = torch.zeros_like(y4)
    good_on_empty = torch.zeros_like(y4)          # predicts nothing
    wrong_on_empty = torch.ones_like(y4) * 0.999    # confident false positive
    l_ok = soft_dice_loss(good_on_empty, empty)
    l_bad = soft_dice_loss(wrong_on_empty, empty)
    assert torch.isfinite(l_ok) and torch.isfinite(l_bad)
    assert l_bad > 0.9 and l_ok < 0.01

    # batch size one is finite
    assert torch.isfinite(soft_dice_loss(good[:1], y4[:1]))

    # reduction is per image: batch loss == mean of per-image losses
    pair_p = torch.cat([good, bad])
    pair_y = torch.cat([y4, y4])
    solo = (soft_dice_loss(good, y4) + soft_dice_loss(bad, y4)) / 2
    assert torch.allclose(soft_dice_loss(pair_p, pair_y), solo, atol=1e-7)


# ---------------------------------------------------------------------------
# 9.5 clDice properties
# ---------------------------------------------------------------------------

def test_9_5_cldice():
    y = _vessel()
    y4 = y[None, None]
    good = torch.sigmoid(4 * (2 * y - 1))[None, None]

    intact = soft_cldice_loss(good, y4, iterations=10)
    # a prediction with a gap must score worse against the INTACT target:
    # the target skeleton crosses the gap where the prediction is ~0
    gapped_pred = y.clone()
    gapped_pred[:, 30:35] = 0.0
    gap_pred = torch.sigmoid(4 * (2 * gapped_pred - 1))[None, None]
    cl_gap = soft_cldice_loss(gap_pred, y4, iterations=10)
    cl_miss = soft_cldice_loss(torch.zeros_like(y4), y4, iterations=10)
    assert intact < 0.05, intact.item()
    assert cl_gap > intact, (cl_gap.item(), intact.item())
    assert cl_miss > 0.9 and cl_miss > cl_gap, (cl_miss.item(), cl_gap.item())

    # empty targets: explicit differentiable zero, finite, zero loss
    empty = torch.zeros_like(y4)
    assert soft_cldice_loss(good, empty, 10).item() == 0.0
    assert torch.isfinite(soft_cldice_loss(torch.zeros_like(y4), empty, 10))
    # mixed batch averages across the original batch size
    mixed = soft_cldice_loss(torch.cat([good, good]), torch.cat([y4, empty]), 10)
    assert torch.allclose(mixed, intact / 2, atol=1e-7)

    # gradient reaches the logits and is finite
    lg = (3.0 * (2 * y - 1))[None, None].requires_grad_(True)
    loss = soft_cldice_loss(torch.sigmoid(lg), y4, 10)
    loss.backward()
    assert torch.isfinite(lg.grad).all() and lg.grad.abs().sum() > 0

    # skeleton on a thick vessel: support inside the vessel, smaller than it
    thick = torch.zeros(1, 1, 64, 64)
    thick[:, :, 20:40, 10:50] = 1.0  # 20px-wide bar
    skel = soft_skeleton(thick, 10)
    assert (skel[:, :, thick[:, :] == 0] == 0).all() if False else True
    bg = thick == 0
    assert (skel[bg] == 0).all(), 'skeleton leaks outside the structure'
    vessel_sum = thick.sum()
    assert 0 < skel.sum() < vessel_sum, (skel.sum().item(), vessel_sum.item())
    assert skel.max() > 0


# ---------------------------------------------------------------------------
# 9.6 Boundary properties
# ---------------------------------------------------------------------------

def test_9_6_boundary():
    y = torch.zeros(16, 16)
    y[6:9, 5:13] = 1.0  # 3px-thick, 8px-long rectangle
    y4 = y[None, None]
    dist = signed_distance_map(y4)

    # sign convention: positive outside, nonpositive inside, zero on the
    # inner one-pixel boundary
    assert (dist[y4 == 0] > 0).all(), 'background distances must be positive'
    assert (dist[y4 == 1] <= 0).all(), 'foreground distances must be nonpositive'
    assert (dist[y4 == 1] == 0).any(), 'inner boundary must be exactly zero'
    assert (dist[y4 == 1] < 0).any(), 'deep interior must be negative'
    assert dist.dtype == torch.float32

    # directional gradients: increasing a distant background logit increases
    # the loss; increasing a deep-foreground logit decreases it
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    loss = boundary_loss(torch.sigmoid(logits), dist)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] > 0, 'far background pixel gradient must be positive'
    assert logits.grad[0, 0, 7, 8] < 0, 'deep foreground pixel gradient must be negative'

    # correct foreground can produce a negative loss (signed objective)
    assert boundary_loss(y4, dist) < 0

    # degenerate masks: explicit zero map, zero contribution, kept in the mean
    all_bg = torch.zeros(1, 1, 16, 16)
    all_fg = torch.ones(1, 1, 16, 16)
    assert torch.equal(signed_distance_map(all_bg), torch.zeros_like(all_bg))
    assert torch.equal(signed_distance_map(all_fg), torch.zeros_like(all_fg))
    p = torch.sigmoid(torch.randn(1, 1, 16, 16))
    assert boundary_loss(p, signed_distance_map(all_bg)) == 0
    assert boundary_loss(p, signed_distance_map(all_fg)) == 0
    mixed_map = torch.cat([dist, torch.zeros_like(dist)])
    mixed_p = torch.cat([p, p])
    mixed = boundary_loss(mixed_p, mixed_map)
    assert torch.isfinite(mixed)
    assert torch.allclose(mixed, boundary_loss(p, dist) / 2, atol=1e-7)

    # [B, H, W] / [H, W] input shapes of the map utility
    assert signed_distance_map(y).shape == (16, 16)
    assert signed_distance_map(y4[0]).shape == (1, 16, 16)


# ---------------------------------------------------------------------------
# 9.7 Geometry: map is recomputed from the transformed (final) label
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(tmp, size=40):
    from PIL import Image
    rng = np.random.RandomState(0)
    im = rng.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[18:22, :] = 255  # 4px-thick vessel across the full width
    mask = np.full((size, size), 255, dtype=np.uint8)
    paths = []
    for name, arr in (('im', im), ('gt', gt), ('mask', mask)):
        p = os.path.join(tmp, name + '.png')
        Image.fromarray(arr).save(p)
        paths.append(p)
    csv = os.path.join(tmp, 'train.csv')
    with open(csv, 'w') as f:
        f.write('im_paths,gt_paths,mask_paths\n')
        f.write(','.join(paths) + '\n')
    return csv


def test_9_7_geometry():
    from utils.get_loaders import get_train_val_datasets
    with tempfile.TemporaryDirectory() as tmp:
        csv = _make_synthetic_dataset(tmp)
        # vanilla path, boundary enabled: sample carries the extra map and it
        # equals a recomputation from the returned final (resized) label
        tr, va = get_train_val_datasets(csv, csv, tg_size=(48, 48),
                                        label_values=[0, 255],
                                        need_distance_map=True)
        img, target, dist = tr[0]
        assert dist.shape == (1, 48, 48) and dist.dtype == torch.float32
        assert torch.equal(dist, signed_distance_map(target.unsqueeze(0)))
        img_v, target_v, dist_v = va[0]
        assert torch.equal(dist_v, signed_distance_map(target_v.unsqueeze(0)))

        # disabled: existing two-tensor interface is preserved
        tr0, va0 = get_train_val_datasets(csv, csv, tg_size=(48, 48),
                                          label_values=[0, 255])
        assert len(tr0[0]) == 2 and len(va0[0]) == 2

        # FreeSDG FMAug train path resolves the same requirement generically
        freesdg_cfg = {'enabled': True, 'seed': 0, 'raw_prob': 0.0,
                       'test_input': 'raw', 'anchor_w': 27, 'anchor_sigma': 9,
                       'ratio': 4.0, 'mixup_size': -1, 'mix_policy': 'repo',
                       'aug_mode': 'fmaug', 'lwnet_aug_profile': 'original'}
        trf, vaf = get_train_val_datasets(csv, csv, tg_size=(48, 48),
                                          label_values=[0, 255],
                                          freesdg_cfg=freesdg_cfg,
                                          need_distance_map=True)
        img_f, target_f, dist_f = trf[0]
        assert torch.equal(dist_f, signed_distance_map(target_f.unsqueeze(0)))
        assert trf.need_distance_map and vaf.need_distance_map

    # the map utility itself is geometry-consistent (flip equivariance)
    y = _vessel()[None, None]
    m = signed_distance_map(y)
    assert torch.equal(signed_distance_map(torch.flip(y, dims=[3])), torch.flip(m, dims=[3]))


# ---------------------------------------------------------------------------
# 9.8 Integration: one short forward/backward smoke run
# ---------------------------------------------------------------------------

def _run_epoch(loader, model, crit, train=True):
    import train_cyclical
    if train:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        res = train_cyclical.run_one_epoch(loader, model, crit, optimizer=opt, scheduler=sch, assess=True)
    else:
        res = train_cyclical.run_one_epoch(loader, model, crit, assess=True)
    return res


def test_9_8_integration_real():
    from torch.utils.data import DataLoader, TensorDataset
    from models.get_model import get_arch
    import train_cyclical

    torch.manual_seed(0)
    n = 4
    imgs = torch.rand(n, 3, 64, 64)
    tgts = torch.stack([_vessel(64, 64, cols=(6 + 4 * i, 50 + 4 * i)) for i in range(n)])
    dists = signed_distance_map(tgts.unsqueeze(1))  # [B, 1, H, W]
    loader = DataLoader(TensorDataset(imgs, tgts, dists), batch_size=2)
    loader_plain = DataLoader(TensorDataset(imgs, tgts), batch_size=2)

    # vanilla unet, all four terms: train-mode smoke run (forward/backward/step)
    model = get_arch('unet')
    crit = SegmentationLoss(1.0, 1.0, 0.2, 0.01, cldice_iters=10)
    _, _, run_loss, _, comps = _run_epoch(loader, model, crit, train=True)
    assert np.isfinite(run_loss)
    assert set(comps) == {'bce', 'dice', 'cldice', 'boundary'}
    assert all(np.isfinite(v) for v in comps.values())
    total_from_comps = sum(crit.weights[k] * v for k, v in comps.items())
    assert abs(total_from_comps - run_loss) < 1e-5

    # wnet (two supervised heads, unit aggregation weights)
    wmodel = get_arch('wnet')
    _, _, run_loss_w, _, comps_w = _run_epoch(loader, wmodel, crit, train=True)
    assert np.isfinite(run_loss_w)
    assert set(comps_w) == {'bce', 'dice', 'cldice', 'boundary'}

    # parameter count and checkpoint keys unchanged by the objective
    assert len(list(SegmentationLoss(1, 1, 0.2, 0.01).parameters())) == 0
    assert list(get_arch('unet').state_dict().keys()) == list(model.state_dict().keys())

    # BCE-only run_one_epoch total equals the original criterion on the same
    # eval-mode model outputs
    model.eval()
    _, _, run_loss_bce, _, comps_bce = _run_epoch(loader_plain, model,
                                                  SegmentationLoss(), train=False)
    with torch.no_grad():
        ref = torch.stack([torch.nn.BCEWithLogitsLoss()(model(imgs[i:i+2]),
                                                        tgts[i:i+2].unsqueeze(1).float())
                           for i in range(0, n, 2)]).mean()
    assert abs(run_loss_bce - ref.item()) < 1e-6, (run_loss_bce, ref.item())
    assert list(comps_bce) == ['bce']

    # identical weight resolution with and without FreeSDG flags appended
    base = ['--loss_bce_weight', '1', '--loss_dice_weight', '1',
            '--loss_cldice_weight', '0.2', '--loss_boundary_weight', '0.01',
            '--loss_cldice_iters', '10']
    a = build_segmentation_loss(train_cyclical.parser.parse_args(base))
    b = build_segmentation_loss(train_cyclical.parser.parse_args(base + ['--freesdg']))
    assert a.weights == b.weights and a.cldice_iters == b.cldice_iters
    assert a.need_distance_map and b.need_distance_map

    # indicative CPU overhead of the geometric terms (report, not a gate)
    timings = {}
    for label, criterion in (('bce-only', SegmentationLoss()),
                             ('bce+dice+cldice+boundary', crit)):
        model_t = get_arch('unet')
        opt = torch.optim.Adam(model_t.parameters(), lr=1e-3)
        start = time.perf_counter()
        for _ in range(3):
            train_cyclical.run_one_epoch(loader if criterion.need_distance_map else loader_plain,
                                         model_t, criterion, optimizer=opt,
                                         scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100))
        timings[label] = (time.perf_counter() - start) / 3
    print('    [info] CPU sec/epoch (2 batches of 2, 64x64, unet): ' +
          ', '.join('{}={:.3f}s'.format(k, v) for k, v in timings.items()))


# ---------------------------------------------------------------------------

SECTIONS = [
    ('9.1 BCE-only compatibility', test_9_1_bce_only_compatibility),
    ('9.2 composition + per-term grads', test_9_2_composition),
    ('9.3 disabled work + validation', test_9_3_disabled_work_and_validation),
    ('9.4 dice properties', test_9_4_dice),
    ('9.5 cldice properties', test_9_5_cldice),
    ('9.6 boundary properties', test_9_6_boundary),
    ('9.7 geometry / dataset maps', test_9_7_geometry),
    ('9.8 integration smoke', test_9_8_integration_real),
]


def main():
    failed = 0
    for name, fn in SECTIONS:
        try:
            fn()
            print('[PASS] {}'.format(name))
        except Exception:
            failed += 1
            print('[FAIL] {}'.format(name))
            traceback.print_exc()
    if failed:
        print('\n{} section(s) failed'.format(failed))
        sys.exit(1)
    print('\nall sections passed')
    sys.exit(0)


if __name__ == '__main__':
    main()
