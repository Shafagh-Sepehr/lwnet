"""Ticket 11 (FDPR arm S1): soft-clDice term added to the compound loss.

Plain-Python runnable -- no pytest dependency:

    python unit-test/test_loss_cldice_term.py

CPU-only synthetic tensors; no dataset, training run, or experiment directory
is required.  Exits 0 when every section passes, non-zero on the first
failure.

The soft-clDice term already existed in ``segmentation_losses.py`` (it backs
the historical ``bce_cldice020_i10`` control family).  These tests pin the
properties the FDPR arm depends on: feature-off identity, four-term
composition, finite gradients at odd/rectangular shapes, config round trip,
and unchanged strict checkpoint loading.
"""

import json
import math
import os
import sys
import tempfile
import traceback
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from segmentation_losses import (  # noqa: E402
    LOSS_DEFAULTS,
    SegmentationLoss,
    boundary_loss,
    build_segmentation_loss,
    signed_distance_map,
    soft_cldice_loss,
    soft_dice_loss,
)
import torch.nn.functional as F  # noqa: E402


# Reference compound configuration (clDice weight 0 = feature off).
REFERENCE_WEIGHTS = dict(bce_weight=1.0, dice_weight=0.25, boundary_weight=0.01)
# Historical clDice control weight and iteration count.
CONTROL_CLDICE_WEIGHT = 0.2
CONTROL_CLDICE_ITERS = 10


def _reference_loss(**overrides):
    kwargs = dict(REFERENCE_WEIGHTS)
    kwargs.update(overrides)
    return SegmentationLoss(**kwargs)


def _synthetic_batch(batch_size, height, width, seed=0):
    """Binary vessel-like targets plus logits; the last sample is empty.

    The bar-and-branch shapes give the soft skeleton real structure, while the
    all-background sample exercises the explicit zero-contribution path.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, 1, height, width, generator=generator) * 2.0
    target = torch.zeros(batch_size, 1, height, width)
    for i in range(batch_size):
        if i == batch_size - 1 and batch_size > 1:
            continue  # all-background sample
        row = 2 + i
        col = 1 + i
        thickness = max(1, height // 8)
        branch_width = max(1, width // 6)
        target[i, 0, row:row + thickness, col:width - col] = 1.0
        target[i, 0, row:height - row, col:col + branch_width] = 1.0
    return logits, target


def _as_reference_target(target):
    assert ((target == 0) | (target == 1)).all()
    return target


# ---------------------------------------------------------------------------
# 1. Weight 0 is the reference path: identical values, gradients, RNG state
# ---------------------------------------------------------------------------

def test_1_weight_zero_identity():
    torch.manual_seed(0)
    logits, target = _synthetic_batch(4, 48, 64, seed=0)
    target = _as_reference_target(target)
    distance_map = signed_distance_map(target)

    reference = _reference_loss()
    arm_off = _reference_loss(cldice_weight=0.0)
    assert arm_off.weights['cldice'] == 0.0
    assert arm_off.need_distance_map == reference.need_distance_map
    assert arm_off._need_probs == reference._need_probs

    state = torch.get_rng_state().clone()
    ref_logits = logits.clone().requires_grad_(True)
    ref_total, ref_comps = reference(ref_logits, target, distance_map=distance_map)
    assert torch.equal(torch.get_rng_state(), state), \
        'reference forward consumed torch RNG'

    arm_logits = logits.clone().requires_grad_(True)
    arm_total, arm_comps = arm_off(arm_logits, target, distance_map=distance_map)
    assert torch.equal(torch.get_rng_state(), state), \
        'clDice weight 0 forward consumed torch RNG'

    assert torch.equal(arm_total, ref_total), (arm_total.item(), ref_total.item())
    assert list(arm_comps) == list(ref_comps) == ['bce', 'dice', 'boundary']
    for key in ref_comps:
        assert torch.equal(arm_comps[key], ref_comps[key]), key

    arm_total.backward()
    ref_total.backward()
    assert torch.equal(arm_logits.grad, ref_logits.grad)

    # Iteration count stays irrelevant when the term is off (old configs may
    # omit --loss_cldice_iters entirely).
    arm_off_zero_iters = _reference_loss(cldice_weight=0.0, cldice_iters=0)
    again = logits.clone().requires_grad_(True)
    total_again, _ = arm_off_zero_iters(again, target, distance_map=distance_map)
    assert torch.equal(total_again, ref_total.detach())
    assert 'cldice' not in arm_off.formula


# ---------------------------------------------------------------------------
# 2. Batch 1 and 4, odd/rectangular shapes: finite outputs and gradients
# ---------------------------------------------------------------------------

def test_2_batch_and_odd_shapes():
    cases = ((1, 65, 47), (4, 33, 81))
    for batch_size, height, width in cases:
        logits, target = _synthetic_batch(batch_size, height, width, seed=batch_size)
        target = _as_reference_target(target)
        distance_map = signed_distance_map(target)
        criterion = _reference_loss(cldice_weight=CONTROL_CLDICE_WEIGHT,
                                    cldice_iters=CONTROL_CLDICE_ITERS)
        shifted = logits.clone().requires_grad_(True)
        total, comps = criterion(shifted, target, distance_map=distance_map)
        label = 'b={}, {}x{}'.format(batch_size, height, width)
        assert torch.isfinite(total), 'nonfinite total at ' + label
        assert set(comps) == {'bce', 'dice', 'cldice', 'boundary'}, (label, sorted(comps))
        for key, value in comps.items():
            assert torch.isfinite(value), 'nonfinite {} at {}'.format(key, label)
        assert 0.0 < comps['cldice'] < 1.0, (label, comps['cldice'].item())
        total.backward()
        assert torch.isfinite(shifted.grad).all(), 'nonfinite grad at ' + label
        assert shifted.grad.abs().sum() > 0, 'zero grad at ' + label
        # target shape [B, H, W] is accepted at this boundary too
        bare = logits.clone().requires_grad_(True)
        total_bare, _ = criterion(bare, target[:, 0], distance_map=distance_map[:, 0])
        assert torch.allclose(total_bare, criterion(logits, target,
                                                    distance_map=distance_map)[0])

    # Parametrised double check that a single sample is finite for both terms.
    one_logits, one_target = _synthetic_batch(1, 41, 29, seed=7)
    criterion = _reference_loss(cldice_weight=CONTROL_CLDICE_WEIGHT)
    probs = torch.sigmoid(one_logits)
    assert torch.isfinite(soft_cldice_loss(probs, one_target, CONTROL_CLDICE_ITERS))
    assert torch.isfinite(soft_dice_loss(probs, one_target))


# ---------------------------------------------------------------------------
# 3. Four-term composition equals the explicit weighted sum
# ---------------------------------------------------------------------------

def test_3_four_term_combination():
    torch.manual_seed(3)
    logits, target = _synthetic_batch(4, 64, 64, seed=3)
    target = _as_reference_target(target)
    distance_map = signed_distance_map(target)
    criterion = _reference_loss(cldice_weight=CONTROL_CLDICE_WEIGHT,
                                cldice_iters=CONTROL_CLDICE_ITERS)

    shifted = logits.clone().requires_grad_(True)
    total, comps = criterion(shifted, target, distance_map=distance_map)
    probs = torch.sigmoid(logits)
    manual = (REFERENCE_WEIGHTS['bce_weight'] * F.binary_cross_entropy_with_logits(logits, target)
              + REFERENCE_WEIGHTS['dice_weight'] * soft_dice_loss(probs, target)
              + CONTROL_CLDICE_WEIGHT * soft_cldice_loss(probs, target, CONTROL_CLDICE_ITERS)
              + REFERENCE_WEIGHTS['boundary_weight'] * boundary_loss(probs, distance_map))
    assert torch.allclose(total, manual, atol=1e-6), (total.item(), manual.item())
    total_from_comps = sum(criterion.weights[key] * comps[key] for key in comps)
    assert torch.allclose(total_from_comps, total, atol=1e-6)
    assert '0.2*cldice(iters=10)' in criterion.formula


# ---------------------------------------------------------------------------
# 4. Config round trip: parser -> config.cfg JSON -> criterion, and defaults
# ---------------------------------------------------------------------------

def _arm_args():
    import train_cyclical
    return train_cyclical.parser.parse_args([
        '--csv_train', 'data/DRIVE/train.csv',
        '--loss_bce_weight', '1.0',
        '--loss_dice_weight', '0.25',
        '--loss_cldice_weight', '0.2',
        '--loss_boundary_weight', '0.01',
        '--loss_cldice_iters', '10',
    ])


def test_4_config_round_trip():
    import train_cyclical
    from models.get_model import get_arch_options

    args = _arm_args()
    criterion = build_segmentation_loss(args)
    assert criterion.weights == {'bce': 1.0, 'dice': 0.25,
                                 'cldice': 0.2, 'boundary': 0.01}
    assert criterion.cldice_iters == 10

    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, 'config.cfg')
        train_cyclical.write_training_config(args, config_path)
        with open(config_path) as handle:
            serialized = json.load(handle)
    restored = build_segmentation_loss(types.SimpleNamespace(**serialized))
    assert restored.weights == criterion.weights
    assert restored.cldice_iters == criterion.cldice_iters
    assert restored.formula == criterion.formula

    # Feature-off reference values survive the same round trip.
    reference = build_segmentation_loss(types.SimpleNamespace(
        loss_bce_weight=1.0, loss_dice_weight=0.25, loss_cldice_weight=0.0,
        loss_boundary_weight=0.01, loss_cldice_iters=10))
    assert reference.weights['cldice'] == 0.0
    assert reference.formula == 'total = 1*bce + 0.25*dice + 0.01*boundary  (eps=1e-06)'

    # A config that predates the loss flags resolves to the documented defaults.
    old = build_segmentation_loss(types.SimpleNamespace(csv_train='data/DRIVE/train.csv'))
    assert old.weights == {'bce': 1.0, 'dice': 0.0, 'cldice': 0.0, 'boundary': 0.0}
    assert old.cldice_iters == LOSS_DEFAULTS['loss_cldice_iters'] == 10

    # Loss keys never reach model reconstruction.
    base = {'model_name': 'wnet'}
    with_loss = dict(base, loss_bce_weight=1.0, loss_dice_weight=0.25,
                     loss_cldice_weight=0.2, loss_boundary_weight=0.01,
                     loss_cldice_iters=10)
    assert get_arch_options(base) == get_arch_options(with_loss)


# ---------------------------------------------------------------------------
# 5. Strict checkpoint reconstruction and CPU smoke (scratch dir only)
# ---------------------------------------------------------------------------

def test_5_checkpoint_and_cpu_smoke():
    from torch.utils.data import DataLoader, TensorDataset
    from models.get_model import get_arch, get_arch_from_config
    from utils.model_saving_loading import load_model, save_model
    import train_cyclical

    torch.manual_seed(0)
    base_model = get_arch('wnet')
    fresh = get_arch_from_config({'model_name': 'wnet'})
    fresh.load_state_dict(base_model.state_dict(), strict=True)
    assert list(fresh.state_dict()) == list(base_model.state_dict())

    criterion = _reference_loss(cldice_weight=CONTROL_CLDICE_WEIGHT)
    assert len(list(criterion.parameters())) == 0
    assert len(criterion.state_dict()) == 0

    with tempfile.TemporaryDirectory() as tmp:
        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
        save_model(tmp, base_model, optimizer)
        rebuilt = get_arch_from_config({'model_name': 'wnet'})
        rebuilt, _ = load_model(rebuilt, tmp, device='cpu')
        assert list(rebuilt.state_dict()) == list(base_model.state_dict())
        for key, value in base_model.state_dict().items():
            assert torch.equal(value, rebuilt.state_dict()[key]), key

        images = torch.rand(2, 3, 64, 64)
        targets = torch.stack([_synthetic_batch(1, 64, 64, seed=i)[1][0, 0]
                               for i in range(2)])
        distance_maps = signed_distance_map(targets.unsqueeze(1))
        loader = DataLoader(TensorDataset(images, targets, distance_maps), batch_size=2)

        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        _, _, train_loss, _, train_comps = train_cyclical.run_one_epoch(
            loader, base_model, criterion, optimizer=optimizer,
            scheduler=scheduler, assess=True)
        _, _, val_loss, _, val_comps = train_cyclical.run_one_epoch(
            loader, base_model, criterion, assess=True)
        assert math.isfinite(train_loss) and math.isfinite(val_loss)
        assert set(train_comps) == set(val_comps) == {'bce', 'dice', 'cldice', 'boundary'}
        total_from_comps = sum(criterion.weights[key] * train_comps[key]
                               for key in train_comps)
        assert abs(total_from_comps - train_loss) < 1e-5

        config_path = os.path.join(tmp, 'config.cfg')
        train_cyclical.write_training_config(_arm_args(), config_path)
        assert sorted(os.listdir(tmp)) == ['config.cfg', 'model_checkpoint.pth']


# ---------------------------------------------------------------------------

SECTIONS = [
    ('1 weight-0 identity', test_1_weight_zero_identity),
    ('2 batch/odd-shape finiteness', test_2_batch_and_odd_shapes),
    ('3 four-term composition', test_3_four_term_combination),
    ('4 config round trip', test_4_config_round_trip),
    ('5 checkpoint + CPU smoke', test_5_checkpoint_and_cpu_smoke),
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
