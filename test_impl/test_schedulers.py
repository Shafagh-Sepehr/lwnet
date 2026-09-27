import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.schedulers import (
    DampedCosineError, DampedCosineLRSchedule, damped_cosine_lr,
    damped_cosine_max_lr, damped_cosine_schedule_max,
    validate_damped_cosine_config,
)


def _make_optimizer(lrs=(0.01,)):
    params = [torch.nn.Parameter(torch.zeros(3)) for _ in lrs]
    return torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, lrs)], lr=lrs[0])


class ClosedFormTests(unittest.TestCase):
    def test_initial_value_is_lr_max(self):
        # lr(0): envelope=1, oscillation=1 -> exactly lr_max
        self.assertAlmostEqual(
            damped_cosine_lr(0, 100, 40, 3.0, 0.95, 0.01, 1e-8), 0.01, places=15)

    def test_alpha_zero_has_no_envelope_decay(self):
        # alpha=0 envelope is constant 1, so the envelope ceiling never decays
        for t in (0, 10, 50, 99):
            envelope = damped_cosine_lr(t, 100, 40, 0.0, 0.0, 0.01, 0.0)
            osc = (1.0 + 0.0 * math.cos(2 * math.pi * t / 40)) / 1.0
            self.assertAlmostEqual(envelope, 0.01 * osc, places=15)

    def test_damping_shrinks_peaks_over_training(self):
        early = damped_cosine_lr(0, 200, 50, 3.0, 0.95, 0.01, 1e-8)
        late = damped_cosine_lr(150, 200, 50, 3.0, 0.95, 0.01, 1e-8)
        self.assertLess(late, early)

    def test_inflation_capped_at_inflate_max_lr(self):
        total, period, alpha, d = 400, 40, -0.5, 0.95
        cap = 0.02
        for t in range(total):
            lr = damped_cosine_lr(t, total, period, alpha, d, 0.01, 0.0,
                                  inflate_max_lr=cap)
            self.assertLessEqual(lr, cap + 1e-15)
        # without the cap the envelope would exceed lr_max at the end
        uncapped = damped_cosine_lr(total - 1, total, period, alpha, d, 0.01, 0.0)
        self.assertGreater(uncapped, 0.01)

    def test_t_equals_one_no_division_by_zero(self):
        self.assertAlmostEqual(
            damped_cosine_lr(0, 1, 1, 3.0, 0.95, 0.01, 1e-8), 0.01, places=15)

    def test_parameter_group_ratio_preserved(self):
        optimizer = _make_optimizer(lrs=(0.01, 0.005))
        sched = DampedCosineLRSchedule(optimizer, total_updates=100, period=25,
                                       alpha=3.0, d=0.95, lr_max=0.01, lr_min=1e-8)
        for _ in range(30):
            sched.step()
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']
        # ratio 0.005/0.01 == 0.5 is preserved at every step
        self.assertAlmostEqual(lr1 / lr0, 0.5, places=12)

    def test_one_step_per_optimizer_update(self):
        optimizer = _make_optimizer()
        total, period = 10, 4
        sched = DampedCosineLRSchedule(optimizer, total_updates=total, period=period,
                                       alpha=2.0, d=0.9, lr_max=0.01, lr_min=0.0)
        # construction sets update 0
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],
                               damped_cosine_lr(0, total, period, 2.0, 0.9, 0.01, 0.0),
                               places=15)
        for t in range(1, total):
            sched.step()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],
                                   damped_cosine_lr(t, total, period, 2.0, 0.9, 0.01, 0.0),
                                   places=15)

    def test_steps_past_budget_hold_last_value(self):
        optimizer = _make_optimizer()
        sched = DampedCosineLRSchedule(optimizer, total_updates=3, period=2,
                                       alpha=3.0, d=0.95, lr_max=0.01, lr_min=0.0)
        for _ in range(10):
            sched.step()
        self.assertEqual(sched.last_update, 2)

    def test_state_dict_round_trip(self):
        optimizer = _make_optimizer(lrs=(0.01, 0.005))
        sched = DampedCosineLRSchedule(optimizer, total_updates=50, period=10,
                                       alpha=1.5, d=0.5, lr_max=0.01, lr_min=1e-8)
        for _ in range(17):
            sched.step()
        state = sched.state_dict()
        fresh = _make_optimizer(lrs=(0.01, 0.005))
        restored = DampedCosineLRSchedule(fresh, total_updates=50, period=10,
                                          alpha=1.5, d=0.5, lr_max=0.01, lr_min=1e-8)
        restored.load_state_dict(state)
        self.assertEqual(restored.last_update, sched.last_update)
        self.assertAlmostEqual(fresh.param_groups[0]['lr'],
                               restored._lr_for_group(17, 0.01, 0.01 * 1e-8 / 0.01), places=15)


class MaxQueryTests(unittest.TestCase):
    def test_query_matches_scanned_sequence_damped(self):
        total, period, alpha, d = 200, 37, 3.0, 0.95
        best_lr, best_t, _ = damped_cosine_schedule_max(total, period, alpha, d, 0.01, 1e-8)
        scanned = [damped_cosine_lr(t, total, period, alpha, d, 0.01, 1e-8)
                   for t in range(total)]
        self.assertAlmostEqual(best_lr, max(scanned), places=15)
        self.assertEqual(best_t, scanned.index(max(scanned)))

    def test_query_matches_scanned_sequence_inflating(self):
        total, period, alpha, d, cap = 300, 25, -0.4, 0.95, 0.03
        best_lr, best_t, ties = damped_cosine_schedule_max(
            total, period, alpha, d, 0.01, 0.0, inflate_max_lr=cap)
        scanned = [damped_cosine_lr(t, total, period, alpha, d, 0.01, 0.0,
                                    inflate_max_lr=cap) for t in range(total)]
        self.assertAlmostEqual(best_lr, max(scanned), places=15)
        self.assertEqual(best_t, scanned.index(max(scanned)))
        self.assertEqual(ties, scanned.count(max(scanned)))

    def test_query_t_equals_one(self):
        best_lr, best_t, ties = damped_cosine_schedule_max(1, 1, 3.0, 0.95, 0.01, 1e-8)
        self.assertAlmostEqual(best_lr, 0.01, places=15)
        self.assertEqual((best_t, ties), (0, 1))

    def test_envelope_ceiling_bounds_oscillation(self):
        total, period, alpha, d = 120, 30, 3.0, 0.95
        for t in range(total):
            ceiling = damped_cosine_max_lr(t, total, alpha, 0.01, 0.0)
            self.assertGreaterEqual(
                ceiling + 1e-15,
                damped_cosine_lr(t, total, period, alpha, d, 0.01, 0.0))
        # at a cosine peak the ceiling and the actual rate coincide
        self.assertAlmostEqual(
            damped_cosine_max_lr(0, total, alpha, 0.01, 0.0),
            damped_cosine_lr(0, total, period, alpha, d, 0.01, 0.0), places=15)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_total_and_period(self):
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(0, 10, 3.0, 0.95, 0.01, 0.0)
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(10, 0, 3.0, 0.95, 0.01, 0.0)

    def test_rejects_alpha_at_or_below_minus_one(self):
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(10, 10, -1.0, 0.95, 0.01, 0.0)

    def test_rejects_depth_and_rate_bounds(self):
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(10, 10, 3.0, 1.5, 0.01, 0.0)
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(10, 10, 3.0, 0.95, 0.01, 0.02)

    def test_rejects_cap_at_or_below_max_lr(self):
        with self.assertRaises(DampedCosineError):
            validate_damped_cosine_config(10, 10, -0.5, 0.95, 0.01, 0.0,
                                          inflate_max_lr=0.005)

    def test_constructor_propagates_validation(self):
        optimizer = _make_optimizer()
        with self.assertRaises(DampedCosineError):
            DampedCosineLRSchedule(optimizer, total_updates=0, period=1,
                                   alpha=3.0, d=0.95, lr_max=0.01, lr_min=0.0)


if __name__ == '__main__':
    unittest.main()
