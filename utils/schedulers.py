"""
Optional damped-cosine learning-rate schedule for LwNet.

lr(t) = lr_min + (lr_max - lr_min)
        * [1 / (1 + alpha * t / (T - 1))]
        * [(1 + d * cos(2*pi*t/P)) / (1 + d)]

where
    t      : zero-based optimizer-update index, 0 .. T-1
    T      : planned total optimizer updates (fixed before training)
    P      : oscillation period in optimizer updates
    alpha  : nonnegative decay strength of the multiplicative envelope
    d      : oscillation depth in [0, 1]
    lr_max : learning rate at t=0 (upper bound)
    lr_min : lower bound reached only asymptotically (envelope floor)

Properties worth stating explicitly (they differ from the baseline
CosineAnnealingLR schedule in train_cyclical.py):
    * This is a smooth *full*-cosine oscillation with a shrinking envelope,
      not a sawtooth sequence of half-cosine restarts.
    * With d < 1 the trough does not reach lr_min exactly: the minimum of the
      oscillating factor is (1 - d)/(1 + d) (e.g. ~0.0256 for d = 0.95).
    * alpha = 0 removes the envelope decay; it does NOT reproduce the
      original scheduler (different lower bound and no eta_min=0 floor).
    * The final learning rate depends on the phase (T mod P) and need not
      equal lr_min.
    * lr(0) = lr_max exactly, so the first optimizer update runs at lr_max.

If the optimizer has several parameter groups, each group's upper bound is
its own initial learning rate and its lower bound is scaled by the same
ratio lr_min/lr_max, so the groups' intended learning-rate ratios are
preserved at every step t.
"""

import math


class DampedCosineError(ValueError):
    """Raised when damped-cosine scheduler settings are invalid."""


def validate_damped_cosine_config(total_updates, period, alpha, d, lr_max, lr_min):
    """Validate damped-cosine settings; raise DampedCosineError with a clear message."""
    if not isinstance(total_updates, int) or total_updates < 1:
        raise DampedCosineError(
            f"total planned optimizer updates must be an integer >= 1, got {total_updates!r}")
    if not isinstance(period, int) or period < 1:
        raise DampedCosineError(
            f"oscillation period must be an integer >= 1 optimizer update, got {period!r}")
    if alpha < 0:
        raise DampedCosineError(f"alpha (decay strength) must be >= 0, got {alpha}")
    if not (0.0 <= d <= 1.0):
        raise DampedCosineError(f"d (oscillation depth) must be in [0, 1], got {d}")
    if lr_min < 0:
        raise DampedCosineError(f"lr_min must be >= 0, got {lr_min}")
    if lr_max <= lr_min:
        raise DampedCosineError(
            f"lr_max must be > lr_min, got lr_max={lr_max}, lr_min={lr_min}")


def damped_cosine_lr(t, total_updates, period, alpha, d, lr_max, lr_min):
    """Closed-form damped-cosine learning rate at zero-based update index t."""
    if total_updates == 1:
        progress = 0.0  # single-update schedule: avoid division by zero
    else:
        progress = t / (total_updates - 1)
    envelope = 1.0 / (1.0 + alpha * progress)
    oscillation = (1.0 + d * math.cos(2.0 * math.pi * t / period)) / (1.0 + d)
    return lr_min + (lr_max - lr_min) * envelope * oscillation


class DampedCosineLRSchedule:
    """Sets each optimizer update t to lr(t) of the damped-cosine formula.

    Step accounting: construction writes lr(0) into the optimizer (update 0
    then runs at lr(0) = lr_max); every scheduler.step() call right after an
    optimizer.step() advances the internal counter, so update t always uses
    lr(t). Exactly one step() per optimizer update -- no extra steps for
    gradient accumulation.
    """

    def __init__(self, optimizer, total_updates, period, alpha, d, lr_max, lr_min):
        validate_damped_cosine_config(total_updates, period, alpha, d, lr_max, lr_min)
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.period = period
        self.alpha = float(alpha)
        self.d = float(d)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        # Per-parameter-group bounds: keep each group's intended lr ratio
        # (group_lr_min / group_lr_max == lr_min / lr_max for every group).
        self.group_lr_max = [float(group['lr']) for group in optimizer.param_groups]
        self.group_lr_min = [lr * self.lr_min / self.lr_max for lr in self.group_lr_max]
        self.last_update = 0  # zero-based index of the NEXT optimizer update
        self._apply_lr(self.last_update)

    def _lr_for_group(self, t, group_lr_max, group_lr_min):
        if self.total_updates == 1:
            progress = 0.0
        else:
            progress = t / (self.total_updates - 1)
        envelope = 1.0 / (1.0 + self.alpha * progress)
        oscillation = (1.0 + self.d * math.cos(2.0 * math.pi * t / self.period)) / (1.0 + self.d)
        return group_lr_min + (group_lr_max - group_lr_min) * envelope * oscillation

    def _apply_lr(self, t):
        for group, g_max, g_min in zip(self.optimizer.param_groups, self.group_lr_max,
                                       self.group_lr_min):
            group['lr'] = self._lr_for_group(t, g_max, g_min)

    def step(self):
        """Advance by one completed optimizer update."""
        self.last_update += 1
        if self.last_update >= self.total_updates:
            # Training should not exceed the planned budget; if it does, hold
            # the last planned value rather than extrapolating the formula.
            self.last_update = self.total_updates - 1
        self._apply_lr(self.last_update)

    def get_last_lr(self):
        return [self._lr_for_group(self.last_update, g_max, g_min)
                for g_max, g_min in zip(self.group_lr_max, self.group_lr_min)]

    def state_dict(self):
        return {
            'total_updates': self.total_updates,
            'period': self.period,
            'alpha': self.alpha,
            'd': self.d,
            'lr_max': self.lr_max,
            'lr_min': self.lr_min,
            'group_lr_max': list(self.group_lr_max),
            'group_lr_min': list(self.group_lr_min),
            'last_update': self.last_update,
        }

    def load_state_dict(self, state):
        self.total_updates = state['total_updates']
        self.period = state['period']
        self.alpha = state['alpha']
        self.d = state['d']
        self.lr_max = state['lr_max']
        self.lr_min = state['lr_min']
        self.group_lr_max = list(state['group_lr_max'])
        self.group_lr_min = list(state['group_lr_min'])
        self.last_update = state['last_update']
        self._apply_lr(self.last_update)
