"""
Optional damped-cosine learning-rate schedule for LwNet.

lr(t) = lr_min + (lr_max - lr_min)
        * [1 / (1 + alpha * t / (T - 1))]
        * [(1 + d * cos(2*pi*t/P)) / (1 + d)]

where
    t      : zero-based optimizer-update index, 0 .. T-1
    T      : planned total optimizer updates (fixed before training)
    P      : oscillation period in optimizer updates
    alpha  : envelope decay strength; must be > -1
             * alpha > 0: envelope shrinks over training (damping)
             * alpha = 0: constant envelope (no decay)
             * -1 < alpha < 0: envelope GROWS over training (inflation):
               lr peaks exceed lr_max, up to lr_max/(1 + alpha) at the end.
               Here lr_max is the *initial* peak, not a global upper bound.
             (alpha <= -1 is invalid: the envelope divides by zero at t = T-1)
    d      : oscillation depth in [0, 1]
    lr_max : learning rate at t=0 (upper bound for alpha >= 0)
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


def validate_damped_cosine_config(total_updates, period, alpha, d, lr_max, lr_min,
                                  inflate_max_lr=None):
    """Validate damped-cosine settings; raise DampedCosineError with a clear message."""
    if not isinstance(total_updates, int) or total_updates < 1:
        raise DampedCosineError(
            f"total planned optimizer updates must be an integer >= 1, got {total_updates!r}")
    if not isinstance(period, int) or period < 1:
        raise DampedCosineError(
            f"oscillation period must be an integer >= 1 optimizer update, got {period!r}")
    if alpha <= -1:
        raise DampedCosineError(
            f"alpha must be > -1, got {alpha}: alpha <= -1 makes the envelope "
            f"divide by zero at t = T-1 (and grow unboundedly before that); "
            f"use -1 < alpha < 0 for an inflating envelope")
    if not (0.0 <= d <= 1.0):
        raise DampedCosineError(f"d (oscillation depth) must be in [0, 1], got {d}")
    if lr_min < 0:
        raise DampedCosineError(f"lr_min must be >= 0, got {lr_min}")
    if lr_max <= lr_min:
        raise DampedCosineError(
            f"lr_max must be > lr_min, got lr_max={lr_max}, lr_min={lr_min}")
    if inflate_max_lr is not None:
        # lr(0) == lr_max already, so a cap at or below lr_max is contradictory
        if inflate_max_lr <= lr_max:
            raise DampedCosineError(
                f"dc_inflate_max_lr must be > lr_max (lr(0) alone starts at lr_max={lr_max}), "
                f"got {inflate_max_lr}")


def _envelope(t, total_updates, alpha):
    if total_updates == 1:
        progress = 0.0  # single-update schedule: avoid division by zero
    else:
        progress = t / (total_updates - 1)
    return 1.0 / (1.0 + alpha * progress)


def damped_cosine_lr(t, total_updates, period, alpha, d, lr_max, lr_min,
                     inflate_max_lr=None):
    """Closed-form damped-cosine learning rate at zero-based update index t.

    With inflate_max_lr set, the envelope is clamped so the lr never exceeds
    inflate_max_lr (relevant for alpha < 0, where it would otherwise grow to
    lr_max / (1 + alpha) at the end).
    """
    envelope = _envelope(t, total_updates, alpha)
    if inflate_max_lr is not None:
        cap = (inflate_max_lr - lr_min) / (lr_max - lr_min)
        if envelope > cap:
            envelope = cap
    oscillation = (1.0 + d * math.cos(2.0 * math.pi * t / period)) / (1.0 + d)
    return lr_min + (lr_max - lr_min) * envelope * oscillation


def damped_cosine_max_lr(t, total_updates, alpha, lr_max, lr_min, inflate_max_lr=None):
    """Peak lr reachable at update index t (envelope ceiling; no oscillation).

    This is the local maximum of the oscillation around t: with alpha > 0 it
    decays over training, with -1 < alpha < 0 it grows (capped at
    inflate_max_lr when given).
    """
    envelope = _envelope(t, total_updates, alpha)
    if inflate_max_lr is not None:
        cap = (inflate_max_lr - lr_min) / (lr_max - lr_min)
        if envelope > cap:
            envelope = cap
    return lr_min + (lr_max - lr_min) * envelope


def damped_cosine_schedule_max(total_updates, period, alpha, d, lr_max, lr_min,
                               inflate_max_lr=None):
    """Exact global maximum of the DISCRETE schedule and where it happens.

    Scans the lr actually used at every optimizer update t in [0, T-1]
    (including the inflation cap). Returns (max_lr, argmax_t, n_ties) where
    argmax_t is the first update attaining the max and n_ties counts updates
    reaching the same value (e.g. repeated flat peaks once a cap binds).
    """
    best_t, best_lr, ties = 0, None, 0
    for t in range(total_updates):
        lr = damped_cosine_lr(t, total_updates, period, alpha, d, lr_max, lr_min,
                              inflate_max_lr=inflate_max_lr)
        if best_lr is None or lr > best_lr:
            best_t, best_lr, ties = t, lr, 1
        elif lr == best_lr:
            ties += 1
    return best_lr, best_t, ties


class DampedCosineLRSchedule:
    """Sets each optimizer update t to lr(t) of the damped-cosine formula.

    Step accounting: construction writes lr(0) into the optimizer (update 0
    then runs at lr(0) = lr_max); every scheduler.step() call right after an
    optimizer.step() advances the internal counter, so update t always uses
    lr(t). Exactly one step() per optimizer update -- no extra steps for
    gradient accumulation.
    """

    def __init__(self, optimizer, total_updates, period, alpha, d, lr_max, lr_min,
                 inflate_max_lr=None):
        validate_damped_cosine_config(total_updates, period, alpha, d, lr_max, lr_min,
                                      inflate_max_lr=inflate_max_lr)
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.period = period
        self.alpha = float(alpha)
        self.d = float(d)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.inflate_max_lr = None if inflate_max_lr is None else float(inflate_max_lr)
        # Per-parameter-group bounds: keep each group's intended lr ratio
        # (group_lr_min / group_lr_max == lr_min / lr_max for every group).
        self.group_lr_max = [float(group['lr']) for group in optimizer.param_groups]
        self.group_lr_min = [lr * self.lr_min / self.lr_max for lr in self.group_lr_max]
        self.last_update = 0  # zero-based index of the NEXT optimizer update
        self._apply_lr(self.last_update)

    def _lr_for_group(self, t, group_lr_max, group_lr_min):
        envelope = _envelope(t, self.total_updates, self.alpha)
        if self.inflate_max_lr is not None:
            # group-agnostic cap on the envelope multiplier keeps group ratios
            cap = (self.inflate_max_lr - self.lr_min) / (self.lr_max - self.lr_min)
            if envelope > cap:
                envelope = cap
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
            'inflate_max_lr': self.inflate_max_lr,
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
        self.inflate_max_lr = state.get('inflate_max_lr')
        self.group_lr_max = list(state['group_lr_max'])
        self.group_lr_min = list(state['group_lr_min'])
        self.last_update = state['last_update']
        self._apply_lr(self.last_update)
