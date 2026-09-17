"""Independent, composable segmentation losses for LwNet vessel segmentation.

Implements the fixed-weight objective

    total = w_bce * bce + w_dice * soft_dice + w_cldice * soft_cldice
            + w_boundary * boundary

The module is deliberately self-contained: it knows nothing about FreeSDG,
Raffe, HFC, augmentation modes, image representations, or the training
script.  It receives binary segmentation logits, labels, and optional
precomputed ground-truth signed-distance maps; nothing else.  Vanilla LwNet
and every augmentation variant use exactly the same criterion.

Conventions (inspection of this checkout)
-----------------------------------------
* Model heads emit logits (final 1x1 convs, no activation); ``sigmoid`` is
  computed here exactly once when any probability-based term is enabled.
  BCE always receives raw logits, preserving the original operation and
  dtype behavior of the replaced ``BCEWithLogitsLoss()``.
* Shapes are ``[B, 1, H, W]``; ``[B, H, W]`` labels are adapted once at
  this boundary.  Accidental channel broadcasting and spatial mismatches
  are rejected.
* The existing pipeline supervises the full rectangle: the dataset zeroes
  labels outside the retinal FOV (that exterior is trained as background)
  and the original BCE used a plain mean over all pixels.  No spatial
  validity mask exists anywhere in the loss path, so none is introduced
  here.  Note the documented artificial edge this creates: vessels cut by
  the FOV boundary terminate at an artificial background edge which
  participates in skeletons and distance maps like any other boundary.
* A fixed epsilon of ``1e-6`` is used in all soft terms.
* Soft-term math runs in float32; prediction gradients stay connected.
* The geometric terms require binary labels.  Fractional targets (the
  pseudo-label mode built by ``--csv_test``/``--path_test_preds``) are
  rejected with an explicit error rather than silently thresholded.

References / attribution
------------------------
* soft-clDice topology term and the soft-skeleton construction are adapted
  from the official clDice implementation: https://github.com/jocpae/clDice
  (Cloux et al., "clDice - a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation").
* The signed-distance boundary term follows the Kervadec-style
  probability-times-distance objective and the discrete distance-map
  convention of https://github.com/LIVIAETS/boundary-loss (utils.py).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Fixed, documented smoothing epsilon for all soft terms (plan: no CLI flag).
EPS = 1e-6

#: Tolerance separating genuine binary labels {0, 1} from fractional
#: (pseudo-label) targets.  Labels are exactly 0.0/1.0 on the GT path;
#: pseudo-labels land anywhere in [0, 1], so 1e-3 separates them robustly.
_BINARY_TOL = 1e-3

FRACTIONAL_LABEL_ERROR = (
    'Geometric loss terms (soft Dice / soft-clDice / boundary) require binary '
    'targets in {0, 1}, but fractional labels were found. This happens in the '
    'pseudo-label training mode (--csv_test / --path_test_preds), whose '
    'supervision mixes continuous predictions into the vessel label. That '
    'mode is rejected for geometric losses until its supervision is '
    'explicitly defined; train it with BCE only.')

#: Resolved defaults for old configs that predate the loss flags.
LOSS_DEFAULTS = {
    'loss_bce_weight': 1.0,
    'loss_dice_weight': 0.0,
    'loss_cldice_weight': 0.0,
    'loss_boundary_weight': 0.0,
    'loss_cldice_iters': 10,
}


def check_binary_labels(target, tol=_BINARY_TOL):
    """Raise ``ValueError`` if ``target`` contains values outside {0, 1}.

    Used on the geometric-loss path only (BCE-only behavior is unchanged and
    keeps accepting whatever targets it accepted before).
    """
    bad = ((target > tol) & (target < 1.0 - tol)) | (target < 0.0) | (target > 1.0)
    if bool(bad.any()):
        raise ValueError(FRACTIONAL_LABEL_ERROR)


# ---------------------------------------------------------------------------
# Foreground soft Dice
# ---------------------------------------------------------------------------

def soft_dice_loss(probs, target, eps=EPS):
    """Foreground soft Dice, reduced per image then averaged over the batch.

    ``dice_i = 1 - (2 * sum(p_i * y_i) + eps) / (sum(p_i) + sum(y_i) + eps)``

    Linear denominator (not squared probabilities).  Empty-target samples
    stay in this term with the same formula and are averaged over the
    original batch; BCE is particularly useful for learning on such samples.
    """
    b = probs.shape[0]
    p = probs.reshape(b, -1)
    y = target.reshape(b, -1)
    inter = (p * y).sum(dim=1)
    denom = p.sum(dim=1) + y.sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


# ---------------------------------------------------------------------------
# soft-clDice topology term
# ---------------------------------------------------------------------------

def _soft_erode(img):
    return -F.max_pool2d(-img, (3, 3), (1, 1), (1, 1))


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(probs, iterations):
    """Differentiable soft skeletonization for ``[B, 1, H, W]`` probabilities.

    Iterative erosion/opening residual accumulation, following the official
    clDice ``soft_skel`` construction (https://github.com/jocpae/clDice).
    ``iterations`` erosion steps fully skeletonize structures up to roughly
    ``2 * iterations`` pixels wide; ten iterations is an initial setting, not
    a guarantee of complete skeletonization for every vessel width -- verify
    on representative resized labels before topology experiments.
    """
    if iterations < 0:
        raise ValueError('iterations must be nonnegative, got {}'.format(iterations))
    img1 = _soft_open(probs)
    skel = F.relu(probs - img1)
    for _ in range(iterations):
        probs = _soft_erode(probs)
        img1 = _soft_open(probs)
        skel = skel + F.relu(probs - img1)
    return skel


def soft_cldice_loss(probs, target, iterations=10, eps=EPS):
    """Foreground soft-clDice topology term (clDice term only, no Dice).

    With soft skeletons ``S(p)`` and ``S(y)``, per image::

        tprec = (sum(S(p) * y) + eps) / (sum(S(p)) + eps)
        tsens = (sum(S(y) * p) + eps) / (sum(S(y)) + eps)
        cldice_i = 1 - 2 * tprec * tsens / (tprec + tsens + eps)

    The target skeleton is computed without autograd.  Empty-target samples
    contribute an explicit differentiable zero and the mean divides by the
    original batch size, so BCE/Dice retain their supervision (project
    choice).  The prediction path stays fully differentiable; no hard
    thresholding, NumPy, or argmax is involved.
    """
    b = probs.shape[0]
    p = probs.reshape(b, -1)
    y = target.reshape(b, -1)
    skel_p = soft_skeleton(probs, iterations)
    with torch.no_grad():
        skel_y = soft_skeleton(target.detach(), iterations)
    sp = skel_p.reshape(b, -1).sum(dim=1)
    sy = skel_y.reshape(b, -1).sum(dim=1)
    tprec = ((skel_p.reshape(b, -1) * y).sum(dim=1) + eps) / (sp + eps)
    tsens = ((skel_y.reshape(b, -1) * p).sum(dim=1) + eps) / (sy + eps)
    cldice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + eps)
    nonempty = y.sum(dim=1) > 0
    cldice = torch.where(nonempty, cldice, torch.zeros_like(cldice))
    return cldice.mean()


# ---------------------------------------------------------------------------
# Signed-distance boundary term
# ---------------------------------------------------------------------------

def signed_distance_map(binary_target):
    """Ground-truth signed distance map (Kervadec-style discrete convention).

    ``phi = EDT(1 - y) * (1 - y) - (EDT(y) - 1) * y``

    where ``EDT`` is the Euclidean distance transform in output-grid pixels
    (SciPy ``distance_transform_edt``, ground truth only).  Distances are
    positive outside vessels, nonpositive inside, and zero on the inner
    one-pixel boundary.  Maps keep pixel units: no absolute value,
    squaring, or per-image normalization, so boundary weights depend on
    resolution and loss scale.

    Degenerate masks (all-background or all-foreground) explicitly return a
    zero distance map -- the internal class interface is absent -- and such
    samples stay in the batch mean with BCE/Dice providing supervision.

    Accepts a torch tensor or ndarray shaped ``[H, W]``, ``[B, H, W]`` or
    ``[B, 1, H, W]`` and returns a float32 torch tensor with the same
    leading shape.  Runs on CPU (called from dataset target preparation
    after all spatial augmentation has finished).
    """
    from scipy.ndimage import distance_transform_edt  # lazy: boundary prep only

    was_4d = False
    if isinstance(binary_target, torch.Tensor):
        y = binary_target.detach().cpu().numpy()
    else:
        y = np.asarray(binary_target)
    if y.ndim == 4:
        if y.shape[1] != 1:
            raise ValueError('expected channel dim 1, got shape {}'.format(y.shape))
        y = y[:, 0]
        was_4d = True
    elif y.ndim != 2 and y.ndim != 3:
        raise ValueError('expected [H, W], [B, H, W] or [B, 1, H, W], got ndim {}'.format(y.ndim))
    was_2d = y.ndim == 2
    if was_2d:
        y = y[None]

    b, h, w = y.shape
    phi = np.zeros((b, h, w), dtype=np.float32)
    for i in range(b):
        m = y[i].astype(bool)
        if m.any() and not m.all():
            outside = distance_transform_edt(~m)  # EDT(1 - y): distance to nearest vessel
            inside = distance_transform_edt(m)    # EDT(y): distance to nearest background
            phi[i] = np.where(m, 1.0 - inside, outside).astype(np.float32)

    if was_2d:
        return torch.from_numpy(phi[0])
    if was_4d:
        return torch.from_numpy(phi[:, None])
    return torch.from_numpy(phi)


def boundary_loss(probs, distance_map):
    """Kervadec-style boundary term: per-image mean of ``p * phi``.

    ``boundary_i = mean(p_i * phi_i)``; ``L_boundary = mean_i(boundary_i)``.

    A signed boundary loss can be negative; its sign is not an error.
    Degenerate-mask samples carry an all-zero map and therefore contribute
    exactly zero while remaining in the batch mean.
    """
    n_pix = probs.shape[-1] * probs.shape[-2]
    per_image = (probs * distance_map).sum(dim=(1, 2, 3)) / n_pix
    return per_image.mean()


# ---------------------------------------------------------------------------
# Weighted wrapper
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """Fixed-weight combination of the independent terms above.

    ``forward(logits, target, *, distance_map=None)`` returns
    ``(total, components)``: a scalar differentiable total and a dict of
    detached scalar components for the *enabled* terms (unweighted values;
    ``self.weights`` holds the CLI weights so callers can report weighted
    contributions).  A zero weight means the component is not evaluated,
    including its target preparation.

    BCE-only configuration preserves the replaced ``BCEWithLogitsLoss()``
    operation and dtype behavior exactly (same op, plain mean, no casting),
    so numerical equivalence is testable.
    """

    def __init__(self, bce_weight=1.0, dice_weight=0.0, cldice_weight=0.0,
                 boundary_weight=0.0, cldice_iters=10, eps=EPS, debug=False):
        super().__init__()
        weights = {'bce': float(bce_weight), 'dice': float(dice_weight),
                   'cldice': float(cldice_weight), 'boundary': float(boundary_weight)}
        for name, value in weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError('loss weight {!r} must be finite and nonnegative, got {!r}'
                                 .format(name, value))
        if sum(weights.values()) <= 0:
            raise ValueError('at least one loss weight must be positive')
        geometric = weights['cldice'] > 0 or weights['boundary'] > 0
        if geometric and weights['bce'] <= 0 and weights['dice'] <= 0:
            raise ValueError('soft-clDice and boundary terms require a positive BCE or '
                             'Dice weight: topology-only and boundary-only training are '
                             'outside the scope of this implementation')
        cldice_iters = int(cldice_iters)
        if weights['cldice'] > 0 and cldice_iters <= 0:
            raise ValueError('loss_cldice_iters must be a positive integer when clDice '
                             'is enabled, got {}'.format(cldice_iters))

        self.weights = weights
        self.cldice_iters = cldice_iters
        self.eps = eps
        #: optional nonfinite-input rejection for tests/debug validation
        self.debug = debug
        #: generic flag for the target-preparation path: boundary term only
        self.need_distance_map = weights['boundary'] > 0
        self._need_probs = any(weights[w] > 0 for w in ('dice', 'cldice', 'boundary'))

    def extra_repr(self):
        return ('bce={}, dice={}, cldice={}, boundary={}, iters={}, eps={}'
                .format(self.weights['bce'], self.weights['dice'],
                        self.weights['cldice'], self.weights['boundary'],
                        self.cldice_iters, self.eps))

    @property
    def formula(self):
        """Human-readable resolved formula printed once at startup."""
        parts = []
        parts.append('{:g}*bce'.format(self.weights['bce']))
        if self.weights['dice'] > 0:
            parts.append('{:g}*dice'.format(self.weights['dice']))
        if self.weights['cldice'] > 0:
            parts.append('{:g}*cldice(iters={})'.format(self.weights['cldice'], self.cldice_iters))
        if self.weights['boundary'] > 0:
            parts.append('{:g}*boundary'.format(self.weights['boundary']))
        return 'total = {}  (eps={:g})'.format(' + '.join(parts), self.eps)

    def forward(self, logits, target, *, distance_map=None):
        if target.dim() == 3:  # [B, H, W] -> [B, 1, H, W], adapted once here
            target = target.unsqueeze(1)
        if logits.dim() != 4 or logits.size(1) != 1:
            raise ValueError('expected logits [B, 1, H, W], got shape {}'
                             .format(tuple(logits.shape)))
        if tuple(target.shape) != tuple(logits.shape):
            raise ValueError('target shape {} does not match logits shape {}'
                             .format(tuple(target.shape), tuple(logits.shape)))
        if not torch.is_floating_point(target):
            target = target.float()
        if self.debug:
            for name, t in (('logits', logits), ('target', target),
                            ('distance_map', distance_map)):
                if t is not None and not torch.isfinite(t).all():
                    raise ValueError('nonfinite values in {}'.format(name))

        components = {}

        # BCE: exactly the op performed by the replaced BCEWithLogitsLoss()
        # (raw logits, plain mean over all elements, original dtypes).
        bce = F.binary_cross_entropy_with_logits(logits, target)
        total = self.weights['bce'] * bce
        components['bce'] = bce.detach()

        if self._need_probs:
            y32 = target.to(torch.float32)
            check_binary_labels(y32)
            p32 = torch.sigmoid(logits.to(torch.float32))
            if self.weights['dice'] > 0:
                dice = soft_dice_loss(p32, y32, self.eps)
                total = total + self.weights['dice'] * dice
                components['dice'] = dice.detach()
            if self.weights['cldice'] > 0:
                cldice = soft_cldice_loss(p32, y32, self.cldice_iters, self.eps)
                total = total + self.weights['cldice'] * cldice
                components['cldice'] = cldice.detach()
            if self.weights['boundary'] > 0:
                if distance_map is None:
                    raise ValueError('boundary term enabled (--loss_boundary_weight > 0) '
                                     'but no distance map was provided')
                if distance_map.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
                    distance_map = distance_map.unsqueeze(1)
                if tuple(distance_map.shape) != tuple(logits.shape):
                    raise ValueError('distance map shape {} does not match logits shape {}'
                                     .format(tuple(distance_map.shape), tuple(logits.shape)))
                bnd = boundary_loss(p32, distance_map.to(torch.float32))
                total = total + self.weights['boundary'] * bnd
                components['boundary'] = bnd.detach()

        return total, components


def build_segmentation_loss(args):
    """Build the composable criterion from parsed arguments.

    Reads ``--loss_*`` settings with :data:`LOSS_DEFAULTS` fallbacks so old
    configs (missing keys) resolve to the documented defaults.  Validation
    happens once, here: weights finite, nonnegative, at least one positive;
    BCE or Dice positive when a geometric term is enabled; positive integer
    iteration count when clDice is active.
    """
    cfg = {key: getattr(args, key, default) for key, default in LOSS_DEFAULTS.items()}
    return SegmentationLoss(
        bce_weight=cfg['loss_bce_weight'],
        dice_weight=cfg['loss_dice_weight'],
        cldice_weight=cfg['loss_cldice_weight'],
        boundary_weight=cfg['loss_boundary_weight'],
        cldice_iters=cfg['loss_cldice_iters'])
