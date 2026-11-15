r"""Strict-reconstruction check for saved experiment checkpoints.

Read-only: nothing is written.  For each experiment directory this reads
``config.cfg``, resolves the loss from its serialized ``loss_*`` values,
rebuilds the model with ``get_arch_from_config``, and loads
``model_checkpoint.pth`` with ``strict=True`` -- proving that adding the
loss flags to the config does not disturb checkpoint reconstruction.

Usage:
    python unit-test/check_reference_checkpoint.py <experiment_dir> [<experiment_dir> ...]

Example (paths outside this checkout are passed explicitly):
    python unit-test/check_reference_checkpoint.py ^
        ..\lwnet\experiments\final_t4_src_ss_raffeSmooth_scalar_sal_bce_d025_bnd001_c20x50_ck20_seed0
"""

import argparse
import json
import os
import sys
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.get_model import get_arch_from_config  # noqa: E402
from segmentation_losses import build_segmentation_loss  # noqa: E402


def check(run_dir, forward_size=64):
    config_path = os.path.join(run_dir, 'config.cfg')
    checkpoint_path = os.path.join(run_dir, 'model_checkpoint.pth')
    with open(config_path) as handle:
        cfg = json.load(handle)

    criterion = build_segmentation_loss(types.SimpleNamespace(**cfg))
    model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, forward_size, forward_size))

    print(os.path.basename(os.path.normpath(run_dir)))
    print('  loss weights:', criterion.weights, 'iters:', criterion.cldice_iters)
    print('  formula:', criterion.formula)
    print('  loss parameters:', len(list(criterion.parameters())),
          'loss state entries:', len(criterion.state_dict()))
    print('  checkpoint keys:', len(checkpoint['model_state_dict']),
          'model keys:', len(model.state_dict()))
    print('  total params:', sum(p.numel() for p in model.parameters()),
          'trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('  eval forward shape:', tuple(output.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('experiment_dirs', nargs='+',
                        help='saved experiment directories containing config.cfg '
                             'and model_checkpoint.pth')
    parser.add_argument('--forward_size', type=int, default=64,
                        help='square spatial size for the eval-mode forward check')
    args = parser.parse_args()
    for run_dir in args.experiment_dirs:
        check(run_dir, forward_size=args.forward_size)


if __name__ == '__main__':
    main()
