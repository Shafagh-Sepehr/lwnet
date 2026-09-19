import sys, json, os, argparse, copy, random, shutil
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import tqdm
import numpy as np
import torch
from models.get_model import get_arch, get_arch_options, validate_cross_stage_bridge

from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate, ewma
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds
from utils.checkpoint_selection import parse_metric_policy, compare_checkpoint

from segmentation_losses import build_segmentation_loss

from torch.optim.lr_scheduler import CosineAnnealingLR

# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--csv_train', type=str, default='data/DRIVE/train.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='wnet', help='architecture')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--grad_acc_steps', type=int, default=0, help='gradient accumulation steps (0)')
parser.add_argument('--min_lr', type=float, default=1e-8, help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cycle_lens', type=str, default='20/50', help='cycling config (nr cycles/cycle len')
parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
parser.add_argument('--checkpoint_interval', choices=['cycle', 'epoch'], default='cycle', help='when to assess and compare checkpoints')
parser.add_argument('--metric_tolerances', type=str, default=None, help='optional comma-separated metric=value absolute tolerances')
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
# these three are for training with pseudo-segmentations
# e.g. --csv_test data/DRIVE/test.csv --path_test_preds results/DRIVE/experiments/wnet_drive
# e.g. --csv_test data/LES_AV/test_all.csv --path_test_preds results/LES_AV/experiments/wnet_drive
parser.add_argument('--csv_test', type=str, default=None, help='path to test data csv (for using pseudo labels)')
parser.add_argument('--path_test_preds', type=str, default=None, help='path to test predictions (for using pseudo labels)')
parser.add_argument('--checkpoint_folder', type=str, default=None, help='path to model to start training (with pseudo labels now)')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--seed', type=int, default=0, help='seed')
# FreeSDG-style Frequency-Mixed Augmentation (FMAug); see utils/freesdg_aug.py
parser.add_argument('--freesdg', action='store_true', help='enable FreeSDG FMAug training-time augmentation (dataset-side)')
parser.add_argument('--freesdg_raw_prob', type=float, default=0.0, help='probability of feeding the raw image instead of the FMAug view during training')
parser.add_argument('--freesdg_test_input', type=str, default='raw', choices=['raw', 'anchor'], help='validation/inference input policy')
parser.add_argument('--freesdg_anchor_w', type=int, default=27, help='anchor HFC Gaussian kernel width')
parser.add_argument('--freesdg_anchor_sigma', type=float, default=9, help='anchor HFC Gaussian sigma')
parser.add_argument('--freesdg_ratio', type=float, default=4.0, help='HFC residual amplification ratio (bank and anchor)')
parser.add_argument('--freesdg_mixup_size', type=int, default=-1, help='FMAug rectangle size: -1 random (repo policy) or >0 fixed square; 0 is invalid')
parser.add_argument('--freesdg_mix_policy', type=str, default='repo', choices=['repo', 'paper'], help='FMAug rectangle sampling policy')
parser.add_argument('--freesdg_seed', type=int, default=0, help='dedicated seed for the isolated FMAug RNG stream')
# Diagnostic-stage selectors (defaults preserve existing behavior)
parser.add_argument('--freesdg_aug_mode', type=str, default='fmaug', choices=['fmaug', 'fixed_hfc', 'random_hfc', 'raffe_filter', 'raffe_smooth_mix'], help='training augmentation mode: full FreeSDG FMAug, deterministic fixed anchor HFC, single random bank HFC view, official RaffeSDG random frequency filtering, or RaffeSDG smooth blending')
parser.add_argument('--freesdg_lwnet_aug_profile', type=str, default='original', choices=['original', 'flips_only'], help='LwNet augmentation applied after the frequency transform: original pipeline or flips only (diagnostic)')
# Composable segmentation losses (independent of FreeSDG/augmentation):
# total = w_bce*bce + w_dice*dice + w_cldice*cldice + w_boundary*boundary
parser.add_argument('--loss_bce_weight', type=float, default=1.0, help='weight of the BCE term (the existing baseline term)')
parser.add_argument('--loss_dice_weight', type=float, default=0.0, help='weight of the foreground soft Dice term')
parser.add_argument('--loss_cldice_weight', type=float, default=0.0, help='weight of the foreground soft-clDice topology term')
parser.add_argument('--loss_boundary_weight', type=float, default=0.0, help='weight of the signed-distance boundary term (distances in pixels: resolution/scale dependent, pilot at 0.01)')
parser.add_argument('--loss_cldice_iters', type=int, default=10, help='soft-skeleton erosion iterations for the clDice term')
# Zero-initialized gated cross-stage decoder bridges (A1)
parser.add_argument('--cross_stage_bridge', choices=['none', 'scalar', 'channel'], default='none',
                    help='U1-decoder to U2-decoder gated residual bridge type')
parser.add_argument('--cross_stage_bridge_scales', type=str, default='all',
                    help='all or a comma-separated subset of quarter,half,full')
parser.add_argument('--cross_stage_bridge_init', type=float, default=0.0,
                    help='initial raw bridge gate value; A1 default is exactly 0.0')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'tr_auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def format_loss_components(split, comps, criterion):
    # One line of unweighted values, CLI weights, and weighted contributions;
    # disabled terms are simply absent from comps.
    weights = getattr(criterion, 'weights', {})
    parts = []
    for name, value in comps.items():
        w = weights.get(name)
        if w is None:
            parts.append('{:s}={:.4f}'.format(name, value))
        else:
            parts.append('{:s}={:.4f} (w={:g} -> {:.4f})'.format(name, value, w, w * value))
    return '{} loss components: {}'.format(split, ' | '.join(parts))


def format_metric_transition(previous, candidate):
    """Format two metrics precisely enough to show where they diverge."""
    previous_text = '{:.16f}'.format(float(previous))
    candidate_text = '{:.16f}'.format(float(candidate))
    previous_decimals = previous_text.split('.')[1]
    candidate_decimals = candidate_text.split('.')[1]
    shared = 0
    for left, right in zip(previous_decimals, candidate_decimals):
        if left != right:
            break
        shared += 1
    precision = min(shared + 2, 16)
    template = '{:.' + str(precision) + 'f}'
    return template.format(float(previous)), template.format(float(candidate))

def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
        grad_acc_steps=0, assess=False):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    # if train:
    #     model.train()
    # else:
    #     model.eval()

    model.train() if train else model.eval()

    if assess: logits_all, labels_all = [], []
    n_elems, running_loss, tr_lr = 0, 0, 0
    # Detached component sums kept as device tensors; they are transferred to
    # Python floats once per epoch (no per-batch GPU synchronization).
    comp_running = {}

    for i_batch, batch in enumerate(loader):
        inputs, labels = batch[0], batch[1]
        # Optional distance maps collated with the labels (boundary loss only).
        distance_maps = batch[2] if len(batch) > 2 else None
        inputs, labels = inputs.to(device), labels.to(device)
        if distance_maps is not None:
            distance_maps = distance_maps.to(device)
        logits = model(inputs)
        components = None
        if isinstance(logits, tuple): # wnet
            logits_aux, logits = logits
            if model.n_classes == 1: # SegmentationLoss (same criterion on each head)
                tgt = labels.unsqueeze(dim=1).float()
                loss_aux, comps_aux = criterion(logits_aux, tgt, distance_map=distance_maps)
                loss, comps_main = criterion(logits, tgt, distance_map=distance_maps)
                loss = loss_aux + loss
                # aggregate reported components with the same (unit) head weights
                # so their weighted sum matches the total
                components = {k: comps_aux[k] + comps_main[k] for k in comps_main}
            else: # CrossEntropyLoss()
                loss_aux = criterion(logits_aux, labels)
                loss = loss_aux + criterion(logits, labels)
        else: # not wnet
            if model.n_classes == 1:
                loss, components = criterion(logits, labels.unsqueeze(dim=1).float(), distance_map=distance_maps)  # SegmentationLoss
            else:
                loss = criterion(logits, labels)  # CrossEntropyLoss()

        # if train:  # only in training mode
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()

        if train:  # only in training mode
            (loss / (grad_acc_steps + 1)).backward() # for grad_acc_steps=0, this is just loss
            tr_lr = get_lr(optimizer)
            if i_batch % (grad_acc_steps+1) == 0:  # for grad_acc_steps=0, this is always True
                optimizer.step()
                for _ in range(grad_acc_steps+1):
                    scheduler.step() # for grad_acc_steps=0, this means once
                optimizer.zero_grad()
        if assess:
            logits_all.extend(logits.detach())
            labels_all.extend(labels.detach())

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems
        if components is not None:
            for k, v in components.items():
                comp_running[k] = comp_running.get(k, 0.0) + v * inputs.size(0)

    comp_means = None
    if comp_running:
        comp_means = {k: (v / n_elems).item() for k, v in comp_running.items()}

    if assess: return logits_all, labels_all, run_loss, tr_lr, comp_means
    return None, None, run_loss, tr_lr, comp_means

def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, grad_acc_steps=0,
                    cycle=0, checkpoint_interval='cycle', epoch_callback=None):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    deferred_assessment = None

    with tqdm(range(cycle_len)) as t:
        for epoch in t:
            is_cycle_end = epoch == cycle_len - 1
            assess = checkpoint_interval == 'epoch' or is_cycle_end
            tr_logits, tr_labels, tr_loss, tr_lr, tr_comps = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                          scheduler=scheduler, grad_acc_steps=grad_acc_steps, assess=assess)
            t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(tr_loss), tr_lr))
            if assess and epoch_callback is not None:
                assessment = (tr_logits, tr_labels, tr_loss, tr_comps, cycle + 1, epoch + 1, is_cycle_end)
                if checkpoint_interval == 'cycle':
                    deferred_assessment = assessment
                else:
                    epoch_callback(*assessment, t.write)

    if deferred_assessment is not None and epoch_callback is not None:
        epoch_callback(*deferred_assessment, print)

    return tr_logits, tr_labels, tr_loss, tr_comps

def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, exp_path,
                checkpoint_interval='cycle', metric_tolerances=None, do_not_save=False):

    n_cycles = len(scheduler.cycle_lens)
    policy = parse_metric_policy(metric, metric_tolerances)
    order = policy['metric_order']
    tolerances = policy['metric_tolerances']
    incumbent = None
    selection_stats = None
    max_validation_auc_seen = None
    global_epoch = 0
    completed_epochs = 0
    history_path = osp.join(exp_path, 'checkpoint_history.jsonl') if exp_path and not do_not_save else None

    def rng_state():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    def restore_rng(state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])

    def assess_and_select(tr_logits, tr_labels, tr_loss, tr_comps, cycle, epoch_in_cycle,
                          is_cycle_end, progress_write=print):
        nonlocal incumbent, selection_stats, max_validation_auc_seen, global_epoch
        tr_auc, tr_dice = evaluate(tr_logits, tr_labels, model.n_classes)
        del tr_logits, tr_labels
        state = None if is_cycle_end else rng_state()
        was_training = model.training
        try:
            with torch.no_grad():
                vl_logits, vl_labels, vl_loss, _, vl_comps = run_one_epoch(val_loader, model, criterion, assess=True)
                vl_auc, vl_dice = evaluate(vl_logits, vl_labels, model.n_classes)
                del vl_logits, vl_labels
        finally:
            model.train(was_training)
            if state is not None:
                restore_rng(state)
        current = {'auc': float(vl_auc), 'dice': float(vl_dice), 'loss': float(vl_loss), 'tr_auc': float(tr_auc)}
        max_validation_auc_seen = current['auc'] if max_validation_auc_seen is None else max(max_validation_auc_seen, current['auc'])
        selected, reason, details = compare_checkpoint(current, incumbent, order, tolerances)
        gate_summary = None
        if hasattr(model, 'bridge_gate_summary'):
            gate_summary = model.bridge_gate_summary()
        record = {'cycle': cycle, 'epoch_in_cycle': epoch_in_cycle, 'global_epoch': global_epoch,
                  'metrics': current, 'incumbent_metrics': incumbent, 'comparison_reason': reason,
                  'selected': selected, 'saved': False}
        if gate_summary:
            record['bridge_gates'] = gate_summary
        if selected:
            if checkpoint_interval == 'epoch':
                progress_write('------------------------- Epoch {} -------------------------'.format(global_epoch))
            if gate_summary:
                progress_write('Bridge gates: {}'.format(gate_summary))
            progress_write('Train/Val Loss: {:.4f}/{:.4f} -- Train/Val AUC: {:.4f}/{:.4f} -- Train/Val DICE: {:.4f}/{:.4f} -- LR={:.6f}'.format(
                tr_loss, vl_loss, tr_auc, vl_auc, tr_dice, vl_dice, get_lr(optimizer)).rstrip('0'))
            if tr_comps: progress_write(format_loss_components('Train', tr_comps, criterion))
            if vl_comps: progress_write(format_loss_components('Val', vl_comps, criterion))
            previous = incumbent
            incumbent = current.copy()
            selection_stats = {'selection_policy_version': 1, 'checkpoint_interval': checkpoint_interval,
                'metric_order': order, 'metric_tolerances': tolerances, 'selected_metrics': incumbent,
                'best_cycle': cycle, 'best_epoch_in_cycle': epoch_in_cycle, 'best_global_epoch': global_epoch,
                'selection_reason': reason, 'max_validation_auc_seen': max_validation_auc_seen}
            if gate_summary:
                selection_stats['bridge_gates'] = gate_summary
            if previous is None:
                progress_write('Best checkpoint initialized: {}={:.8f} (cycle {}, epoch {}, global epoch {})'.format(
                    reason if reason in current else order[0], current[reason] if reason in current else current[order[0]],
                    cycle, epoch_in_cycle, global_epoch))
            else:
                previous_metric, candidate_metric = format_metric_transition(
                    previous.get(reason, float('nan')), current.get(reason, float('nan')))
                progress_write('Best {} attained: incumbent={} -> candidate={} (cycle {}, epoch {}, global epoch {})'.format(
                    reason, previous_metric, candidate_metric,
                    cycle, epoch_in_cycle, global_epoch))
            if exp_path and not do_not_save:
                progress_write('-------------------------  Checkpointing  -------------------------')
                save_model(exp_path, model, optimizer, stats=selection_stats)
                record['saved'] = True
        if history_path:
            with open(history_path, 'a') as history:
                history.write(json.dumps(record) + '\n')
        return current

    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle+1, n_cycles))
        cycle_start = completed_epochs
        def callback(*args):
            nonlocal global_epoch
            global_epoch = cycle_start + args[5]
            return assess_and_select(*args)
        train_one_cycle(train_loader, model, criterion, optimizer, scheduler, grad_acc_steps, cycle,
                        checkpoint_interval=checkpoint_interval, epoch_callback=callback)
        completed_epochs += scheduler.cycle_lens[cycle]
        print('-' * shutil.get_terminal_size(fallback=(80, 24)).columns)

    del model
    torch.cuda.empty_cache()
    if selection_stats is None:
        return {'val_auc': None, 'val_dice': None, 'val_loss': None, 'best_cycle': 0,
                'best_epoch_in_cycle': 0, 'best_global_epoch': 0, 'selected_metrics': None}
    return {'val_auc': selection_stats['selected_metrics'].get('auc'),
            'val_dice': selection_stats['selected_metrics'].get('dice'),
            'val_loss': selection_stats['selected_metrics'].get('loss'),
            'best_cycle': selection_stats['best_cycle'], 'best_epoch_in_cycle': selection_stats['best_epoch_in_cycle'],
            'best_global_epoch': selection_stats['best_global_epoch'], 'selected_metrics': selection_stats['selected_metrics']}

if __name__ == '__main__':
    '''
    Example:
    python train_cyclical.py --csv_train data/DRIVE/train.csv --save_path unet_DRIVE
    '''

    args = parser.parse_args()

    try:
        selection_policy = parse_metric_policy(args.metric, args.metric_tolerances)
    except ValueError as e:
        parser.error(str(e))
    args.metric = selection_policy['metric']
    args.metric_order = selection_policy['metric_order']
    args.resolved_metric_tolerances = selection_policy['metric_tolerances']

    # FreeSDG FMAug argument validation (plan §11)
    if args.freesdg:
        if not (0.0 <= args.freesdg_raw_prob <= 1.0):
            sys.exit('--freesdg_raw_prob must be within [0, 1]')
        if args.freesdg_mixup_size == 0 or args.freesdg_mixup_size < -1:
            sys.exit('--freesdg_mixup_size must be -1 (random) or > 0 (fixed square); 0 is not a valid mode')

    # Cross-stage bridge validation/canonicalization (A1): canonical values are
    # what get serialized into config.cfg and passed to get_arch().
    try:
        args.cross_stage_bridge, args.cross_stage_bridge_scales, args.cross_stage_bridge_init = \
            validate_cross_stage_bridge(
                args.cross_stage_bridge, args.cross_stage_bridge_scales,
                args.cross_stage_bridge_init, args.model_name)
    except ValueError as e:
        parser.error(str(e))

    im_size_tmp = tuple([int(item) for item in args.im_size.split(',')])
    tg_size_tmp = (im_size_tmp[0], im_size_tmp[0]) if len(im_size_tmp) == 1 else tuple(im_size_tmp[:2])
    if args.freesdg and args.freesdg_mix_policy == 'paper' and tg_size_tmp != (512, 512):
        sys.exit('--freesdg_mix_policy paper requires 512x512 input (--im_size 512)')

    if args.freesdg:
        freesdg_cfg = {
            'enabled': True,
            'raw_prob': args.freesdg_raw_prob,
            'test_input': args.freesdg_test_input,
            'anchor_w': args.freesdg_anchor_w,
            'anchor_sigma': args.freesdg_anchor_sigma,
            'ratio': args.freesdg_ratio,
            'mixup_size': args.freesdg_mixup_size,
            'mix_policy': args.freesdg_mix_policy,
            'seed': args.freesdg_seed,
            'aug_mode': args.freesdg_aug_mode,
            'lwnet_aug_profile': args.freesdg_lwnet_aug_profile,
            'device': args.device,
        }
    else:
        freesdg_cfg = None

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print('* Training on device {}'.format(args.device))
        device = torch.device(args.device)
    else:  #cpu
        device = torch.device(args.device)

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, args.device.startswith("cuda"))

    # gather parser parameters
    model_name = args.model_name
    max_lr, min_lr, bs, grad_acc_steps = args.max_lr, args.min_lr, args.batch_size, args.grad_acc_steps
    cycle_lens, metric = args.cycle_lens.split('/'), selection_policy['metric']
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens)==2: # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0]*[cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path=None

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    # training for artery-vein segmentation
    if 'av' in csv_train:
        n_classes=4
        label_values=[0, 85, 170, 255]
    else:
        n_classes=1
        label_values = [0, 255]

    # Composable segmentation criterion, built once through the generic builder,
    # outside all FreeSDG/Raffe conditionals: vanilla LwNet and every
    # augmentation variant use exactly the same criterion. Validation runs once
    # here so misconfiguration fails before anything is created.
    if n_classes == 1:
        try:
            criterion = build_segmentation_loss(args)
        except ValueError as e:
            sys.exit('Invalid loss configuration: {}'.format(e))
    else: # artery-vein segmentation keeps its original criterion untouched
        criterion = torch.nn.CrossEntropyLoss()
    # Generic boundary-loss requirement for the target-preparation path:
    # derived only from loss_boundary_weight > 0 (never from --freesdg).
    need_distance_map = bool(getattr(criterion, 'need_distance_map', False))

    print('* Instantiating loss function', str(criterion))
    if hasattr(criterion, 'formula'):
        print('* Loss formula:', criterion.formula)


    print("* Creating Dataloaders, batch size = {}, workers = {}".format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs, tg_size=tg_size, label_values=label_values, num_workers=args.num_workers, freesdg_cfg=freesdg_cfg, need_distance_map=need_distance_map)

    # grad_acc_steps: if I want to train with a fake_bs=K but the actual bs I want is bs=N, then you use
    # grad_acc_steps = N/K - 1.
    # Example: bs=4, fake_bs=4 -> grad_acc_steps = 0 (default)
    # Example: bs=4, fake_bs=2 -> grad_acc_steps = 1
    # Example: bs=4, fake_bs=1 -> grad_acc_steps = 3


    print('* Instantiating a {} model'.format(model_name))
    arch_opts = get_arch_options(args)
    model = get_arch(model_name, in_c=args.in_c, n_classes=n_classes, **arch_opts)
    model = model.to(device)

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Architecture summary: cross_stage_bridge={} scales={} init={} trainable_params={}'.format(
        args.cross_stage_bridge, args.cross_stage_bridge_scales, args.cross_stage_bridge_init,
        sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    ### TRAINING WITH PSEUDO-LABELS
    csv_test = args.csv_test
    path_test_preds = args.path_test_preds
    checkpoint_folder = args.checkpoint_folder
    if csv_test is not None:
        print('Training with pseudo-labels, completing training set with predictions on test set')
        from utils.get_loaders import build_pseudo_dataset
        tr_im_list, tr_gt_list, tr_mask_list = build_pseudo_dataset(csv_train, csv_test, path_test_preds)
        train_loader.dataset.im_list = tr_im_list
        train_loader.dataset.gt_list = tr_gt_list
        train_loader.dataset.mask_list = tr_mask_list
        print('* Loading weights from previous checkpoint={}'.format(checkpoint_folder))
        model, stats, optimizer_state_dict = load_model(model, checkpoint_folder, device=device, with_opt=True)
        optimizer.load_state_dict(optimizer_state_dict)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = max_lr
            param_group['initial_lr'] = max_lr


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_lens[0] * len(train_loader), eta_min=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=0)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)


    print('* Starting to train\n','-' * 10)


    result = train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps,
                         metric, experiment_path, checkpoint_interval=args.checkpoint_interval,
                         metric_tolerances=args.metric_tolerances, do_not_save=do_not_save)

    print("val_auc: %s" % result['val_auc'])
    print("val_dice: %s" % result['val_dice'])
    print("val_loss: %s" % result['val_loss'])
    print("best_cycle: %d" % result['best_cycle'])
    print("best_epoch_in_cycle: %d" % result['best_epoch_in_cycle'])
    print("best_global_epoch: %d" % result['best_global_epoch'])
    if do_not_save is False:
        if result['selected_metrics'] is None:
            print('No valid checkpoint was selected; validation metrics were non-finite.')
            sys.exit(1)
        # file = open(osp.join(experiment_path, 'val_metrics.txt'), 'w')
        # file.write(str(m1)+ '\n')
        # file.write(str(m2)+ '\n')
        # file.write(str(m3)+ '\n')
        # file.close()

        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best AUC = {:.2f}\nBest DICE = {:.2f}\nBest loss = {:.8f}\nBest cycle = {}\nBest epoch in cycle = {}\nBest global epoch = {}'.format(
                100 * result['val_auc'], 100 * result['val_dice'], result['val_loss'], result['best_cycle'],
                result['best_epoch_in_cycle'], result['best_global_epoch']), file=f)
