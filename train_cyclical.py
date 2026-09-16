import sys, json, os, argparse, math, time
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import tqdm
import numpy as np
import torch
from models.get_model import get_arch

from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate, ewma
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds, get_rng_states, set_rng_states

from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.schedulers import (DampedCosineLRSchedule, DampedCosineError,
                              validate_damped_cosine_config)

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
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
# these three are for training with pseudo-segmentations
# e.g. --csv_test data/DRIVE/test.csv --path_test_preds results/DRIVE/experiments/wnet_drive
# e.g. --csv_test data/LES_AV/test_all.csv --path_test_preds results/DRIVE/experiments/wnet_drive
parser.add_argument('--csv_test', type=str, default=None, help='path to test data csv (for using pseudo labels)')
parser.add_argument('--path_test_preds', type=str, default=None, help='path to test predictions (for using pseudo labels)')
parser.add_argument('--checkpoint_folder', type=str, default=None, help='path to model to start training (with pseudo labels now)')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--seed', type=int, default=0, help='seed')
# learning-rate schedule selection
parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'damped_cosine'],
                    help="learning-rate schedule: 'cosine' (original CosineAnnealingLR behavior, default) "
                         "or 'damped_cosine' (decaying full-cosine oscillation)")
parser.add_argument('--dc_alpha', type=float, default=3.0, help='damped_cosine: nonnegative envelope decay strength')
parser.add_argument('--dc_d', type=float, default=0.95, help='damped_cosine: oscillation depth in [0, 1]')
parser.add_argument('--dc_period', type=int, default=0,
                    help='damped_cosine: oscillation period in optimizer updates '
                         '(0 = automatic: 2 x cycle_len x updates/epoch, matching the original scheduler oscillation)')
parser.add_argument('--resume_from', type=str, default=None,
                    help='path to a previous experiment folder whose training_state.pth should be resumed')


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

def is_damped(scheduler):
    return getattr(scheduler, 'kind', 'cosine') == 'damped_cosine'

def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
        grad_acc_steps=0, assess=False, lr_log=None):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    # if train:
    #     model.train()
    # else:
    #     model.eval()

    model.train() if train else model.eval()

    if assess: logits_all, labels_all = [], []
    n_elems, running_loss, tr_lr = 0, 0, 0


    for i_batch, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        if isinstance(logits, tuple): # wnet
            logits_aux, logits = logits
            if model.n_classes == 1: # BCEWithLogitsLoss()/DiceLoss()
                loss_aux = criterion(logits_aux, labels.unsqueeze(dim=1).float())
                loss = loss_aux + criterion(logits, labels.unsqueeze(dim=1).float())
            else: # CrossEntropyLoss()
                loss_aux = criterion(logits_aux, labels)
                loss = loss_aux + criterion(logits, labels)
        else: # not wnet
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()/DiceLoss()
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
                if lr_log is not None:
                    lr_log.append(float(tr_lr))  # lr actually used by this optimizer update
                optimizer.step()
                if is_damped(scheduler):
                    scheduler.step()  # exactly one scheduler step per optimizer update
                else:
                    # original behavior, preserved for the default scheduler
                    for _ in range(grad_acc_steps+1):
                        scheduler.step() # for grad_acc_steps=0, this means once
                optimizer.zero_grad()
        if assess:
            logits_all.extend(logits)
            labels_all.extend(labels)

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems

    if assess: return logits_all, labels_all, run_loss, tr_lr
    return None, None, run_loss, tr_lr

def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, grad_acc_steps=0, cycle=0,
                    lr_log=None, epoch_log=None, first_epoch=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    with tqdm(range(cycle_len)) as t:
        for epoch in t:
            if epoch == cycle_len-1: assess=True # only get logits/labels on last cycle
            else: assess = False
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                          scheduler=scheduler, grad_acc_steps=grad_acc_steps, assess=assess,
                                                          lr_log=lr_log)
            t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(tr_loss), tr_lr))
            if epoch_log is not None:
                epoch_log.append({'cycle': cycle+1, 'epoch_in_cycle': epoch,
                                  'global_epoch': first_epoch + epoch + 1,
                                  'train_loss': float(tr_loss), 'lr': float(tr_lr)})

    return tr_logits, tr_labels, tr_loss

def save_training_state(path, model, optimizer, scheduler, completed_cycles, completed_updates,
                        total_planned_updates, best_state, lr_history, rng_states, loader_generator_state,
                        elapsed_time):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_type': getattr(scheduler, 'kind', 'cosine'),
        'scheduler_state_dict': scheduler.state_dict(),
        'completed_cycles': completed_cycles,
        'completed_updates': completed_updates,
        'total_planned_updates': total_planned_updates,
        'best_state': best_state,
        'lr_history': lr_history,
        'rng_states': rng_states,
        'loader_generator_state': loader_generator_state,
        'elapsed_time': elapsed_time,
        }, osp.join(path, 'training_state.pth'))

def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, exp_path,
                start_cycle=0, best_state=None, lr_history=None, elapsed_time=0.0, resume_from=None):

    n_cycles = len(scheduler.cycle_lens)
    if best_state is None:
        best_state = {'best_auc': 0, 'best_dice': 0, 'best_cycle': 0, 'best_monitoring_metric': 0}
    best_auc, best_dice, best_cycle = best_state['best_auc'], best_state['best_dice'], best_state['best_cycle']
    best_monitoring_metric = best_state['best_monitoring_metric']
    is_better, _ = compare_op(metric)

    train_log = None
    if exp_path is not None:
        train_log = open(osp.join(exp_path, 'train_log.jsonl'), 'a')

    train_time_start = time.time()

    for cycle in range(start_cycle, n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle, retrieve segmentation data and compute metrics at the end of cycle
        first_epoch = sum(scheduler.cycle_lens[:cycle])
        epoch_records = []
        tr_logits, tr_labels, tr_loss = train_one_cycle(train_loader, model, criterion, optimizer, scheduler, grad_acc_steps, cycle,
                                                        lr_log=lr_history, epoch_log=epoch_records, first_epoch=first_epoch)
        if train_log is not None:
            for rec in epoch_records:
                train_log.write(json.dumps(dict(rec, type='epoch')) + '\n')

        # classification metrics at the end of cycle
        print(25 * '-' + '  End of cycle, evaluating ' + 25 * '-')
        tr_auc, tr_dice = evaluate(tr_logits, tr_labels, model.n_classes)  # for n_classes>1, will need to redo evaluate
        del tr_logits, tr_labels
        with torch.no_grad():
            assess=True
            vl_logits, vl_labels, vl_loss, _ = run_one_epoch(val_loader, model, criterion, assess=assess)
            vl_auc, vl_dice = evaluate(vl_logits, vl_labels, model.n_classes)  # for n_classes>1, will need to redo evaluate
            del vl_logits, vl_labels
        print('Train/Val Loss: {:.4f}/{:.4f}  -- Train/Val AUC: {:.4f}/{:.4f}  -- Train/Val DICE: {:.4f}/{:.4f} -- LR={:.6f}'.format(
                tr_loss, vl_loss, tr_auc, vl_auc, tr_dice, vl_dice, get_lr(optimizer)).rstrip('0'))

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'auc':
            monitoring_metric = vl_auc
        elif metric == 'tr_auc':
            monitoring_metric = tr_auc
        elif metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_dice
        checkpointed = False
        if is_better(monitoring_metric, best_monitoring_metric):
            print('Best {} attained. {:.2f} --> {:.2f}'.format(metric, 100*best_monitoring_metric, 100*monitoring_metric))
            best_auc, best_dice, best_cycle = vl_auc, vl_dice, cycle+1
            best_monitoring_metric = monitoring_metric
            if exp_path is not None:
                print(25 * '-', ' Checkpointing ', 25 * '-')
                save_model(exp_path, model, optimizer)
                checkpointed = True

        completed_updates = len(lr_history) if lr_history is not None else None
        if train_log is not None:
            train_log.write(json.dumps({'type': 'cycle_end', 'cycle': cycle+1,
                                        'completed_updates': completed_updates,
                                        'tr_loss': float(tr_loss), 'vl_loss': float(vl_loss),
                                        'tr_auc': float(tr_auc), 'vl_auc': float(vl_auc),
                                        'tr_dice': float(tr_dice), 'vl_dice': float(vl_dice),
                                        'monitoring_metric': metric,
                                        'monitoring_metric_value': float(monitoring_metric),
                                        'checkpointed': checkpointed,
                                        'lr_end_of_cycle': float(get_lr(optimizer))}) + '\n')
            train_log.flush()
        if exp_path is not None and lr_history is not None:
            with open(osp.join(exp_path, 'lr_history.json'), 'w') as f:
                json.dump(lr_history, f)
        # full state for exact resumption (model, optimizer, schedule position, RNGs)
        if exp_path is not None and lr_history is not None:
            elapsed = elapsed_time + (time.time() - train_time_start)
            generator_state = None
            if getattr(train_loader, 'generator', None) is not None:
                generator_state = train_loader.generator.get_state()
            save_training_state(exp_path, model, optimizer, scheduler,
                                completed_cycles=cycle+1, completed_updates=completed_updates,
                                total_planned_updates=getattr(scheduler, 'total_updates', None),
                                best_state={'best_auc': best_auc, 'best_dice': best_dice,
                                            'best_cycle': best_cycle,
                                            'best_monitoring_metric': best_monitoring_metric},
                                lr_history=lr_history, rng_states=get_rng_states(),
                                loader_generator_state=generator_state, elapsed_time=elapsed)

    if train_log is not None:
        train_log.close()
    total_elapsed = elapsed_time + (time.time() - train_time_start)
    if exp_path is not None and lr_history is not None:
        with open(osp.join(exp_path, 'training_time.txt'), 'w') as f:
            print('total_optimizer_updates: {}'.format(len(lr_history)), file=f)
            print('planned_total_optimizer_updates: {}'.format(getattr(scheduler, 'total_updates', 'n/a')), file=f)
            print('wall_time_seconds: {:.1f}'.format(total_elapsed), file=f)
            print('resumed_from: {}'.format(resume_from), file=f)

    del model
    torch.cuda.empty_cache()
    return best_auc, best_dice, best_cycle

if __name__ == '__main__':
    '''
    Example:
    python train_cyclical.py --csv_train data/DRIVE/train.csv --save_path unet_DRIVE
    '''

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print('* Training on device '.format(args.device))
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    # reproducibility
    seed_value = args.seed
    set_seeds(seed_value, args.device.startswith("cuda"))

    # gather parser parameters
    model_name = args.model_name
    max_lr, min_lr, bs, grad_acc_steps = args.max_lr, args.min_lr, args.batch_size, args.grad_acc_steps
    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
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


    print("* Creating Dataloaders, batch size = {}, workers = {}".format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs, tg_size=tg_size, label_values=label_values, num_workers=args.num_workers, seed=seed_value)

    # grad_acc_steps: if I want to train with a fake_bs=K but the actual bs I want is bs=N, then you use
    # grad_acc_steps = N/K - 1.
    # Example: bs=4, fake_bs=4 -> grad_acc_steps = 0 (default)
    # Example: bs=4, fake_bs=2 -> grad_acc_steps = 1
    # Example: bs=4, fake_bs=1 -> grad_acc_steps = 3

    print('* Instantiating a {} model'.format(model_name))
    model = get_arch(model_name, in_c=args.in_c, n_classes=n_classes)
    model = model.to(device)

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
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
    # number of optimizer updates per epoch: one update every (grad_acc_steps+1) batches
    # (computed after the pseudo-label branch above, so an extended train set is accounted for)
    updates_per_epoch = math.ceil(len(train_loader) / (grad_acc_steps + 1))
    total_planned_updates = updates_per_epoch * sum(cycle_lens)

    if args.scheduler == 'damped_cosine':
        # original scheduler oscillation spans TWO cycles (decay over one cycle of
        # T_max = cycle_lens[0]*len(train_loader) steps, rise over the next), so the
        # matching damped-cosine period is 2*cycle_lens[0]*updates_per_epoch.
        dc_period = args.dc_period if args.dc_period != 0 else 2 * cycle_lens[0] * updates_per_epoch
        try:
            validate_damped_cosine_config(total_planned_updates, dc_period, args.dc_alpha, args.dc_d,
                                          max_lr, min_lr)
        except DampedCosineError as e:
            sys.exit('invalid damped_cosine configuration: {}'.format(e))
        scheduler = DampedCosineLRSchedule(optimizer, total_updates=total_planned_updates,
                                           period=dc_period, alpha=args.dc_alpha, d=args.dc_d,
                                           lr_max=max_lr, lr_min=min_lr)
    else:
        # original scheduler, behavior preserved exactly (note: eta_min=0 regardless of --min_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=0)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)
    setattr(scheduler, 'kind', args.scheduler)
    setattr(scheduler, 'total_updates', total_planned_updates)

    print('* Scheduler: {} -- {} planned optimizer updates ({} per epoch x {} epochs)'.format(
        args.scheduler, total_planned_updates, updates_per_epoch, sum(cycle_lens)))
    if args.scheduler == 'damped_cosine':
        print('  damped_cosine: alpha={}, d={}, period={} updates, lr_max={}, lr_min={}'.format(
            args.dc_alpha, args.dc_d, dc_period, max_lr, min_lr))

    # (re)save config including resolved schedule information
    if do_not_save is False:
        with open(config_file_path, 'w') as f:
            cfg = dict(vars(args))
            cfg['total_planned_updates'] = total_planned_updates
            cfg['updates_per_epoch'] = updates_per_epoch
            cfg['dc_period_resolved'] = dc_period if args.scheduler == 'damped_cosine' else None
            json.dump(cfg, f, indent=2)

    ### RESUMING A PREVIOUS RUN
    start_cycle, best_state, lr_history, elapsed_time = 0, None, [], 0.0
    if args.resume_from is not None:
        state_path = osp.join(args.resume_from, 'training_state.pth')
        if not osp.isfile(state_path):
            sys.exit('cannot resume: {} not found (training_state.pth is saved at the end of each cycle)'.format(state_path))
        state = torch.load(state_path, map_location=device, weights_only=False)
        # consistency checks: never resume a run whose plan differs from the saved one
        if state['scheduler_type'] != args.scheduler:
            sys.exit('cannot resume: run was trained with scheduler "{}", but --scheduler {} was given'.format(
                state['scheduler_type'], args.scheduler))
        if state['total_planned_updates'] != total_planned_updates:
            sys.exit('cannot resume: planned total optimizer updates changed (saved {}, now {})'.format(
                state['total_planned_updates'], total_planned_updates))
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # restores the schedule position without resetting decay or recomputing
        # the horizon (DampedCosineLRSchedule.load_state_dict also restores T)
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_cycle = state['completed_cycles']
        best_state = state['best_state']
        lr_history = list(state['lr_history'])
        elapsed_time = state.get('elapsed_time', 0.0)
        set_rng_states(state['rng_states'])
        if getattr(train_loader, 'generator', None) is not None and state.get('loader_generator_state') is not None:
            train_loader.generator.set_state(state['loader_generator_state'])
        print('* Resuming from {} at cycle {}/{} ({} optimizer updates done)'.format(
            args.resume_from, start_cycle, len(cycle_lens), state['completed_updates']))

    criterion = torch.nn.BCEWithLogitsLoss() if model.n_classes == 1 else torch.nn.CrossEntropyLoss()


    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n','-' * 10)


    m1, m2, m3=train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, grad_acc_steps, metric, experiment_path,
                           start_cycle=start_cycle, best_state=best_state, lr_history=lr_history,
                           elapsed_time=elapsed_time, resume_from=args.resume_from)

    print("val_auc: %f" % m1)
    print("val_dice: %f" % m2)
    print("best_cycle: %d" % m3)
    if do_not_save is False:
        # file = open(osp.join(experiment_path, 'val_metrics.txt'), 'w')
        # file.write(str(m1)+ '\n')
        # file.write(str(m2)+ '\n')
        # file.write(str(m3)+ '\n')
        # file.close()

        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best AUC = {:.2f}\nBest DICE = {:.2f}\nBest cycle = {}'.format(100*m1, 100*m2, m3), file=f)
