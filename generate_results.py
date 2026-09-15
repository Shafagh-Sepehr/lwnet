import os, json, sys
import os.path as osp
import argparse
import warnings
from tqdm import tqdm
import numpy
import time

import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.transform import resize
import torch
from utils.model_saving_loading import str2bool
from models.get_model import get_arch
from utils.get_loaders import get_test_dataset
from utils.model_saving_loading import load_model

# argument parsing
parser = argparse.ArgumentParser()
required_named = parser.add_argument_group('required arguments')
required_named.add_argument('--dataset', type=str, help='generate results for which dataset', required=True)
parser.add_argument('--public', type=str2bool, nargs='?', const=True, default=True, help='public or private data')
# parser.add_argument('--experiment_path', help='experiments/subfolder where checkpoint is', default=None)
parser.add_argument('--tta', type=str, default='from_preds', help='test-time augmentation (no/from_logits/from_preds)')
parser.add_argument('--binarize', type=str, default='otsu', help='binarization scheme (\'otsu\')')
parser.add_argument('--config_file', type=str, default=None, help='experiments/name_of_config_file, overrides everything')
# im_size overrides config file
parser.add_argument('--im_size', help='delimited list input, could be 600,400', type=str, default='512')
parser.add_argument('--device', type=str, default='cpu', help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
parser.add_argument('--result_path', type=str, default='results', help='path to save predictions (defaults to results')
# FreeSDG FMAug eval-input override (plan §12): None means "use the training
# config value, falling back to raw"; explicit values override the config.
parser.add_argument('--freesdg_test_input', type=str, default=None, choices=['raw', 'anchor'], help='override FreeSDG eval input policy (raw/anchor); default: training-config value or raw')

def flip_ud(tens):
    return torch.flip(tens, dims=[1])

def flip_lr(tens):
    return torch.flip(tens, dims=[2])

def flip_lrud(tens):
    return torch.flip(tens, dims=[1, 2])

def create_pred(model, tens, mask, coords_crop, original_sz, tta='no', device=None):
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=0)
    # NOTE (unrelated upstream bug fix, kept separate from FreeSDG logic):
    # create_pred previously always picked cuda-if-available, ignoring the
    # CLI-selected --device; it now receives the caller's device and only
    # falls back to the old behavior when not passed.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)

    if tta!='no':
        with torch.no_grad():
            logits_lr = model(tens.flip(-1).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1)
            logits_ud = model(tens.flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-2)
            logits_lrud = model(tens.flip(-1).flip(-2).unsqueeze(dim=0).to(device)).squeeze(dim=0).flip(-1).flip(-2)

        if tta == 'from_logits':
            mean_logits = torch.mean(torch.stack([logits, logits_lr, logits_ud, logits_lrud]), dim=0)
            pred = act(mean_logits)
        elif tta == 'from_preds':
            pred_lr = act(logits_lr)
            pred_ud = act(logits_ud)
            pred_lrud = act(logits_lrud)
            pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)
        else: raise NotImplementedError
    pred = pred.detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic
    pred = resize(pred, output_shape=original_sz, order=3)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0

    return full_pred

def save_pred(full_pred, save_results_path, im_name):
    im_name = im_name.rsplit('/', 1)[-1]
    save_name = osp.join(save_results_path, im_name[:-4] + '.png')
    # NOTE (unrelated upstream bug fix, kept separate from FreeSDG logic):
    # Windows-generated CSVs mix '/' and '\' in paths, so im_name may still
    # contain a directory component; create the actual target directory
    # instead of assuming save_results_path is the final directory.
    os.makedirs(osp.dirname(save_name), exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # this casts preds to int, loses precision but meh
        imsave(save_name, img_as_ubyte(full_pred))

if __name__ == '__main__':
    '''
    Example:
    python generate_results.py --config_file experiments/unet_drive/config.cfg --dataset DRIVE
    '''

    args = parser.parse_args()
    # capture before the config file overwrites the argparse namespace (§12)
    cli_freesdg_test_input = args.freesdg_test_input

    if args.device.startswith("cuda"):
        # In case one has multiple devices, we must first set the one
        # we would like to use so pytorch can find it.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":",1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is not currently available!")
        print(f"* Running prediction on device '{args.device}'...")
        device = torch.device("cuda")
    else:  #cpu
        device = torch.device(args.device)

    dataset = args.dataset
    binarize = args.binarize
    tta = args.tta
    public = str2bool(args.public)

    # parse config file if provided
    config_file = args.config_file
    if config_file is not None:
        if not osp.isfile(config_file): raise Exception('non-existent config file')
        with open(args.config_file, 'r') as f:
            args.__dict__.update(json.load(f))
    experiment_path = args.experiment_path # these should exist in a config file
    model_name = args.model_name
    in_c = args.in_c

    # FreeSDG eval-input resolution (plan §12): CLI value -> training-config
    # value -> raw. Anchor preprocessing applies only when the training config
    # indicates FreeSDG AND the resolved policy is 'anchor'.
    cfg_freesdg = bool(args.__dict__.get('freesdg', False))
    if cli_freesdg_test_input is not None:
        freesdg_test_input = cli_freesdg_test_input
    elif args.__dict__.get('freesdg_test_input') is not None:
        freesdg_test_input = args.freesdg_test_input
    else:
        freesdg_test_input = 'raw'
    use_freesdg_anchor = cfg_freesdg and freesdg_test_input == 'anchor'
    if freesdg_test_input == 'anchor' and not cfg_freesdg:
        print("* WARNING: --freesdg_test_input anchor requested, but the training "
              "config has no active FreeSDG fields (freesdg=false or absent); "
              "anchor preprocessing is skipped and raw input is used (plan §29.2).")

    if experiment_path is None: raise Exception('must specify path to experiment')

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    if public: data_path = osp.join('data', dataset)
    else: data_path = osp.join('private_data', dataset)

    csv_path = 'test_all.csv'
    print('* Reading test data from ' + osp.join(data_path, csv_path))
    test_dataset = get_test_dataset(data_path, csv_path=csv_path, tg_size=tg_size)
    print('* Instantiating model  = ' + str(model_name))
    model = get_arch(model_name, in_c=in_c).to(device)
    if model_name == 'wnet': model.mode='eval'

    print('* Loading trained weights from ' + experiment_path)
    try:
        model, stats = load_model(model, experiment_path, device)
    except RuntimeError:
        sys.exit('---- bad config specification (check layers, n_classes, etc.) ---- ')
    model.eval()

    save_results_path = osp.join(args.result_path, dataset, experiment_path)
    print('* Saving predictions to ' + save_results_path)

    # FreeSDG fixed-anchor inference preprocessing (plan §12/§15): float only,
    # no uint8 round-trip; applied once per image before the existing flip
    # TTA, which is valid because the Gaussian kernel and mask operations are
    # reflection-equivariant under the existing TTA flips (plan §12.1).
    if use_freesdg_anchor:
        from utils.freesdg_aug import FreeSDGAugmentor
        from torchvision.transforms import functional as tvF
        from torchvision.transforms import InterpolationMode
        from PIL import Image
        print('* Using FreeSDG anchor eval input (w={}, sigma={}, ratio={})'.format(
            args.__dict__.get('freesdg_anchor_w', 27),
            args.__dict__.get('freesdg_anchor_sigma', 9),
            args.__dict__.get('freesdg_ratio', 4.0)))
        anchoror = FreeSDGAugmentor(
            ratio=args.__dict__.get('freesdg_ratio', 4.0),
            anchor_w=args.__dict__.get('freesdg_anchor_w', 27),
            anchor_sigma=args.__dict__.get('freesdg_anchor_sigma', 9))

        def align_fov_mask(mask, coords_crop, hw):
            # FOV flow (§15): original mask -> same bbox crop as the image ->
            # NEAREST resize -> binary {0,1} tensor
            minr, minc, maxr, maxc = coords_crop
            mask_crop = mask[minr:maxr, minc:maxc]
            mask_pil = Image.fromarray((mask_crop.astype(np.uint8)) * 255, mode='L')
            mask_rsz = tvF.resize(mask_pil, hw, InterpolationMode.NEAREST)
            return tvF.to_tensor(mask_rsz)

    times = []
    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        if use_freesdg_anchor:
            mask01 = align_fov_mask(mask, coords_crop, tuple(im_tens.shape[-2:]))
            im_tens = anchoror.anchor(im_tens, mask01)
        start_time = time.perf_counter()
        full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta, device=device)
        times.append(time.perf_counter() - start_time)
        save_pred(full_pred, save_results_path, im_name)

    print(f"* Average image time: {numpy.mean(times):g}s")
    print('* Done')
