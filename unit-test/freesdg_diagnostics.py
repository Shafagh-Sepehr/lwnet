"""FreeSDG/RaffeSDG visual diagnostics for LwNet (diagnostic stage).

Saves, for several DRIVE training images, the full panel of intermediate
representations so augmentation quality can be inspected by eye:

 1. raw image (crop-to-FOV + resize, as the training pipeline sees it)
 2. fixed HFC(27, 9)                          [M2 test/train representation]
 3. random HFC sample A                        [M3, Gaussian bank]
 4. random HFC sample B                        [M3, second draw]
 5. full FMAug before LwNet transforms         [current training input]
 6. FMAug after the ORIGINAL LwNet transforms  [what W-Net actually sees]
 7. FMAug after flips-only transforms          [M4 profile]
 8. RaffeSDG random frequency filter           [U3A]
 9. RaffeSDG smooth blend                      [U3B]
10. GT vessel mask (aligned)

Usage:
    python freesdg_diagnostics.py [--n 3] [--out .scratch/freesdg-fmaug/diag2]
"""

import argparse
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from skimage.measure import regionprops
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvtF

from utils import paired_transforms_tv04 as p_tr
from utils.freesdg_aug import FreeSDGAugmentor


def load_sample(row, tg_size):
    img = Image.open(row.im_paths)
    mask = Image.open(row.mask_paths).convert('L')
    gt = Image.open(row.gt_paths).convert('L')
    m = np.array(mask)
    minr, minc, maxr, maxc = regionprops(m.astype(int))[0].bbox
    img_c = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
    mask_c = Image.fromarray(m[minr:maxr, minc:maxc])
    gt_c = Image.fromarray(np.array(gt)[minr:maxr, minc:maxc])
    rsz = p_tr.Resize(tg_size)
    img_r = rsz(img_c)
    mask_r = tvtF.resize(mask_c, tg_size, InterpolationMode.NEAREST)
    gt_r = tvtF.resize(gt_c, tg_size, InterpolationMode.NEAREST)
    return img_r, mask_r, gt_r


def tensor_to_pil(t01):
    return FreeSDGAugmentor.to_pil_uint8(
        t01 if torch.is_tensor(t01) else tvtF.to_tensor(t01))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3, help='number of images')
    parser.add_argument('--csv', type=str, default='data/DRIVE/train.csv')
    parser.add_argument('--im_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str,
                        default='.scratch/freesdg-fmaug/diag2')
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv).head(args.n)
    tg = (args.im_size, args.im_size)
    os.makedirs(args.out, exist_ok=True)

    # dedicated FMAug RNG (seeded); global RNG seeded for the LwNet transforms
    aug_fixed = FreeSDGAugmentor(seed=args.seed, aug_mode='fixed_hfc')
    aug_rand = FreeSDGAugmentor(seed=args.seed, aug_mode='random_hfc')
    aug_fmaug = FreeSDGAugmentor(seed=args.seed, aug_mode='fmaug')
    aug_raffe = FreeSDGAugmentor(seed=args.seed, aug_mode='raffe_filter')
    aug_smooth = FreeSDGAugmentor(seed=args.seed, aug_mode='raffe_smooth_mix')

    # exact replicas of the two post-frequency transform profiles
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=(0, 0, 0), fill_tg=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    jitter = p_tr.ColorJitter(0.25, 0.25, 0.25, 0.01)
    tensorizer = p_tr.ToTensor()
    orig_profile = p_tr.Compose(
        [scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    flips_profile = p_tr.Compose([h_flip, v_flip, tensorizer])

    for i, row in df.iterrows():
        img_r, mask_r, gt_r = load_sample(row, tg)
        x01 = tvtF.to_tensor(img_r)
        m01 = tvtF.to_tensor(mask_r)

        fixed = aug_fixed.augment_train(x01, m01)
        randA = aug_rand.augment_train(x01, m01)
        randB = aug_rand.augment_train(x01, m01)
        fmaug = aug_fmaug.augment_train(x01, m01)
        raffe = aug_raffe.augment_train(x01, m01)
        smooth = aug_smooth.augment_train(x01, m01)

        # post-transform views: replicate TrainDataset flow (uint8 PIL in)
        random.seed(args.seed + 1000 + i)
        fmaug_pil = aug_fmaug.to_pil_uint8(fmaug)
        gt_pil = gt_r
        img_after_orig, _ = orig_profile(fmaug_pil, gt_pil)
        random.seed(args.seed + 1000 + i)
        img_after_flips, _ = flips_profile(fmaug_pil, gt_pil)

        panels = [
            ('01_raw', img_r),
            ('02_fixed_hfc', tensor_to_pil(fixed)),
            ('03_random_hfc_A', tensor_to_pil(randA)),
            ('04_random_hfc_B', tensor_to_pil(randB)),
            ('05_fmaug_pre_transforms', fmaug_pil),
            ('06_fmaug_after_original_lwnet', tensor_to_pil(img_after_orig)),
            ('07_fmaug_after_flips_only', tensor_to_pil(img_after_flips)),
            ('08_raffe_filter', tensor_to_pil(raffe)),
            ('09_raffe_smooth_mix', tensor_to_pil(smooth)),
            ('10_gt_vessels', gt_r),
        ]
        for name, pil in panels:
            pil.save(osp.join(args.out, 'img{}_{}.png'.format(i, name)))
        print('saved {} panels for image {}'.format(len(panels), i))
    print('* diagnostics written to ' + args.out)


if __name__ == '__main__':
    main()
