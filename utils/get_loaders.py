from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr

import os
import os.path as osp
import random
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops
import torch
from torchvision.transforms import functional as tvtF
from torchvision.transforms import InterpolationMode

from .freesdg_aug import FreeSDGAugmentor

# Root-level loss module (scripts run from the repo root; torch-only import,
# SciPy stays lazy inside signed_distance_map for boundary preparation).
from segmentation_losses import signed_distance_map, check_binary_labels


def _capture_transform_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def _restore_transform_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None, freesdg_cfg=None,
                 need_distance_map=False, need_structural_saliency=False):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.label_values = label_values  # for use in label_encoding
        # Generic supervision metadata for the signed-distance boundary loss:
        # when True, each sample additionally returns a distance map computed
        # on CPU from the FINAL label (after resize/crop/rotation/deformation,
        # at the supervised output resolution). Derived only from
        # loss_boundary_weight > 0; never from --freesdg.
        self.need_distance_map = need_distance_map
        # Structural-saliency self-supervision: when True, each
        # training sample additionally returns a geometrically synchronized
        # raw RGB reference and its FOV mask. Training-only; never set for
        # validation.
        self.need_structural_saliency = need_structural_saliency
        # Opt-in FreeSDG FMAug configuration. None/disabled ->
        # the original code path below executes verbatim.
        self.freesdg_cfg = freesdg_cfg
        self.freesdg_resize = None  # hoisted deterministic resize (set by get_train_val_datasets)
        self.freesdg_raw_transforms = None
        self.freesdg_augmented_transforms = None
        self._freesdg_augmentor = None
        self._freesdg_augmentor_args = None
        self._freesdg_mask_resize = None
        self._freesdg_last_raffe_device = None

    def _freesdg_enabled(self):
        return self.freesdg_cfg is not None and self.freesdg_cfg.get('enabled', False)

    def _get_freesdg_augmentor(self):
        args = (
            self.freesdg_cfg.get('seed', 0),
            self.freesdg_cfg.get('ratio', 4.0),
            self.freesdg_cfg.get('mixup_size', -1),
            self.freesdg_cfg.get('mix_policy', 'repo'),
            self.freesdg_cfg.get('anchor_w', 27),
            self.freesdg_cfg.get('anchor_sigma', 9),
            self.freesdg_cfg.get('aug_mode', 'fmaug'),
        )
        if self._freesdg_augmentor is None or self._freesdg_augmentor_args != args:
            self._freesdg_augmentor_args = args
            self._freesdg_augmentor = FreeSDGAugmentor(
                seed=args[0], ratio=args[1], mixup_size=args[2],
                mix_policy=args[3], anchor_w=args[4], anchor_sigma=args[5],
                aug_mode=args[6])
        return self._freesdg_augmentor

    def _freesdg_mask_pil(self, mask, hw):
        # NEAREST resize keeps the FOV mask binary.
        if self._freesdg_mask_resize is None:
            self._freesdg_mask_resize = p_tr.Resize(hw, interpolation=InterpolationMode.NEAREST)
        elif self._freesdg_mask_resize.size != hw:
            self._freesdg_mask_resize = p_tr.Resize(hw, interpolation=InterpolationMode.NEAREST)
        return self._freesdg_mask_resize(mask)

    def _freesdg_mask_tensor(self, mask, hw):
        return tvtF.to_tensor(self._freesdg_mask_pil(mask, hw))

    def _apply_transforms_with_structural_reference(
            self, network_img, segmentation_target, structural_raw_img,
            structural_fov_mask, transforms=None):
        # Replay the exact same random transform realization on the raw RGB
        # reference and its FOV mask. The second transform call must
        # not advance the global RNG permanently.
        before = _capture_transform_rng_state()
        transforms = transforms or self.transforms
        network_img_t, segmentation_t = transforms(network_img, segmentation_target)
        after = _capture_transform_rng_state()
        _restore_transform_rng_state(before)
        try:
            structural_ref_t, structural_mask_t = transforms(
                structural_raw_img, structural_fov_mask)
        finally:
            _restore_transform_rng_state(after)
        return network_img_t, segmentation_t, structural_ref_t, structural_mask_t

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(self.im_list[index])
        target = Image.open(self.gt_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')

        img, target, mask = self.crop_to_fov(img, target, mask)

        target = self.label_encoding(target)

        target = np.array(self.label_encoding(target))

        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        structural_ref = None
        structural_mask = None

        if self._freesdg_enabled() and self.freesdg_cfg.get('role', 'train') == 'train':
            # FMAug training path: hoisted deterministic
            # resize of image+target, NEAREST-resized FOV mask, FMAug on the
            # [0,1] image, uint8 PIL round-trip (~1/255 quantization, §19.6),
            # then exactly the remaining original transform order.
            augmentor = self._get_freesdg_augmentor()
            img, target = self.freesdg_resize(img, target)
            if self.need_structural_saliency:
                structural_raw_img = img.copy()
            img01 = tvtF.to_tensor(img)
            mask01 = self._freesdg_mask_tensor(mask, tuple(img01.shape[-2:]))
            if self.need_structural_saliency:
                structural_fov_mask = self._freesdg_mask_pil(mask, tuple(img01.shape[-2:]))
            # Raffe is intentionally executed per sample before the existing
            # PIL/LwNet transforms.  Keep all other FreeSDG modes on their
            # original CPU path and return to CPU for the required PIL bridge.
            raffe_mode = self.freesdg_cfg.get('aug_mode') in (
                'raffe_filter', 'raffe_smooth_mix')
            raffe_device = torch.device(
                self.freesdg_cfg.get('device', 'cpu')) if raffe_mode else None
            if raffe_device is not None and raffe_device.type == 'cuda':
                img01 = img01.to(raffe_device)
                mask01 = mask01.to(raffe_device)
            disable_color_jitter = self.freesdg_cfg.get(
                'disable_aug_color_jitter', False)
            if disable_color_jitter:
                img01, branch = augmentor.augment_train(
                    img01, mask01, raw_prob=self.freesdg_cfg.get('raw_prob', 0.0),
                    return_branch=True)
            else:
                img01 = augmentor.augment_train(
                    img01, mask01, raw_prob=self.freesdg_cfg.get('raw_prob', 0.0))
                branch = None
            if raffe_device is not None:
                self._freesdg_last_raffe_device = img01.device
                img01 = img01.cpu()
            img = augmentor.to_pil_uint8(img01)
            if self.transforms is not None:
                selected_transforms = self.transforms
                if disable_color_jitter:
                    if branch == 'raw':
                        selected_transforms = self.freesdg_raw_transforms
                    elif branch == 'augmented':
                        selected_transforms = self.freesdg_augmented_transforms
                    else:
                        raise RuntimeError('unexpected FreeSDG branch: {!r}'.format(branch))
                if self.need_structural_saliency:
                    img, target, structural_ref, structural_mask = \
                        self._apply_transforms_with_structural_reference(
                            img, target, structural_raw_img, structural_fov_mask,
                            transforms=selected_transforms)
                else:
                    img, target = selected_transforms(img, target)
        elif self._freesdg_enabled() and self.freesdg_cfg.get('role') == 'val' \
                and self.freesdg_cfg.get('test_input', 'raw') == 'anchor':
            # Validation anchor path: deterministic Resize +
            # ToTensor as before, then fixed-anchor HFC on the float [0,1]
            # tensor (no uint8 round-trip on evaluation paths, §8).
            img, target = self.transforms(img, target)
            augmentor = self._get_freesdg_augmentor()
            mask01 = self._freesdg_mask_tensor(mask, tuple(img.shape[-2:]))
            img = augmentor.anchor(img, mask01)
        elif self.transforms is not None:
            if self.need_structural_saliency:
                structural_raw_img = img.copy()
                structural_fov_mask = mask.copy()
                img, target, structural_ref, structural_mask = \
                    self._apply_transforms_with_structural_reference(
                        img, target, structural_raw_img, structural_fov_mask)
            else:
                img, target = self.transforms(img, target)


        # QUICK HACK FOR PSEUDO_SEG IN VESSELS, BUT IT SPOILS A/V
        if len(self.label_values)==2: # vessel segmentation case
            target = target.float()
            if torch.max(target) >1:
                target= target.float()/255

        if structural_ref is not None:
            structural_mask = (structural_mask > 0).float().unsqueeze(0)

        # Optional boundary-loss supervision metadata: the signed-distance map
        # is recomputed from the final (post-augmentation) label here on CPU
        # -- no original-size/stale map is cached or deformed alongside -- then
        # collated and transferred to the device together with the labels.
        # The existing two-tensor sample format is preserved when disabled.
        # Degenerate masks (all-background/all-foreground) yield a zero map.
        if self.need_distance_map:
            check_binary_labels(target)
            dist_map = signed_distance_map(target.unsqueeze(0))  # [1, H, W]
            if self.need_structural_saliency:
                return img, target, dist_map, structural_ref, structural_mask
            return img, target, dist_map

        if self.need_structural_saliency:
            return img, target, structural_ref, structural_mask

        return img, target

    def __len__(self):
        return len(self.im_list)

class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.mask_list = df.mask_paths
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # # load image and mask
        img = Image.open(self.im_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        # # load image and mask
        # img = Image.open(self.im_list[index])
        # original_sz = img.size[1], img.size[0]  # in numpy convention
        # mask = Image.open(self.mask_list[index]).convert('L')
        # img, coords_crop = self.crop_to_fov(img, mask)
        # print(self.im_list[index], 'original size inside dataset', original_sz)

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)


def build_pseudo_dataset(train_csv_path, test_csv_path, path_to_preds):
    # assumes predictions are in path_to_preds and have the same name as images in the test csv
    # image extension does not matter
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # If there are more pseudo-segmentations than training segmentations
    # we bootstrap training images to get same numbers
    missing = test_df.shape[0] - train_df.shape[0]
    if missing > 0:
        extra_segs = train_df.sample(n=missing, replace=True, random_state=42)
        train_df = pd.concat([train_df, extra_segs])


    train_im_list = list(train_df.im_paths)
    train_gt_list = list(train_df.gt_paths)
    train_mask_list = list(train_df.mask_paths)

    test_im_list = list(test_df.im_paths)
    test_mask_list = list(test_df.mask_paths)

    test_preds = [n for n in os.listdir(path_to_preds) if 'binary' not in n and 'perf' not in n]
    test_pseudo_gt_list = []

    for n in test_im_list:
        im_name_no_extension = n.split('/')[-1][:-4]
        for pred_name in test_preds:
            pred_name_no_extension = pred_name.split('/')[-1][:-4]
            if im_name_no_extension == pred_name_no_extension:
                test_pseudo_gt_list.append(osp.join(path_to_preds, pred_name))
                break
    train_im_list.extend(test_im_list)
    train_gt_list.extend(test_pseudo_gt_list)
    train_mask_list.extend(test_mask_list)
    return train_im_list, train_gt_list, train_mask_list


def get_train_val_datasets(csv_path_train, csv_path_val, tg_size=(512, 512), label_values=(0, 255), freesdg_cfg=None, need_distance_map=False, need_structural_saliency=False):

    freesdg_enabled = freesdg_cfg is not None and freesdg_cfg.get('enabled', False)
    if freesdg_enabled:
        train_freesdg_cfg = dict(freesdg_cfg, role='train')
        val_freesdg_cfg = dict(freesdg_cfg, role='val')
        train_dataset = TrainDataset(csv_path=csv_path_train, label_values=label_values, freesdg_cfg=train_freesdg_cfg, need_distance_map=need_distance_map, need_structural_saliency=need_structural_saliency)
        val_dataset = TrainDataset(csv_path=csv_path_val, label_values=label_values, freesdg_cfg=val_freesdg_cfg, need_distance_map=need_distance_map, need_structural_saliency=False)
    else:
        train_dataset = TrainDataset(csv_path=csv_path_train, label_values=label_values, need_distance_map=need_distance_map, need_structural_saliency=need_structural_saliency)
        val_dataset = TrainDataset(csv_path=csv_path_val, label_values=label_values, need_distance_map=need_distance_map, need_structural_saliency=False)
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=(0, 0, 0), fill_tg=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    if freesdg_enabled:
        # The deterministic resize is hoisted ahead of FMAug in
        # TrainDataset.__getitem__ (same resize instance/interpolations);
        # the remaining original transform order is kept verbatim (§10.1).
        # Diagnostic profile: 'flips_only' disables LwNet's
        # scale/translation/rotation and ColorJitter after FMAug, keeping
        # only the flips + tensor conversion. The vanilla baseline
        # transforms are never altered.
        if freesdg_cfg.get('lwnet_aug_profile', 'original') == 'flips_only':
            train_transforms = p_tr.Compose([h_flip, v_flip, tensorizer])
        else:
            train_transforms = p_tr.Compose([scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
        train_dataset.freesdg_resize = resize
        if (freesdg_cfg.get('lwnet_aug_profile', 'original') == 'original'
                and freesdg_cfg.get('disable_aug_color_jitter', False)):
            train_dataset.freesdg_raw_transforms = p_tr.Compose([
                scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
            train_dataset.freesdg_augmented_transforms = p_tr.Compose([
                scale_transl_rot, h_flip, v_flip, tensorizer])
            train_transforms = train_dataset.freesdg_raw_transforms
    else:
        train_transforms = p_tr.Compose([resize,  scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), label_values=(0, 255), num_workers=0, freesdg_cfg=None, need_distance_map=False, need_structural_saliency=False):
    # need_distance_map is the generic boundary-loss requirement; it is applied
    # to the validation loader as well because validation loss is calculated.
    # need_structural_saliency is training-only and is forced off for validation.
    train_dataset, val_dataset = get_train_val_datasets(csv_path_train, csv_path_val, tg_size=tg_size, label_values=label_values, freesdg_cfg=freesdg_cfg, need_distance_map=need_distance_map, need_structural_saliency=need_structural_saliency)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def get_test_dataset(data_path, csv_path='test.csv', tg_size=(512, 512)):
    # csv_path will only not be test.csv when we want to build training set predictions
    path_test_csv = osp.join(data_path, csv_path)
    test_dataset = TestDataset(csv_path=path_test_csv, tg_size=tg_size)

    return test_dataset



