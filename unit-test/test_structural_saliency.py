import math
import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.get_model import (
    get_arch, get_arch_options, set_eval_mode, validate_structural_saliency)
from models.res_unet_adrian import UNet, AuxiliarySaliencyDecoder
from utils.structural_saliency import (
    StructuralSaliencyTarget, build_gaussian_kernel2d)
from utils.get_loaders import (
    TrainDataset, get_train_val_datasets, get_train_val_loaders,
    _capture_transform_rng_state, _restore_transform_rng_state)
from utils.model_saving_loading import save_model, load_model
from segmentation_losses import SegmentationLoss, build_segmentation_loss


def _num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_input(in_c=3, size=64, batch=1):
    return torch.randn(batch, in_c, size, size)


def _make_synthetic_sample(size=64):
    """Return (rgb PIL, gt PIL, mask PIL) with a circular FOV and a marker."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    r = size // 2 - 2
    fov = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[fov] = [100, 120, 140]
    img[:, cx - 1:cx + 2] = [255, 0, 0]
    img = img * fov[..., None]
    mask = (fov * 255).astype(np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[:, cx - 1:cx + 2] = 255
    gt = gt * fov
    return Image.fromarray(img, 'RGB'), Image.fromarray(gt, 'L'), Image.fromarray(mask, 'L')


def _write_synthetic_dataset(tmpdir, n=4, size=64):
    img_dir = os.path.join(tmpdir, 'images')
    gt_dir = os.path.join(tmpdir, 'gt')
    mask_dir = os.path.join(tmpdir, 'mask')
    for d in (img_dir, gt_dir, mask_dir):
        os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(n):
        img, gt, mask = _make_synthetic_sample(size)
        ip = os.path.join(img_dir, '{}.png'.format(i))
        gp = os.path.join(gt_dir, '{}.png'.format(i))
        mp = os.path.join(mask_dir, '{}.png'.format(i))
        img.save(ip)
        gt.save(gp)
        mask.save(mp)
        rows.append('{},{},{}'.format(ip, gp, mp))
    csv_path = os.path.join(tmpdir, 'train.csv')
    with open(csv_path, 'w') as f:
        f.write('im_paths,gt_paths,mask_paths\n')
        f.write('\n'.join(rows) + '\n')
    return csv_path


def _make_marker_sample(size=96, pos=(20, 20)):
    """RGB image with a bright 5x5 marker at ``pos``, a matching vessel label,
    and a full FOV mask. The marker is off-center so rotation/translation/scale
    move it, and it is bright enough to survive the geometric transforms."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    r, c = pos
    img[r - 2:r + 3, c - 2:c + 3] = [255, 255, 255]
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[r - 2:r + 3, c - 2:c + 3] = 255
    mask = np.full((size, size), 255, dtype=np.uint8)
    return Image.fromarray(img, 'RGB'), Image.fromarray(gt, 'L'), Image.fromarray(mask, 'L')


def _write_marker_dataset(tmpdir, size=96, pos=(20, 20)):
    img_dir = os.path.join(tmpdir, 'images')
    gt_dir = os.path.join(tmpdir, 'gt')
    mask_dir = os.path.join(tmpdir, 'mask')
    for d in (img_dir, gt_dir, mask_dir):
        os.makedirs(d, exist_ok=True)
    img, gt, mask = _make_marker_sample(size, pos)
    ip = os.path.join(img_dir, '0.png')
    gp = os.path.join(gt_dir, '0.png')
    mp = os.path.join(mask_dir, '0.png')
    img.save(ip)
    gt.save(gp)
    mask.save(mp)
    csv_path = os.path.join(tmpdir, 'train.csv')
    with open(csv_path, 'w') as f:
        f.write('im_paths,gt_paths,mask_paths\n')
        f.write('{},{},{}\n'.format(ip, gp, mp))
    return csv_path


def _marker_location(tensor2d):
    """Argmax location of a [H, W] tensor, returned as (row, col)."""
    idx = int(torch.argmax(tensor2d))
    return idx // tensor2d.shape[1], idx % tensor2d.shape[1]


class TestA_UNetRefactorRegression(unittest.TestCase):
    def test_encode_decode_equals_forward(self):
        torch.manual_seed(0)
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        u.eval()
        x = _make_input()
        with torch.no_grad():
            ref = u(x)
            bottleneck, skips = u.encode(x)
            new = u.decode(bottleneck, skips)
        torch.testing.assert_close(new, ref, rtol=0, atol=0)

    def test_encode_decode_with_decoder_additions(self):
        torch.manual_seed(0)
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        u.eval()
        x = _make_input()
        bottleneck, skips = u.encode(x)
        additions = (torch.zeros(1, 16, 32, 32), torch.zeros(1, 8, 64, 64))
        with torch.no_grad():
            ref = u(x, decoder_additions=additions)
            new = u.decode(bottleneck, skips, decoder_additions=additions)
        torch.testing.assert_close(new, ref, rtol=0, atol=0)

    def test_encode_decode_return_features(self):
        torch.manual_seed(0)
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        u.eval()
        x = _make_input()
        bottleneck, skips = u.encode(x)
        with torch.no_grad():
            ref_logits, ref_feats = u(x, return_decoder_features=True)
            new_logits, new_feats = u.decode(bottleneck, skips, return_decoder_features=True)
        torch.testing.assert_close(new_logits, ref_logits, rtol=0, atol=0)
        self.assertEqual(len(new_feats), len(ref_feats))
        for a, b in zip(new_feats, ref_feats):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class TestB_DisabledFeatureIsNoOp(unittest.TestCase):
    def test_state_dict_keys_unchanged(self):
        m = get_arch('wnet')
        keys = list(m.state_dict().keys())
        self.assertFalse(any('saliency_decoder' in k for k in keys))

    def test_train_forward_returns_two_values(self):
        m = get_arch('wnet')
        m.train()
        m.mode = 'train'
        out = m(_make_input())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_param_count_unchanged(self):
        self.assertEqual(_num_params(get_arch('wnet')), 68482)

    def test_eval_behavior_unchanged(self):
        m = get_arch('wnet')
        m.eval()
        m.mode = 'eval'
        out = m(_make_input())
        self.assertIsInstance(out, torch.Tensor)


class TestC_StructuralTargetShapeRange(unittest.TestCase):
    def test_shape_and_range(self):
        torch.manual_seed(0)
        builder = StructuralSaliencyTarget(kernel_size=27, sigma=9.0, ratio=4.0)
        img = torch.rand(2, 3, 64, 64)
        mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
        target = builder(img, mask)
        self.assertEqual(tuple(target.shape), (2, 3, 64, 64))
        self.assertGreaterEqual(float(target.min()), -1.0)
        self.assertLessEqual(float(target.max()), 1.0)
        self.assertTrue(torch.isfinite(target).all())


class TestD_OutsideFOV(unittest.TestCase):
    def test_outside_fov_is_exactly_minus_one(self):
        builder = StructuralSaliencyTarget(kernel_size=27, sigma=9.0, ratio=4.0)
        size = 64
        yy, xx = np.mgrid[0:size, 0:size]
        cy, cx = size // 2, size // 2
        fov = ((yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 2 - 2) ** 2).astype(np.float32)
        mask = torch.from_numpy(fov).unsqueeze(0).unsqueeze(0)
        img = torch.rand(1, 3, size, size)
        target = builder(img, mask)
        outside = target[:, :, mask[0, 0] == 0]
        self.assertTrue(torch.all(outside == -1.0))


class TestE_ConstantImage(unittest.TestCase):
    def test_constant_image_residual_near_zero(self):
        builder = StructuralSaliencyTarget(kernel_size=27, sigma=9.0, ratio=4.0)
        img = torch.full((1, 3, 64, 64), 0.5)
        mask = torch.ones(1, 1, 64, 64)
        target = builder(img, mask)
        # Away from the border, the residual should be ~0 (target ~ -1 + 0).
        interior = target[:, :, 10:-10, 10:-10]
        self.assertLess(float(interior.abs().max()), 1e-3)


class TestF_GaussianCorrectness(unittest.TestCase):
    def test_kernel_normalized_and_symmetric(self):
        k = build_gaussian_kernel2d(27, 9.0)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=6)
        k2 = k[0, 0]
        self.assertTrue(torch.allclose(k2, k2.flip(0)))
        self.assertTrue(torch.allclose(k2, k2.flip(1)))

    def test_depthwise_matches_slow_reference(self):
        torch.manual_seed(0)
        builder = StructuralSaliencyTarget(kernel_size=27, sigma=9.0, ratio=4.0)
        img = torch.rand(1, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        fast = builder(img, mask)

        # Slow reference: median fill + per-channel replication-padded conv.
        median = img.flatten(2).median(dim=2).values.view(1, 3, 1, 1) + 0.2
        x_filled = img * mask + median * (1.0 - mask)
        pad = 13
        padded = F.pad(x_filled, (pad, pad, pad, pad), mode='replicate')
        kernel = builder.kernel
        slow = F.conv2d(padded, kernel.repeat(3, 1, 1, 1), groups=3)
        residual = 4.0 * (x_filled - slow)
        residual = torch.clamp(residual, -1.0, 1.0)
        slow_target = (residual + 1.0) * mask - 1.0
        self.assertLess(float((fast - slow_target).abs().max()), 1e-6)


class TestH_RNGPreservation(unittest.TestCase):
    def test_replay_does_not_consume_extra_rng(self):
        import utils.paired_transforms_tv04 as p_tr
        resize = p_tr.Resize(64)
        tensorizer = p_tr.ToTensor()
        h_flip = p_tr.RandomHorizontalFlip()
        v_flip = p_tr.RandomVerticalFlip()
        rotate = p_tr.RandomRotation(degrees=45, fill=(0, 0, 0), fill_tg=(0,))
        scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
        scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
        jitter = p_tr.ColorJitter(0.25, 0.25, 0.25, 0.01)
        transforms = p_tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])

        img, gt, mask = _make_synthetic_sample(96)

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        # Path 1: ordinary transforms once.
        transforms(img, gt)
        final_1 = _capture_transform_rng_state()

        # Reset to the same initial state.
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        # Path 2: replay.
        before = _capture_transform_rng_state()
        transforms(img, gt)
        after = _capture_transform_rng_state()
        _restore_transform_rng_state(before)
        try:
            transforms(img.copy(), mask.copy())
        finally:
            _restore_transform_rng_state(after)
        final_2 = _capture_transform_rng_state()

        self.assertEqual(final_1['python'], final_2['python'])
        self.assertTrue(np.array_equal(final_1['numpy'][1], final_2['numpy'][1]))
        self.assertTrue(torch.equal(final_1['torch'], final_2['torch']))


class TestI_DatasetTupleContracts(unittest.TestCase):
    def _make_dataset(self, tmpdir, need_distance_map, need_structural_saliency, freesdg_cfg=None):
        csv = _write_synthetic_dataset(tmpdir, n=4, size=64)
        train_ds, val_ds = get_train_val_datasets(
            csv, csv, tg_size=(64, 64), label_values=(0, 255),
            freesdg_cfg=freesdg_cfg, need_distance_map=need_distance_map,
            need_structural_saliency=need_structural_saliency)
        return train_ds

    def test_neither(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._make_dataset(d, False, False)
            sample = ds[0]
            self.assertEqual(len(sample), 2)

    def test_boundary_only(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._make_dataset(d, True, False)
            sample = ds[0]
            self.assertEqual(len(sample), 3)

    def test_structural_only(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._make_dataset(d, False, True)
            sample = ds[0]
            self.assertEqual(len(sample), 4)
            img, target, structural_ref, structural_mask = sample
            self.assertEqual(tuple(structural_ref.shape), (3, 64, 64))
            self.assertEqual(tuple(structural_mask.shape), (1, 64, 64))

    def test_boundary_and_structural(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._make_dataset(d, True, True)
            sample = ds[0]
            self.assertEqual(len(sample), 5)
            img, target, dist_map, structural_ref, structural_mask = sample
            self.assertEqual(tuple(dist_map.shape), (1, 64, 64))
            self.assertEqual(tuple(structural_ref.shape), (3, 64, 64))
            self.assertEqual(tuple(structural_mask.shape), (1, 64, 64))

    def test_validation_never_returns_structural(self):
        with tempfile.TemporaryDirectory() as d:
            csv = _write_synthetic_dataset(d, n=4, size=64)
            train_ds, val_ds = get_train_val_datasets(
                csv, csv, tg_size=(64, 64), label_values=(0, 255),
                need_distance_map=False, need_structural_saliency=True)
            self.assertEqual(len(train_ds[0]), 4)
            self.assertEqual(len(val_ds[0]), 2)


class TestJ_ModelForwardContracts(unittest.TestCase):
    def test_training_returns_three(self):
        m = get_arch('wnet', structural_saliency=True)
        m.train()
        m.mode = 'train'
        x1, x2, saliency = m(_make_input())
        self.assertEqual(tuple(x1.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(x2.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(saliency.shape), (1, 3, 64, 64))
        self.assertGreaterEqual(float(saliency.detach().min()), -1.0)
        self.assertLessEqual(float(saliency.detach().max()), 1.0)

    def test_validation_style_returns_two(self):
        m = get_arch('wnet', structural_saliency=True)
        m.eval()
        m.mode = 'train'
        out = m(_make_input())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_inference_style_returns_tensor(self):
        m = get_arch('wnet', structural_saliency=True)
        m.eval()
        set_eval_mode(m)
        out = m(_make_input())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class TestK_DecoderNotExecutedDuringInference(unittest.TestCase):
    def test_saliency_decoder_forward_count_zero(self):
        m = get_arch('wnet', structural_saliency=True)
        counter = {'n': 0}

        def hook(module, inp, out):
            counter['n'] += 1

        handle = m.saliency_decoder.register_forward_hook(hook)
        try:
            m.eval()
            set_eval_mode(m)
            with torch.no_grad():
                m(_make_input())
        finally:
            handle.remove()
        self.assertEqual(counter['n'], 0)


class TestL_GradientFlow(unittest.TestCase):
    def test_structural_mse_gradients_reach_encoder_and_decoder(self):
        torch.manual_seed(0)
        m = get_arch('wnet', structural_saliency=True)
        m.train()
        m.mode = 'train'
        x = _make_input()
        x1, x2, saliency = m(x)
        target = torch.zeros_like(saliency)
        loss = F.mse_loss(saliency, target)
        loss.backward()

        # Saliency decoder and U1 encoder receive gradients.
        self.assertIsNotNone(m.saliency_decoder.final.weight.grad)
        self.assertTrue(torch.isfinite(m.saliency_decoder.final.weight.grad).all())
        self.assertIsNotNone(m.unet1.first.block[0].weight.grad)
        self.assertTrue(torch.isfinite(m.unet1.first.block[0].weight.grad).all())

    def test_structural_mse_does_not_grad_u2_or_u1_seg_decoder(self):
        torch.manual_seed(0)
        m = get_arch('wnet', structural_saliency=True)
        m.train()
        m.mode = 'train'
        x = _make_input()
        x1, x2, saliency = m(x)
        loss = F.mse_loss(saliency, torch.zeros_like(saliency))
        m.zero_grad()
        loss.backward()
        # U2 final conv and U1 segmentation final conv must have no grad.
        self.assertIsNone(m.unet2.final.weight.grad)
        self.assertIsNone(m.unet1.final.weight.grad)

    def test_structural_target_requires_grad_false(self):
        builder = StructuralSaliencyTarget()
        target = builder(torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32))
        self.assertFalse(target.requires_grad)


class TestM_Bridges(unittest.TestCase):
    def _check(self, bridge):
        torch.manual_seed(0)
        m = get_arch('wnet', structural_saliency=True, cross_stage_bridge=bridge)
        m.train()
        m.mode = 'train'
        x = _make_input()
        x1, x2, saliency = m(x)
        self.assertEqual(tuple(x1.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(x2.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(saliency.shape), (1, 3, 64, 64))
        loss = x1.sum() + x2.sum() + saliency.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_none(self):
        self._check('none')

    def test_scalar(self):
        self._check('scalar')

    def test_channel(self):
        self._check('channel')


class TestQ_CheckpointCompatibility(unittest.TestCase):
    def test_old_checkpoint_strict_loads_without_saliency(self):
        torch.manual_seed(0)
        base = get_arch('wnet')
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(base.parameters(), lr=0.01)
            save_model(d, base, opt)
            fresh = get_arch('wnet')
            fresh, stats = load_model(fresh, d, device='cpu')
        self.assertEqual(list(base.state_dict().keys()), list(fresh.state_dict().keys()))

    def test_new_checkpoint_strict_loads_from_config(self):
        torch.manual_seed(0)
        cfg = {'model_name': 'wnet', 'structural_saliency': True,
               'cross_stage_bridge': 'none', 'cross_stage_bridge_scales': 'all',
               'cross_stage_bridge_init': 0.0}
        arch_opts = get_arch_options(cfg)
        model = get_arch('wnet', in_c=3, n_classes=1, **arch_opts)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(d, model, opt)
            model2 = get_arch('wnet', in_c=3, n_classes=1, **get_arch_options(cfg))
            model2, stats = load_model(model2, d, device='cpu')
        self.assertEqual(list(model.state_dict().keys()), list(model2.state_dict().keys()))

    def test_saliency_checkpoint_fails_strict_load_without_saliency(self):
        torch.manual_seed(0)
        model = get_arch('wnet', structural_saliency=True)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(d, model, opt)
            plain = get_arch('wnet')
            with self.assertRaises(RuntimeError):
                load_model(plain, d, device='cpu')


class TestR_InferenceFromStructuralCheckpoint(unittest.TestCase):
    def test_inference_output_format(self):
        torch.manual_seed(0)
        model = get_arch('wnet', structural_saliency=True)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(d, model, opt)
            model2 = get_arch('wnet', structural_saliency=True)
            model2, stats = load_model(model2, d, device='cpu')
        set_eval_mode(model2)
        model2.eval()
        with torch.no_grad():
            out = model2(_make_input())
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class TestT_DeterministicRaffeRegression(unittest.TestCase):
    def test_raffe_output_unchanged_when_structural_disabled(self):
        from utils.freesdg_aug import FreeSDGAugmentor
        torch.manual_seed(0)
        img, gt, mask = _make_synthetic_sample(64)
        img01 = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        mask01 = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)

        def run():
            aug = FreeSDGAugmentor(seed=0, aug_mode='raffe_filter')
            return aug.augment_train(img01.unsqueeze(0), mask01, raw_prob=0.0)

        a = run()
        b = run()
        self.assertTrue(torch.equal(a, b))


class TestU_NoNaNs(unittest.TestCase):
    def test_target_and_prediction_finite(self):
        torch.manual_seed(0)
        builder = StructuralSaliencyTarget()
        img = torch.rand(2, 3, 64, 64)
        mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
        target = builder(img, mask)
        self.assertTrue(torch.isfinite(target).all())

        m = get_arch('wnet', structural_saliency=True)
        m.train()
        m.mode = 'train'
        x1, x2, saliency = m(_make_input(batch=2))
        self.assertTrue(torch.isfinite(saliency).all())
        loss = F.mse_loss(saliency, target)
        self.assertTrue(torch.isfinite(loss))


class TestG_SynchronizedAlignment(unittest.TestCase):
    def test_baseline_structural_reference_equals_network_image(self):
        # In the baseline (no FreeSDG) path the network image and the raw
        # structural reference are the same image; the replay must produce
        # identical tensors, proving the exact same transform realization.
        with tempfile.TemporaryDirectory() as d:
            csv = _write_synthetic_dataset(d, n=4, size=64)
            train_ds, _ = get_train_val_datasets(
                csv, csv, tg_size=(64, 64), label_values=(0, 255),
                need_distance_map=False, need_structural_saliency=True)
            img, target, structural_ref, structural_mask = train_ds[0]
            self.assertTrue(torch.equal(img, structural_ref))

    def test_structural_mask_is_binary(self):
        with tempfile.TemporaryDirectory() as d:
            csv = _write_synthetic_dataset(d, n=4, size=64)
            train_ds, _ = get_train_val_datasets(
                csv, csv, tg_size=(64, 64), label_values=(0, 255),
                need_distance_map=False, need_structural_saliency=True)
            img, target, structural_ref, structural_mask = train_ds[0]
            vals = torch.unique(structural_mask)
            self.assertTrue(torch.all((vals == 0) | (vals == 1)))

    def test_geometric_alignment_with_marker(self):
        # Plan §59: a distinctive off-center marker must remain spatially
        # aligned between the structural reference and the segmentation target
        # after the full random geometric pipeline (rotation/translation/scale/
        # flips). The marker is detected by argmax in both tensors; the target
        # uses NEAREST interpolation while the reference uses BILINEAR, so a
        # small tolerance absorbs the sub-pixel difference.
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            with tempfile.TemporaryDirectory() as d:
                csv = _write_marker_dataset(d, size=96, pos=(20, 20))
                train_ds, _ = get_train_val_datasets(
                    csv, csv, tg_size=(64, 64), label_values=(0, 255),
                    need_distance_map=False, need_structural_saliency=True)
                img, target, structural_ref, structural_mask = train_ds[0]
                ref_r, ref_c = _marker_location(structural_ref.sum(dim=0))
                tgt_r, tgt_c = _marker_location(target)
                self.assertLess(abs(ref_r - tgt_r), 3, 'seed {}'.format(seed))
                self.assertLess(abs(ref_c - tgt_c), 3, 'seed {}'.format(seed))

    def test_freesdg_structural_reference_is_raw_not_raffe(self):
        # Plan §11/§59: in the FreeSDG path the network input is RAFFE-augmented
        # while the structural reference must be the raw (non-frequency-
        # augmented) image. The reference must therefore differ from the
        # network image, and its marker must still align with the segmentation
        # target (both are raw and share the same geometric transform).
        freesdg_cfg = {
            'enabled': True, 'raw_prob': 0.0, 'test_input': 'raw',
            'anchor_w': 27, 'anchor_sigma': 9, 'ratio': 4.0,
            'mixup_size': -1, 'mix_policy': 'repo', 'seed': 0,
            'aug_mode': 'raffe_filter', 'lwnet_aug_profile': 'original',
            'device': 'cpu',
        }
        for seed in range(3):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            with tempfile.TemporaryDirectory() as d:
                csv = _write_marker_dataset(d, size=96, pos=(20, 20))
                train_ds, _ = get_train_val_datasets(
                    csv, csv, tg_size=(64, 64), label_values=(0, 255),
                    freesdg_cfg=freesdg_cfg, need_distance_map=False,
                    need_structural_saliency=True)
                img, target, structural_ref, structural_mask = train_ds[0]
                # The reference is the raw image, not the RAFFE view.
                self.assertFalse(torch.equal(img, structural_ref),
                                 'seed {}'.format(seed))
                # The reference marker aligns with the target marker.
                ref_r, ref_c = _marker_location(structural_ref.sum(dim=0))
                tgt_r, tgt_c = _marker_location(target)
                self.assertLess(abs(ref_r - tgt_r), 3, 'seed {}'.format(seed))
                self.assertLess(abs(ref_c - tgt_c), 3, 'seed {}'.format(seed))


class TestN_AugmentationCombinations(unittest.TestCase):
    def _check_mode(self, aug_mode):
        with tempfile.TemporaryDirectory() as d:
            csv = _write_synthetic_dataset(d, n=4, size=64)
            freesdg_cfg = {
                'enabled': True, 'raw_prob': 0.0, 'test_input': 'raw',
                'anchor_w': 27, 'anchor_sigma': 9, 'ratio': 4.0,
                'mixup_size': -1, 'mix_policy': 'repo', 'seed': 0,
                'aug_mode': aug_mode, 'lwnet_aug_profile': 'original',
                'device': 'cpu',
            }
            train_ds, _ = get_train_val_datasets(
                csv, csv, tg_size=(64, 64), label_values=(0, 255),
                freesdg_cfg=freesdg_cfg, need_distance_map=False,
                need_structural_saliency=True)
            sample = train_ds[0]
            self.assertEqual(len(sample), 4)
            img, target, structural_ref, structural_mask = sample
            self.assertTrue(torch.isfinite(img).all())
            self.assertTrue(torch.isfinite(structural_ref).all())
            self.assertTrue(torch.isfinite(structural_mask).all())

    def test_fmaug(self):
        self._check_mode('fmaug')

    def test_raffe_filter(self):
        self._check_mode('raffe_filter')

    def test_raffe_smooth_mix(self):
        self._check_mode('raffe_smooth_mix')


class TestValidation(unittest.TestCase):
    def test_validator_rejects_unsupported_model(self):
        with self.assertRaises(ValueError):
            validate_structural_saliency(True, 'unet', 1, 3, 1.0, 27, 9.0, 4.0)

    def test_validator_rejects_multiclass(self):
        with self.assertRaises(ValueError):
            validate_structural_saliency(True, 'wnet', 4, 3, 1.0, 27, 9.0, 4.0)

    def test_validator_rejects_non_rgb(self):
        with self.assertRaises(ValueError):
            validate_structural_saliency(True, 'wnet', 1, 1, 1.0, 27, 9.0, 4.0)

    def test_validator_rejects_even_kernel(self):
        with self.assertRaises(ValueError):
            validate_structural_saliency(True, 'wnet', 1, 3, 1.0, 28, 9.0, 4.0)

    def test_validator_rejects_nonpositive_weight(self):
        with self.assertRaises(ValueError):
            validate_structural_saliency(True, 'wnet', 1, 3, 0.0, 27, 9.0, 4.0)

    def test_validator_accepts_valid(self):
        validate_structural_saliency(True, 'wnet', 1, 3, 1.0, 27, 9.0, 4.0)
        validate_structural_saliency(True, 'big_wnet', 1, 3, 1.0, 27, 9.0, 4.0)

    def test_validator_noop_when_disabled(self):
        validate_structural_saliency(False, 'unet', 4, 1, 0.0, 0, 0.0, 0.0)


if __name__ == '__main__':
    unittest.main()
