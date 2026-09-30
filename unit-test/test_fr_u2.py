import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fr_u2 import FullResolutionMultiResU2, MultiResolutionFusionStage
from models.get_model import get_arch, get_arch_options, set_eval_mode
from segmentation_losses import SegmentationLoss
from train_cyclical import format_fr_u2_dilations, run_one_epoch
from utils.model_saving_loading import load_model, save_model
from utils.get_loaders import get_train_val_datasets
from utils.structural_saliency import StructuralSaliencyTarget


class FrU2Tests(unittest.TestCase):
    def test_default_architecture_is_unchanged(self):
        model = get_arch('wnet')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 68482)
        self.assertEqual(model.u2_arch, 'unet')

    def test_forward_supports_rectangular_and_odd_inputs(self):
        model = get_arch('wnet', u2_arch='fr_multi')
        model.eval()
        model.mode = 'eval'
        for height, width in ((128, 160), (129, 161)):
            with torch.no_grad():
                output = model(torch.randn(1, 3, height, width))
            self.assertEqual(tuple(output.shape), (1, 1, height, width))
            self.assertTrue(torch.isfinite(output).all())

    def test_odd_size_padding_is_explicit_boundary_behavior(self):
        model = get_arch('wnet', u2_arch='fr_multi')
        observed = []
        handle = model.unet1.register_forward_pre_hook(
            lambda module, inputs: observed.append(tuple(inputs[0].shape[-2:]))
        )
        try:
            with torch.no_grad():
                model.eval()
                model.mode = 'eval'
                model(torch.randn(1, 3, 129, 161))
        finally:
            handle.remove()
        self.assertEqual(observed, [(132, 164)])

    def test_divisible_size_passes_original_tensor_to_u1(self):
        model = get_arch('wnet', u2_arch='fr_multi')
        input_tensor = torch.randn(1, 3, 128, 160)
        observed = []
        handle = model.unet1.register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0])
        )
        try:
            with torch.no_grad():
                model.eval()
                model.mode = 'eval'
                model(input_tensor)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], input_tensor)

    def test_dilation_summary_format_is_canonical(self):
        self.assertEqual(
            format_fr_u2_dilations([1, 2, 4, 2, 1]), '[1,2,4,2,1]')

    def test_train_forward_and_gradients_cover_all_streams(self):
        model = get_arch('wnet', u2_arch='fr_multi')
        model.train()
        x1, x2 = model(torch.randn(2, 3, 128, 128))
        self.assertEqual(tuple(x1.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(x2.shape), (2, 1, 128, 128))
        (x1.mean() + x2.mean()).backward()

        for name in ('stem', 'half_init', 'quarter_init', 'detail', 'final'):
            parameter = next(getattr(model.unet2, name).parameters())
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for stage in model.unet2.fusion_stages:
            parameter = next(stage.parameters())
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fusion_stage_preserves_stream_shapes(self):
        stage = MultiResolutionFusionStage(8, dilation=4)
        inputs = (
            torch.randn(2, 8, 33, 41),
            torch.randn(2, 16, 16, 20),
            torch.randn(2, 32, 8, 10),
        )
        outputs = stage(*inputs)
        self.assertEqual(tuple(outputs[0].shape), tuple(inputs[0].shape))
        self.assertEqual(tuple(outputs[1].shape), tuple(inputs[1].shape))
        self.assertEqual(tuple(outputs[2].shape), tuple(inputs[2].shape))

    def test_validation_rejects_a1_and_big_wnet(self):
        for bridge in ('scalar', 'channel'):
            with self.assertRaisesRegex(ValueError, 'does not support A1'):
                get_arch('wnet', u2_arch='fr_multi', cross_stage_bridge=bridge)
        with self.assertRaises(ValueError):
            get_arch('big_wnet', u2_arch='fr_multi')

    def test_validation_rejects_unused_or_malformed_fr_options(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='unet', fr_u2_base_channels=16)
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='unet', fr_u2_dilations='1,2,4,2,2')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', fr_u2_dilations='1,2,4')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', fr_u2_dilations='1,2,4,2,nope')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', fr_u2_dilations='1,2,4,2,1.5')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', fr_u2_base_channels=16.5)

    def test_structural_saliency_contract(self):
        model = get_arch('wnet', u2_arch='fr_multi', structural_saliency=True)
        model.train()
        x1, x2, saliency = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(x1.shape), (1, 1, 64, 80))
        self.assertEqual(tuple(x2.shape), (1, 1, 64, 80))
        self.assertEqual(tuple(saliency.shape), (1, 3, 64, 80))
        model = set_eval_mode(model)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 80))

    def test_strict_checkpoint_round_trip(self):
        config = {
            'u2_arch': 'fr_multi',
            'fr_u2_base_channels': 8,
            'fr_u2_dilations': '1,2,4,2,1',
        }
        model = set_eval_mode(get_arch('wnet', **get_arch_options(config)))
        model.eval()
        optimizer = torch.optim.Adam(model.parameters())
        folder = tempfile.mkdtemp()
        try:
            save_model(folder, model, optimizer)
            restored = set_eval_mode(get_arch('wnet', **get_arch_options(config)))
            restored, _ = load_model(restored, folder)
            restored.eval()
            x = torch.randn(1, 3, 64, 80)
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        finally:
            shutil.rmtree(folder)

    def test_baseline_checkpoint_cannot_load_into_fr_u2(self):
        folder = tempfile.mkdtemp()
        try:
            baseline = get_arch('wnet')
            save_model(folder, baseline, torch.optim.Adam(baseline.parameters()))
            fr_model = get_arch('wnet', u2_arch='fr_multi')
            with self.assertRaises(RuntimeError):
                load_model(fr_model, folder)
        finally:
            shutil.rmtree(folder)

    def test_run_one_epoch_fr_u2_training_validation_and_checkpoint(self):
        inputs = torch.rand(1, 3, 32, 32)
        labels = (torch.rand(1, 32, 32) > 0.5).float()
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=1)
        model = get_arch('wnet', u2_arch='fr_multi')
        criterion = SegmentationLoss(bce_weight=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        _, _, train_loss, _, _ = run_one_epoch(
            loader, model, criterion, optimizer=optimizer, scheduler=scheduler)
        _, _, val_loss, _, _ = run_one_epoch(loader, model, criterion)
        self.assertTrue(np.isfinite(train_loss))
        self.assertTrue(np.isfinite(val_loss))

        folder = tempfile.mkdtemp()
        try:
            save_model(folder, model, optimizer)
            restored = get_arch('wnet', u2_arch='fr_multi')
            restored, _ = load_model(restored, folder)
            restored.eval()
            restored.mode = 'eval'
            with torch.no_grad():
                output = restored(inputs)
            self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        finally:
            shutil.rmtree(folder)

    def test_run_one_epoch_fr_u2_structural_saliency(self):
        inputs = torch.rand(1, 3, 32, 32)
        labels = (torch.rand(1, 32, 32) > 0.5).float()
        reference = torch.rand(1, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        loader = DataLoader(
            TensorDataset(inputs, labels, reference, mask), batch_size=1)
        model = get_arch('wnet', u2_arch='fr_multi', structural_saliency=True)
        criterion = SegmentationLoss(bce_weight=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        _, _, loss, _, components = run_one_epoch(
            loader,
            model,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            structural_target_builder=StructuralSaliencyTarget(),
            structural_saliency_weight=1.0,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertIn('structural_saliency', components)

    def test_fr_u2_raffe_dataset_path(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.full((32, 32, 3), 128, dtype=np.uint8)
            label = np.zeros((32, 32), dtype=np.uint8)
            label[8:24, 8:24] = 255
            mask = np.full((32, 32), 255, dtype=np.uint8)
            image_path = os.path.join(folder, 'image.png')
            label_path = os.path.join(folder, 'label.png')
            mask_path = os.path.join(folder, 'mask.png')
            Image.fromarray(image).save(image_path)
            Image.fromarray(label).save(label_path)
            Image.fromarray(mask).save(mask_path)
            csv_path = os.path.join(folder, 'data.csv')
            with open(csv_path, 'w') as handle:
                handle.write('im_paths,gt_paths,mask_paths\n')
                handle.write('{},{},{}\n'.format(image_path, label_path, mask_path))

            cfg = {
                'enabled': True,
                'aug_mode': 'raffe_filter',
                'lwnet_aug_profile': 'flips_only',
                'seed': 0,
                'device': 'cpu',
            }
            train_dataset, _ = get_train_val_datasets(
                csv_path,
                csv_path,
                tg_size=(32, 32),
                freesdg_cfg=cfg,
            )
            loader = DataLoader(train_dataset, batch_size=1)
            model = get_arch('wnet', u2_arch='fr_multi')
            criterion = SegmentationLoss(bce_weight=1.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            _, _, loss, _, _ = run_one_epoch(
                loader, model, criterion, optimizer=optimizer, scheduler=scheduler)
            self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
