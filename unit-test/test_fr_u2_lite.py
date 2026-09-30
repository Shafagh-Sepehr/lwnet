import os
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fr_u2 import FullResolutionMultiResU2
from models.fr_u2_lite import FullResolutionMultiResU2Lite
from models.get_model import (
    get_arch,
    get_arch_from_config,
    get_arch_options,
    set_eval_mode,
    wnet,
)
from models.res_unet_adrian import UNet
from segmentation_losses import SegmentationLoss
from train_cyclical import (
    parser,
    resolve_arch_args,
    run_one_epoch,
    write_training_config,
)
from utils.get_loaders import get_train_val_datasets
from utils.model_saving_loading import load_model, save_model
from utils.structural_saliency import StructuralSaliencyTarget


LEGACY_FULL_FR_SCHEMA = {
    'state_dict_keys': 731,
    'state_dict_numel': 315994,
    'trainable_params': 312354,
    'first_key': 'unet1.first.shortcut.0.weight',
    'last_key': 'unet2.final.bias',
    'key_shapes': {
        'unet1.first.block.0.weight': (8, 3, 3, 3),
        'unet1.final.weight': (1, 8, 1, 1),
        'unet2.stem.conv1.weight': (8, 4, 3, 3),
        'unet2.fusion_stages.2.block1.conv1.weight': (16, 48, 3, 3),
        'unet2.detail.0.weight': (8, 12, 1, 1),
        'unet2.final.weight': (1, 8, 1, 1),
    },
}


def _legacy_full_fr_checkpoint_fixture():
    """Build a frozen pre-Lite schema fixture without using ``get_arch``.

    The fixture values are intentionally zeros: strict loading validates the
    persisted key/shape contract, while the constants above freeze the schema
    independently of the current architecture factory. No archived full-FR
    checkpoint exists in this repository, so this fixture does not claim
    historical numerical output identity.
    """
    u1 = UNet(in_c=3, n_classes=1, layers=[8, 16, 32],
              conv_bridge=True, shortcut=True)
    u2 = FullResolutionMultiResU2(in_c=4, n_classes=1)
    state = {}
    for key, value in u1.state_dict().items():
        state['unet1.' + key] = torch.zeros_like(value)
    for key, value in u2.state_dict().items():
        state['unet2.' + key] = torch.zeros_like(value)
    return state


class FrU2LiteTests(unittest.TestCase):
    def test_mode_selection_and_legacy_full_fr_reconstruction(self):
        original = get_arch('wnet')
        full = get_arch('wnet', u2_arch='fr_multi')
        lite = get_arch('wnet', fr_lite=True)
        both = get_arch('wnet', u2_arch='fr_multi', fr_lite=True)

        self.assertIsInstance(original.unet2, type(original.unet1))
        self.assertIsInstance(full.unet2, FullResolutionMultiResU2)
        self.assertIsInstance(lite.unet2, FullResolutionMultiResU2Lite)
        self.assertIsInstance(both.unet2, FullResolutionMultiResU2Lite)
        self.assertFalse(original.fr_lite)
        self.assertFalse(full.fr_lite)
        self.assertTrue(lite.fr_lite)
        self.assertEqual(lite.u2_arch, 'fr_multi')

        old_full_options = get_arch_options({'u2_arch': 'fr_multi'})
        old_full = get_arch('wnet', **old_full_options)
        self.assertIsInstance(old_full.unet2, FullResolutionMultiResU2)
        self.assertEqual(
            list(full.state_dict().keys()), list(old_full.state_dict().keys()))
        self.assertEqual(
            sum(p.numel() for p in full.parameters()),
            sum(p.numel() for p in old_full.parameters()))

        config_only_lite = get_arch(
            'wnet', **get_arch_options({'fr_lite': True}))
        self.assertIsInstance(config_only_lite.unet2, FullResolutionMultiResU2Lite)
        self.assertFalse(get_arch_options({'u2_arch': 'fr_multi'})['fr_lite'])

    def test_direct_wnet_constructor_canonicalizes_lite(self):
        model = wnet(fr_lite=True)
        self.assertIsInstance(model.unet2, FullResolutionMultiResU2Lite)
        self.assertEqual(model.u2_arch, 'fr_multi')
        self.assertTrue(model.fr_lite)
        with self.assertRaisesRegex(ValueError, 'does not support A1'):
            wnet(fr_lite=True, cross_stage_bridge='scalar')

    def test_pre_lite_full_fr_checkpoint_loads_from_legacy_config(self):
        checkpoint = _legacy_full_fr_checkpoint_fixture()
        keys = list(checkpoint)
        self.assertEqual(len(keys), LEGACY_FULL_FR_SCHEMA['state_dict_keys'])
        self.assertEqual(
            sum(value.numel() for value in checkpoint.values()),
            LEGACY_FULL_FR_SCHEMA['state_dict_numel'])
        self.assertEqual(keys[0], LEGACY_FULL_FR_SCHEMA['first_key'])
        self.assertEqual(keys[-1], LEGACY_FULL_FR_SCHEMA['last_key'])
        for key, shape in LEGACY_FULL_FR_SCHEMA['key_shapes'].items():
            self.assertIn(key, checkpoint)
            self.assertEqual(tuple(checkpoint[key].shape), shape, key)

        old_config = {'model_name': 'wnet', 'u2_arch': 'fr_multi'}
        model = get_arch('wnet', **get_arch_options(old_config))
        model.load_state_dict(checkpoint, strict=True)
        self.assertIsInstance(model.unet2, FullResolutionMultiResU2)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            LEGACY_FULL_FR_SCHEMA['trainable_params'])

    def test_lite_has_fixed_architecture_and_parameter_budget(self):
        model = get_arch('wnet', fr_lite=True)
        lite = model.unet2
        self.assertEqual(lite.base_channels, 4)
        self.assertEqual(lite.dilations, (1, 2, 1))
        self.assertEqual(len(lite.fusion_stages), 3)
        self.assertLess(sum(p.numel() for p in model.parameters()), 100000)
        self.assertTrue(78000 <= sum(p.numel() for p in model.parameters()) <= 85000)
        self.assertFalse(hasattr(lite, 'detail'))
        self.assertFalse(hasattr(lite, 'detail_dilation1'))
        self.assertFalse(hasattr(lite, 'head_fusion'))

    def test_lite_spatial_shapes_and_raw_logits(self):
        model = get_arch('wnet', fr_lite=True)
        model.eval()
        model.mode = 'eval'
        for height, width in ((128, 128), (128, 160), (129, 161), (512, 512)):
            with torch.no_grad():
                output = model(torch.randn(1, 3, height, width))
            self.assertEqual(tuple(output.shape), (1, 1, height, width))
            self.assertTrue(torch.isfinite(output).all())

    def test_lite_gradients_reach_all_required_components(self):
        model = get_arch('wnet', fr_lite=True)
        model.train()
        x1, x2 = model(torch.randn(1, 3, 64, 80))
        (x1.mean() + x2.mean()).backward()

        lite = model.unet2
        modules = [lite.stem, lite.half_init, lite.quarter_init]
        modules.extend(lite.fusion_stages)
        modules.extend([lite.out_up_2_to_1, lite.half_fusion,
                        lite.out_up_1_to_0, lite.full_fusion, lite.final])
        for module in modules:
            parameter = next(module.parameters())
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.unet1.parameters()))

    def test_lite_rejects_bridges_and_uses_fixed_cli_resolution(self):
        with self.assertRaisesRegex(ValueError, 'FR-WNet-Lite does not support'):
            get_arch('wnet', fr_lite=True, cross_stage_bridge='scalar')
        args = parser.parse_args(['--fr_lite'])
        if args.fr_lite:
            args.u2_arch = 'fr_multi'
        model = get_arch(
            'wnet',
            u2_arch=args.u2_arch,
            fr_lite=args.fr_lite,
            fr_u2_base_channels=args.fr_u2_base_channels,
            fr_u2_dilations=args.fr_u2_dilations,
        )
        self.assertIsInstance(model.unet2, FullResolutionMultiResU2Lite)

    def test_lite_cli_resolution_is_serialized_with_effective_values(self):
        args = parser.parse_args([
            '--fr_lite', '--fr_u2_base_channels', '16',
            '--fr_u2_dilations', '1,2,4,2,1',
        ])
        resolve_arch_args(args)
        self.assertEqual(args.u2_arch, 'fr_multi')
        self.assertTrue(args.fr_lite)
        self.assertEqual(args.fr_u2_base_channels, 4)
        self.assertEqual(args.fr_u2_dilations, [1, 2, 1])

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'config.cfg')
            write_training_config(args, path)
            with open(path) as handle:
                saved = json.load(handle)
        self.assertEqual(saved['u2_arch'], 'fr_multi')
        self.assertTrue(saved['fr_lite'])
        self.assertEqual(saved['fr_u2_base_channels'], 4)
        self.assertEqual(saved['fr_u2_dilations'], [1, 2, 1])

    def test_lite_checkpoint_round_trip_and_full_checkpoint_separation(self):
        folder = tempfile.mkdtemp()
        try:
            model = set_eval_mode(get_arch('wnet', fr_lite=True))
            model.eval()
            optimizer = torch.optim.Adam(model.parameters())
            save_model(folder, model, optimizer)
            restored = set_eval_mode(get_arch('wnet', **get_arch_options({
                'u2_arch': 'fr_multi', 'fr_lite': True,
            })))
            restored, _ = load_model(restored, folder)
            restored.eval()
            x = torch.randn(1, 3, 64, 80)
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        finally:
            shutil.rmtree(folder)

        full_folder = tempfile.mkdtemp()
        try:
            full = get_arch('wnet', u2_arch='fr_multi')
            save_model(full_folder, full, torch.optim.Adam(full.parameters()))
            lite = get_arch('wnet', fr_lite=True)
            with self.assertRaises(RuntimeError):
                load_model(lite, full_folder)
        finally:
            shutil.rmtree(full_folder)

    def test_shared_inference_reconstruction_and_cpu_prediction_paths(self):
        config = {
            'model_name': 'wnet',
            'in_c': 3,
            'u2_arch': 'fr_multi',
            'fr_lite': True,
        }
        model = get_arch_from_config(config, in_c=3, device='cpu')
        self.assertIsInstance(model.unet2, FullResolutionMultiResU2Lite)
        model.eval()
        model.mode = 'eval'

        import generate_results
        import generate_av_results
        import predict_one_image
        import predict_one_image_av

        tensor = torch.rand(3, 32, 32)
        mask = np.ones((32, 32), dtype=bool)
        coords = [0, 0, 32, 32]
        with torch.no_grad():
            generated = generate_results.create_pred(
                model, tensor, mask, coords, (32, 32), tta='no',
                device=torch.device('cpu'))
            predicted, predicted_binary = predict_one_image.create_pred(
                model, tensor, mask, coords, (32, 32), bin_thresh=0.5,
                tta='no', device=torch.device('cpu'))
        self.assertEqual(generated.shape, (32, 32))
        self.assertEqual(predicted.shape, (32, 32))
        self.assertEqual(predicted_binary.shape, (32, 32))
        self.assertTrue(np.isfinite(generated).all())
        self.assertTrue(np.isfinite(predicted).all())

        av_model = get_arch_from_config(config, n_classes=4, device='cpu')
        av_model.eval()
        av_model.mode = 'eval'
        with torch.no_grad():
            generated_av, _ = generate_av_results.create_pred(
                av_model, tensor, mask, coords, (32, 32), tta='no',
                device=torch.device('cpu'))
            predicted_av, _ = predict_one_image_av.create_pred(
                av_model, tensor, mask, coords, (32, 32), tta='no',
                device=torch.device('cpu'))
        self.assertEqual(generated_av.shape, (32, 32, 3))
        self.assertEqual(predicted_av.shape, (32, 32, 3))
        self.assertTrue(np.isfinite(generated_av).all())
        self.assertTrue(np.isfinite(predicted_av).all())

    def test_lite_final_conv_receives_only_progressive_full_feature(self):
        model = get_arch('wnet', fr_lite=True)
        lite = model.unet2
        full_outputs = []
        final_inputs = []
        full_handle = lite.full_fusion.register_forward_hook(
            lambda module, inputs, output: full_outputs.append(output))
        final_handle = lite.final.register_forward_pre_hook(
            lambda module, inputs: final_inputs.append(inputs[0]))
        try:
            model.eval()
            model.mode = 'eval'
            with torch.no_grad():
                model(torch.randn(1, 3, 64, 80))
        finally:
            full_handle.remove()
            final_handle.remove()
        self.assertEqual(len(full_outputs), 1)
        self.assertEqual(len(final_inputs), 1)
        self.assertIs(final_inputs[0], full_outputs[0])
        self.assertEqual(final_inputs[0].shape[1], 4)

    def test_lite_structural_saliency_and_raffe_paths(self):
        inputs = torch.rand(1, 3, 32, 32)
        labels = (torch.rand(1, 32, 32) > 0.5).float()
        reference = torch.rand(1, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        loader = DataLoader(
            TensorDataset(inputs, labels, reference, mask), batch_size=1)
        model = get_arch('wnet', fr_lite=True, structural_saliency=True)
        criterion = SegmentationLoss(bce_weight=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        _, _, loss, _, components = run_one_epoch(
            loader, model, criterion, optimizer=optimizer, scheduler=scheduler,
            structural_target_builder=StructuralSaliencyTarget(),
            structural_saliency_weight=1.0)
        self.assertTrue(np.isfinite(loss))
        self.assertIn('structural_saliency', components)

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
                csv_path, csv_path, tg_size=(32, 32), freesdg_cfg=cfg)
            raffe_loader = DataLoader(train_dataset, batch_size=1)
            model = get_arch('wnet', fr_lite=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            _, _, loss, _, _ = run_one_epoch(
                raffe_loader, model, criterion,
                optimizer=optimizer, scheduler=scheduler)
            self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
