"""Tests for the opt-in U2 bottleneck Haar-DWT split/fuse and control.

Label: FDPR pillar-2 main idea in risk-controlled single-stage form. This is
not FDPR's dual-branch encoder and not a FDPR reproduction.

Covers the exact fixed Haar transform, the cross-gated DWT variant, the
exactly parameter-matched plain concat control, the off-path parity that keeps
the retained reference byte-identical, checkpoint reconstruction, and config
round trips.
"""

import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.dwt_fusion import (  # noqa: E402
    DWT_FUSION_TYPES, DWT_STAGE_NAME, HaarDWTFusion, HaarPlainControl,
    HaarTransform,
)
from models.get_model import (  # noqa: E402
    get_arch, get_arch_from_config, get_arch_options, validate_dwt_fusion,
)
from models.res_unet_adrian import UNet  # noqa: E402
from utils.model_saving_loading import save_model, load_model  # noqa: E402

REFERENCE_RUN = os.path.join(
    os.path.dirname(REPO_ROOT), 'lwnet', 'experiments',
    'final_t4_src_ss_raffeSmooth_scalar_sal_bce_d025_bnd001_c20x50_ck20_seed0')

MODULE_CLASSES = {
    'dwt': HaarDWTFusion,
    'plain': HaarPlainControl,
}

STAGE_CHANNELS = 32
DWT_MODULE_PARAMS = 6881
FULL_MODEL_PARAMS = 68482
FULL_MODEL_WITH_DWT = FULL_MODEL_PARAMS + DWT_MODULE_PARAMS
BIG_WNET_BASE_PARAMS = 279170
BIG_WNET_DWT_PARAMS = 26049


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


def _make_input(batch=1, in_c=3, height=64, width=None):
    if width is None:
        width = height
    return torch.randn(batch, in_c, height, width)


def _randomize_module(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name == 'output_gate':
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.1)


class HaarTransformTests(unittest.TestCase):
    def test_variant_enums_and_stage_name(self):
        self.assertEqual(DWT_FUSION_TYPES, ['none', 'dwt', 'plain'])
        self.assertEqual(DWT_STAGE_NAME, 'u2_bottleneck')

    def test_round_trip_is_exact_on_integer_inputs(self):
        haar = HaarTransform(STAGE_CHANNELS)
        for shape in [(1, STAGE_CHANNELS, 128, 128),
                      (2, STAGE_CHANNELS, 64, 96)]:
            x = torch.randint(0, 4, shape).float()
            low, high, (height, width) = haar.split(x)
            reconstructed = haar.reconstruct(low, high)
            self.assertTrue(torch.equal(reconstructed, x), shape)
            self.assertEqual(tuple(reconstructed.shape),
                             (shape[0], shape[1], height, width))

    def test_round_trip_float_precision_even_and_odd(self):
        haar = HaarTransform(STAGE_CHANNELS)
        for shape in [(1, STAGE_CHANNELS, 128, 128),
                      (4, STAGE_CHANNELS, 64, 64),
                      (1, STAGE_CHANNELS, 127, 129),
                      (3, STAGE_CHANNELS, 96, 160),
                      (2, STAGE_CHANNELS, 65, 63)]:
            x = torch.randn(*shape)
            low, high, (height, width) = haar.split(x)
            reconstructed = haar.reconstruct(low, high)
            reconstructed = reconstructed[..., :height, :width]
            self.assertEqual(tuple(reconstructed.shape), shape)
            self.assertTrue(torch.allclose(reconstructed, x, atol=1e-6, rtol=0),
                            (shape, float((reconstructed - x).abs().max())))

    def test_split_group_shapes(self):
        haar = HaarTransform(STAGE_CHANNELS)
        cases = [
            ((1, STAGE_CHANNELS, 128, 128), (1, STAGE_CHANNELS, 64, 64),
             (1, 3 * STAGE_CHANNELS, 64, 64)),
            ((4, STAGE_CHANNELS, 127, 129), (4, STAGE_CHANNELS, 64, 65),
             (4, 3 * STAGE_CHANNELS, 64, 65)),
        ]
        for shape, low_shape, high_shape in cases:
            low, high, spatial = haar.split(torch.randn(*shape))
            self.assertEqual(tuple(low.shape), low_shape)
            self.assertEqual(tuple(high.shape), high_shape)
            self.assertEqual(spatial, (shape[-2], shape[-1]))

    def test_energy_is_preserved(self):
        haar = HaarTransform(3)
        x = torch.randn(2, 3, 32, 40)
        low, high, _ = haar.split(x)
        self.assertTrue(torch.allclose(
            low.pow(2).sum() + high.pow(2).sum(), x.pow(2).sum(),
            atol=1e-4, rtol=0))

    def test_rejects_bad_channel_count(self):
        haar = HaarTransform(3)
        with self.assertRaises(ValueError):
            haar.split(torch.randn(1, 4, 16, 16))
        with self.assertRaises(ValueError):
            haar.reconstruct(torch.randn(1, 3, 8, 8),
                             torch.randn(1, 8, 8, 8))


class ModuleShapeTests(unittest.TestCase):
    def test_bottleneck_32ch_shapes(self):
        cases = [
            (1, STAGE_CHANNELS, 128, 128),
            (4, STAGE_CHANNELS, 128, 128),
            (1, STAGE_CHANNELS, 127, 129),
            (3, STAGE_CHANNELS, 96, 160),
            (2, STAGE_CHANNELS, 65, 63),
        ]
        for name, cls in MODULE_CLASSES.items():
            module = cls(STAGE_CHANNELS)
            for shape in cases:
                x = torch.randn(*shape)
                with torch.no_grad():
                    out = module(x)
                self.assertEqual(tuple(out.shape), shape, (name, shape))
                self.assertTrue(torch.isfinite(out).all(), (name, shape))

    def test_outputs_and_gradients_are_finite(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(STAGE_CHANNELS)
            _randomize_module(module)
            x = torch.randn(2, STAGE_CHANNELS, 32, 40, requires_grad=True)
            out = module(x)
            self.assertTrue(torch.isfinite(out).all(), name)
            out.sum().backward()
            self.assertIsNotNone(x.grad, name)
            self.assertTrue(torch.isfinite(x.grad).all(), name)
            for param_name, param in module.named_parameters():
                self.assertIsNotNone(param.grad, (name, param_name))
                self.assertTrue(torch.isfinite(param.grad).all(),
                                (name, param_name))

    def test_zero_gate_keeps_inner_gradients_zero(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(STAGE_CHANNELS)
            x = torch.randn(1, STAGE_CHANNELS, 32, 32)
            module(x).sum().backward()
            self.assertIsNotNone(module.output_gate.grad, name)
            self.assertTrue(torch.isfinite(module.output_gate.grad), name)
            for param_name, param in module.named_parameters():
                if param_name == 'output_gate':
                    continue
                self.assertEqual(float(param.grad.abs().sum()), 0.0,
                                 (name, param_name))


class InitIdentityTests(unittest.TestCase):
    def test_module_identity_at_init(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(STAGE_CHANNELS)
            x = torch.randn(2, STAGE_CHANNELS, 32, 32)
            with torch.no_grad():
                out = module(x)
            self.assertTrue(torch.equal(out, x), name)
            self.assertEqual(float(module.output_gate.detach()), 0.0, name)

    def _full_model_parity(self, variant, bridge):
        torch.manual_seed(0)
        base = get_arch('wnet', cross_stage_bridge=bridge)
        torch.manual_seed(0)
        fused = get_arch('wnet', cross_stage_bridge=bridge,
                         dwt_fusion=variant)
        fused.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        fused.eval()
        base.mode = 'eval'
        fused.mode = 'eval'
        x = _make_input()
        with torch.no_grad():
            self.assertTrue(torch.equal(base(x), fused(x)), (variant, bridge))

    def test_full_model_eval_parity_all_variants(self):
        for variant in ('dwt', 'plain'):
            self._full_model_parity(variant, 'none')

    def test_full_model_eval_parity_with_a1_bridge(self):
        for variant in ('dwt', 'plain'):
            self._full_model_parity(variant, 'scalar')

    def test_structural_saliency_train_parity(self):
        torch.manual_seed(0)
        base = get_arch('wnet', structural_saliency=True)
        torch.manual_seed(0)
        fused = get_arch('wnet', structural_saliency=True, dwt_fusion='dwt')
        fused.load_state_dict(base.state_dict(), strict=False)
        base.train()
        fused.train()
        base.mode = 'train'
        fused.mode = 'train'
        x = _make_input()
        b1, b2, bs = base(x)
        f1, f2, fs = fused(x)
        self.assertTrue(torch.equal(b1, f1))
        self.assertTrue(torch.equal(b2, f2))
        self.assertTrue(torch.equal(bs, fs))


class FullModelStageTests(unittest.TestCase):
    def test_dwt_applies_at_u2_bottleneck_32ch_at_128_for_512(self):
        torch.manual_seed(0)
        model = get_arch('wnet', dwt_fusion='dwt')
        self.assertEqual(model.dwt_stage_name, 'u2_bottleneck')
        self.assertEqual(model.dwt_fusion_channels, STAGE_CHANNELS)
        seen = []
        model.dwt_fusion_module.register_forward_hook(
            lambda mod, inputs, output: seen.append(tuple(inputs[0].shape)))
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(out.shape), (1, 1, 512, 512))
        self.assertEqual(seen, [(1, STAGE_CHANNELS, 128, 128)])

    def test_full_model_batch_and_rectangular(self):
        for variant in ('dwt', 'plain'):
            for batch, height, width in [(1, 64, 64), (4, 64, 64),
                                         (1, 68, 52), (2, 96, 128)]:
                torch.manual_seed(0)
                model = get_arch('wnet', dwt_fusion=variant)
                model.eval()
                model.mode = 'eval'
                with torch.no_grad():
                    out = model(torch.randn(batch, 3, height, width))
                self.assertEqual(tuple(out.shape), (batch, 1, height, width))
                self.assertTrue(torch.isfinite(out).all(),
                                (variant, batch, height, width))

    def test_full_model_gradients_are_finite(self):
        for variant in ('dwt', 'plain'):
            torch.manual_seed(0)
            model = get_arch('wnet', dwt_fusion=variant)
            _randomize_module(model.dwt_fusion_module)
            model.train()
            x1, x2 = model(_make_input(batch=2))
            (x1.sum() + x2.sum()).backward()
            for name, param in model.dwt_fusion_module.named_parameters():
                self.assertIsNotNone(param.grad, (variant, name))
                self.assertTrue(torch.isfinite(param.grad).all(),
                                (variant, name))

    def test_train_returns_tuple_and_eval_returns_tensor(self):
        model = get_arch('wnet', dwt_fusion='dwt')
        model.mode = 'train'
        self.assertIsInstance(model(_make_input()), tuple)
        model.mode = 'eval'
        self.assertIsInstance(model(_make_input()), torch.Tensor)

    def test_combines_with_existing_opt_in_flags(self):
        torch.manual_seed(0)
        model = get_arch('wnet', cross_stage_bridge='scalar',
                         cross_gate_fusion='corrected', proto_gate='corrected',
                         dwt_fusion='dwt')
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(_make_input())
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all())


class ParameterCountTests(unittest.TestCase):
    def test_module_counts_are_exactly_matched(self):
        dwt = _num_params(HaarDWTFusion(STAGE_CHANNELS))
        plain = _num_params(HaarPlainControl(STAGE_CHANNELS))
        self.assertEqual(dwt, DWT_MODULE_PARAMS)
        self.assertEqual(plain, DWT_MODULE_PARAMS)
        self.assertEqual(dwt - plain, 0)

    def test_full_model_counts(self):
        self.assertEqual(_num_params(get_arch('wnet')), FULL_MODEL_PARAMS)
        self.assertEqual(
            _num_params(get_arch('wnet', dwt_fusion='dwt')),
            FULL_MODEL_WITH_DWT)
        self.assertEqual(
            _num_params(get_arch('wnet', dwt_fusion='plain')),
            FULL_MODEL_WITH_DWT)

    def test_big_wnet_delta(self):
        self.assertEqual(
            _num_params(get_arch('big_wnet', dwt_fusion='dwt')) -
            _num_params(get_arch('big_wnet')),
            BIG_WNET_DWT_PARAMS)


class OffParityTests(unittest.TestCase):
    def test_default_state_dict_keys_unchanged(self):
        default = get_arch('wnet')
        explicit_off = get_arch('wnet', dwt_fusion='none')
        self.assertEqual(list(default.state_dict().keys()),
                         list(explicit_off.state_dict().keys()))
        self.assertFalse(any('dwt' in key for key in default.state_dict()))

    def test_no_extra_rng_draws_when_off(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', dwt_fusion='none')
        expected = default.state_dict()
        actual = explicit_off.state_dict()
        self.assertEqual(list(expected.keys()), list(actual.keys()))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_enabling_dwt_does_not_disturb_existing_init(self):
        torch.manual_seed(0)
        base = get_arch('wnet')
        torch.manual_seed(0)
        fused = get_arch('wnet', dwt_fusion='dwt')
        base_state = base.state_dict()
        fused_state = fused.state_dict()
        for key in base_state:
            self.assertTrue(torch.equal(base_state[key], fused_state[key]), key)

    def test_off_forward_parity(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', dwt_fusion='none')
        x = _make_input()
        b1, b2 = default(x)
        o1, o2 = explicit_off(x)
        self.assertTrue(torch.equal(b1, o1))
        self.assertTrue(torch.equal(b2, o2))

    def test_get_arch_options_default_off(self):
        self.assertEqual(get_arch_options({})['dwt_fusion'], 'none')
        self.assertEqual(get_arch_options({'dwt_fusion': 'dwt'})['dwt_fusion'],
                         'dwt')
        self.assertEqual(get_arch_options({'dwt_fusion': 'plain'})['dwt_fusion'],
                         'plain')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_strict_loads_unchanged(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        self.assertEqual(model.dwt_fusion, 'none')
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.assertEqual(_num_params(model), 83519)
        self.assertEqual(len(model.state_dict()), len(checkpoint['model_state_dict']))
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_warm_starts_dwt_at_identity(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        off_model = get_arch_from_config(dict(cfg), in_c=cfg.get('in_c', 3),
                                         n_classes=1)
        off_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        off_model.mode = 'eval'
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            reference_out = off_model(x)
        for variant in ('dwt', 'plain'):
            warm_cfg = dict(cfg)
            warm_cfg['dwt_fusion'] = variant
            warm_model = get_arch_from_config(
                warm_cfg, in_c=cfg.get('in_c', 3), n_classes=1)
            warm_model.load_state_dict(checkpoint['model_state_dict'],
                                       strict=False)
            warm_model.mode = 'eval'
            with torch.no_grad():
                self.assertTrue(torch.equal(reference_out, warm_model(x)),
                                variant)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_rejects_enabled_dwt(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        cfg['dwt_fusion'] = 'dwt'
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        with self.assertRaises(RuntimeError):
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)


class ConfigRoundTripTests(unittest.TestCase):
    def _cfg(self, variant, **extra):
        cfg = {'model_name': 'wnet', 'dwt_fusion': variant,
               'structural_saliency': True}
        cfg.update(extra)
        return cfg

    def test_each_variant_round_trips(self):
        for variant in ('dwt', 'plain'):
            model = get_arch_from_config(self._cfg(variant), in_c=3, n_classes=1)
            self.assertEqual(model.dwt_fusion, variant)
            self.assertEqual(model.dwt_stage_name, 'u2_bottleneck')
            self.assertEqual(model.dwt_fusion_channels, STAGE_CHANNELS)
            self.assertIn('dwt_fusion_module.output_gate', model.state_dict())
            model.mode = 'eval'
            with torch.no_grad():
                out = model(torch.randn(1, 3, 64, 64))
            self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    def test_old_config_without_key_stays_off(self):
        cfg = {'model_name': 'wnet', 'structural_saliency': True}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.dwt_fusion, 'none')
        self.assertFalse(any('dwt' in key for key in model.state_dict()))

    def test_reference_family_plus_dwt_round_trips(self):
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0, 'structural_saliency': True,
               'dwt_fusion': 'dwt'}
        torch.manual_seed(0)
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.dwt_fusion, 'dwt')
        self.assertIn('bridge_gates.half', model.state_dict())
        self.assertIn('dwt_fusion_module.low_proj.weight', model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(directory, model, optimizer)
            rebuilt = get_arch_from_config(cfg, in_c=3, n_classes=1)
            rebuilt, _ = load_model(rebuilt, directory, device='cpu')
        self.assertEqual(list(model.state_dict().keys()),
                         list(rebuilt.state_dict().keys()))
        for key in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[key],
                                        rebuilt.state_dict()[key]), key)

    def test_validation_rejects_unsupported_combinations(self):
        with self.assertRaises(ValueError):
            validate_dwt_fusion('bogus', 'wnet', 'unet')
        with self.assertRaises(ValueError):
            get_arch('unet', dwt_fusion='dwt')
        with self.assertRaises(ValueError):
            get_arch('big_unet', dwt_fusion='dwt')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', dwt_fusion='dwt')
        with self.assertRaises(ValueError):
            get_arch('wnet', fr_lite=True, dwt_fusion='dwt')

    def test_big_wnet_round_trips(self):
        cfg = {'model_name': 'big_wnet', 'dwt_fusion': 'dwt'}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.dwt_fusion_channels, 64)
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class EncoderSeamTests(unittest.TestCase):
    def test_bottleneck_transform_applies_to_a_single_unet(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        transform = HaarDWTFusion(32)
        out = model(_make_input(), bottleneck_transform=transform)
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    def test_bottleneck_transform_shape_mismatch_raises(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])

        class BadTransform(torch.nn.Module):
            def forward(self, x):
                return x[:, :16]

        with self.assertRaises(RuntimeError):
            model(_make_input(), bottleneck_transform=BadTransform())


class OptimizerSmokeTests(unittest.TestCase):
    def test_one_step_cpu_synthetic_with_dwt(self):
        torch.manual_seed(0)
        model = get_arch('wnet', dwt_fusion='dwt')
        _randomize_module(model.dwt_fusion_module)
        model.train()
        inputs = torch.randn(2, 3, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.8).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x1, x2 = model(inputs)
        loss = F.binary_cross_entropy_with_logits(x1, target) + \
            F.binary_cross_entropy_with_logits(x2, target)
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.dwt_fusion_module.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        optimizer.step()
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(inputs)
        self.assertEqual(tuple(out.shape), (2, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
