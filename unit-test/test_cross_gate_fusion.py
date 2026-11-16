"""Tests for the opt-in U1-to-U2 half-scale C3-corrected fusion and controls.

Covers the corrected cross-gated module (CFF prose form), the matched-budget
plain concat+3x3 control, and the labeled CFF-literal degenerate comparator.
All variants are default-off; the disabled path must stay identical to the
retained reference in state-dict keys, parameter count, RNG use, and forward
behavior.
"""

import json
import os
import sys
import tempfile
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.cross_gate_fusion import (  # noqa: E402
    CFFLiteralFusion, CrossGatedFusion, PlainConcatFusion, FUSION_TYPES,
)
from models.get_model import (  # noqa: E402
    get_arch, get_arch_from_config, get_arch_options, validate_cross_gate_fusion,
)
from models.res_unet_adrian import UNet  # noqa: E402
from utils.model_saving_loading import save_model, load_model  # noqa: E402

REFERENCE_RUN = os.path.join(
    os.path.dirname(REPO_ROOT), 'lwnet', 'experiments',
    'final_t4_src_ss_raffeSmooth_scalar_sal_bce_d025_bnd001_c20x50_ck20_seed0')

MODULE_CLASSES = {
    'corrected': CrossGatedFusion,
    'plain': PlainConcatFusion,
    'literal': CFFLiteralFusion,
}


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


def _make_input(batch=1, in_c=3, size=64):
    return torch.randn(batch, in_c, size, size)


class ModuleShapeTests(unittest.TestCase):
    def test_variant_enums(self):
        self.assertEqual(FUSION_TYPES, ['none', 'corrected', 'plain', 'literal'])

    def test_half_scale_16ch_shapes(self):
        cases = [
            (1, 16, 256, 256),
            (4, 16, 256, 256),
            (1, 16, 127, 129),
            (3, 16, 96, 160),
        ]
        for name, cls in MODULE_CLASSES.items():
            module = cls(16)
            for shape in cases:
                b = torch.randn(*shape)
                a = torch.randn(*shape)
                with torch.no_grad():
                    out = module(b, a)
                self.assertEqual(tuple(out.shape), shape, (name, shape))
                self.assertTrue(torch.isfinite(out).all(), (name, shape))

    def test_outputs_and_gradients_finite(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(16)
            module.output_gate.data.fill_(1.0)
            b = torch.randn(2, 16, 32, 40)
            a = torch.randn(2, 16, 32, 40)
            out = module(b, a)
            self.assertTrue(torch.isfinite(out).all(), name)
            out.sum().backward()
            for param_name, param in module.named_parameters():
                self.assertIsNotNone(param.grad, (name, param_name))
                self.assertTrue(torch.isfinite(param.grad).all(), (name, param_name))

    def test_zero_gate_keeps_upstream_gradients_zero_until_gate_moves(self):
        module = CrossGatedFusion(16)
        out = module(torch.randn(1, 16, 16, 16), torch.randn(1, 16, 16, 16))
        out.sum().backward()
        self.assertTrue(torch.isfinite(module.output_gate.grad))
        self.assertEqual(float(module.project.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(module.gate_a.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(module.gate_b.weight.grad.abs().sum()), 0.0)


class HalfScaleBridgeTests(unittest.TestCase):
    def test_full_model_fuses_16ch_at_256_for_512_input(self):
        torch.manual_seed(0)
        model = get_arch('wnet', cross_gate_fusion='corrected')
        seen = []
        model.fusion_modules['half'].register_forward_hook(
            lambda mod, inputs, output: seen.append(tuple(inputs[1].shape)))
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(out.shape), (1, 1, 512, 512))
        self.assertEqual(seen, [(1, 16, 256, 256)])

    def test_full_model_batch_and_rectangular(self):
        for batch, height, width in [(1, 64, 64), (4, 64, 64), (1, 68, 52)]:
            torch.manual_seed(0)
            model = get_arch('wnet', cross_gate_fusion='corrected')
            model.eval()
            model.mode = 'eval'
            with torch.no_grad():
                out = model(torch.randn(batch, 3, height, width))
            self.assertEqual(tuple(out.shape), (batch, 1, height, width))

    def test_full_model_gradients_finite(self):
        torch.manual_seed(0)
        model = get_arch('wnet', cross_gate_fusion='corrected')
        model.train()
        model.fusion_modules['half'].output_gate.data.fill_(1.0)
        x1, x2 = model(_make_input(batch=2, size=64))
        (x1.sum() + x2.sum()).backward()
        for name, param in model.fusion_modules['half'].named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_train_returns_tuple_and_eval_returns_tensor(self):
        model = get_arch('wnet', cross_gate_fusion='corrected')
        model.mode = 'train'
        self.assertIsInstance(model(_make_input()), tuple)
        model.mode = 'eval'
        self.assertIsInstance(model(_make_input()), torch.Tensor)


class ZeroInitIdentityTests(unittest.TestCase):
    def test_module_identity_at_init(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(16)
            b = torch.randn(2, 16, 32, 32)
            a = torch.randn(2, 16, 32, 32)
            with torch.no_grad():
                out = module(b, a)
            self.assertTrue(torch.equal(out, b), name)
            self.assertEqual(float(module.output_gate.detach()), 0.0, name)

    def _full_model_parity(self, variant, bridge):
        torch.manual_seed(0)
        base = get_arch('wnet', cross_stage_bridge=bridge)
        fused = get_arch('wnet', cross_stage_bridge=bridge,
                         cross_gate_fusion=variant)
        fused.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        fused.eval()
        base.mode = 'eval'
        fused.mode = 'eval'
        x = _make_input()
        with torch.no_grad():
            self.assertTrue(torch.equal(base(x), fused(x)), (variant, bridge))

    def test_full_model_eval_parity_all_variants(self):
        for variant in ('corrected', 'plain', 'literal'):
            self._full_model_parity(variant, 'none')

    def test_full_model_eval_parity_with_a1_bridge(self):
        for variant in ('corrected', 'plain', 'literal'):
            self._full_model_parity(variant, 'scalar')

    def test_structural_saliency_train_parity(self):
        torch.manual_seed(0)
        base = get_arch('wnet', structural_saliency=True)
        fused = get_arch('wnet', structural_saliency=True,
                         cross_gate_fusion='corrected')
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


class ParameterCountTests(unittest.TestCase):
    def test_module_counts_exact(self):
        self.assertEqual(_num_params(CrossGatedFusion(16)), 4945)
        self.assertEqual(_num_params(PlainConcatFusion(16)), 4625)
        self.assertEqual(_num_params(CFFLiteralFusion(16)), 4945)

    def test_corrected_and_literal_are_param_identical(self):
        self.assertEqual(_num_params(CrossGatedFusion(16)),
                         _num_params(CFFLiteralFusion(16)))

    def test_plain_control_is_matched_budget(self):
        corrected = _num_params(CrossGatedFusion(16))
        plain = _num_params(PlainConcatFusion(16))
        self.assertEqual(corrected - plain, 320)
        self.assertLessEqual(abs(corrected - plain), 0.10 * corrected)

    def test_full_model_counts_exact(self):
        self.assertEqual(_num_params(get_arch('wnet')), 68482)
        self.assertEqual(_num_params(get_arch('wnet', cross_gate_fusion='corrected')), 73427)
        self.assertEqual(_num_params(get_arch('wnet', cross_gate_fusion='plain')), 73107)
        self.assertEqual(_num_params(get_arch('wnet', cross_gate_fusion='literal')), 73427)

    def test_full_model_counts_with_a1_bridge(self):
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='scalar',
                                               cross_gate_fusion='corrected')), 73429)
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='scalar',
                                               cross_gate_fusion='plain')), 73109)


class OffParityTests(unittest.TestCase):
    def test_default_state_dict_keys_unchanged(self):
        default = get_arch('wnet')
        explicit_off = get_arch('wnet', cross_gate_fusion='none')
        self.assertEqual(list(default.state_dict().keys()),
                         list(explicit_off.state_dict().keys()))
        self.assertFalse(any('fusion' in k for k in default.state_dict()))

    def test_no_extra_rng_draws_when_off(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', cross_gate_fusion='none')
        expected = default.state_dict()
        actual = explicit_off.state_dict()
        self.assertEqual(list(expected.keys()), list(actual.keys()))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_enabling_fusion_does_not_disturb_existing_init(self):
        torch.manual_seed(0)
        base = get_arch('wnet')
        torch.manual_seed(0)
        fused = get_arch('wnet', cross_gate_fusion='corrected')
        base_state = base.state_dict()
        fused_state = fused.state_dict()
        for key in base_state:
            self.assertTrue(torch.equal(base_state[key], fused_state[key]), key)

    def test_enabling_fusion_does_not_shift_downstream_rng_stream(self):
        for variant in ('corrected', 'plain', 'literal'):
            torch.manual_seed(0)
            get_arch('wnet')
            expected = torch.rand(16)
            torch.manual_seed(0)
            get_arch('wnet', cross_gate_fusion=variant)
            actual = torch.rand(16)
            self.assertTrue(torch.equal(expected, actual), variant)

    def test_off_forward_parity(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', cross_gate_fusion='none')
        x = _make_input()
        b1, b2 = default(x)
        o1, o2 = explicit_off(x)
        self.assertTrue(torch.equal(b1, o1))
        self.assertTrue(torch.equal(b2, o2))

    def test_get_arch_options_default_off(self):
        self.assertEqual(get_arch_options({})['cross_gate_fusion'], 'none')
        self.assertEqual(get_arch_options({'cross_gate_fusion': 'corrected'})
                         ['cross_gate_fusion'], 'corrected')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_strict_loads_unchanged(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        self.assertEqual(model.cross_gate_fusion, 'none')
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
    def test_reference_checkpoint_warm_starts_fusion_at_identity(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        off_model = get_arch_from_config(dict(cfg), in_c=cfg.get('in_c', 3), n_classes=1)
        off_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        warm_cfg = dict(cfg)
        warm_cfg['cross_gate_fusion'] = 'corrected'
        warm_model = get_arch_from_config(warm_cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        warm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        off_model.mode = 'eval'
        warm_model.mode = 'eval'
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(off_model(x), warm_model(x)))

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_rejects_enabled_fusion(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        cfg['cross_gate_fusion'] = 'corrected'
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        with self.assertRaises(RuntimeError):
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)


class ConfigRoundTripTests(unittest.TestCase):
    def _cfg(self, variant, **extra):
        cfg = {'model_name': 'wnet', 'cross_gate_fusion': variant,
               'structural_saliency': True}
        cfg.update(extra)
        return cfg

    def test_each_variant_round_trips(self):
        for variant in ('corrected', 'plain', 'literal'):
            model = get_arch_from_config(self._cfg(variant), in_c=3, n_classes=1)
            self.assertEqual(model.cross_gate_fusion, variant)
            self.assertIn('fusion_modules.half.output_gate', model.state_dict())
            model.mode = 'eval'
            with torch.no_grad():
                out = model(torch.randn(1, 3, 64, 64))
            self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    def test_old_config_without_key_stays_off(self):
        cfg = {'model_name': 'wnet', 'structural_saliency': True}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.cross_gate_fusion, 'none')
        self.assertFalse(any('fusion' in k for k in model.state_dict()))

    def test_reference_family_plus_fusion_round_trips(self):
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0, 'structural_saliency': True,
               'cross_gate_fusion': 'corrected'}
        torch.manual_seed(0)
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.cross_gate_fusion, 'corrected')
        self.assertIn('bridge_gates.half', model.state_dict())
        self.assertIn('fusion_modules.half.project.weight', model.state_dict())
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
            validate_cross_gate_fusion('bogus', 'wnet', 'unet')
        with self.assertRaises(ValueError):
            get_arch('unet', cross_gate_fusion='corrected')
        with self.assertRaises(ValueError):
            get_arch('big_unet', cross_gate_fusion='corrected')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', cross_gate_fusion='corrected')
        with self.assertRaises(ValueError):
            get_arch('wnet', fr_lite=True, cross_gate_fusion='corrected')

    def test_big_wnet_round_trips(self):
        cfg = {'model_name': 'big_wnet', 'cross_gate_fusion': 'corrected'}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class DecoderSeamTests(unittest.TestCase):
    def test_decoder_fusion_length_mismatch(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        with self.assertRaises(ValueError):
            model(_make_input(), decoder_fusions=(None,),
                  decoder_fusion_inputs=(None,))

    def test_decoder_fusion_inputs_are_required(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        with self.assertRaises(ValueError):
            model(_make_input(), decoder_fusions=(None, None))

    def test_decoder_fusion_shape_mismatch(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        bad = torch.zeros(1, 16, 8, 8)
        fusion = CrossGatedFusion(16)
        with self.assertRaises(RuntimeError):
            model(_make_input(), decoder_fusions=(fusion, None),
                  decoder_fusion_inputs=(bad, None))

    def test_decoder_fusion_applies_at_selected_stage_only(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        fusion = PlainConcatFusion(16)
        fusion.output_gate.data.fill_(0.0)
        feature = torch.randn(1, 16, 32, 32)
        out = model(_make_input(), decoder_fusions=(fusion, None),
                    decoder_fusion_inputs=(feature, None))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


if __name__ == '__main__':
    unittest.main()
