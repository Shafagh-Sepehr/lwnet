"""Tests for the opt-in U2 decoder C2-corrected prototype gate and controls.

Label: C2-corrected -- FDPR prose intent (recalibration incl. suppression).
FDPR-literal MPCR is amplification-only and is not implemented.

Covers the corrected gain (0, 2), the amplification-only SE control (1, 2), and
the parameter-matched plain convolution control. All variants are default-off;
the disabled path must stay identical to the retained reference in state-dict
keys, parameter count, RNG use, and forward behavior. Every variant is identity
at initialization.
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

from models.get_model import (  # noqa: E402
    get_arch, get_arch_from_config, get_arch_options, validate_proto_gate,
)
from models.proto_gate import (  # noqa: E402
    CorrectedProtoGate, PROTO_GATE_TYPES, PlainProtoControl, SEProtoGate,
)
from models.res_unet_adrian import UNet  # noqa: E402
from utils.model_saving_loading import save_model, load_model  # noqa: E402

REFERENCE_RUN = os.path.join(
    os.path.dirname(REPO_ROOT), 'lwnet', 'experiments',
    'final_t4_src_ss_raffeSmooth_scalar_sal_bce_d025_bnd001_c20x50_ck20_seed0')

MODULE_CLASSES = {
    'corrected': CorrectedProtoGate,
    'se': SEProtoGate,
    'plain': PlainProtoControl,
}

CORRECTED_16 = 16 * 16 + 16
CORRECTED_8 = 8 * 8 + 8
CORRECTED_TOTAL = CORRECTED_16 + CORRECTED_8


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


def _make_input(batch=1, in_c=3, height=64, width=None):
    if width is None:
        width = height
    return torch.randn(batch, in_c, height, width)


def _randomize_gate(module):
    with torch.no_grad():
        if hasattr(module, 'prototype'):
            module.prototype.weight.normal_(0.0, 0.5)
            module.prototype.bias.normal_(0.0, 0.5)
        if hasattr(module, 'project'):
            module.project.weight.normal_(0.0, 0.5)
            module.project.bias.normal_(0.0, 0.5)
        if hasattr(module, 'gate_scale'):
            module.gate_scale.fill_(1.0)


class ModuleShapeTests(unittest.TestCase):
    def test_variant_enums(self):
        self.assertEqual(PROTO_GATE_TYPES, ['none', 'corrected', 'se', 'plain'])

    def test_stage_shapes_16ch_and_8ch(self):
        cases = [
            (1, 16, 64, 64),
            (4, 16, 64, 64),
            (1, 16, 33, 41),
            (4, 16, 32, 40),
            (1, 8, 128, 128),
            (4, 8, 64, 64),
            (1, 8, 37, 53),
        ]
        for name, cls in MODULE_CLASSES.items():
            for channels in (16, 8):
                module = cls(channels)
                for batch, case_channels, height, width in cases:
                    if case_channels != channels:
                        continue
                    x = torch.randn(batch, channels, height, width)
                    with torch.no_grad():
                        out = module(x)
                    self.assertEqual(tuple(out.shape), tuple(x.shape),
                                     (name, channels, (batch, height, width)))
                    self.assertTrue(torch.isfinite(out).all(),
                                    (name, channels, (batch, height, width)))

    def test_outputs_and_gradients_finite(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(16)
            _randomize_gate(module)
            x = torch.randn(2, 16, 32, 40)
            out = module(x)
            self.assertTrue(torch.isfinite(out).all(), name)
            out.sum().backward()
            for param_name, param in module.named_parameters():
                self.assertIsNotNone(param.grad, (name, param_name))
                self.assertTrue(torch.isfinite(param.grad).all(), (name, param_name))

    def test_gain_ranges(self):
        torch.manual_seed(0)
        corrected = CorrectedProtoGate(16)
        _randomize_gate(corrected)
        x = torch.randn(2, 16, 32, 32)
        corrected_gain = corrected.channel_gain(x)
        self.assertGreater(float(corrected_gain.min()), 0.0)
        self.assertLess(float(corrected_gain.min()), 1.0)
        self.assertGreater(float(corrected_gain.max()), 1.0)
        self.assertLess(float(corrected_gain.max()), 2.0)

        se = SEProtoGate(16)
        _randomize_gate(se)
        se_gain = se.channel_gain(x)
        self.assertGreater(float(se_gain.min()), 1.0)
        self.assertLess(float(se_gain.max()), 2.0)

    def test_init_gain_is_identity(self):
        corrected = CorrectedProtoGate(16)
        se = SEProtoGate(16)
        x = torch.randn(1, 16, 32, 32)
        with torch.no_grad():
            self.assertEqual(float(corrected.channel_gain(x).min()), 1.0)
            self.assertEqual(float(corrected.channel_gain(x).max()), 1.0)
            self.assertEqual(float(se.channel_gain(x).min()), 1.0)
            self.assertEqual(float(se.channel_gain(x).max()), 1.0)

    def test_module_identity_at_init(self):
        for name, cls in MODULE_CLASSES.items():
            module = cls(16)
            x = torch.randn(2, 16, 32, 32)
            with torch.no_grad():
                out = module(x)
            self.assertTrue(torch.equal(out, x), name)


class FullModelStageTests(unittest.TestCase):
    def test_gates_apply_at_half_16ch_and_full_8ch_only(self):
        torch.manual_seed(0)
        model = get_arch('wnet', proto_gate='corrected')
        seen = []
        for scale_name, gate in model.proto_gate_modules.items():
            gate.register_forward_hook(
                lambda mod, inputs, output, scale=scale_name: seen.append(
                    (scale, tuple(inputs[0].shape))))
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(_make_input())
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))
        self.assertEqual(seen, [('half', (1, 16, 32, 32)),
                                ('full', (1, 8, 64, 64))])

    def test_full_model_batch_and_rectangular(self):
        for batch, height, width in [(1, 64, 64), (4, 64, 64), (1, 68, 52)]:
            torch.manual_seed(0)
            model = get_arch('wnet', proto_gate='corrected')
            model.eval()
            model.mode = 'eval'
            with torch.no_grad():
                out = model(torch.randn(batch, 3, height, width))
            self.assertEqual(tuple(out.shape), (batch, 1, height, width))

    def test_full_model_gradients_finite(self):
        for variant in ('corrected', 'se', 'plain'):
            torch.manual_seed(0)
            model = get_arch('wnet', proto_gate=variant)
            for _, gate in model.proto_gate_modules.items():
                _randomize_gate(gate)
            model.train()
            x1, x2 = model(_make_input(batch=2))
            (x1.sum() + x2.sum()).backward()
            for name, param in model.proto_gate_modules.named_parameters():
                self.assertIsNotNone(param.grad, (variant, name))
                self.assertTrue(torch.isfinite(param.grad).all(), (variant, name))
            for name, param in model.proto_gate_modules.named_parameters():
                if param.numel() > 1:
                    self.assertGreater(float(param.grad.abs().sum()), 0.0,
                                       (variant, name))

    def test_combines_with_a1_bridge_and_cross_gate_fusion(self):
        torch.manual_seed(0)
        model = get_arch('wnet', cross_stage_bridge='scalar',
                         cross_gate_fusion='corrected', proto_gate='corrected')
        model.eval()
        model.mode = 'eval'
        with torch.no_grad():
            out = model(_make_input())
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(out).all())


class InitIdentityAndOffParityTests(unittest.TestCase):
    def _full_model_parity(self, variant, bridge):
        torch.manual_seed(0)
        base = get_arch('wnet', cross_stage_bridge=bridge)
        gated = get_arch('wnet', cross_stage_bridge=bridge, proto_gate=variant)
        gated.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        gated.eval()
        base.mode = 'eval'
        gated.mode = 'eval'
        x = _make_input()
        with torch.no_grad():
            self.assertTrue(torch.equal(base(x), gated(x)), (variant, bridge))

    def test_full_model_eval_parity_all_variants(self):
        for variant in ('corrected', 'se', 'plain'):
            self._full_model_parity(variant, 'none')

    def test_full_model_eval_parity_with_a1_bridge(self):
        for variant in ('corrected', 'se', 'plain'):
            self._full_model_parity(variant, 'scalar')

    def test_structural_saliency_train_parity(self):
        torch.manual_seed(0)
        base = get_arch('wnet', structural_saliency=True)
        gated = get_arch('wnet', structural_saliency=True, proto_gate='corrected')
        gated.load_state_dict(base.state_dict(), strict=False)
        base.train()
        gated.train()
        base.mode = 'train'
        gated.mode = 'train'
        x = _make_input()
        b1, b2, bs = base(x)
        g1, g2, gs = gated(x)
        self.assertTrue(torch.equal(b1, g1))
        self.assertTrue(torch.equal(b2, g2))
        self.assertTrue(torch.equal(bs, gs))

    def test_default_state_dict_keys_unchanged(self):
        default = get_arch('wnet')
        explicit_off = get_arch('wnet', proto_gate='none')
        self.assertEqual(list(default.state_dict().keys()),
                         list(explicit_off.state_dict().keys()))
        self.assertFalse(any('proto_gate' in k for k in default.state_dict()))

    def test_no_extra_rng_draws_when_off(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', proto_gate='none')
        expected = default.state_dict()
        actual = explicit_off.state_dict()
        self.assertEqual(list(expected.keys()), list(actual.keys()))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_enabling_gate_does_not_disturb_existing_init(self):
        torch.manual_seed(0)
        base = get_arch('wnet')
        torch.manual_seed(0)
        gated = get_arch('wnet', proto_gate='corrected')
        base_state = base.state_dict()
        gated_state = gated.state_dict()
        for key in base_state:
            self.assertTrue(torch.equal(base_state[key], gated_state[key]), key)

    def test_off_forward_parity(self):
        torch.manual_seed(0)
        default = get_arch('wnet')
        torch.manual_seed(0)
        explicit_off = get_arch('wnet', proto_gate='none')
        x = _make_input()
        b1, b2 = default(x)
        o1, o2 = explicit_off(x)
        self.assertTrue(torch.equal(b1, o1))
        self.assertTrue(torch.equal(b2, o2))

    def test_get_arch_options_default_off(self):
        self.assertEqual(get_arch_options({})['proto_gate'], 'none')
        self.assertEqual(get_arch_options({'proto_gate': 'corrected'})['proto_gate'],
                         'corrected')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_strict_loads_unchanged(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        self.assertEqual(model.proto_gate, 'none')
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
    def test_reference_checkpoint_warm_starts_gate_at_identity(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        off_model = get_arch_from_config(dict(cfg), in_c=cfg.get('in_c', 3), n_classes=1)
        off_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        x = torch.randn(1, 3, 64, 64)
        off_model.mode = 'eval'
        with torch.no_grad():
            reference_out = off_model(x)
        for variant in ('corrected', 'se', 'plain'):
            warm_cfg = dict(cfg)
            warm_cfg['proto_gate'] = variant
            warm_model = get_arch_from_config(
                warm_cfg, in_c=cfg.get('in_c', 3), n_classes=1)
            warm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            warm_model.mode = 'eval'
            with torch.no_grad():
                self.assertTrue(torch.equal(reference_out, warm_model(x)), variant)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth')),
        'reference checkpoint not present on this machine')
    def test_reference_checkpoint_rejects_enabled_gate(self):
        with open(os.path.join(REFERENCE_RUN, 'config.cfg')) as handle:
            cfg = json.load(handle)
        cfg['proto_gate'] = 'corrected'
        model = get_arch_from_config(cfg, in_c=cfg.get('in_c', 3), n_classes=1)
        checkpoint = torch.load(os.path.join(REFERENCE_RUN, 'model_checkpoint.pth'),
                                map_location='cpu')
        with self.assertRaises(RuntimeError):
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)


class ParameterCountTests(unittest.TestCase):
    def test_module_counts_exact(self):
        self.assertEqual(_num_params(CorrectedProtoGate(16)), CORRECTED_16)
        self.assertEqual(_num_params(CorrectedProtoGate(8)), CORRECTED_8)
        self.assertEqual(_num_params(SEProtoGate(16)), CORRECTED_16 + 1)
        self.assertEqual(_num_params(SEProtoGate(8)), CORRECTED_8 + 1)
        self.assertEqual(_num_params(PlainProtoControl(16)), CORRECTED_16)
        self.assertEqual(_num_params(PlainProtoControl(8)), CORRECTED_8)

    def test_full_model_counts_exact(self):
        self.assertEqual(_num_params(get_arch('wnet')), 68482)
        self.assertEqual(_num_params(get_arch('wnet', proto_gate='corrected')),
                         68482 + CORRECTED_TOTAL)
        self.assertEqual(_num_params(get_arch('wnet', proto_gate='plain')),
                         68482 + CORRECTED_TOTAL)
        self.assertEqual(_num_params(get_arch('wnet', proto_gate='se')),
                         68482 + CORRECTED_TOTAL + 2)

    def test_variants_are_closely_matched(self):
        counts = [
            _num_params(get_arch('wnet', proto_gate=variant))
            for variant in ('corrected', 'se', 'plain')
        ]
        self.assertLessEqual(max(counts) - min(counts), 2)
        self.assertLessEqual(max(counts) - min(counts), 0.01 * counts[0])

    def test_combined_count_with_bridge_and_fusion(self):
        count = _num_params(get_arch('wnet', cross_stage_bridge='scalar',
                                     cross_gate_fusion='corrected',
                                     proto_gate='corrected'))
        self.assertEqual(count, 68482 + 2 + 4945 + CORRECTED_TOTAL)

    def test_big_wnet_delta_count(self):
        base = _num_params(get_arch('big_wnet'))
        gated = _num_params(get_arch('big_wnet', proto_gate='corrected'))
        self.assertEqual(gated - base, CORRECTED_TOTAL)


class ConfigRoundTripTests(unittest.TestCase):
    def _cfg(self, variant, **extra):
        cfg = {'model_name': 'wnet', 'proto_gate': variant,
               'structural_saliency': True}
        cfg.update(extra)
        return cfg

    def test_each_variant_round_trips(self):
        for variant in ('corrected', 'se', 'plain'):
            model = get_arch_from_config(self._cfg(variant), in_c=3, n_classes=1)
            self.assertEqual(model.proto_gate, variant)
            self.assertEqual(model.proto_gate_stage_names, {0: 'half', 1: 'full'})
            self.assertTrue(any(k.startswith('proto_gate_modules.half')
                                for k in model.state_dict()))
            model.mode = 'eval'
            with torch.no_grad():
                out = model(torch.randn(1, 3, 64, 64))
            self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    def test_old_config_without_key_stays_off(self):
        cfg = {'model_name': 'wnet', 'structural_saliency': True}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.proto_gate, 'none')
        self.assertFalse(any('proto_gate' in k for k in model.state_dict()))

    def test_reference_family_plus_gate_round_trips(self):
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0, 'structural_saliency': True,
               'proto_gate': 'corrected'}
        torch.manual_seed(0)
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.proto_gate, 'corrected')
        self.assertIn('bridge_gates.half', model.state_dict())
        self.assertIn('proto_gate_modules.full.prototype.weight', model.state_dict())
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
            validate_proto_gate('bogus', 'wnet', 'unet')
        with self.assertRaises(ValueError):
            get_arch('unet', proto_gate='corrected')
        with self.assertRaises(ValueError):
            get_arch('big_unet', proto_gate='corrected')
        with self.assertRaises(ValueError):
            get_arch('wnet', u2_arch='fr_multi', proto_gate='corrected')
        with self.assertRaises(ValueError):
            get_arch('wnet', fr_lite=True, proto_gate='corrected')

    def test_big_wnet_round_trips(self):
        cfg = {'model_name': 'big_wnet', 'proto_gate': 'corrected'}
        model = get_arch_from_config(cfg, in_c=3, n_classes=1)
        self.assertEqual(model.proto_gate_stage_names, {1: 'half', 2: 'full'})
        model.mode = 'eval'
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class DecoderSeamTests(unittest.TestCase):
    def test_decoder_gates_length_mismatch(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        with self.assertRaises(ValueError):
            model(_make_input(), decoder_gates=(None,))

    def test_decoder_gate_applies_at_selected_stage_only(self):
        model = UNet(in_c=3, n_classes=1, layers=[8, 16, 32])
        gate = CorrectedProtoGate(16)
        out = model(_make_input(), decoder_gates=(gate, None))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))


class OptimizerSmokeTests(unittest.TestCase):
    def test_one_step_corrected_train_and_eval(self):
        torch.manual_seed(0)
        model = get_arch('wnet', proto_gate='corrected')
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
        for name, param in model.proto_gate_modules.named_parameters():
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
