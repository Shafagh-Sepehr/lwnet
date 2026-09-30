import math
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.get_model import get_arch, get_arch_options, set_eval_mode, validate_cross_stage_bridge
from models.res_unet_adrian import UNet
from utils.model_saving_loading import save_model, load_model


def _num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_input(in_c=3, size=64, batch=1):
    return torch.randn(batch, in_c, size, size)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_default_wnet_has_none_bridge(self):
        m = get_arch('wnet')
        self.assertEqual(m.cross_stage_bridge, 'none')

    def test_default_wnet_state_dict_keys_match_baseline(self):
        m = get_arch('wnet')
        keys = list(m.state_dict().keys())
        self.assertFalse(any('bridge_gates' in k for k in keys))
        self.assertTrue(any(k.startswith('unet1.') for k in keys))
        self.assertTrue(any(k.startswith('unet2.') for k in keys))

    def test_default_wnet_param_count(self):
        self.assertEqual(_num_params(get_arch('wnet')), 68482)

    def test_baseline_checkpoint_loads_strict(self):
        base = get_arch('wnet')
        fresh = get_arch('wnet')
        fresh.load_state_dict(base.state_dict(), strict=True)

    def test_old_config_resolves_to_defaults(self):
        opts = get_arch_options({})
        self.assertEqual(opts['cross_stage_bridge'], 'none')
        self.assertEqual(opts['cross_stage_bridge_scales'], 'all')
        self.assertEqual(opts['cross_stage_bridge_init'], 0.0)

    def test_plain_unet_returns_logits_tensor(self):
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        out = u(_make_input())
        self.assertIsInstance(out, torch.Tensor)


class ParameterCountTests(unittest.TestCase):
    def test_scalar_all(self):
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='scalar')), 68484)

    def test_scalar_full(self):
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='scalar',
                                              cross_stage_bridge_scales='full')), 68483)

    def test_scalar_half(self):
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='scalar',
                                              cross_stage_bridge_scales='half')), 68483)

    def test_channel_all(self):
        self.assertEqual(_num_params(get_arch('wnet', cross_stage_bridge='channel')), 68506)

    def test_big_wnet_scalar_delta(self):
        base = _num_params(get_arch('big_wnet'))
        scalar = _num_params(get_arch('big_wnet', cross_stage_bridge='scalar'))
        self.assertEqual(scalar - base, 3)

    def test_big_wnet_channel_delta(self):
        base = _num_params(get_arch('big_wnet'))
        channel = _num_params(get_arch('big_wnet', cross_stage_bridge='channel'))
        self.assertEqual(channel - base, 56)


class ZeroGateParityTests(unittest.TestCase):
    def _check_parity(self, bridge):
        torch.manual_seed(0)
        base = get_arch('wnet')
        bridged = get_arch('wnet', cross_stage_bridge=bridge)
        bridged.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        bridged.eval()
        x = _make_input()
        with torch.no_grad():
            b1, b2 = base(x)
            c1, c2 = bridged(x)
        self.assertTrue(torch.equal(b1, c1))
        self.assertTrue(torch.equal(b2, c2))
        # inference mode
        base.mode = 'eval'
        bridged.mode = 'eval'
        with torch.no_grad():
            o1 = base(x)
            o2 = bridged(x)
        self.assertTrue(torch.equal(o1, o2))

    def test_scalar_zero_gate_parity(self):
        self._check_parity('scalar')

    def test_channel_zero_gate_parity(self):
        self._check_parity('channel')


class FunctionalBridgeTests(unittest.TestCase):
    def test_scalar_gate_changes_u2_only(self):
        torch.manual_seed(0)
        base = get_arch('wnet')
        m = get_arch('wnet', cross_stage_bridge='scalar')
        m.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        m.eval()
        x = _make_input()
        with torch.no_grad():
            m.bridge_gates['full'].fill_(1.0)
            m.bridge_gates['half'].fill_(0.0)
            o1, o2 = m(x)
            b1, b2 = base(x)
        self.assertTrue(torch.equal(o1, b1))
        self.assertFalse(torch.equal(o2, b2))

    def test_full_only_has_no_half_gate(self):
        m = get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='full')
        keys = list(m.state_dict().keys())
        self.assertIn('bridge_gates.full', keys)
        self.assertNotIn('bridge_gates.half', keys)

    def test_half_only_has_no_full_gate(self):
        m = get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='half')
        keys = list(m.state_dict().keys())
        self.assertIn('bridge_gates.half', keys)
        self.assertNotIn('bridge_gates.full', keys)

    def test_gate_shapes(self):
        scalar = get_arch('wnet', cross_stage_bridge='scalar')
        self.assertEqual(scalar.bridge_gates['half'].shape, torch.Size([]))
        self.assertEqual(scalar.bridge_gates['full'].shape, torch.Size([]))
        channel = get_arch('wnet', cross_stage_bridge='channel')
        self.assertEqual(channel.bridge_gates['half'].shape, torch.Size([1, 16, 1, 1]))
        self.assertEqual(channel.bridge_gates['full'].shape, torch.Size([1, 8, 1, 1]))

    def test_backward_produces_finite_gate_grads(self):
        torch.manual_seed(0)
        m = get_arch('wnet', cross_stage_bridge='scalar')
        m.train()
        x = _make_input()
        x1, x2 = m(x)
        loss = x2.sum()
        loss.backward()
        for name, gate in m.bridge_gates.items():
            self.assertIsNotNone(gate.grad)
            self.assertTrue(torch.isfinite(gate.grad).all(), name)

    def test_optimizer_step_changes_zero_gate(self):
        torch.manual_seed(0)
        m = get_arch('wnet', cross_stage_bridge='scalar')
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        x = _make_input()
        x1, x2 = m(x)
        loss = x2.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        changed = any(float(g.item()) != 0.0 for g in m.bridge_gates.values())
        self.assertTrue(changed)

    def test_bridge_features_not_detached(self):
        torch.manual_seed(0)
        m = get_arch('wnet', cross_stage_bridge='scalar')
        m.train()
        x = _make_input()
        x1, feats = m.unet1(x, return_decoder_features=True)
        for f in feats:
            self.assertTrue(f.requires_grad)

    def test_train_returns_tuple_inference_returns_tensor(self):
        m = get_arch('wnet', cross_stage_bridge='scalar')
        x = _make_input()
        m.mode = 'train'
        out = m(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        m.mode = 'eval'
        out = m(x)
        self.assertIsInstance(out, torch.Tensor)

    def test_binary_and_four_class_forward_backward(self):
        torch.manual_seed(0)
        for name, n_classes, in_c in [('wnet', 1, 3), ('big_wnet', 4, 3)]:
            m = get_arch(name, n_classes=n_classes, cross_stage_bridge='scalar')
            m.train()
            x = _make_input(in_c=in_c)
            x1, x2 = m(x)
            self.assertEqual(x2.shape[1], n_classes)
            (x1.sum() + x2.sum()).backward()
            for p in m.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())


class ValidationErrorTests(unittest.TestCase):
    def test_quarter_on_wnet(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='quarter')

    def test_bridge_on_unet(self):
        with self.assertRaises(ValueError):
            get_arch('unet', cross_stage_bridge='scalar')

    def test_bridge_on_big_unet(self):
        with self.assertRaises(ValueError):
            get_arch('big_unet', cross_stage_bridge='channel')

    def test_duplicate_scales(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='full,full')

    def test_all_combined(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='all,full')

    def test_blank_scale(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='half,,full')

    def test_unknown_scale(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='bogus')

    def test_nan_init(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_init=float('nan'))

    def test_inf_init(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_init=float('inf'))

    def test_none_with_nondefault_scales(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='none', cross_stage_bridge_scales='full')

    def test_none_with_nonzero_init(self):
        with self.assertRaises(ValueError):
            get_arch('wnet', cross_stage_bridge='none', cross_stage_bridge_init=0.5)

    def test_wrong_number_of_decoder_additions(self):
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        with self.assertRaises(ValueError):
            u(_make_input(), decoder_additions=(None,))

    def test_bridge_shape_mismatch(self):
        u = UNet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        bad = torch.zeros(1, 16, 16, 16)  # wrong spatial size for stage 0
        with self.assertRaises(RuntimeError):
            u(_make_input(), decoder_additions=(bad, None))

    def test_load_bridge_checkpoint_with_different_scales(self):
        torch.manual_seed(0)
        a = get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='full')
        b = get_arch('wnet', cross_stage_bridge='scalar', cross_stage_bridge_scales='half')
        with self.assertRaises(RuntimeError):
            b.load_state_dict(a.state_dict(), strict=True)

    def test_load_bridge_checkpoint_with_different_type(self):
        torch.manual_seed(0)
        a = get_arch('wnet', cross_stage_bridge='scalar')
        b = get_arch('wnet', cross_stage_bridge='channel')
        with self.assertRaises(RuntimeError):
            b.load_state_dict(a.state_dict(), strict=True)


class SaveLoadRoundTripTests(unittest.TestCase):
    def _round_trip(self, bridge, scales='all'):
        torch.manual_seed(0)
        m = get_arch('wnet', cross_stage_bridge=bridge, cross_stage_bridge_scales=scales)
        for name, gate in m.bridge_gates.items():
            gate.data.fill_(0.7)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(m.parameters(), lr=0.01)
            save_model(d, m, opt, stats={'bridge_gates': m.bridge_gate_summary()})
            m2 = get_arch('wnet', cross_stage_bridge=bridge, cross_stage_bridge_scales=scales)
            m2, stats = load_model(m2, d, device='cpu')
        self.assertEqual(list(m.state_dict().keys()), list(m2.state_dict().keys()))
        for k in m.state_dict():
            self.assertTrue(torch.equal(m.state_dict()[k], m2.state_dict()[k]), k)
        x = _make_input()
        m.eval()
        m2.eval()
        m.mode = 'eval'
        m2.mode = 'eval'
        with torch.no_grad():
            o1 = m(x)
            o2 = m2(x)
        self.assertTrue(torch.equal(o1, o2))

    def test_scalar_round_trip(self):
        self._round_trip('scalar')

    def test_channel_round_trip(self):
        self._round_trip('channel')


class InferenceReconstructionTests(unittest.TestCase):
    # Simulates the config-driven reconstruction performed by the inference
    # scripts (generate_results.py, generate_av_results.py, predict_one_image.py,
    # predict_one_image_av.py): recover arch options from a config dict, build the
    # model, set eval mode, and load the checkpoint strictly.

    def _reconstruct(self, cfg, n_classes=1, in_c=3):
        model_name = cfg.get('model_name', 'wnet')
        arch_opts = get_arch_options(cfg)
        model = get_arch(model_name, in_c=in_c, n_classes=n_classes, **arch_opts)
        return set_eval_mode(model)

    def test_big_wnet_binary_returns_tensor_after_eval_mode(self):
        cfg = {'model_name': 'big_wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['quarter', 'half', 'full'],
               'cross_stage_bridge_init': 0.0}
        model = self._reconstruct(cfg, n_classes=1)
        out = model(_make_input())
        self.assertIsInstance(out, torch.Tensor)

    def test_wnet_binary_returns_tensor_after_eval_mode(self):
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0}
        model = self._reconstruct(cfg, n_classes=1)
        out = model(_make_input())
        self.assertIsInstance(out, torch.Tensor)

    def test_old_config_reconstructs_no_bridge_model(self):
        cfg = {'model_name': 'wnet'}  # no bridge fields
        model = self._reconstruct(cfg, n_classes=1)
        self.assertEqual(model.cross_stage_bridge, 'none')
        self.assertFalse(any('bridge_gates' in k for k in model.state_dict()))

    def test_bridge_config_reconstructs_bridge_model(self):
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'channel',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0}
        model = self._reconstruct(cfg, n_classes=1)
        self.assertEqual(model.cross_stage_bridge, 'channel')
        self.assertIn('bridge_gates.half', model.state_dict())
        self.assertIn('bridge_gates.full', model.state_dict())

    def test_config_checkpoint_round_trip_strict(self):
        torch.manual_seed(0)
        cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
               'cross_stage_bridge_scales': ['half', 'full'],
               'cross_stage_bridge_init': 0.0}
        model = self._reconstruct(cfg, n_classes=1)
        for name, gate in model.bridge_gates.items():
            gate.data.fill_(0.3)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(d, model, opt)
            model2 = self._reconstruct(cfg, n_classes=1)
            model2, stats = load_model(model2, d, device='cpu')
        self.assertEqual(list(model.state_dict().keys()), list(model2.state_dict().keys()))

    def test_config_mismatch_fails_strict_load(self):
        torch.manual_seed(0)
        bridge_cfg = {'model_name': 'wnet', 'cross_stage_bridge': 'scalar',
                      'cross_stage_bridge_scales': ['half', 'full'],
                      'cross_stage_bridge_init': 0.0}
        model = self._reconstruct(bridge_cfg, n_classes=1)
        with tempfile.TemporaryDirectory() as d:
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            save_model(d, model, opt)
            # Reconstruct a no-bridge model from an old config and try to load
            # the bridge checkpoint strictly: must fail, not silently predict.
            no_bridge = self._reconstruct({'model_name': 'wnet'}, n_classes=1)
            with self.assertRaises(RuntimeError):
                load_model(no_bridge, d, device='cpu')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class AmpCudaTests(unittest.TestCase):
    def test_amp_forward_backward_step(self):
        torch.manual_seed(0)
        m = get_arch('wnet', cross_stage_bridge='scalar').cuda()
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        x = _make_input().cuda()
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        with torch.cuda.amp.autocast(enabled=True):
            x1, x2 = m(x)
            loss = x2.sum()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        self.assertTrue(torch.isfinite(loss))
        for name, gate in m.bridge_gates.items():
            self.assertTrue(torch.isfinite(gate).all(), name)


if __name__ == '__main__':
    unittest.main()
