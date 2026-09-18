import sys
import math
from .res_unet_adrian import UNet as unet

import torch

# from .res_unet_adrian import WNet as wnet

SCALE_ORDER = ['quarter', 'half', 'full']
BRIDGE_TYPES = ['none', 'scalar', 'channel']


def _supported_scales(model_name):
    if model_name == 'wnet':
        return ['half', 'full']
    if model_name == 'big_wnet':
        return ['quarter', 'half', 'full']
    return []


def validate_cross_stage_bridge(cross_stage_bridge, cross_stage_bridge_scales, cross_stage_bridge_init, model_name):
    if cross_stage_bridge not in BRIDGE_TYPES:
        raise ValueError('cross_stage_bridge must be one of {}, got {!r}'.format(
            BRIDGE_TYPES, cross_stage_bridge))

    init = float(cross_stage_bridge_init)
    if not math.isfinite(init):
        raise ValueError('cross_stage_bridge_init must be finite, got {!r}'.format(cross_stage_bridge_init))

    supported = _supported_scales(model_name)

    if isinstance(cross_stage_bridge_scales, (list, tuple)):
        tokens = [str(t).strip() for t in cross_stage_bridge_scales]
    else:
        raw = str(cross_stage_bridge_scales).strip()
        tokens = [t.strip() for t in raw.split(',')] if raw else []

    if any(t == '' for t in tokens):
        raise ValueError('cross_stage_bridge_scales contains an empty token: {!r}'.format(
            cross_stage_bridge_scales))

    if 'all' in tokens:
        if len(tokens) != 1:
            raise ValueError("'all' cannot be combined with other scales: {!r}".format(
                cross_stage_bridge_scales))
        scales = list(supported)
    else:
        scales = tokens

    if len(set(scales)) != len(scales):
        raise ValueError('duplicate scales in cross_stage_bridge_scales: {!r}'.format(
            cross_stage_bridge_scales))

    for s in scales:
        if s not in SCALE_ORDER:
            raise ValueError('unknown scale {!r} in cross_stage_bridge_scales'.format(s))

    for s in scales:
        if s not in supported:
            raise ValueError('scale {!r} is not supported by model {!r} (supported: {})'.format(
                s, model_name, supported))

    if cross_stage_bridge != 'none' and model_name in ('unet', 'big_unet'):
        raise ValueError('cross-stage bridges are not supported for single U-Net model {!r}'.format(
            model_name))

    if cross_stage_bridge == 'none':
        if scales != supported or init != 0.0:
            raise ValueError(
                "cross_stage_bridge='none' requires the default scales ('all') and init 0.0")

    scales = sorted(scales, key=SCALE_ORDER.index)

    return cross_stage_bridge, scales, init


def get_arch_options(cfg):
    if cfg is None:
        cfg = {}
    if isinstance(cfg, dict):
        get = cfg.get
    else:
        get = lambda k, d=None: getattr(cfg, k, d)
    return {
        'cross_stage_bridge': get('cross_stage_bridge', 'none'),
        'cross_stage_bridge_scales': get('cross_stage_bridge_scales', 'all'),
        'cross_stage_bridge_init': get('cross_stage_bridge_init', 0.0),
    }


def set_eval_mode(model):
    # W-Net models carry a `mode` attribute that switches forward() between the
    # training tuple (x1, x2) and the inference logits tensor. Single U-Net
    # models have no such attribute and already return logits directly.
    if hasattr(model, 'mode'):
        model.mode = 'eval'
    return model


class _BridgeGateContainer(torch.nn.Module):
    # A minimal parameter container that permits the semantic scale name 'half',
    # which collides with the reserved nn.Module.half() method and therefore
    # cannot be used as a plain ParameterDict key. Parameters are registered
    # directly into _parameters so their state-dict names stay predictable
    # (bridge_gates.half, bridge_gates.full, ...).
    def __init__(self):
        super().__init__()
        self._gate_names = []

    def add(self, name, param):
        self._parameters[name] = param
        self._gate_names.append(name)

    def items(self):
        return [(n, self._parameters[n]) for n in self._gate_names]

    def values(self):
        return [self._parameters[n] for n in self._gate_names]

    def __getitem__(self, name):
        return self._parameters[name]

    def __contains__(self, name):
        return name in self._parameters


class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train',
                 cross_stage_bridge='none', cross_stage_bridge_scales='all', cross_stage_bridge_init=0.0):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode = mode
        self.cross_stage_bridge = cross_stage_bridge
        self.cross_stage_bridge_scales = cross_stage_bridge_scales
        self.cross_stage_bridge_init = cross_stage_bridge_init

        decoder_channels = list(reversed(layers))[1:]
        scale_names = SCALE_ORDER[-len(decoder_channels):]

        if isinstance(cross_stage_bridge_scales, (list, tuple)):
            selected = [str(s).strip() for s in cross_stage_bridge_scales]
        else:
            raw = str(cross_stage_bridge_scales).strip()
            if raw == 'all':
                selected = list(scale_names)
            else:
                selected = [s.strip() for s in raw.split(',')] if raw else []

        self.bridge_stage_names = {}
        self.bridge_gates = _BridgeGateContainer()

        if cross_stage_bridge != 'none':
            for scale_name in selected:
                stage_index = scale_names.index(scale_name)
                channels = decoder_channels[stage_index]
                self.bridge_stage_names[stage_index] = scale_name
                if cross_stage_bridge == 'scalar':
                    self.bridge_gates.add(scale_name, torch.nn.Parameter(
                        torch.full((), float(cross_stage_bridge_init))))
                elif cross_stage_bridge == 'channel':
                    self.bridge_gates.add(scale_name, torch.nn.Parameter(
                        torch.full((1, channels, 1, 1), float(cross_stage_bridge_init))))

    def forward(self, x):
        if self.cross_stage_bridge == 'none':
            x1 = self.unet1(x)
            x2 = self.unet2(torch.cat([x, x1], dim=1))
        else:
            x1, u1_decoder_features = self.unet1(x, return_decoder_features=True)

            additions = [None] * len(u1_decoder_features)
            for stage_index, scale_name in self.bridge_stage_names.items():
                gate = self.bridge_gates[scale_name]
                additions[stage_index] = gate * u1_decoder_features[stage_index]

            x2 = self.unet2(
                torch.cat([x, x1], dim=1),
                decoder_additions=tuple(additions))

        if self.mode != 'train':
            return x2
        return x1, x2

    def bridge_gate_summary(self):
        if self.cross_stage_bridge == 'none':
            return None
        summary = {}
        for scale_name, gate in self.bridge_gates.items():
            if self.cross_stage_bridge == 'scalar':
                summary[scale_name] = float(gate.detach().cpu().item())
            else:
                g = gate.detach().cpu()
                summary[scale_name] = {
                    'min': float(g.min().item()),
                    'mean': float(g.mean().item()),
                    'max': float(g.max().item()),
                    'l2_norm': float(g.norm().item()),
                }
        return summary


def get_arch(model_name, in_c=3, n_classes=1,
             cross_stage_bridge='none', cross_stage_bridge_scales='all', cross_stage_bridge_init=0.0):

    cross_stage_bridge, scales, init = validate_cross_stage_bridge(
        cross_stage_bridge, cross_stage_bridge_scales, cross_stage_bridge_init, model_name)

    if model_name == 'unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[12,24,48], conv_bridge=True, shortcut=True)
    elif model_name == 'wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True,
                     cross_stage_bridge=cross_stage_bridge, cross_stage_bridge_scales=scales,
                     cross_stage_bridge_init=init)
    elif model_name == 'big_wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32,64], conv_bridge=True, shortcut=True,
                     cross_stage_bridge=cross_stage_bridge, cross_stage_bridge_scales=scales,
                     cross_stage_bridge_init=init)

    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model
if __name__ == '__main__':
    import time
    batch_size = 1
    batch = torch.zeros([batch_size, 1, 80, 80], dtype=torch.float32)
    model = get_arch('unet')
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
    start_time = time.time()
    logits = model(batch)
    print("--- %s seconds ---" % (time.time() - start_time))
