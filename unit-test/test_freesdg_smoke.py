"""FreeSDG FMAug smoke tests for LwNet (plan §20).

Plain-Python runnable — no pytest dependency:

    python test_freesdg_smoke.py

Exits 0 when every section passes, non-zero on the first failure.
Section numbering follows plan §20.1-20.12 plus the TTA equivalence test of
plan §12.1.
"""

import json
import os
import random
import sys
import tempfile
import traceback

import numpy as np
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TEST_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(1, _TEST_DIR)

from utils.freesdg_aug import (  # noqa: E402
    FREESDG_FILTER_BANK,
    HFCFilter,
    GaussianMixUp,
    FreeSDGAugmentor,
    build_gaussian_kernel2d,
    fmaug_seed_for_worker,
    median_padding,
)

# Numerical tolerance for HFC-vs-reference comparisons (plan §20.2: ~1e-5
# after checking PyTorch behavior; the optimized and reference paths run the
# same float32 ops so agreement is far tighter than 1e-5 in practice).
TOL = 1e-5


# ---------------------------------------------------------------------------
# Reference implementation (plan §9 `hfc_reference`; naive per-channel loops)
# ---------------------------------------------------------------------------

def hfc_reference(x_pm1, mask, kernel_size, sigma, ratio=4.0):
    """Naive per-sample, per-channel HFC reference.

    Mirrors the released FreeSDG logic directly: per-channel median over ALL
    spatial pixels + 0.2 background fill, replication padding, per-channel
    Gaussian convolution, ratio-scaled residual, clamp, FOV re-application.
    Used to prove the optimized depthwise implementation numerically
    equivalent (plan §20.2).
    """
    assert x_pm1.dim() == 4 and mask.dim() == 4 and mask.shape[1] == 1
    b, c, h, w = x_pm1.shape
    kernel = build_gaussian_kernel2d(kernel_size, sigma, dtype=x_pm1.dtype)
    pad = kernel_size // 2
    out = torch.empty_like(x_pm1)
    for s in range(b):
        for ch in range(c):
            xch = x_pm1[s, ch]
            med = xch.reshape(-1).median() + 0.2
            xpad = torch.where(
                mask[s, 0] > 0.5, xch, med.expand_as(xch).to(xch.dtype)
            )
            xpad4 = xpad.view(1, 1, h, w)
            padded = torch.nn.functional.pad(
                xpad4, (pad, pad, pad, pad), mode="replicate"
            )
            blur = torch.nn.functional.conv2d(padded, kernel)
            res = ratio * (xpad4 - blur)
            res = torch.clamp(res, -1.0, 1.0)
            out[s, ch] = (res + 1.0) * mask[s, 0] - 1.0
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_img_mask(batch=1, channels=3, height=64, width=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand((batch, channels, height, width), generator=g) * 2.0 - 1.0
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    cy, cx = height // 2, width // 2
    r = min(height, width) // 2 - 2
    circle = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()
    mask = circle.unsqueeze(0).unsqueeze(0).expand(batch, 1, height, width
                                                   ).clone()
    return img, mask


# ---------------------------------------------------------------------------
# §20.1 Gaussian kernel
# ---------------------------------------------------------------------------

def test_20_1_gaussian_kernel():
    for k, sigma in [(5, 2), (9, 4), (27, 9), (43, 21), (8, 3)]:
        kern = build_gaussian_kernel2d(k, sigma)
        assert kern.shape == (1, 1, k, k), kern.shape
        assert kern.dtype == torch.float32
        assert abs(kern.sum().item() - 1.0) < 1e-6, kern.sum().item()
        flipped = torch.flip(kern, dims=[2, 3])
        assert torch.allclose(kern, flipped, atol=0), "kernel not symmetric"
        # device/dtype transfer works (plan §4.1)
        k64 = build_gaussian_kernel2d(k, sigma, dtype=torch.float64)
        assert k64.dtype == torch.float64
        assert torch.allclose(kern.to(torch.float64), k64, atol=1e-7)
    # even a size-1 kernel is normalized
    assert abs(build_gaussian_kernel2d(1, 2.0).sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# §20.2 HFC == reference; §20.3 HFC properties
# ---------------------------------------------------------------------------

def test_20_2_hfc_reference_equivalence():
    cases = [
        # (B, C, H, W, kernel, sigma, ratio)
        (1, 1, 48, 48, 5, 2, 4.0),
        (2, 3, 64, 64, 27, 9, 4.0),
        (3, 3, 96, 64, 43, 21, 4.0),   # B > 1, non-square
        (2, 1, 64, 96, 13, 7, 4.0),    # non-square, C=1
        (1, 3, 64, 64, 27, 9, 2.5),    # different ratio
    ]
    for i, (b, c, h, w, k, s, ratio) in enumerate(cases):
        img, mask = make_img_mask(batch=b, channels=c, height=h, width=w,
                                  seed=100 + i)
        filt = HFCFilter(k, s, ratio=ratio)
        opt = filt(img, mask)
        ref = hfc_reference(img, mask, k, s, ratio=ratio)
        assert opt.shape == img.shape, (opt.shape, img.shape)
        assert torch.allclose(opt, ref, atol=TOL), \
            "HFC vs reference mismatch for case {}".format(i)
    # varied masks: all-valid and random-threshold
    img, _ = make_img_mask(batch=2, channels=3, height=64, width=64, seed=7)
    g = torch.Generator().manual_seed(7)
    rnd_mask = (torch.rand((2, 1, 64, 64), generator=g) > 0.3).float()
    full_mask = torch.ones(2, 1, 64, 64)
    for mask in (rnd_mask, full_mask):
        filt = HFCFilter(27, 9)
        assert torch.allclose(filt(img, mask),
                              hfc_reference(img, mask, 27, 9), atol=TOL)


def test_20_3_hfc_properties():
    img, mask = make_img_mask(batch=2, channels=3, height=80, width=80,
                              seed=11)
    filt = HFCFilter(27, 9)
    out = filt(img, mask)
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6
    # mask == 0 -> output == -1 exactly
    bg = mask == 0
    assert torch.all(out[bg.expand_as(out)] == -1.0)
    # deterministic output for identical inputs
    assert torch.equal(out, filt(img, mask))
    # median_padding semantics: per-sample/per-channel all-pixel median +0.2
    x = torch.tensor([[[[0.2, 0.4], [0.6, -0.8]]]])
    m = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    padded = median_padding(x, m)
    med = x.reshape(-1).median() + 0.2
    assert torch.allclose(padded[0, 0, 1, 1], med, atol=1e-7)
    assert torch.equal(padded[0, 0, :2, :2].reshape(-1)[:3],
                       x.reshape(-1)[:3])


# ---------------------------------------------------------------------------
# §20.4 GaussianMixUp controlled test (injected filter indices + rectangle)
# ---------------------------------------------------------------------------

def test_20_4_mixup_controlled():
    img, mask = make_img_mask(batch=2, channels=3, height=64, width=64,
                              seed=21)
    mix = GaussianMixUp(ratio=4.0, mixup_size=-1, mix_policy="repo")
    rect = (10, 20, 24, 16)  # (y0, x0, h, w)
    out = mix(img, mask, filter_idx_1=3, filter_idx_2=7, rectangle=rect)
    v1 = mix.filters[3](img, mask)
    v2 = mix.filters[7](img, mask)
    y0, x0, h, w = rect
    inside = torch.zeros_like(out, dtype=torch.bool)
    inside[:, :, y0:y0 + h, x0:x0 + w] = True
    # inside rectangle == view 2 (analytically equal, same op graph)
    assert torch.equal(out[inside], v2[inside])
    # outside rectangle == view 1
    assert torch.equal(out[~inside], v1[~inside])
    # shared-median-padding optimization verified against the straightforward
    # two-view reference: each view computed by an independent full HFCFilter
    # forward (own median_padding) must match the shared-x_pad path.
    naive = v1.clone()
    naive[:, :, y0:y0 + h, x0:x0 + w] = v2[:, :, y0:y0 + h, x0:x0 + w]
    assert torch.allclose(out, naive, atol=TOL)


# ---------------------------------------------------------------------------
# §20.5 repo-policy RNG; filter bank
# ---------------------------------------------------------------------------

def test_20_5_repo_policy_rng():
    # bank: exactly 20 filters (5,2)..(43,21) — plan §3.1/§19.8
    assert len(FREESDG_FILTER_BANK) == 20
    assert FREESDG_FILTER_BANK[0] == (5, 2)
    assert FREESDG_FILTER_BANK[-1] == (43, 21)
    assert all(k % 2 == 1 for k, _ in FREESDG_FILTER_BANK)
    assert len(GaussianMixUp().filters) == 20

    # fixed seed -> reproducible rectangle sequence (repo policy)
    rng_a, rng_b = random.Random(42), random.Random(42)
    mix = GaussianMixUp(mixup_size=-1, mix_policy="repo")
    seqs = ([mix.sample_rectangle(rng_a, 512, 512) for _ in range(50)],
            [mix.sample_rectangle(rng_b, 512, 512) for _ in range(50)])
    assert seqs[0] == seqs[1]
    assert seqs[0] != [mix.sample_rectangle(random.Random(43), 512, 512)
                       for _ in range(50)]
    for (y0, x0, h, w) in seqs[0]:
        assert 0 <= y0 and y0 + h <= 512
        assert 0 <= x0 and x0 + w <= 512

    # fixed-size repo mode: square + positioning guard (plan §5.2)
    rng = random.Random(0)
    for _ in range(10):
        y0, x0, h, w = mix_fixed = GaussianMixUp(
            mixup_size=32, mix_policy="repo"
        ).sample_rectangle(rng, 64, 64)
        assert h == w == 32
        assert 0 <= x0 and x0 + 32 <= 64
        assert 0 <= y0 and y0 + 32 <= 64
    try:
        GaussianMixUp(mixup_size=64, mix_policy="repo").sample_rectangle(
            random.Random(0), 64, 64
        )
        raise AssertionError("mixup_size == width must be rejected")
    except ValueError:
        pass

    # second augmentor with same seed produces the same FMAug sequence
    img, mask = make_img_mask(batch=1, channels=3, height=64, width=64,
                              seed=5)
    img01, mask01 = (img + 1) / 2, mask
    aug1 = FreeSDGAugmentor(seed=7)
    aug2 = FreeSDGAugmentor(seed=7)
    outs1 = [aug1.augment_train(img01, mask01) for _ in range(8)]
    outs2 = [aug2.augment_train(img01, mask01) for _ in range(8)]
    assert all(torch.equal(a, b) for a, b in zip(outs1, outs2))


# ---------------------------------------------------------------------------
# §20.6 paper-policy RNG
# ---------------------------------------------------------------------------

def test_20_6_paper_policy_rng():
    rng = random.Random(123)
    mix = GaussianMixUp(mix_policy="paper")
    for _ in range(100):
        y0, x0, h, w = mix.sample_rectangle(rng, 512, 512)
        s = h
        assert h == w
        assert 32 <= s <= 256
        cx, cy = x0 + s // 2, y0 + s // 2
        assert 128 <= cx <= 384
        assert 128 <= cy <= 384
        assert 0 <= y0 and y0 + h <= 512
        assert 0 <= x0 and x0 + w <= 512
    # non-512 input raises a clear error
    try:
        mix.sample_rectangle(random.Random(0), 256, 256)
        raise AssertionError("paper policy must require 512x512")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# §20.7 global RNG isolation (plan §18)
# ---------------------------------------------------------------------------

def test_20_7_global_rng_isolation():
    img, mask = make_img_mask(batch=1, channels=3, height=64, width=64,
                              seed=9)
    img01, mask01 = (img + 1) / 2, mask
    aug = FreeSDGAugmentor(seed=7)

    random.seed(1234)
    before = random.getstate()
    for _ in range(5):
        _ = aug.augment_train(img01, mask01)
        _ = aug.anchor(img01, mask01)
    assert random.getstate() == before, \
        "FMAug must not advance the global Python RNG"


# ---------------------------------------------------------------------------
# §20.3(b) range contract + §8 identity round-trip (augmentor level)
# ---------------------------------------------------------------------------

def test_range_contract():
    img, mask = make_img_mask(batch=1, channels=3, height=64, width=64,
                              seed=3)
    img01, mask01 = (img.clamp(-1, 1) + 1) / 2, mask
    aug = FreeSDGAugmentor(seed=0)
    out = aug.augment_train(img01, mask01)  # batch form in, batch form out
    assert out.shape == img01.shape
    assert out.min() >= 0.0 and out.max() <= 1.0, "output must be [0,1]"
    anchored = aug.anchor(img01, mask01)
    assert anchored.min() >= 0.0 and anchored.max() <= 1.0
    # per-sample form [C, H, W] used by TrainDataset
    out_s = aug.augment_train(img01[0], mask01[0])
    assert out_s.shape == img01[0].shape
    assert out_s.min() >= 0.0 and out_s.max() <= 1.0
    # anchor background maps to exactly 0 after the [0,1] remap (plan §3.3)
    bg = (mask[0, 0] == 0)
    assert torch.all(anchored[0][:, bg] == 0.0)
    # identity conversion round-trip
    x = torch.rand(3, 32, 32)
    assert torch.allclose(((2 * x - 1) + 1) / 2, x, atol=1e-7)
    # to_pil_uint8 round-trip quantization is within 1/255 + epsilon
    pil = aug.to_pil_uint8(out_s)
    arr = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
    assert (arr - out_s).abs().max() <= 0.5 / 255 + 1e-6
    assert pil.mode == "RGB"


# ---------------------------------------------------------------------------
# §20.8 multi-worker RNG (helper-level, plan §6.3/§29.1)
# ---------------------------------------------------------------------------

class _FakeWorkerInfo:
    def __init__(self, id, seed):
        self.id = id
        self.seed = seed


def test_20_8_worker_rng():
    base = 11
    # main-process: plain seed
    assert fmaug_seed_for_worker(base, None) == base
    # workers: distinct streams, deterministic per configuration
    w0, w1, w0b = (
        _FakeWorkerInfo(0, 1000),
        _FakeWorkerInfo(1, 1001),
        _FakeWorkerInfo(0, 1000),
    )
    s0, s1, s0b = (fmaug_seed_for_worker(base, w) for w in (w0, w1, w0b))
    assert s0 != s1
    assert s0 == s0b
    r0a, r0b, r1 = (random.Random(s) for s in (s0, s0b, s1))
    assert [r0a.random() for _ in range(5)] == [r0b.random() for _ in range(5)]
    assert [r0a.random() for _ in range(5)] != [r1.random() for _ in range(5)]
    # an augmentor inside a "worker" derives its stream from the worker seed
    img, mask = make_img_mask(seed=2)
    img01 = (img + 1) / 2
    torch.manual_seed(0)
    aug_a, aug_b = FreeSDGAugmentor(seed=base), FreeSDGAugmentor(seed=base)
    a1 = aug_a.augment_train(img01, mask)
    a2 = aug_b.augment_train(img01, mask)
    assert torch.equal(a1, a2)  # deterministic under num_workers=0 semantics


# ---------------------------------------------------------------------------
# §20.9 loader / FOV tests (synthetic mini dataset)
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(tmpdir, n=2, size=80):
    """Tiny vessel-style dataset: RGB image, binary GT, FOV ellipse mask."""
    import pandas as pd
    from PIL import Image as PILImage

    im_paths, gt_paths, mask_paths = [], [], []
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    r = size // 2 - 4
    circle = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)
    for i in range(n):
        rng = np.random.default_rng(i)
        base = (yy + xx) / (2.0 * size)
        img = np.stack([
            (base * 200 + rng.uniform(0, 40, (size, size))) * circle,
            (base * 180 + rng.uniform(0, 40, (size, size))) * circle,
            (base * 160 + rng.uniform(0, 40, (size, size))) * circle,
        ], axis=-1).astype(np.uint8)
        gt = np.zeros((size, size), dtype=np.uint8)
        gt[size // 3, size // 4:size - size // 4] = 255  # one vessel
        gt[:, size // 2] = np.where(circle[:, size // 2], 255, 0)
        mask = (circle * 255).astype(np.uint8)
        im = os.path.join(tmpdir, "im_{}.png".format(i))
        gtp = os.path.join(tmpdir, "gt_{}.png".format(i))
        mp = os.path.join(tmpdir, "mask_{}.png".format(i))
        PILImage.fromarray(img).save(im)
        PILImage.fromarray(gt).save(gtp)
        PILImage.fromarray(mask, mode="L").save(mp)
        im_paths.append(im)
        gt_paths.append(gtp)
        mask_paths.append(mp)
    csv_path = os.path.join(tmpdir, "train.csv")
    pd.DataFrame(
        {"im_paths": im_paths, "gt_paths": gt_paths, "mask_paths": mask_paths}
    ).to_csv(csv_path, index=False)
    return csv_path


def test_20_9_loader_fov():
    from utils.get_loaders import get_train_val_datasets

    tmp = tempfile.mkdtemp(prefix="freesdg_smoke_")
    csv_path = _build_synthetic_dataset(tmp)
    tg = (64, 64)

    # baseline: original path, two-value return (plan §17/§20.9)
    tr_base, _ = get_train_val_datasets(csv_path, csv_path, tg_size=tg)
    random.seed(0)
    item = tr_base[0]
    assert isinstance(item, tuple) and len(item) == 2
    img_b, tg_b = item
    assert torch.is_tensor(img_b) and img_b.shape == (3, 64, 64)
    assert img_b.min() >= 0.0 and img_b.max() <= 1.0
    assert tg_b.shape == (64, 64)

    # FMAug enabled (train role): shapes, ranges, alignment
    cfg = {"enabled": True, "raw_prob": 0.0, "test_input": "raw",
           "anchor_w": 27, "anchor_sigma": 9, "ratio": 4.0,
           "mixup_size": -1, "mix_policy": "repo", "seed": 0}
    tr_aug, vl_aug = get_train_val_datasets(csv_path, csv_path, tg_size=tg,
                                            freesdg_cfg=cfg)
    img_a, tg_a = tr_aug[0]
    assert img_a.shape == (3, 64, 64) and tg_a.shape == (64, 64)
    assert img_a.min() >= 0.0 and img_a.max() <= 1.0
    # NEAREST FOV resize: binary values, spatially aligned with image tensor
    from PIL import Image as PILImage
    raw_mask = PILImage.open(tr_aug.mask_list[0]).convert("L")
    mask_t = tr_aug._freesdg_mask_tensor(raw_mask, (64, 64))
    assert mask_t.shape == (1, 64, 64)
    assert set(mask_t.unique().tolist()) <= {0.0, 1.0}

    # global Python RNG stream: with the same global seed, the raw-branch
    # FMAug dataset must consume the identical paired-transform draws as the
    # baseline (differences limited to the ~1/255 uint8 round-trip, §19.6)
    random.seed(0)
    img_b2, _ = tr_base[0]
    random.seed(0)
    tr_raw, _ = get_train_val_datasets(
        csv_path, csv_path, tg_size=tg, freesdg_cfg=dict(cfg, raw_prob=1.0))
    img_r, _ = tr_raw[0]
    assert (img_b2 - img_r).abs().max() <= 1.0 / 255 + 1e-6, \
        "FMAug path must not alter the global augmentation RNG stream"

    # validation anchor: float [0,1] output; FOV background maps to exactly 0
    _, vl_anc = get_train_val_datasets(
        csv_path, csv_path, tg_size=tg,
        freesdg_cfg=dict(cfg, test_input="anchor"))
    img_v, tg_v = vl_anc[0]
    assert img_v.shape == (3, 64, 64)
    assert img_v.min() >= 0.0 and img_v.max() <= 1.0
    # replicate the dataset's FOV treatment: bbox crop (crop_to_fov) then
    # NEAREST resize — the same mask the anchor HFC received
    from skimage.measure import regionprops
    full_mask = np.array(PILImage.open(vl_anc.mask_list[0]).convert("L"))
    minr, minc, maxr, maxc = regionprops(full_mask.astype(int))[0].bbox
    mask_v = vl_anc._freesdg_mask_tensor(
        PILImage.fromarray(full_mask[minr:maxr, minc:maxc]), (64, 64))
    bg = mask_v[0] == 0
    assert bg.any()
    assert torch.all(img_v[:, bg] == 0.0), \
        "anchor background must be 0 after [0,1] remap"


# ---------------------------------------------------------------------------
# §20.10 W-Net integration (unchanged architecture, §14/§16)
# ---------------------------------------------------------------------------

def test_20_10_wnet_integration():
    from models.get_model import get_arch

    m1, m2 = get_arch("wnet"), get_arch("wnet")
    n1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    assert n1 == n2 and n1 > 0
    assert list(m1.state_dict().keys()) == list(m2.state_dict().keys())
    assert not any(k.startswith("base_net") for k in m1.state_dict())
    x = torch.rand(1, 3, 64, 64)
    m1.mode = "train"
    out = m1(x)
    assert isinstance(out, tuple) and len(out) == 2  # (aux, main)
    m1.mode = "eval"
    out = m1(x)
    assert torch.is_tensor(out)  # single output in eval mode
    print("       wnet trainable params: {:,}".format(n1))


# ---------------------------------------------------------------------------
# §20.11 checkpoint round-trip (tiny synthetic FMAug-style training)
# ---------------------------------------------------------------------------

def test_20_11_checkpoint_roundtrip():
    from models.get_model import get_arch
    from utils.model_saving_loading import save_model, load_model

    device = torch.device("cpu")
    model = get_arch("wnet").to(device)
    model.mode = "train"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.rand(1, 3, 64, 64)
    y = (torch.rand(1, 64, 64) > 0.5).float().unsqueeze(dim=1)  # as run_one_epoch
    for _ in range(2):  # forward/backward succeeds (FMAug-equivalent input)
        logits_aux, logits = model(x)
        loss = crit(logits_aux, y) + crit(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    tmp = tempfile.mkdtemp(prefix="freesdg_ckpt_")
    save_model(tmp, model, opt)
    fresh = get_arch("wnet").to(device)
    fresh, _ = load_model(fresh, tmp, device=device)  # strict=True by default
    ckpt = torch.load(os.path.join(tmp, "model_checkpoint.pth"),
                      map_location=device)
    assert list(ckpt["model_state_dict"].keys()) == \
        list(fresh.state_dict().keys())


# ---------------------------------------------------------------------------
# §20.12 saved-config coverage of the 9 FreeSDG arguments
# ---------------------------------------------------------------------------

def test_20_12_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_cyclical_for_test", os.path.join(_ROOT_DIR, "train_cyclical.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    defaults = vars(mod.parser.parse_args([]))
    expected = {
        "freesdg": False,
        "freesdg_raw_prob": 0.0,
        "freesdg_test_input": "raw",
        "freesdg_anchor_w": 27,
        "freesdg_anchor_sigma": 9,
        "freesdg_ratio": 4.0,
        "freesdg_mixup_size": -1,
        "freesdg_mix_policy": "repo",
        "freesdg_seed": 0,
        "freesdg_aug_mode": "fmaug",
        "freesdg_lwnet_aug_profile": "original",
    }
    for key, value in expected.items():
        assert defaults.get(key, "<missing>") == value, key
    enabled = mod.parser.parse_args([
        '--freesdg', '--freesdg_disable_aug_color_jitter'])
    assert enabled.freesdg_disable_aug_color_jitter is True
    # training writes json.dump(vars(args)) -> all keys serialize
    json.dumps(defaults)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, 'config.cfg')
        mod.write_training_config(enabled, config_path)
        with open(config_path) as f:
            saved = json.load(f)
        assert saved['freesdg_disable_aug_color_jitter'] is True


# ---------------------------------------------------------------------------
# §12.1 anchor-then-flip == flip-then-anchor (TTA equivariance)
# ---------------------------------------------------------------------------

def test_12_1_tta_flip_anchor_equivalence():
    img, mask = make_img_mask(batch=1, channels=3, height=96, width=96,
                              seed=31)
    img01 = (img + 1) / 2
    aug = FreeSDGAugmentor(seed=0)
    anchored = aug.anchor(img01, mask)
    for dims in ([3], [2], [2, 3]):  # lr, ud, lrud — the create_pred TTA flips
        flipped_anchor = torch.flip(anchored, dims=dims)
        anchor_of_flipped = aug.anchor(torch.flip(img01, dims=dims),
                                       torch.flip(mask, dims=dims))
        assert torch.allclose(flipped_anchor, anchor_of_flipped, atol=TOL), \
            "anchor HFC must be reflection-equivariant for dims={}".format(dims)


# ---------------------------------------------------------------------------
# Diagnostic stage: aug_mode dispatch (M2/M3), Raffe modes (U3), profiles
# ---------------------------------------------------------------------------

def test_diag_aug_modes():
    img, mask = make_img_mask(batch=1, channels=3, height=64, width=64,
                              seed=41)
    img01, mask01 = (img + 1) / 2, mask

    # fixed_hfc: deterministic, equal to anchor()
    a1 = FreeSDGAugmentor(seed=5, aug_mode='fixed_hfc')
    a2 = FreeSDGAugmentor(seed=5, aug_mode='fixed_hfc')
    f1 = a1.augment_train(img01, mask01)
    assert torch.equal(f1, a2.augment_train(img01, mask01))
    assert torch.allclose(f1, a1.anchor(img01, mask01))

    # random_hfc: exactly one Gaussian-bank filter, hook-controlled
    r = FreeSDGAugmentor(seed=5, aug_mode='random_hfc')
    out5 = r.augment_train(img01, mask01, filter_idx=5)
    expect = (r.mixup.filters[5].forward(2 * img01 - 1, mask01) + 1) / 2
    assert torch.allclose(out5, expect, atol=1e-6)
    out6 = r.augment_train(img01, mask01, filter_idx=6)
    assert not torch.allclose(out5, out6)
    # seeded sequence reproducibility + variation across draws
    r2 = FreeSDGAugmentor(seed=9, aug_mode='random_hfc')
    r3 = FreeSDGAugmentor(seed=9, aug_mode='random_hfc')
    seqs = ([r2.augment_train(img01, mask01) for _ in range(6)],
            [r3.augment_train(img01, mask01) for _ in range(6)])
    assert all(torch.equal(a, b) for a, b in zip(*seqs))
    assert not all(torch.equal(seqs[0][0], s) for s in seqs[0][1:])

    # all modes: output in [0,1]
    for mode in ("fmaug", "fixed_hfc", "random_hfc", "raffe_filter",
                 "raffe_smooth_mix"):
        a = FreeSDGAugmentor(seed=3, aug_mode=mode)
        o = a.augment_train(img01, mask01)
        assert o.shape == img01.shape
        assert o.min() >= 0.0 and o.max() <= 1.0, mode


def test_diag_raffe_components():
    from utils.freesdg_aug import (RaffeFilterBank, DT2SmoothMask,
                                   FourierButterworthView, raffe_d0_list,
                                   build_corner_butterworth_map)

    # official bank: 36 params, d0 endpoints 0 -> ~0.1, n in {1,2,3}
    d0s = raffe_d0_list()
    assert len(d0s) == 12
    assert abs(d0s[0]) < 1e-12 and abs(d0s[-1] - 0.1) < 1e-9
    assert all(d0s[i] < d0s[i + 1] for i in range(11))
    bank = RaffeFilterBank()
    assert len(bank.params) == 36
    assert bank.params[0][1] == 1 and bank.params[1][1] == 2 \
        and bank.params[2][1] == 3 and bank.params[3][1] == 1

    # vectorized map == literal reference loop (incl. [j, i] transpose).
    # Non-square input is undefined in the official code (transposed
    # assignment overflows), so the equivalence check uses a square size.
    H = W = 24
    ratio, n = 0.05, 2
    ref = np.zeros((H, W), dtype=np.float32)
    d0 = int(max((ratio * min(H, W)) // 2, 1))
    corners = ((0, 0), (0, W - 1), (H - 1, W - 1), (H - 1, 0))
    for i in range(H):
        for j in range(W):
            d = min(np.sqrt((i - x) ** 2 + (j - y) ** 2) for x, y in corners)
            ref[j, i] = 1.0 / (1.0 + (d0 / (d + 1.0)) ** (2 * n))
    ours = build_corner_butterworth_map(H, W, ratio, n).numpy()
    assert np.allclose(ours, ref, atol=1e-6)
    try:
        build_corner_butterworth_map(20, 28, ratio, n)
        raise AssertionError("non-square map must be rejected")
    except ValueError:
        pass

    # view: range, masking; map is a corner-centered high-pass
    img, mask = make_img_mask(batch=2, channels=3, height=48, width=48,
                              seed=17)
    view = FourierButterworthView(48, 48, 0.05, 2)
    out = view(img, mask)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.all(out[:, :, mask[0, 0] == 0] == 0.0)
    gains = build_corner_butterworth_map(48, 48, 0.05, 2)
    # with fft2 (no fftshift) the low frequencies live at the corners:
    # corner gain must be low, far-from-corner gain ~1 (high-pass)
    assert float(gains[0, 0]) < 0.6 and float(gains[24, 24]) > 0.999

    # smooth blend: injected mask reproduces m*v1 + (1-m)*v2 exactly
    x01 = (img + 1) / 2
    idx1, idx2 = [0, 3, 6], [1, 4, 7]
    v1 = bank.apply(x01, mask, idx1)
    v2 = bank.apply(x01, mask, idx2)
    g = torch.linspace(0, 1, 48).view(1, 1, 1, 48).expand(1, 1, 48, 48)
    a = FreeSDGAugmentor(seed=0, aug_mode='raffe_smooth_mix')
    out = a.augment_train(
        x01, mask, raffe_indices_1=idx1, raffe_indices_2=idx2,
        blend_mask=g[0])
    assert torch.allclose(out, (g * v1 + (1 - g) * v2).clamp(0, 1) * mask,
                          atol=1e-6)

    # DT2 mask: normalized, smooth, deterministic per rng, varies by rng
    dt2 = DT2SmoothMask()
    m1 = dt2.generate(random.Random(0), 64, 64)
    m1b = dt2.generate(random.Random(0), 64, 64)
    m2 = dt2.generate(random.Random(1), 64, 64)
    assert torch.equal(m1, m1b)
    assert not torch.equal(m1, m2)
    assert abs(float(m1.min())) < 1e-6 and abs(float(m1.max()) - 1.0) < 1e-6
    jump = (m1[:, 1:] - m1[:, :-1]).abs().max()
    assert float(jump) <= 0.101, "DT2 mask must be smooth (max step 1/100)"

    # raffe modes through the augmentor: deterministic per seed, ranges
    a1 = FreeSDGAugmentor(seed=7, aug_mode='raffe_filter')
    a2 = FreeSDGAugmentor(seed=7, aug_mode='raffe_filter')
    o1 = a1.augment_train(x01, mask)
    assert torch.equal(o1, a2.augment_train(x01, mask))
    s1 = FreeSDGAugmentor(seed=7, aug_mode='raffe_smooth_mix')
    s2 = FreeSDGAugmentor(seed=7, aug_mode='raffe_smooth_mix')
    os_ = s1.augment_train(x01, mask)
    assert torch.equal(os_, s2.augment_train(x01, mask))
    assert o1.min() >= 0.0 and o1.max() <= 1.0
    assert os_.min() >= 0.0 and os_.max() <= 1.0


def test_diag_profiles_and_rawprob():
    from utils.get_loaders import get_train_val_datasets

    tmp = tempfile.mkdtemp(prefix="freesdg_diag_")
    csv_path = _build_synthetic_dataset(tmp)
    base_cfg = {"enabled": True, "raw_prob": 0.0, "test_input": "raw",
                "anchor_w": 27, "anchor_sigma": 9, "ratio": 4.0,
                "mixup_size": -1, "mix_policy": "repo", "seed": 0}

    def profile_of(cfg):
        tr, _ = get_train_val_datasets(csv_path, csv_path, tg_size=(64, 64),
                                       freesdg_cfg=cfg)
        return [type(t).__name__ for t in tr.transforms.transforms]

    # flips_only: exactly [RandomHorizontalFlip, RandomVerticalFlip, ToTensor]
    names = profile_of(dict(base_cfg, lwnet_aug_profile='flips_only'))
    assert names == ['RandomHorizontalFlip', 'RandomVerticalFlip', 'ToTensor'], names
    # original profile (freesdg enabled): resize hoisted, rest verbatim
    names = profile_of(dict(base_cfg, lwnet_aug_profile='original'))
    assert names == ['RandomChoice', 'ColorJitter', 'RandomHorizontalFlip',
                     'RandomVerticalFlip', 'ToTensor'], names
    # vanilla baseline unchanged: resize still first
    tr_v, _ = get_train_val_datasets(csv_path, csv_path, tg_size=(64, 64))
    names = [type(t).__name__ for t in tr_v.transforms.transforms]
    assert names == ['Resize', 'RandomChoice', 'ColorJitter',
                     'RandomHorizontalFlip', 'RandomVerticalFlip', 'ToTensor']

    # Structural-saliency replay must use selected branch pipeline.
    tr_struct, _ = get_train_val_datasets(
        csv_path, csv_path, tg_size=(64, 64),
        freesdg_cfg=dict(base_cfg, disable_aug_color_jitter=True),
        need_structural_saliency=True)
    structural_sample = tr_struct[0]
    assert len(structural_sample) == 4

    # Omitted and explicit False preserve identical dataset behavior.
    cfg_legacy = dict(base_cfg)
    cfg_false = dict(base_cfg, disable_aug_color_jitter=False)
    tr_legacy, _ = get_train_val_datasets(csv_path, csv_path,
                                          tg_size=(64, 64),
                                          freesdg_cfg=cfg_legacy)
    tr_false, _ = get_train_val_datasets(csv_path, csv_path,
                                         tg_size=(64, 64),
                                         freesdg_cfg=cfg_false)
    random.seed(123)
    torch.manual_seed(123)
    legacy_sample = tr_legacy[0]
    random.seed(123)
    torch.manual_seed(123)
    false_sample = tr_false[0]
    assert all(torch.equal(a, b) for a, b in zip(legacy_sample, false_sample))

    # raw_prob behavior: 0 -> never raw, 1 -> always raw, 0.5 deterministic
    img, mask = make_img_mask(batch=1, channels=3, height=64, width=64,
                              seed=13)
    img01 = (img + 1) / 2
    a0 = FreeSDGAugmentor(seed=0, aug_mode='fmaug')
    assert all(not torch.equal(a0.augment_train(img01, mask, raw_prob=0.0),
                               img01) for _ in range(5))
    a1 = FreeSDGAugmentor(seed=0, aug_mode='fmaug')
    assert all(torch.equal(a1.augment_train(img01, mask, raw_prob=1.0), img01)
               for _ in range(5))
    counts = []
    for seed in (0, 1, 2):
        a = FreeSDGAugmentor(seed=seed, aug_mode='fmaug')
        counts.append(sum(torch.equal(a.augment_train(img01, mask,
                                                      raw_prob=0.5), img01)
                          for _ in range(200)))
    b = FreeSDGAugmentor(seed=0, aug_mode='fmaug')
    recount = sum(torch.equal(b.augment_train(img01, mask, raw_prob=0.5),
                              img01) for _ in range(200))
    assert recount == counts[0], "raw counts must be seed-deterministic"
    assert all(70 <= c <= 130 for c in counts), counts

    # global RNG isolation across ALL aug modes
    random.seed(99)
    before = random.getstate()
    for mode in ("fmaug", "fixed_hfc", "random_hfc", "raffe_filter",
                 "raffe_smooth_mix"):
        a = FreeSDGAugmentor(seed=2, aug_mode=mode)
        a.augment_train(img01, mask, raw_prob=0.3)
    assert random.getstate() == before


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

SECTIONS = [
    ("20.1 gaussian kernel", test_20_1_gaussian_kernel),
    ("20.2 HFC vs reference", test_20_2_hfc_reference_equivalence),
    ("20.3 HFC properties + median semantics", test_20_3_hfc_properties),
    ("20.4 GaussianMixUp controlled", test_20_4_mixup_controlled),
    ("20.5 repo policy RNG + bank", test_20_5_repo_policy_rng),
    ("20.6 paper policy RNG", test_20_6_paper_policy_rng),
    ("20.7 global RNG isolation", test_20_7_global_rng_isolation),
    ("20.3b range contract", test_range_contract),
    ("20.8 worker RNG", test_20_8_worker_rng),
    ("20.9 loader / FOV", test_20_9_loader_fov),
    ("20.10 W-Net integration", test_20_10_wnet_integration),
    ("20.11 checkpoint round-trip", test_20_11_checkpoint_roundtrip),
    ("20.12 config coverage", test_20_12_config),
    ("12.1 TTA flip-anchor equivariance", test_12_1_tta_flip_anchor_equivalence),
    ("diag aug_mode dispatch", test_diag_aug_modes),
    ("diag raffe components", test_diag_raffe_components),
    ("diag profiles + raw_prob", test_diag_profiles_and_rawprob),
]


def main():
    for name, fn in SECTIONS:
        try:
            fn()
        except Exception:
            print("FAIL: {}".format(name))
            traceback.print_exc()
            sys.exit(1)
        print("PASS: {}".format(name))
    print("All {} smoke sections passed.".format(len(SECTIONS)))


if __name__ == "__main__":
    main()
