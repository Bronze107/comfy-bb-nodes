import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

KodakC713 = _impl.KodakC713


def run(image, seed=0, **kwargs):
    params = dict(softness=0.0, flash=0.0, glow_strength=0.0, glow_radius=12, warmth=0.0,
                  saturation=1.0, noise_strength=0.0, vignette=0.0, resolution_cap=False)
    params.update(kwargs)
    return KodakC713.execute(image, seed, params["softness"], params["flash"], params["glow_strength"],
                             params["glow_radius"], params["warmth"], params["saturation"],
                             params["noise_strength"], params["vignette"], params["resolution_cap"])[0]


def test_shape_and_range_preserved():
    image = torch.rand(1, 64, 80, 3)
    out = run(image, softness=0.5, flash=0.5, glow_strength=0.5, warmth=0.1,
              saturation=1.2, noise_strength=0.05, vignette=0.2, resolution_cap=True)
    assert out.shape == image.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_identity_when_all_effects_off():
    image = torch.rand(1, 64, 80, 3)
    out = run(image)
    assert torch.equal(out, image)


def test_alpha_channel_untouched():
    image = torch.rand(1, 32, 32, 4)
    out = run(image, flash=0.5, noise_strength=0.05, glow_strength=0.5)
    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image[..., 3])


def test_resolution_cap_downscales_to_7mp():
    image = torch.rand(1, 3000, 4000, 3)  # 12 MP
    out = run(image, resolution_cap=True)
    assert out.shape[1] * out.shape[2] <= 3072 * 2304
    assert abs(out.shape[1] * out.shape[2] - 3072 * 2304) < 4000


def test_flash_brightens_center_and_sinks_edges():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, flash=0.4)
    # the meshgrid has no exact center pixel, so the boost is only approximately 1.2
    assert out[0, 32, 32, 0].item() == pytest.approx(0.6, abs=1e-3)
    # corner factor: 1 + 0.4 * (0.5 - 0.75) = 0.9
    assert out[0, 0, 0, 0].item() == pytest.approx(0.45, abs=1e-6)


def test_glow_blooms_flash_highlights():
    image = torch.full((1, 64, 64, 3), 0.05)
    image[0, 28:36, 28:36] = 1.0
    out = run(image, glow_strength=0.8, glow_radius=8)
    assert out[0, 26, 30, 0].item() > 0.1
    assert torch.allclose(out[0, 2, 2], image[0, 2, 2], atol=1e-5)


def test_warmth_shifts_white_balance():
    image = torch.full((1, 8, 8, 3), 0.5)
    out = run(image, warmth=0.1)
    assert out[0, 0, 0, 0].item() > 0.5
    assert out[0, 0, 0, 1].item() == 0.5
    assert out[0, 0, 0, 2].item() < 0.5


def test_saturation_pushes_away_from_luminance():
    image = torch.zeros(1, 1, 1, 3)
    image[0, 0, 0] = torch.tensor([0.8, 0.2, 0.2])
    out = run(image, saturation=1.2)
    lum = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.2
    assert out[0, 0, 0, 0].item() > 0.8  # above-luma channel is pushed further up
    assert abs(out[0, 0, 0, 0].item() - (lum + (0.8 - lum) * 1.2)) < 1e-5


def test_ccd_noise_is_colored():
    # independent per-channel noise: deltas must differ between channels
    image = torch.full((1, 32, 32, 3), 0.5)
    out = run(image, seed=3, noise_strength=0.02)
    delta = out - image
    assert not torch.allclose(delta[..., 0], delta[..., 1], atol=1e-4)
    a = run(image, seed=3, noise_strength=0.02)
    b = run(image, seed=3, noise_strength=0.02)
    c = run(image, seed=4, noise_strength=0.02)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_vignette_darkens_corners():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, vignette=0.3)
    assert out[0, 0, 0, 0].item() == pytest.approx(0.35, abs=1e-6)
    assert out[0, 0, 0, 0].item() < out[0, 32, 32, 0].item()


def test_batch_processed():
    image = torch.rand(2, 32, 32, 3)
    out = run(image, seed=5, noise_strength=0.05, flash=0.3)
    assert out.shape == (2, 32, 32, 3)
    assert not torch.equal(out[0], out[1])
    single = run(image[0:1], seed=5, noise_strength=0.05, flash=0.3)
    assert torch.allclose(out[0:1], single, atol=1e-5)
