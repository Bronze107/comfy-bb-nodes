import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

RicohGR3HDF = _impl.RicohGR3HDF


def run(image, seed=0, **kwargs):
    params = dict(glow_strength=0.0, glow_threshold=0.65, glow_radius=16, mist_strength=0.0,
                  contrast=1.0, saturation=1.0, vignette=0.0, grain_strength=0.0)
    params.update(kwargs)
    return RicohGR3HDF.execute(image, seed, params["glow_strength"], params["glow_threshold"],
                               params["glow_radius"], params["mist_strength"], params["contrast"],
                               params["saturation"], params["vignette"], params["grain_strength"])[0]


def test_shape_and_range_preserved():
    image = torch.rand(1, 64, 80, 3)
    out = run(image, glow_strength=0.8, mist_strength=0.3, contrast=0.8,
              saturation=0.7, vignette=0.3, grain_strength=0.02)
    assert out.shape == image.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_identity_when_all_effects_off():
    image = torch.rand(1, 64, 80, 3)
    out = run(image)
    assert torch.equal(out, image)


def test_alpha_channel_untouched():
    image = torch.rand(1, 32, 32, 4)
    out = run(image, glow_strength=0.8, mist_strength=0.3, grain_strength=0.02)
    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image[..., 3])


def test_glow_blooms_bright_block_into_dark_surround():
    image = torch.full((1, 64, 64, 3), 0.05)
    image[0, 28:36, 28:36] = 1.0
    out = run(image, glow_strength=0.8, glow_radius=8)
    # adjacent background receives bloom; a far dark corner does not
    assert out[0, 26, 30, 0].item() > 0.1
    assert torch.allclose(out[0, 2, 2], image[0, 2, 2], atol=1e-5)


def test_glow_threshold_gates_dim_scene():
    image = torch.full((1, 32, 32, 3), 0.5)  # below the 0.65 threshold
    out = run(image, glow_strength=0.8, glow_radius=8)
    assert torch.allclose(out, image, atol=1e-5)


def test_mist_lifts_flat_gray():
    image = torch.full((1, 32, 32, 3), 0.5)
    out = run(image, mist_strength=0.2)
    # screen blend of the flat bloom: 1 - 0.5 * (1 - 0.1) = 0.55
    assert out[0, 0, 0, 0].item() == pytest.approx(0.55, abs=1e-5)


def test_contrast_pivots_at_mid_gray():
    image = torch.full((1, 8, 8, 3), 0.4)
    out = run(image, contrast=0.8)
    assert out[0, 0, 0, 0].item() == pytest.approx(0.42, abs=1e-6)
    assert run(torch.full((1, 8, 8, 3), 0.5), contrast=0.8)[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_saturation_pulls_toward_luminance():
    image = torch.zeros(1, 1, 1, 3)
    image[0, 0, 0] = torch.tensor([0.8, 0.2, 0.2])
    out = run(image, saturation=0.5)
    lum = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.2
    assert out[0, 0, 0, 0].item() == pytest.approx(lum + (0.8 - lum) * 0.5, abs=1e-5)


def test_vignette_darkens_corners():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, vignette=0.3)
    assert out[0, 0, 0, 0].item() == pytest.approx(0.35, abs=1e-5)
    assert out[0, 0, 0, 0].item() < out[0, 32, 32, 0].item()


def test_grain_deterministic_per_seed_and_gray():
    image = torch.full((1, 32, 32, 3), 0.5)
    a = run(image, seed=7, grain_strength=0.02)
    b = run(image, seed=7, grain_strength=0.02)
    c = run(image, seed=8, grain_strength=0.02)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    delta = a - image
    assert torch.allclose(delta[..., 0], delta[..., 1], atol=1e-5)
    assert torch.allclose(delta[..., 1], delta[..., 2], atol=1e-5)


def test_zero_radius_disables_diffusion():
    image = torch.full((1, 64, 64, 3), 0.05)
    image[0, 28:36, 28:36] = 1.0
    out = run(image, glow_strength=0.8, glow_radius=0)
    assert torch.equal(out, image)


def test_batch_processed():
    image = torch.rand(2, 32, 32, 3)
    out = run(image, seed=5, grain_strength=0.02)
    assert out.shape == (2, 32, 32, 3)
    assert not torch.equal(out[0], out[1])
    single = run(image[0:1], seed=5, grain_strength=0.02)
    assert torch.allclose(out[0:1], single, atol=1e-6)
