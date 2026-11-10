import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

CCTVBodycam = _impl.CCTVBodycam


def run(image, seed=0, **kwargs):
    params = dict(noise_strength=0.0, blockiness=0.0, fisheye=0.0, scanlines=0.0, saturation=1.0,
                  color_cast="off", cast_strength=0.0, night_ir=False, vignette=0.0)
    params.update(kwargs)
    return CCTVBodycam.execute(image, seed, params["noise_strength"], params["blockiness"],
                               params["fisheye"], params["scanlines"], params["saturation"],
                               params["color_cast"], params["cast_strength"], params["night_ir"],
                               params["vignette"])[0]


def test_shape_and_range_preserved():
    image = torch.rand(1, 64, 80, 3)
    out = run(image, noise_strength=0.05, blockiness=0.7, fisheye=0.3, scanlines=0.5,
              saturation=0.7, color_cast="green", cast_strength=0.8, night_ir=True, vignette=0.3)
    assert out.shape == image.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_identity_when_all_effects_off():
    image = torch.rand(1, 64, 80, 3)
    out = run(image)
    assert torch.equal(out, image)


def test_alpha_channel_untouched():
    image = torch.rand(1, 32, 32, 4)
    out = run(image, noise_strength=0.05, blockiness=0.5, fisheye=0.2, night_ir=True)
    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image[..., 3])


def test_blockiness_flattens_into_8x8_blocks():
    image = torch.rand(1, 64, 64, 3)
    out = run(image, blockiness=1.0)
    # pixels inside one macroblock share the block average
    assert torch.equal(out[0, 3, 1], out[0, 3, 6])
    assert torch.equal(out[0, 1, 3], out[0, 6, 3])
    # blend factor 0 is a no-op
    assert torch.allclose(run(image, blockiness=0.0), image)


def test_fisheye_compresses_corners_toward_center():
    # channel 0 encodes the x coordinate in [-1, 1]
    image = torch.linspace(-1, 1, 64).view(1, 1, 64, 1).expand(1, 64, 64, 3).contiguous()
    out = run(image, fisheye=0.15)
    # the left-edge pixel samples from x' = -1 * (1 - 0.15 * 2) = -0.7
    assert out[0, 0, 0, 0].item() > -0.75
    # the center pixel samples (almost) its own position
    assert abs(out[0, 32, 32, 0].item()) < 0.05


def test_scanlines_darken_alternating_rows():
    image = torch.full((1, 8, 8, 3), 0.5)
    out = run(image, scanlines=0.5)
    assert torch.allclose(out[0, 0::2], image[0, 0::2])
    assert torch.allclose(out[0, 1::2], image[0, 1::2] * 0.8)


def test_saturation_pulls_toward_luminance():
    image = torch.zeros(1, 1, 1, 3)
    image[0, 0, 0] = torch.tensor([0.8, 0.2, 0.2])
    out = run(image, saturation=0.5)
    lum = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.2
    assert abs(out[0, 0, 0, 0].item() - (lum + (0.8 - lum) * 0.5)) < 1e-5


def test_color_cast_shifts_channels():
    image = torch.full((1, 4, 4, 3), 0.5)
    out = run(image, color_cast="green", cast_strength=1.0)
    assert out[0, 0, 0, 1].item() > out[0, 0, 0, 0].item() > out[0, 0, 0, 2].item()
    assert out[0, 0, 0, 0].item() == pytest.approx(0.47, abs=1e-6)


def test_night_ir_renders_green_monochrome():
    # dim scene: nothing clamps at 0/1, so the tint ratio must be exact
    image = 0.2 + 0.2 * torch.rand(1, 32, 32, 3)
    out = run(image, night_ir=True)
    # the IR response tint: green strongest, blue in between, red weakest
    assert (out[..., 1] > out[..., 2]).all()
    assert (out[..., 2] > out[..., 0]).all()
    # tint ratio is constant: the image is monochrome under the green ramp
    ratio = out[..., 1] / out[..., 0]
    assert (ratio.max() - ratio.min()) < 0.05


def test_night_ir_center_hotspot():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, night_ir=True)
    assert out[0, 32, 32, 1].item() > out[0, 0, 0, 1].item()


def test_vignette_darkens_corners():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, vignette=0.3)
    assert out[0, 0, 0, 0].item() < out[0, 32, 32, 0].item()


def test_noise_deterministic_per_seed():
    image = torch.rand(1, 32, 32, 3)
    a = run(image, seed=7, noise_strength=0.05)
    b = run(image, seed=7, noise_strength=0.05)
    c = run(image, seed=8, noise_strength=0.05)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_batch_processed():
    image = torch.rand(2, 32, 32, 3)
    out = run(image, seed=5, noise_strength=0.05, fisheye=0.2)
    assert out.shape == (2, 32, 32, 3)
    assert not torch.equal(out[0], out[1])
    single = run(image[0:1], seed=5, noise_strength=0.05, fisheye=0.2)
    assert torch.allclose(out[0:1], single, atol=1e-5)
