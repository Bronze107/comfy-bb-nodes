import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

IPhone5Camera = _impl.IPhone5Camera


def run(image, seed=0, **kwargs):
    params = dict(noise_strength=0.0, softness=0.0, vignette=0.0, color_shading=0.0,
                  purple_haze=0.0, warmth=0.0, resolution_cap=False, chroma_subsampling=False)
    params.update(kwargs)
    return IPhone5Camera.execute(image, seed, params["noise_strength"], params["softness"],
                                 params["vignette"], params["color_shading"], params["purple_haze"],
                                 params["warmth"], params["resolution_cap"], params["chroma_subsampling"])[0]


def test_shape_preserved():
    image = torch.rand(1, 64, 80, 3)
    out = run(image, noise_strength=0.05, softness=0.5, vignette=0.2,
              color_shading=1.0, purple_haze=0.5, warmth=0.1, chroma_subsampling=True)
    assert out.shape == image.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_identity_when_all_effects_off():
    image = torch.rand(1, 64, 80, 3)
    out = run(image)
    assert torch.equal(out, image)


def test_alpha_channel_untouched():
    image = torch.rand(1, 32, 32, 4)
    out = run(image, noise_strength=0.1, vignette=0.3, purple_haze=0.5)
    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image[..., 3])


def test_noise_deterministic_per_seed():
    image = torch.rand(1, 32, 32, 3)
    a = run(image, seed=7, noise_strength=0.05)
    b = run(image, seed=7, noise_strength=0.05)
    c = run(image, seed=8, noise_strength=0.05)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_noise_is_gray_luminance():
    # flat gray keeps chroma constant through subsampling, so the only change
    # is the noise: the same amount added to every channel of a pixel
    image = torch.full((1, 32, 32, 3), 0.5)
    out = run(image, seed=3, noise_strength=0.02, chroma_subsampling=True)
    delta = out - image
    assert torch.allclose(delta[..., 0], delta[..., 1], atol=1e-5)
    assert torch.allclose(delta[..., 1], delta[..., 2], atol=1e-5)


def test_vignette_darkens_corners_more_than_center():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, vignette=0.3)
    assert out[0, 0, 0, 0].item() < out[0, 32, 32, 0].item()
    assert out[0, 0, 0, 0].item() == pytest.approx(0.35, abs=1e-6)


def test_color_shading_greens_corners():
    image = torch.full((1, 64, 64, 3), 0.5)
    out = run(image, color_shading=1.0)
    corner_ratio = (out[0, 0, 0, 1] / out[0, 0, 0, 0]).item()
    center_ratio = (out[0, 32, 32, 1] / out[0, 32, 32, 0]).item()
    assert corner_ratio > center_ratio
    # the meshgrid has no exact center pixel, so the ratio is only approximately 1
    assert center_ratio == pytest.approx(1.0, abs=1e-3)


def test_purple_haze_lifts_r_and_b_more_than_g():
    image = torch.full((1, 32, 32, 3), 0.9)
    out = run(image, purple_haze=0.5)
    delta = out - image
    assert (delta[..., 0] > delta[..., 1]).all()
    assert (delta[..., 2] > delta[..., 1]).all()


def test_warmth_shifts_white_balance():
    image = torch.full((1, 32, 32, 3), 0.5)
    out = run(image, warmth=0.1)
    assert out[0, 0, 0, 0].item() > 0.5
    assert out[0, 0, 0, 1].item() == 0.5
    assert out[0, 0, 0, 2].item() < 0.5


def test_resolution_cap_downscales_to_8mp():
    image = torch.rand(1, 3000, 4000, 3)  # 12 MP
    out = run(image, resolution_cap=True)
    assert out.shape[1] * out.shape[2] <= 3264 * 2448
    assert abs(out.shape[1] * out.shape[2] - 3264 * 2448) < 4000
    assert out.shape[2] == round(4000 * (3264 * 2448 / 12000000) ** 0.5)


def test_resolution_cap_off_and_small_image_untouched():
    image = torch.rand(1, 512, 640, 3)
    out = run(image, resolution_cap=True)
    assert out.shape == image.shape


def test_chroma_subsampling_keeps_flat_image():
    # uniform color: chroma is constant, 2x2 averaging must be a no-op
    image = torch.rand(1, 32, 32, 3).mean(dim=3, keepdim=True).expand(1, 32, 32, 3).contiguous()
    out = run(image, chroma_subsampling=True)
    assert torch.allclose(out, image, atol=1e-6)


def test_chroma_subsampling_blurs_color_edges():
    # hard vertical red/blue split on an odd column: the 2x2 chroma block
    # straddles the seam and must soften it
    image = torch.zeros(1, 32, 64, 3)
    image[..., 31:, 0] = 1.0
    image[..., :31, 2] = 1.0
    out = run(image, chroma_subsampling=True)
    assert (out[0, 16, 30] - image[0, 16, 30]).abs().sum() > 0.1
    # luma detail passes through unchanged
    gray = torch.linspace(0, 1, 64).view(1, 1, 64, 1).expand(1, 32, 64, 3).contiguous()
    out_gray = run(gray, chroma_subsampling=True)
    assert torch.allclose(out_gray, gray, atol=1e-6)


def test_batch_processed():
    image = torch.rand(2, 32, 32, 3)
    out = run(image, seed=5, noise_strength=0.05)
    assert out.shape == (2, 32, 32, 3)
    assert not torch.equal(out[0], out[1])
    # the generator draws samples in batch order, so frame 0 matches a solo run
    single = run(image[0:1], seed=5, noise_strength=0.05)
    assert torch.allclose(out[0:1], single, atol=1e-6)
