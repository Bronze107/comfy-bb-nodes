import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

ScaleImageToTotalPixelsCoverCrop = _impl.ScaleImageToTotalPixelsCoverCrop


def run(image, megapixels=1.0, steps=1, method="bilinear", only_downscale=False):
    return ScaleImageToTotalPixelsCoverCrop.execute(image, method, megapixels, steps, only_downscale)[0]


def test_output_shape_step_aligned():
    image = torch.rand(1, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    # scale_by = sqrt(1048576 / (1920*1080)) = 0.71109
    # width: round(1365.29 / 8) * 8 = 1368, height: round(767.98 / 8) * 8 = 768
    assert out.shape == (1, 768, 1368, 3)


def test_total_pixels_close_to_target():
    image = torch.rand(1, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    target = 1.0 * 1024 * 1024
    assert abs(out.shape[1] * out.shape[2] - target) < 1368 * 8 + 768 * 8


def test_batch_frames_share_output_size():
    image = torch.rand(2, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    assert out.shape == (2, 768, 1368, 3)


def test_identity_when_scale_is_one():
    image = torch.rand(1, 1024, 1024, 3)
    out = run(image, megapixels=1.0, steps=1, method="nearest-exact")
    assert out.shape == image.shape
    assert torch.equal(out, image)


def test_center_crop_trims_edges_symmetrically():
    # Pixel value encodes its original row position: 0 at top, 1 at bottom.
    image = torch.linspace(0.0, 1.0, 1080).view(1, 1080, 1, 1).expand(1, 1080, 1920, 3).contiguous()
    out = run(image, megapixels=1.0, steps=8, method="bilinear")
    assert out.shape == (1, 768, 1368, 3)
    # 1368x770 scaled result is cropped to 1368x768: one row trimmed from each
    # edge. First and last visible rows are therefore symmetric around 0.5.
    assert out[0, 0, 0, 0].item() + out[0, -1, 0, 0].item() == pytest.approx(1.0, abs=0.01)
    assert out[0, 383, 0, 0].item() == pytest.approx(0.5, abs=0.01)
    assert out[0, 384, 0, 0].item() == pytest.approx(0.5, abs=0.01)


def test_only_downscale_keeps_small_image_size_aligned():
    image = torch.rand(1, 512, 768, 3)  # 0.39 MP, below the 1 MP target, divisible by 8
    out = run(image, megapixels=1.0, steps=8, only_downscale=True)
    assert out.shape == image.shape
    assert torch.equal(out, image)


def test_only_downscale_aligns_without_upscaling():
    image = torch.rand(1, 500, 500, 3)  # below target; aligned size must not exceed the input
    out = run(image, megapixels=1.0, steps=8, only_downscale=True)
    assert out.shape[1] <= 500 and out.shape[2] <= 500
    assert out.shape[1] % 8 == 0 and out.shape[2] % 8 == 0


def test_only_downscale_still_downscales_large_image():
    image = torch.rand(1, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8, only_downscale=True)
    assert out.shape == (1, 768, 1368, 3)


def test_only_downscale_off_still_upscales():
    image = torch.rand(1, 512, 768, 3)
    out = run(image, megapixels=1.0, steps=8)
    assert out.shape[1] * out.shape[2] > 512 * 768
