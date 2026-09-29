import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

ScaleImageToTotalPixelsCoverCrop = _impl.ScaleImageToTotalPixelsCoverCrop


def run(image, megapixels=1.0, steps=1, method="bilinear"):
    return ScaleImageToTotalPixelsCoverCrop.execute(image, method, megapixels, steps)[0]


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
