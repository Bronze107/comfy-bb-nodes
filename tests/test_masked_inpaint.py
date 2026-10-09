import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

MaskedInpaintCrop = _impl.MaskedInpaintCrop
MaskedInpaintPaste = _impl.MaskedInpaintPaste
masked_region_box = _impl.masked_region_box


def blob_mask(height, width, y0, y1, x0, x1, value=1.0):
    m = torch.zeros(1, height, width)
    m[:, y0:y1, x0:x1] = value
    return m


def crop(image, mask, padding=32, megapixels=0):
    return MaskedInpaintCrop.execute(image, mask, padding, megapixels)


def test_box_grown_by_padding_and_aligned_to_8():
    # mask pixels [200:300, 100:260], padding 32
    box = masked_region_box(blob_mask(512, 512, 200, 300, 100, 260), 32, 512, 512)
    assert box == (64, 168, 296, 336)


def test_box_clamped_at_image_edges():
    # mask bbox [0:10, 0:10] + 32 padding ends at 42, rounds up to the 48px VAE stride
    box = masked_region_box(blob_mask(64, 64, 0, 10, 0, 10), 32, 64, 64)
    assert box == (0, 0, 48, 48)


def test_box_aligns_to_32_for_qwen_2_1():
    # Qwen-Image 2.1 VAE stride: [200:300, 100:260] + 32 padding on a 512x512 frame
    box = masked_region_box(blob_mask(512, 512, 200, 300, 100, 260), 32, 512, 512, alignment=32)
    assert box == (64, 160, 320, 352)


def test_box_clamps_x_by_width_and_y_by_height_on_non_square_images():
    # portrait 960x952: x must stay inside the 952px width, not the 960px height
    box = masked_region_box(blob_mask(960, 952, 200, 800, 400, 948), 32, 960, 952)
    assert box == (368, 168, 952, 832)


def test_soft_and_empty_masks_raise():
    # 0.4 stays below the 0.5 threshold, so this mask is treated as empty
    with pytest.raises(ValueError):
        masked_region_box(blob_mask(512, 512, 100, 200, 100, 200, value=0.4), 0, 512, 512)
    with pytest.raises(ValueError):
        masked_region_box(torch.zeros(1, 512, 512), 0, 512, 512)


def test_crop_slices_image_and_mask():
    image = torch.rand(1, 512, 512, 3)
    mask = blob_mask(512, 512, 200, 300, 100, 260)
    out_image, out_mask = crop(image, mask)
    assert out_image.shape == (1, 168, 232, 3)
    assert torch.equal(out_image, image[:, 168:336, 64:296, :])
    assert torch.equal(out_mask, mask[:, 168:336, 64:296])


def test_crop_scales_to_megapixels():
    image = torch.rand(1, 512, 512, 3)
    mask = blob_mask(512, 512, 200, 300, 100, 260)
    out_image, out_mask = crop(image, mask, megapixels=1.0)
    # crop is 232x168; sqrt(1048576 / (232*168)) then 8px alignment gives 1200x872
    assert out_image.shape == (1, 872, 1200, 3)
    assert out_mask.shape == (1, 872, 1200)


def test_crop_keeps_size_at_zero_megapixels():
    image = torch.rand(1, 512, 512, 3)
    mask = blob_mask(512, 512, 200, 300, 100, 260)
    out_image, _ = crop(image, mask)
    assert out_image.shape == (1, 168, 232, 3)


def test_paste_composites_through_mask():
    image = torch.zeros(1, 128, 128, 3)
    mask = blob_mask(128, 128, 32, 96, 32, 96)
    inpainted = torch.ones(1, 128, 128, 3)
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 0)[0]
    assert out.shape == image.shape
    assert torch.all(out[:, 40, 40] == 1.0)  # inside the mask
    assert torch.all(out[:, 10, 10] == 0.0)  # outside the mask, untouched


def test_paste_resizes_inpainted_back_to_box():
    image = torch.zeros(1, 512, 512, 3)
    mask = blob_mask(512, 512, 200, 300, 100, 260)
    # any scaled-up inpaint result gets resized back to the box
    inpainted = torch.ones(1, 368, 512, 3)
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 32)[0]
    assert out.shape == image.shape
    assert torch.all(out[:, 250, 150] == 1.0)


def test_crop_resamples_mask_to_image_size():
    image = torch.rand(1, 128, 128, 3)
    mask = torch.ones(1, 64, 64)  # half resolution, covers the same frame
    out_image, out_mask = crop(image, mask, padding=0)
    assert out_image.shape == (1, 128, 128, 3)
    assert out_mask.shape == (1, 128, 128)


def test_paste_resamples_mask_to_image_size():
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.ones(1, 64, 64)
    out = MaskedInpaintPaste.execute(image, mask, torch.ones(1, 128, 128, 3), 32)[0]
    assert out.shape == image.shape
    assert torch.all(out == 1.0)  # full-frame mask covers the whole image


def test_paste_composites_on_non_square_image():
    image = torch.zeros(1, 960, 952, 3)
    mask = blob_mask(960, 952, 200, 800, 400, 948)
    inpainted = torch.ones(1, 664, 584, 3)  # box is 584x664
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 32)[0]
    assert out.shape == image.shape
    assert torch.all(out[:, 500, 700] == 1.0)  # inside the mask
    assert torch.all(out[:, 100, 100] == 0.0)  # outside the box, untouched


def test_paste_makes_inpainted_opaque_on_an_alpha_image():
    image = torch.zeros(1, 128, 128, 4)
    image[..., 3] = 0.5
    mask = blob_mask(128, 128, 32, 96, 32, 96)
    inpainted = torch.ones(1, 128, 128, 3)  # VAE decode output is RGB
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 0)[0]
    assert out.shape == image.shape
    assert torch.all(out[:, 64, 64] == 1.0)  # pasted rgb, alpha forced opaque
    assert torch.all(out[:, 10, 10] == torch.tensor([0.0, 0.0, 0.0, 0.5]))  # outside untouched


def test_paste_drops_alpha_from_inpainted():
    image = torch.zeros(1, 128, 128, 3)
    mask = blob_mask(128, 128, 32, 96, 32, 96)
    inpainted = torch.ones(1, 128, 128, 4)
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 0)[0]
    assert out.shape == image.shape
    assert torch.all(out[:, 64, 64] == 1.0)


def test_full_mask_pastes_everything():
    image = torch.zeros(1, 64, 64, 3)
    mask = torch.ones(1, 64, 64)
    inpainted = torch.ones(1, 64, 64, 3)
    out = MaskedInpaintPaste.execute(image, mask, inpainted, 32)[0]
    assert torch.all(out == 1.0)
