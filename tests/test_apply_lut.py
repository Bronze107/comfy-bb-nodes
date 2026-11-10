import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

ApplyLut = _impl.ApplyLut
parse_cube = _impl.parse_cube
apply_lut = _impl.apply_lut
LUTS_DIR = _impl.LUTS_DIR


def identity_cube(size):
    lines = [f'LUT_3D_SIZE {size}', 'TITLE pytest identity']
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}")
    return "\n".join(lines)


def run(image, cube_text, strength=1.0):
    lut_path = Path(LUTS_DIR) / "pytest_tmp_lut.cube"
    lut_path.write_text(cube_text, encoding="utf-8")
    try:
        return ApplyLut.execute(image, "pytest_tmp_lut.cube", strength)[0]
    finally:
        lut_path.unlink()


def test_parse_cube_3d_reshapes_blue_green_red():
    lut, domain_min, domain_max = parse_cube(identity_cube(2))
    assert lut.shape == (2, 2, 2, 3)
    # rows vary red fastest: row 3 of a size-2 file is (b=0, g=1, r=1), value (1, 1, 0)
    assert lut[0, 1, 1].tolist() == [1.0, 1.0, 0.0]
    assert lut[1, 1, 1].tolist() == [1.0, 1.0, 1.0]
    assert (domain_min == 0).all() and (domain_max == 1).all()


def test_parse_cube_1d_and_domain():
    text = "LUT_1D_SIZE 2\nDOMAIN_MIN 0.1 0.1 0.1\nDOMAIN_MAX 1.1 1.1 1.1\n0 0 0\n1 1 1"
    lut, domain_min, domain_max = parse_cube(text)
    assert lut.shape == (2, 3)
    assert domain_min.tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert domain_max.tolist() == pytest.approx([1.1, 1.1, 1.1])


def test_parse_cube_rejects_missing_header():
    with pytest.raises(ValueError):
        parse_cube("0 0 0\n1 1 1")


def test_parse_cube_rejects_wrong_entry_count():
    with pytest.raises(ValueError):
        parse_cube("LUT_3D_SIZE 2\n" + "\n".join(["0.5 0.5 0.5"] * 7))


def test_identity_lut_preserves_image():
    image = torch.rand(1, 32, 32, 3)
    out = run(image, identity_cube(17))
    assert torch.allclose(out, image, atol=1e-6)


def test_red_fastest_row_order():
    # every entry is the blue coordinate of its grid position: a pure blue pixel
    # must read node (r=0, g=0, b=1), a pure red pixel node (r=1, g=0, b=0)
    lines = ["LUT_3D_SIZE 2"]
    for b in (0, 1):
        for g in (0, 1):
            for r in (0, 1):
                lines.append(f"{b}.0 {b}.0 {b}.0")
    image = torch.zeros(1, 1, 2, 3)
    image[0, 0, 0] = torch.tensor([0.0, 0.0, 1.0])
    image[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    out = run(image, "\n".join(lines))
    assert out[0, 0, 0].tolist() == [1.0, 1.0, 1.0]
    assert out[0, 0, 1].tolist() == [0.0, 0.0, 0.0]


def test_strength_blends_toward_graded_color():
    # constant red LUT: gray 0.5 grades to (1, 0, 0)
    image = torch.full((1, 4, 4, 3), 0.5)
    out = run(image, "LUT_3D_SIZE 2\n" + "1 0 0\n" * 8, strength=0.5)
    assert out[0, 0, 0].tolist() == pytest.approx([0.75, 0.25, 0.25])


def test_zero_strength_returns_input_unchanged():
    image = torch.rand(1, 8, 8, 3)
    out = run(image, "LUT_3D_SIZE 2\n" + "1 0 0\n" * 8, strength=0.0)
    assert torch.equal(out, image)


def test_alpha_channel_untouched():
    image = torch.rand(1, 8, 8, 4)
    out = run(image, "LUT_3D_SIZE 2\n" + "1 0 0\n" * 8)
    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image[..., 3])


def test_1d_lut_maps_each_channel_through_its_column():
    # table columns: red = 0..1, green and blue = constant 0
    text = "LUT_1D_SIZE 2\n0 0 0\n1 0 0"
    image = torch.zeros(1, 1, 1, 3)
    image[0, 0, 0] = torch.tensor([0.25, 0.5, 0.75])
    out = run(image, text)
    assert out[0, 0, 0].tolist() == pytest.approx([0.25, 0.0, 0.0])


def test_domain_rescales_pixel_coordinates():
    # inside a 0..0.5 domain, pixel 0.25 sits at the halfway point of the grid
    image = torch.full((1, 1, 1, 3), 0.25)
    out = run(image, "LUT_3D_SIZE 2\nDOMAIN_MAX 0.5 0.5 0.5\n" + identity_cube(2).split("\n", 1)[1])
    assert out[0, 0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_batch_matches_solo_frames():
    image = torch.rand(2, 16, 16, 3)
    out = run(image, identity_cube(5), strength=0.8)
    assert out.shape == image.shape
    single = run(image[0:1], identity_cube(5), strength=0.8)
    assert torch.allclose(out[0:1], single, atol=1e-6)


def test_missing_lut_file_raises():
    with pytest.raises(FileNotFoundError):
        ApplyLut.execute(torch.rand(1, 4, 4, 3), "pytest_does_not_exist.cube", 1.0)
