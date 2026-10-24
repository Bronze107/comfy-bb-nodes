import importlib.util
import random
import re
from pathlib import Path

import pytest

_IMPL = Path(__file__).resolve().parent.parent / "bbox_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_bbox_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

random_color = _impl.random_color
box_metadata = _impl.box_metadata
random_bboxes = _impl.random_bboxes
boxes_from_rows = _impl.boxes_from_rows

HEX = re.compile(r"^#[0-9A-F]{6}$")


def test_same_seed_is_deterministic():
    a = random_bboxes(5, 512, 512, 0.1, 0.5, 42)
    b = random_bboxes(5, 512, 512, 0.1, 0.5, 42)
    assert a == b


def test_different_seed_differs():
    a = random_bboxes(5, 512, 512, 0.1, 0.5, 42)
    b = random_bboxes(5, 512, 512, 0.1, 0.5, 43)
    assert a != b


def test_count_boxes_produced():
    assert len(random_bboxes(7, 256, 256, 0.1, 0.5, 0)) == 7


def test_boxes_stay_inside_canvas():
    for box in random_bboxes(50, 640, 480, 0.05, 0.9, 7):
        assert 0 <= box["x"]
        assert 0 <= box["y"]
        assert box["x"] + box["width"] <= 640
        assert box["y"] + box["height"] <= 480


def test_size_within_fraction_range():
    for box in random_bboxes(50, 1000, 800, 0.1, 0.5, 7):
        # +/- 1px absorbs the round() in random_bboxes
        assert 0.1 * 1000 - 1 <= box["width"] <= 0.5 * 1000 + 1
        assert 0.1 * 800 - 1 <= box["height"] <= 0.5 * 800 + 1


def test_width_and_height_use_their_own_axis():
    # A wide-short canvas makes the independent-axis rule observable.
    boxes = random_bboxes(200, 1000, 100, 0.2, 0.2, 3)
    assert all(box["width"] == pytest.approx(200, abs=1) for box in boxes)
    assert all(box["height"] == pytest.approx(20, abs=1) for box in boxes)


def test_metadata_palette_is_uppercase_hex():
    for box in random_bboxes(20, 256, 256, 0.1, 0.5, 0):
        palette = box["metadata"]["palette"]
        assert len(palette) == 1
        assert HEX.match(palette[0]), palette[0]


def test_random_color_format():
    for seed in range(20):
        assert HEX.match(random_color(random.Random(seed)))


def test_inverted_size_range_still_works():
    boxes = random_bboxes(50, 1000, 800, 0.5, 0.1, 7)
    for box in boxes:
        assert 0.1 * 1000 - 1 <= box["width"] <= 0.5 * 1000 + 1
        assert 0.1 * 800 - 1 <= box["height"] <= 0.5 * 800 + 1


def test_zero_size_clamps_to_one_pixel():
    for box in random_bboxes(10, 256, 256, 0.0, 0.0, 0):
        assert box["width"] == 1
        assert box["height"] == 1


def test_size_above_one_clamps_to_canvas():
    # min/max are widget-side only; a linked FLOAT can deliver max_size > 1.0.
    boxes = random_bboxes(20, 256, 128, 0.9, 1.5, 0)
    for box in boxes:
        assert 1 <= box["width"] <= 256
        assert 1 <= box["height"] <= 128
        assert box["x"] + box["width"] <= 256
        assert box["y"] + box["height"] <= 128


def test_non_positive_count_gives_empty_list():
    assert random_bboxes(0, 256, 256, 0.1, 0.5, 0) == []
    assert random_bboxes(-3, 256, 256, 0.1, 0.5, 0) == []


def test_execute_outputs():
    out = _impl.RandomBoundingBoxes.execute(
        width=256, height=128, count=3, min_size=0.1, max_size=0.5, seed=0
    )
    preview, bboxes, elements = out[0], out[1], out[2]
    assert tuple(preview.shape) == (1, 128, 256, 3)
    assert len(bboxes) == 1  # single per-frame list
    assert bboxes[0] == random_bboxes(3, 256, 128, 0.1, 0.5, 0)
    assert len(elements) == 3
    for element in elements:
        ymin, xmin, ymax, xmax = element["bbox"]
        assert 0 <= ymin < ymax <= 1000
        assert 0 <= xmin < xmax <= 1000
        assert len(element["color_palette"]) == 1
        assert HEX.match(element["color_palette"][0]), element["color_palette"]


def test_execute_empty_count():
    out = _impl.RandomBoundingBoxes.execute(
        width=256, height=128, count=0, min_size=0.1, max_size=0.5, seed=0
    )
    assert tuple(out[0].shape) == (1, 128, 256, 3)
    assert out[1] == [[]]
    assert out[2] == []


def test_random_color_off_drops_palette():
    for box in random_bboxes(10, 256, 256, 0.1, 0.5, 0, use_random_color=False):
        assert box["metadata"]["palette"] == []


def test_random_color_off_uses_color_palette():
    boxes = random_bboxes(5, 256, 256, 0.1, 0.5, 0, use_random_color=False,
                          color_palette=["#ff8800", "#00ff88"])
    for box in boxes:
        assert box["metadata"]["palette"] == ["#FF8800", "#00FF88"]


def test_random_color_off_skips_color_palette_in_elements():
    out = _impl.RandomBoundingBoxes.execute(
        width=256, height=128, count=2, min_size=0.1, max_size=0.5, seed=0,
        random_color=False,
    )
    for element in out[2]:
        assert "color_palette" not in element


def test_box_metadata_infers_type_from_text():
    assert box_metadata()["type"] == "obj"
    assert box_metadata(text="hi")["type"] == "text"
    assert box_metadata(text="hi")["text"] == "hi"
    assert box_metadata(desc="a ball")["desc"] == "a ball"
    assert box_metadata(palette=["#FF8800"])["palette"] == ["#FF8800"]
    assert box_metadata()["palette"] == []


def test_boxes_from_rows_keeps_row_metadata():
    rows = [
        {"text": "hello", "desc": "a sign", "color_palette": ["#FF8800", "#00FF88"]},
        {"text": "", "desc": "a ball", "color_palette": []},
        {"text": "bye", "desc": "", "color_palette": []},
    ]
    boxes = boxes_from_rows(rows, 512, 512, 0.1, 0.5, 0)
    assert len(boxes) == 3
    assert boxes[0]["metadata"] == {
        "type": "text", "text": "hello", "desc": "a sign",
        "palette": ["#FF8800", "#00FF88"],
    }
    assert boxes[1]["metadata"]["type"] == "obj"
    assert boxes[1]["metadata"]["desc"] == "a ball"
    assert boxes[2]["metadata"]["type"] == "text"


def test_boxes_from_rows_blank_palette_gets_a_random_color():
    boxes = boxes_from_rows([{"text": "hi", "desc": "", "color_palette": []}], 256, 256, 0.1, 0.5, 0)
    assert len(boxes[0]["metadata"]["palette"]) == 1
    assert HEX.match(boxes[0]["metadata"]["palette"][0])


def test_boxes_from_rows_normalizes_palette():
    boxes = boxes_from_rows(
        [{"text": "hi", "desc": "", "color_palette": ["#ff8800", "", "#00ff88"]}],
        256, 256, 0.1, 0.5, 0,
    )
    assert boxes[0]["metadata"]["palette"] == ["#FF8800", "#00FF88"]


def test_boxes_from_rows_skips_empty_rows():
    rows = [
        {"text": "", "desc": "", "color_palette": []},
        {"text": None, "desc": None, "color_palette": None},
        {"text": "hi", "desc": "", "color_palette": []},
    ]
    boxes = boxes_from_rows(rows, 256, 256, 0.1, 0.5, 0)
    assert len(boxes) == 1
    assert boxes[0]["metadata"]["text"] == "hi"


def test_boxes_from_rows_empty_input():
    assert boxes_from_rows([], 256, 256, 0.1, 0.5, 0) == []


def test_advanced_execute_outputs():
    rows = [
        {"text": "hello", "desc": "a sign", "color_palette": ["#FF8800", "#00FF88"]},
        {"text": "", "desc": "a ball", "color_palette": []},
        {"text": "", "desc": "", "color_palette": []},
    ]
    out = _impl.RandomBoundingBoxesAdvanced.execute(
        width=256, height=128, min_size=0.1, max_size=0.5, seed=0, boxes=rows
    )
    assert tuple(out[0].shape) == (1, 128, 256, 3)
    assert out[1] == [boxes_from_rows(rows, 256, 128, 0.1, 0.5, 0)]
    assert len(out[2]) == 2  # the empty row is skipped
    assert out[2][0]["type"] == "text"
    assert out[2][0]["text"] == "hello"
    assert out[2][0]["desc"] == "a sign"
    assert out[2][0]["color_palette"] == ["#FF8800", "#00FF88"]


def test_advanced_execute_without_rows():
    out = _impl.RandomBoundingBoxesAdvanced.execute(
        width=256, height=128, min_size=0.1, max_size=0.5, seed=0
    )
    assert tuple(out[0].shape) == (1, 128, 256, 3)
    assert out[1] == [[]]
    assert out[2] == []


def test_schemas_define():
    for cls in (_impl.RandomBoundingBoxes, _impl.RandomBoundingBoxesAdvanced):
        schema = cls.define_schema()
        assert schema.inputs
        assert schema.outputs
