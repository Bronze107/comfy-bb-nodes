import importlib.util
from pathlib import Path

from comfy_extras.nodes_bounding_boxes import boxes_to_regions

_IMPL = Path(__file__).resolve().parent.parent / "bbox_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_bbox_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

describe_box = _impl.describe_box
describe_boxes = _impl.describe_boxes


def det(x=10, y=20, width=30, height=40, label="person", score=0.87):
    box = {"x": x, "y": y, "width": width, "height": height, "score": score}
    if label is not None:
        box["label"] = label
    return box


def test_label_becomes_desc():
    out = describe_box(det())
    assert out["metadata"]["desc"] == "person"
    assert out["metadata"]["type"] == "obj"
    assert out["score"] == 0.87


def test_existing_desc_is_kept():
    box = det()
    box["metadata"] = {"type": "obj", "text": "", "desc": "a man in a coat", "palette": []}
    out = describe_box(box)
    assert out["metadata"]["desc"] == "a man in a coat"


def test_existing_metadata_keys_survive():
    box = det()
    box["metadata"] = {"type": "text", "text": "STOP", "desc": "", "palette": ["#FF8800"]}
    out = describe_box(box)
    assert out["metadata"] == {
        "type": "text", "text": "STOP", "desc": "person", "palette": ["#FF8800"],
    }


def test_box_without_label_gets_empty_desc():
    out = describe_box(det(label=None))
    assert out["metadata"]["desc"] == ""
    assert out["x"] == 10


def test_non_dict_passes_through():
    assert describe_box(None) is None
    assert describe_box([1, 2]) == [1, 2]


def test_describe_boxes_preserves_input_shape():
    single = describe_boxes(det())
    assert single["metadata"]["desc"] == "person"

    flat = describe_boxes([det(label="dog"), det(label="cat")])
    assert [b["metadata"]["desc"] for b in flat] == ["dog", "cat"]

    per_frame = describe_boxes([[det(label="dog")], [], [det(label="cat"), det(label="dog")]])
    assert [b["metadata"]["desc"] for b in per_frame[0]] == ["dog"]
    assert per_frame[1] == []
    assert [b["metadata"]["desc"] for b in per_frame[2]] == ["cat", "dog"]


def test_describe_boxes_empty_and_passthrough():
    assert describe_boxes([]) == []
    assert describe_boxes(None) is None


def test_described_boxes_feed_create_bounding_boxes():
    # CreateBoundingBoxes reads the description from metadata.desc via boxes_to_regions.
    regions = boxes_to_regions(describe_boxes([det()]), 640, 480)
    assert regions[0]["desc"] == "person"
    assert regions[0]["type"] == "obj"


def test_execute_outputs():
    out = _impl.DescribeBoundingBoxes.execute([[det(label="dog", score=0.5)]])
    assert out[0][0][0]["metadata"]["desc"] == "dog"


def test_schema_defines():
    schema = _impl.DescribeBoundingBoxes.define_schema()
    assert schema.inputs
    assert schema.outputs
