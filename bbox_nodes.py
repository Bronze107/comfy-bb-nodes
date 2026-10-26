import random

from comfy_api.latest import io
from comfy_api.latest._io import _DynamicGroup
from comfy_extras.color_util import normalize_palette
from comfy_extras.nodes_bounding_boxes import boxes_to_regions, build_elements, render_preview


def random_color(rng):
    return "#%02X%02X%02X" % (rng.randrange(256), rng.randrange(256), rng.randrange(256))


def box_metadata(text="", desc="", palette=()):
    return {
        "type": "text" if text else "obj",
        "text": text,
        "desc": desc,
        "palette": list(palette),
    }


def random_box(rng, width, height, min_size, max_size):
    # Widget min/max do not constrain linked values, so clamp against the canvas.
    w = max(1, min(width, round(rng.uniform(min_size, max_size) * width)))
    h = max(1, min(height, round(rng.uniform(min_size, max_size) * height)))
    return {
        "x": rng.randint(0, max(0, width - w)),
        "y": rng.randint(0, max(0, height - h)),
        "width": w,
        "height": h,
    }


def random_bboxes(count, width, height, min_size, max_size, seed, use_random_color=True, color_palette=()):
    rng = random.Random(seed)
    shared = normalize_palette(color_palette or [])
    boxes = []
    for _ in range(count):
        box = random_box(rng, width, height, min_size, max_size)
        box["metadata"] = box_metadata(
            palette=[random_color(rng)] if use_random_color else shared
        )
        boxes.append(box)
    return boxes


def boxes_from_rows(rows, width, height, min_size, max_size, seed):
    rng = random.Random(seed)
    boxes = []
    for row in rows:
        text = row.get("text") or ""
        desc = row.get("desc") or ""
        palette = normalize_palette(row.get("color_palette") or [])
        if not (text or desc or palette):
            continue
        box = random_box(rng, width, height, min_size, max_size)
        box["metadata"] = box_metadata(text, desc, palette or [random_color(rng)])
        boxes.append(box)
    return boxes


def render_boxes(boxes, width, height):
    regions = boxes_to_regions(boxes, width, height)
    return io.NodeOutput(
        render_preview(regions, width, height),
        [boxes],
        build_elements(regions),
    )


def describe_box(box):
    if not isinstance(box, dict):
        return box
    meta = box.get("metadata")
    meta = dict(meta) if isinstance(meta, dict) else box_metadata()
    label = box.get("label")
    if label and not meta.get("desc"):
        meta["desc"] = str(label)
    out = dict(box)
    out["metadata"] = meta
    return out


def describe_boxes(bboxes):
    # Same shapes detection nodes emit: dict, list[dict], or list[list[dict]] per frame.
    if isinstance(bboxes, dict):
        return describe_box(bboxes)
    if isinstance(bboxes, list):
        if not bboxes:
            return []
        if isinstance(bboxes[0], list):
            return [describe_boxes(frame) for frame in bboxes]
        return [describe_box(box) for box in bboxes]
    return bboxes


class RandomBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RandomBoundingBoxes",
            display_name="Random Bounding Boxes",
            category="utilities",
            search_aliases=["random bbox", "bounding box", "random boxes", "generate boxes"],
            description="Generate count random bounding boxes inside a width x height canvas. "
                        "Each box gets a random size within min_size/max_size, a random position "
                        "that keeps it inside the canvas, and optionally a random color. Outputs a "
                        "preview image, per-frame pixel boxes, and Ideogram-style elements.",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas width in pixels. Box x and width stay within it."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas height in pixels. Box y and height stay within it."),
                io.Int.Input("count", default=1, min=1, max=64,
                             tooltip="How many random boxes to generate."),
                io.Float.Input("min_size", default=0.1, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum box size as a fraction of the canvas. Width uses "
                                       "canvas width, height uses canvas height, independently - "
                                       "aspect ratio is not constrained."),
                io.Float.Input("max_size", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Maximum box size, same fraction semantics as min_size."),
                io.Boolean.Input("random_color", default=True,
                                 tooltip="Give each box a random color. Off hands every box the "
                                         "color_palette below instead."),
                io.Colors.Input("color_palette", socketless=False,
                                tooltip="Hex colors applied to every box when random_color is off. "
                                        "Empty leaves the boxes without a palette: grey in the "
                                        "preview, no color_palette in elements."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True, tooltip="Random seed."),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes",
                                      tooltip="Per-frame pixel boxes with metadata.palette. "
                                              "DrawBBoxes draws its own colors and ignores "
                                              "metadata; CreateBoundingBoxes and elements keep "
                                              "them."),
                io.Array.Output(display_name="elements"),
            ],
        )

    @classmethod
    def execute(cls, width, height, count, min_size, max_size, seed, random_color=True,
                color_palette=None):
        boxes = random_bboxes(count, width, height, min_size, max_size, seed, random_color,
                              color_palette or [])
        return render_boxes(boxes, width, height)


class RandomBoundingBoxesAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RandomBoundingBoxesAdvanced",
            display_name="Random Bounding Boxes (Advanced)",
            category="utilities",
            search_aliases=["random bbox", "bounding box", "random boxes", "bbox rows"],
            description="One row per bounding box. Each row sets its own text, description and "
                        "color palette while size and position stay random inside the canvas. A "
                        "row left entirely empty is skipped. Outputs a preview image, per-frame "
                        "pixel boxes, and Ideogram-style elements.",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas width in pixels. Box x and width stay within it."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas height in pixels. Box y and height stay within it."),
                io.Float.Input("min_size", default=0.1, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum box size as a fraction of the canvas. Width uses "
                                       "canvas width, height uses canvas height, independently - "
                                       "aspect ratio is not constrained."),
                io.Float.Input("max_size", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Maximum box size, same fraction semantics as min_size."),
                _DynamicGroup.Input(
                    "boxes",
                    template=[
                        io.String.Input("text",
                                        tooltip="Text drawn inside the box; also sets the element "
                                                "type to text so the text reaches elements."),
                        io.String.Input("desc", tooltip="Description of what the box covers."),
                        io.Colors.Input("color_palette",
                                        tooltip="Hex colors for this box. Empty gives it a random "
                                                "color."),
                    ],
                    min=0,
                    max=20,
                    group_name="Box",
                    tooltip="One row per box. Size and position stay random; text, description "
                            "and color palette come from the row.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True, tooltip="Random seed."),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes",
                                      tooltip="Per-frame pixel boxes with metadata.palette. "
                                              "DrawBBoxes draws its own colors and ignores "
                                              "metadata; CreateBoundingBoxes and elements keep "
                                              "them."),
                io.Array.Output(display_name="elements"),
            ],
        )

    @classmethod
    def execute(cls, width, height, min_size, max_size, seed, boxes=None):
        rows = boxes or []
        return render_boxes(boxes_from_rows(rows, width, height, min_size, max_size, seed), width, height)


class DescribeBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DescribeBoundingBoxes",
            display_name="Describe Bounding Boxes",
            category="utilities",
            search_aliases=["bbox", "bounding box", "detection", "label", "description", "rt_detr"],
            description="Copy detection labels into each box's metadata description so Create "
                        "Bounding Boxes shows them in the canvas. Boxes that already carry a "
                        "description keep it.",
            inputs=[
                io.BoundingBox.Input("bboxes", force_input=True,
                                     tooltip="Detection boxes from RT-DETR, SAM3, MediaPipe or any "
                                             "other source, in pixel space."),
            ],
            outputs=[
                io.BoundingBox.Output(display_name="bboxes",
                                      tooltip="Same boxes with metadata.desc filled from the "
                                              "detection labels. Plug into Create Bounding Boxes."),
            ],
        )

    @classmethod
    def execute(cls, bboxes) -> io.NodeOutput:
        return io.NodeOutput(describe_boxes(bboxes))
