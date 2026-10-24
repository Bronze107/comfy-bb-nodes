import random

from comfy_api.latest import io
from comfy_extras.nodes_bounding_boxes import boxes_to_regions, build_elements, render_preview


def random_color(rng):
    return "#%02X%02X%02X" % (rng.randrange(256), rng.randrange(256), rng.randrange(256))


def random_bboxes(count, width, height, min_size, max_size, seed):
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        # Widget min/max do not constrain linked values, so clamp against the canvas.
        w = max(1, min(width, round(rng.uniform(min_size, max_size) * width)))
        h = max(1, min(height, round(rng.uniform(min_size, max_size) * height)))
        boxes.append({
            "x": rng.randint(0, max(0, width - w)),
            "y": rng.randint(0, max(0, height - h)),
            "width": w,
            "height": h,
            "metadata": {"palette": [random_color(rng)]},
        })
    return boxes


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
                        "that keeps it inside the canvas, and a random color carried in "
                        "metadata.palette. Outputs a preview image, per-frame pixel boxes, and "
                        "Ideogram-style elements.",
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
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True, tooltip="Random seed."),
            ],
            outputs=[
                io.Image.Output(display_name="preview",
                                tooltip="Boxes drawn in their random colors."),
                io.BoundingBox.Output(display_name="bboxes",
                                      tooltip="Per-frame pixel boxes with metadata.palette. "
                                              "DrawBBoxes draws its own colors and ignores "
                                              "metadata; CreateBoundingBoxes and elements keep "
                                              "them."),
                io.Array.Output(display_name="elements",
                                tooltip="Ideogram-style elements with color_palette."),
            ],
        )

    @classmethod
    def execute(cls, width, height, count, min_size, max_size, seed):
        boxes = random_bboxes(count, width, height, min_size, max_size, seed)
        regions = boxes_to_regions(boxes, width, height)
        return io.NodeOutput(
            render_preview(regions, width, height),
            [boxes],
            build_elements(regions),
        )
