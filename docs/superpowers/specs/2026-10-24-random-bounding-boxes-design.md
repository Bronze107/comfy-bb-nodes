# Random Bounding Boxes — Design

## Goal

Add a node to `comfy-bb-nodes` that generates N random bounding boxes inside a
canvas — random size, random position, random color — for testing, layout
prototyping, and seeding `CreateBoundingBoxes` / `LayersFromBoundingBoxes`
without drawing boxes by hand.

## Background

Core ComfyUI has `CreateBoundingBoxes` (`comfy_extras/nodes_bounding_boxes.py`)
for hand-drawing boxes on a canvas, and detection nodes that *emit* boxes
(`RTDETR_detect`, `SDPoseFaceBBoxes`), but nothing that fabricates random boxes.

Core's box helpers already do the parts that are not about randomness:
`boxes_to_regions` maps pixel boxes to fraction-space regions, `render_preview`
draws regions on a canvas, `build_elements` builds Ideogram-style elements. This
node reuses all three rather than reimplementing them, so its preview style and
`elements` format are identical to `CreateBoundingBoxes`.

## Behavior (confirmed with user)

- A `count` input generates N boxes per execution.
- Box size range is expressed as canvas fractions (0–1). Width is scaled against
  canvas width and height against canvas height **independently**, so aspect
  ratio is not constrained.
- Outputs mirror `CreateBoundingBoxes`: preview image, per-frame pixel boxes,
  Ideogram-style elements.
- Each box carries a random color in `metadata.palette`.

## Node definition

New file `custom_nodes/comfy-bb-nodes/bbox_nodes.py`, registered in
`custom_nodes/comfy-bb-nodes/__init__.py`.

- **node_id**: `RandomBoundingBoxes`
- **display_name**: "Random Bounding Boxes"
- **category**: `utilities` (same as core `CreateBoundingBoxes`)
- **inputs**:
  - `width` (`io.Int`, default `1024`, min `64`, max `16384`, step `16`)
  - `height` (`io.Int`, default `1024`, min `64`, max `16384`, step `16`)
  - `count` (`io.Int`, default `1`, min `1`, max `64`)
  - `min_size` (`io.Float`, default `0.1`, min `0.0`, max `1.0`, step `0.01`)
  - `max_size` (`io.Float`, default `0.5`, min `0.0`, max `1.0`, step `0.01`)
  - `seed` (`io.Int`, default `0`, min `0`, max `0xFFFFFFFFFFFFFFFF`,
    `control_after_generate=True`)
- **outputs**:
  - `preview` (`io.Image`) — boxes drawn in their random colors
  - `bboxes` (`io.BoundingBox`) — `[[{x, y, width, height, metadata}, ...]]`
  - `elements` (`io.Array`) — Ideogram elements with `color_palette`

## Implementation

```python
def random_bboxes(count, width, height, min_size, max_size, seed):
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
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
```

`execute` delegates rendering to core:

```python
boxes = random_bboxes(count, width, height, min_size, max_size, seed)
regions = boxes_to_regions(boxes, width, height)
return io.NodeOutput(render_preview(regions, width, height), [boxes], build_elements(regions))
```

### Output type

`io.BoundingBox`, not `io.BoundingBoxes`. The two are distinct socket types
(`BOUNDING_BOX` vs `BOUNDING_BOXES`) and `comfy_execution/validation.py` matches
exact type strings only. Nothing in core produces `BOUNDING_BOXES`; every
consumer accepts `BOUNDING_BOX`. The value is the per-frame `list[list[dict]]`
shape documented at `comfy_api_nodes/nodes_bytedance.py` and emitted by
`CreateBoundingBoxes` via `fractions_to_bbox_frame`.

### Metadata is preserved

The `bboxes` output emits the original pixel boxes rather than
`fractions_to_bbox_frame(regions, ...)`, which strips `metadata`. Keeping it
means the random color survives into `CreateBoundingBoxes` and
`LayersFromBoundingBoxes` via `normalize_incoming_boxes`. Geometry-only
consumers (`DrawBBoxes`, `CropByBBoxes`) ignore the extra key. This is
stricter than core, whose own `bboxes` output drops the palette.

### Seed

Local `random.Random(seed)`, matching `text_nodes.py`. Adds
`control_after_generate=True`, a deliberate deviation from `text_nodes.py`:
without it every queue run reproduces identical boxes, which defeats a random
generator. Matches core seed inputs.

## Edge cases

- `min_size`/`max_size` are widget-side only; a linked FLOAT can deliver values
  above 1.0. Box size is clamped to the canvas, otherwise `rng.randint` raises
  `ValueError: empty range in randrange`.
- A fraction that rounds to 0 clamps to 1px rather than a zero-area box.
- Inverted `min_size > max_size` is not rejected — `random.uniform` already
  returns values between the two bounds.
- `count <= 0` yields an empty box list and a plain grey preview. Not rejected:
  empty output is a valid result and the node stays executable.
- `randint(0, width - w)` keeps boxes fully inside the canvas, so large boxes
  cluster toward the left/top. Intended.

## Testing

`tests/test_random_bboxes.py` loads `bbox_nodes.py` standalone and exercises the
pure helpers and `execute`:

- seed determinism and seed variation
- `count` honored
- boxes stay inside the canvas
- sizes within the fraction range, per axis
- `metadata.palette` is a one-element uppercase `#RRGGBB`
- inverted size range, zero size, size above 1.0, non-positive count
- `execute` output shapes and `elements` format

## Non-goals

- Deriving canvas size from an image input (core `CreateBoundingBoxes` has the
  same limitation).
- Separate x/y position bounds.
- Making the node an output node that writes preview files.

## Verification

From the ComfyUI repo root:

```
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```

All tests pass, and the pack still loads as a custom node.
