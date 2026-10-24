# Random Bounding Boxes — Design

## Goal

Add nodes to `comfy-bb-nodes` that generate random bounding boxes inside a canvas
— random size, random position — for testing, layout prototyping, and seeding
`CreateBoundingBoxes` / `LayersFromBoundingBoxes` without drawing boxes by hand.

Two nodes: a simple one driven by a `count`, and an advanced one where each box
gets its own text, description and color.

## Background

Core ComfyUI has `CreateBoundingBoxes` (`comfy_extras/nodes_bounding_boxes.py`)
for hand-drawing boxes on a canvas, and detection nodes that *emit* boxes
(`RTDETR_detect`, `SDPoseFaceBBoxes`), but nothing that fabricates random boxes.

Core's box helpers already do the parts that are not about randomness:
`boxes_to_regions` maps pixel boxes to fraction-space regions, `render_preview`
draws regions on a canvas, `build_elements` builds Ideogram-style elements. Both
nodes reuse all three through `render_boxes`, so preview style and `elements`
format are identical to `CreateBoundingBoxes`.

## Behavior (confirmed with user)

- Size range is expressed as canvas fractions (0–1). Width is scaled against
  canvas width and height against canvas height **independently**, so aspect
  ratio is not constrained.
- Outputs mirror `CreateBoundingBoxes`: preview image, per-frame pixel boxes,
  Ideogram-style elements.
- Colors are palettes, not single hex values: `io.Colors`, the same widget
  `BuildJsonPromptIdeogram` uses for `color_palette`, so a box can carry up to
  several colors the way `CreateBoundingBoxes` does.
- Simple node: `count` boxes, and a `random_color` toggle. Off hands every box
  the `color_palette` input instead; an empty palette leaves the boxes grey in
  the preview with no `color_palette` in elements.
- Advanced node: one row per box, each row setting its own `text`, `desc` and
  `color_palette`. A row left entirely empty is skipped.

## Node definitions

New file `custom_nodes/comfy-bb-nodes/bbox_nodes.py`, registered in
`custom_nodes/comfy-bb-nodes/__init__.py`. Both nodes are category `utilities`,
the same as core `CreateBoundingBoxes`.

### RandomBoundingBoxes

- **node_id**: `RandomBoundingBoxes`
- **display_name**: "Random Bounding Boxes"
- **inputs**: `width`, `height` (`io.Int`, 64–16384, step 16, default 1024),
  `count` (`io.Int`, 1–64, default 1), `min_size` / `max_size` (`io.Float`,
  0–1, defaults 0.1 / 0.5), `random_color` (`io.Boolean`, default `True`),
  `color_palette` (`io.Colors`, `socketless=False`), `seed` (`io.Int`,
  `control_after_generate=True`)
- **outputs**: `preview` (`io.Image`), `bboxes` (`io.BoundingBox`),
  `elements` (`io.Array`)

### RandomBoundingBoxesAdvanced

- **node_id**: `RandomBoundingBoxesAdvanced`
- **display_name**: "Random Bounding Boxes (Advanced)"
- **inputs**: `width`, `height`, `min_size`, `max_size`, `boxes`
  (`_DynamicGroup`, 0–20 rows of `text` / `desc` strings and a `color_palette`
  widget), `seed`
- **outputs**: same three as above

`_DynamicGroup` rather than `io.Autogrow`, matching core's `LoadLoraModel`
(`comfy_extras/nodes_lora_stack.py`): an Autogrow template is a single input and
forces it to a socket, so it cannot group three typed fields into one row. The
group name is "Box" and each submitted row arrives in `execute` as a dict.

## Implementation

Shared geometry, metadata and rendering:

```python
def random_box(rng, width, height, min_size, max_size):
    w = max(1, min(width, round(rng.uniform(min_size, max_size) * width)))
    h = max(1, min(height, round(rng.uniform(min_size, max_size) * height)))
    return {
        "x": rng.randint(0, max(0, width - w)),
        "y": rng.randint(0, max(0, height - h)),
        "width": w,
        "height": h,
    }

def box_metadata(text="", desc="", palette=()):
    return {
        "type": "text" if text else "obj",
        "text": text,
        "desc": desc,
        "palette": list(palette),
    }
```

`metadata.palette` is a list of hex strings, the shape `CreateBoundingBoxes`
uses. Palettes are run through `normalize_palette`
(`comfy_extras/color_util.py`) so entries come back as clean uppercase hex.
`render_preview` draws `palette[0]` as the outline and `palette[:5]` as swatches;
`build_elements` copies `palette[:5]` into `color_palette`.

`type` is inferred from `text` because `build_elements` only emits `text` when
the region type is `"text"`.

Each node's `execute` ends in the same three lines via `render_boxes`, which
delegates to core.

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
means the color survives into `CreateBoundingBoxes` and
`LayersFromBoundingBoxes` via `normalize_incoming_boxes`. Geometry-only
consumers (`DrawBBoxes`, `CropByBBoxes`) ignore the extra key. This is stricter
than core, whose own `bboxes` output drops the palette.

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
- An empty `color_palette` in an advanced row falls back to a random color for
  that box. The simple node does not: with `random_color` off and an empty
  palette the boxes simply carry none.
- An advanced row with every field falsy is skipped. This also drops the
  all-`None` gap rows `_DynamicGroup` inserts for unsubmitted indices.
- `randint(0, width - w)` keeps boxes fully inside the canvas, so large boxes
  cluster toward the left/top. Intended.

## Testing

`tests/test_random_bboxes.py` loads `bbox_nodes.py` standalone and exercises the
pure helpers and both `execute`s:

- seed determinism and seed variation
- `count` honored; boxes stay inside the canvas; sizes within the fraction range
- `random_color` off drops the palette, or applies `color_palette` when set
- `box_metadata` type inference from `text`
- advanced rows keep their text/desc/color_palette, palettes are normalized,
  an empty palette gets a random color, empty rows are skipped
- inverted size range, zero size, size above 1.0, non-positive count
- both schemas define

## Non-goals

- Deriving canvas size from an image input (core `CreateBoundingBoxes` has the
  same limitation).
- Separate x/y position bounds, or per-row size and position in the advanced
  node.
- Making either node an output node that writes preview files.

## Verification

From the ComfyUI repo root:

```
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```

All tests pass, and the pack still loads as a custom node.
