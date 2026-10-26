# Describe Bounding Boxes — Design

## Goal

Detection nodes (`RTDETR_detect`, `MediaPipeFaceLandmarker`, `SAM3_Detect`) put
`label` and `score` as top-level keys on each box dict, but
`CreateBoundingBoxes` only reads descriptions from `metadata.desc`. Feeding
detections straight into the canvas therefore shows the boxes without any text.
Add a small adapter node that copies the detection labels into
`metadata.desc`, so detection boxes display in `CreateBoundingBoxes` with their
description.

## Background

- `boxes_to_regions` (`comfy_extras/nodes_bounding_boxes.py`) builds the canvas
  region from `metadata.type` / `text` / `desc` / `palette` only; top-level
  `label` is ignored.
- `normalize_incoming_boxes` keeps `metadata` but drops every other extra key,
  so `label`/`score` do not survive into the editor on their own.
- `RandomBoundingBoxes` already produces the metadata shape
  (`box_metadata()`), so the adapter targets exactly that shape.

## Behavior

- Each box's `label` becomes `metadata.desc`, e.g. RT-DETR `person` shows as
  `person` in the canvas and flows into Ideogram `elements` as the element
  `desc`.
- A box that already has a non-empty `metadata.desc` keeps it — hand-written
  descriptions win over detection labels.
- Other metadata keys (`type`, `text`, `palette`) survive untouched.
- `score` stays on the box dict but is not written into `desc`: the description
  feeds prompt elements, where a confidence number is noise. Geometry-only
  consumers (DrawBBoxes) keep reading `score` from the top level.
- Input shape is preserved: `dict`, `list[dict]`, or `list[list[dict]]` per
  frame, matching what the detection nodes emit.

## Node definition

Added to `bbox_nodes.py`, registered in `__init__.py`. Category `utilities`,
the same as core `CreateBoundingBoxes`.

### DescribeBoundingBoxes

- **node_id**: `DescribeBoundingBoxes`
- **display_name**: "Describe Bounding Boxes"
- **inputs**: `bboxes` (`io.BoundingBox`, `force_input=True`)
- **outputs**: `bboxes` (`io.BoundingBox`) — same boxes with `metadata.desc`
  filled

No preview or elements outputs: the point is to feed `CreateBoundingBoxes`,
which already renders both.

## Edge cases

- Box without `label` → `metadata.desc` is `""`, matching `box_metadata()`.
- Box with empty-string `label` → treated as no label.
- Non-dict entries inside box lists pass through unchanged.
- Empty list / `None` input passes through as-is, so the node never blocks a
  graph that upstream left empty.

## Testing

`tests/test_describe_bboxes.py` loads `bbox_nodes.py` standalone:

- label becomes desc; existing desc and other metadata keys survive
- no-label boxes get an empty desc
- all three input shapes map through without reshaping
- described boxes actually reach the canvas description
  (`boxes_to_regions` sees `desc == "person"`)
- `execute` round-trips per-frame input; schema defines

## Non-goals

- Changing core `boxes_to_regions` / `normalize_incoming_boxes` to read
  top-level `label` (that would change `CreateBoundingBoxes` behavior for
  everyone).
- Writing `score` into the description.
- Mask → bbox conversion.

## Verification

From the ComfyUI repo root:

```
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```

108 tests pass.
